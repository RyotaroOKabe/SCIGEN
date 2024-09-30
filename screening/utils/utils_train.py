import os
import torch
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
import math
import time
from utils.utils_plot import generate_dataframe, plot_bands, plot_gphonons, plot_loss, plot_test_loss
from utils.utils_model import get_spectra
from config_file import palette, seedn 
torch.autograd.set_detect_anomaly(True)

def evaluate(model, dataloader, loss_fn, device, option='kmvn'):
    model.eval()
    loss_cumulative = 0.
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            if option in ['kmvn']:
                Hs, shifts = model(d)
                output = get_spectra(Hs, shifts, d.qpts)
            else:
                output = model(d)
            loss = loss_fn(output, d.y).cpu()
            loss_cumulative += loss.detach().item()
    return loss_cumulative/len(dataloader)

def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))

def train(model,
          opt,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter,
          scheduler,
          device,
          batch_size,
          k_fold,
          option='kmvn',
          factor=1000,
          conf_dict=None):
    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    record_lines = []

    try:
        print('Use model.load_state_dict to load the existing model: ' + run_name + '.torch')
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        print('There is no existing model')
        results = {}
        history = []
        s0 = 0
    else:
        print('Use torch.load to load the existing model: ' + run_name + '.torch')
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1

    tr_sets = torch.utils.data.random_split(tr_set, tr_nums)
    te_loader = DataLoader(te_set, batch_size = batch_size)
    for step in range(max_iter):
        k = step % k_fold
        tr_loader = DataLoader(torch.utils.data.ConcatDataset(tr_sets[:k] + tr_sets[k+1:]), batch_size = batch_size, shuffle=True)
        va_loader = DataLoader(tr_sets[k], batch_size = batch_size)
        model.train()
        N = len(tr_loader)
        for i, d in enumerate(tr_loader):
            start = time.time()
            d.to(device)
            if option in ['kmvn']:
                Hs, shifts = model(d)
                output = get_spectra(Hs, shifts, d.qpts)
            else:
                output = model(d)
            loss = loss_fn(output, d.y).cpu()
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'num {i+1:4d}/{N}, loss = {loss}, train time = {time.time() - start}', end = '\r')

        end_time = time.time()
        wall = end_time - start_time
        print(wall)
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, va_loader, loss_fn, device, option)
            train_avg_loss = evaluate(model, tr_loader, loss_fn, device, option)

            history.append({
                            'step': s0 + step,
                            'wall': wall,
                            'batch': {
                                    'loss': loss.item(),
                                    },
                            'valid': {
                                    'loss': valid_avg_loss,
                                    },
                            'train': {
                                    'loss': train_avg_loss,
                                    },
                           })

            results = {
                        'history': history,
                        'state': model.state_dict()
                      }
            
            if conf_dict is not None:
                results['conf_dict'] = conf_dict

            print(f"Iteration {step+1:4d}   " +
                  f"train loss = {train_avg_loss:8.20f}   " +
                  f"valid loss = {valid_avg_loss:8.20f}   " +
                  f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            save_name = f'./models/{run_name}'
            with open(save_name + '.torch', 'wb') as f:
                torch.save(results, f)

            record_line = '%d\t%.20f\t%.20f'%(step,train_avg_loss,valid_avg_loss)
            record_lines.append(record_line)
            plot_loss(history, save_name + '_loss')
            plot_test_loss(model, te_loader, loss_fn, device, save_name + '_loss_test', option)

            df_tr = generate_dataframe(model, tr_loader, loss_fn, device, option, factor)
            df_te = generate_dataframe(model, te_loader, loss_fn, device, option, factor)
            if option == 'kmvn':
                fig_tr = plot_bands(df_tr, header=save_name, title='train', n=6, m=2, palette=palette, formula=True, seed=seedn)
                fig_te = plot_bands(df_te, header=save_name, title='test', n=6, m=2, palette=palette, formula=True, seed=seedn)
            elif option in ['mvn', 'vvn']:
                fig_tr = plot_gphonons(df_tr, header=save_name, title='train', n=6, m=2, lwidth=0.5, windowsize=(4, 2), palette=palette, formula=True, seed=seedn)
                fig_te = plot_gphonons(df_te, header=save_name, title='test', n=6, m=2, lwidth=0.5, windowsize=(4, 2), palette=palette, formula=True, seed=seedn)
        text_file = open(save_name + ".txt", "w")
        for line in record_lines:
            text_file.write(line + "\n")
        text_file.close()

        if scheduler is not None:
            scheduler.step()

def load_model(model_class, model_file, device):
    """
    Loads a pre-trained model, its weights, and hyperparameters.
    
    Args:
        model_class: The class of the model to be instantiated.
        model_file: Path to the saved model file.
        device: The device on which to load the model.
        
    Returns:
        model: The model with loaded weights.
        conf_dict: A dictionary of hyperparameters (if available).
        history: A list of training history (if available).
        int: The starting step (for resuming training).
    """
    if os.path.exists(model_file):
        print(f"Loading model from: {model_file}")
        checkpoint = torch.load(model_file)
        
        # Extract hyperparameters and initialize the model
        # conf_dict = checkpoint.get('conf_dict', {})
        conf_dict = checkpoint.get('conf_dict', checkpoint)#['conf_dict']
        # print('conf_dict: ', conf_dict)
        model = model_class(**conf_dict)  # Initialize the model with the saved hyperparameters
        model.load_state_dict(checkpoint['state'])
        model.to(device)

        # Extract history and step number
        history = checkpoint.get('history', [])
        s0 = history[-1]['step'] + 1 if history else 0

        return model, conf_dict, history, s0
    else:
        raise FileNotFoundError(f"No model found at {model_file}")