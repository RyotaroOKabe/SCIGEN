import torch
from torch_geometric.loader import DataLoader
import numpy as np
import math
import time
from sklearn.metrics import accuracy_score
from os.path import join
import wandb
from tqdm import tqdm
from utils.output import generate_dataframe
from utils.plot_data import plot_confusion_matrices
torch.autograd.set_detect_anomaly(True)


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))


def train_model_classifier(model, train_loader, criterion, device, optimizer):
    model.train()
    total_loss, total_accuracy = 0, 0
    for data in train_loader:
        optimizer.zero_grad()
        outputs = model(data.to(device))  # Ensure outputs are logits
        y_true = data.y.to(device).float()  # Ensure targets are the correct shape and type
        y_pred = (outputs >= 0.5).float()
        # print(f"outputs shape: {outputs.shape}, y_true shape: {y_true.shape}")  
        loss = criterion(outputs, y_true).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += accuracy_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    return avg_loss, avg_accuracy

def evaluate_classifier(model, loader, criterion, device):
    start_time = time.time()
    model.eval()
    total_loss, total_accuracy = 0, 0
    with torch.no_grad():
        for data in loader:
            outputs = model(data.to(device))
            y_true = data.y.to(device).float()
            y_pred = (outputs >= 0.5).float()
            loss = criterion(outputs, y_true).mean()
            total_loss += loss.item()
            total_accuracy += accuracy_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    return avg_loss, avg_accuracy


def train_classifier(model,
          optimizer,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter,
          k_fold,  
          scheduler,
          device,
          batch_size,
          num_classes=2,
          ):
    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    te_loader = DataLoader(te_set, batch_size = batch_size)
    num = len(tr_set)

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
    
    # for step in range(max_iter):
    for step in tqdm(range(max_iter), total=max_iter, desc='Training', unit='step'):
        k = step % k_fold
        tr_loader = DataLoader(torch.utils.data.ConcatDataset(tr_sets[:k] + tr_sets[k+1:]), batch_size = batch_size, shuffle=True)
        va_loader = DataLoader(tr_sets[k], batch_size = batch_size)
        train_loss, train_accuracy = train_model_classifier(model, tr_loader, loss_fn, device, optimizer)

        # print(f'num {i+1:4d}/{N}, loss = {loss}, train time = {time.time() - start}', end = '\r')

        end_time = time.time()
        wall = end_time - start_time
        # print(wall)
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            val_loss, val_accuracy = evaluate_classifier(model, va_loader, loss_fn, device)
            # valid_avg_loss = evaluate(model, va_loader, loss_fn, device, option)
            # train_avg_loss = evaluate(model, tr_loader, loss_fn, device, option)

            # df_te = generate_dafaframe(model, te_loader, loss_fn, None, device)
            # fig, ax = plt.subplots(1,1, figsize=(6, 6))
            # reals, preds = np.concatenate(list(df_te['real'])), np.concatenate(list(df_te['pred']))
            # preds = preds >= 0.5#.float()

            df_tr = generate_dataframe(model, tr_loader, loss_fn, None, device, num_classes)
            df_te = generate_dataframe(model, te_loader, loss_fn, None, device, num_classes)
            dfs = {'train': df_tr, 'test': df_te}
            
            fig = plot_confusion_matrices(dfs, run_name, save_path=join('./models', run_name + '_cm_subplot.png'))
            wandb.log({"confusion_matrix": wandb.Image(fig)})
            for k in df_te.keys():  #!
                df_tr[k] = df_tr[k].apply(convert_array_to_float)
                df_te[k] = df_te[k].apply(convert_array_to_float)
            log_dataframe_to_wandb(df_tr, "Training Data")
            log_dataframe_to_wandb(df_te, "Test Data")
            
            run_time = time.time() - start_time
            print(f'(Step {step}: Train Loss: {train_loss}, Validation Loss: {val_loss}')

            history.append({
                            'step': s0 + step,
                            'wall': wall,
                            # 'batch': {
                            #         'loss': loss.item(),
                            #         },
                            'valid': {
                                    'loss': val_loss,
                                    },
                            'train': {
                                    'loss': train_loss,
                                    },
                           })

            results = {
                        'history': history,
                        'state': model.state_dict()
                      }

            # print(f"Iteration {step+1:4d}   " +
            #       f"train loss = {train_avg_loss:8.20f}   " +
            #       f"valid loss = {valid_avg_loss:8.20f}   " +
            #       f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            with open(f'./models/{run_name}.torch', 'wb') as f:
                torch.save(results, f)

            # model_path = f'./models/{run_name}.torch'
            # with open(model_path, 'wb') as f:
            #     torch.save(model, f)  
            wandb.log({
                "epoch": step,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })


        if scheduler is not None:
            scheduler.step()


def convert_array_to_float(x):
    if isinstance(x, np.ndarray) and sum(x.shape) < 2:
        return x.item()  # Or x[0] to convert the single-item array to a float
    else:
        return x
    
def dataframe_to_wandb_table(df):
    # Convert a pandas DataFrame to a wandb Table
    cols = df.columns.tolist()
    wandb_table = wandb.Table(columns=cols)
    for _, row in df.iterrows():
        wandb_table.add_data(*row.values)
    return wandb_table

# Example usage within your training loop or evaluation section
def log_dataframe_to_wandb(df, table_name):
    wandb_table = dataframe_to_wandb_table(df)
    wandb.log({table_name: wandb_table})
