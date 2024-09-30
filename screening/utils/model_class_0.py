import torch
from torch.nn.modules.loss import _Loss
from torch_scatter import scatter
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader
import numpy as np
from e3nn.o3 import Irrep, Irreps, spherical_harmonics, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import Gate, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import r2_score
# from utils.plot_data import loss_plot, loss_test_plot
from utils.utils import *
from os.path import join
import wandb


class MSELoss_Norm(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction: str = 'mean') -> None:
        super(MSELoss_Norm, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        A, B = torch.max(torch.abs(target)).reshape(1, -1), torch.tensor(2e-1).reshape(1, -1)
        denominator = torch.max(torch.concat((A, B), dim=-1))
        return torch.sum(torch.pow(torch.abs(input - target)/denominator, 2)) \
               /torch.numel(target)
               
def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class GraphConvolution(torch.nn.Module):
    def __init__(self,
                 irreps_in,
                 irreps_node_attr,
                 irreps_edge_attr,
                 irreps_out,
                 number_of_basis,
                 radial_layers,
                 radial_neurons):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_out = Irreps(irreps_out)

        self.linear_input = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)
        self.linear_mask = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        
        irreps_mid = []
        instructions = []
        for i, (mul, irrep_in) in enumerate(self.irreps_in):
            for j, (_, irrep_edge_attr) in enumerate(self.irreps_edge_attr):
                for irrep_mid in irrep_in * irrep_edge_attr:
                    if irrep_mid in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, irrep_mid))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for (i_1, i_2, i_out, mode, train) in instructions]

        self.tensor_edge = TensorProduct(self.irreps_in,
                                         self.irreps_edge_attr,
                                         irreps_mid,
                                         instructions,
                                         internal_weights=False,
                                         shared_weights=False)
        
        self.edge2weight = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [self.tensor_edge.weight_numel], torch.nn.functional.silu)
        self.linear_output = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self,
                node_input,
                node_attr,
                node_deg,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedded,
                numb, n):

        node_input_features = self.linear_input(node_input, node_attr)
        node_features = torch.div(node_input_features, torch.pow(node_deg, 0.5))

        node_mask = self.linear_mask(node_input, node_attr)

        edge_weight = self.edge2weight(edge_length_embedded)
        edge_features = self.tensor_edge(node_features[edge_src], edge_attr, edge_weight)

        node_features = scatter(edge_features, edge_dst, dim = 0, dim_size = node_features.shape[0])
        node_features = torch.div(node_features, torch.pow(node_deg, 0.5))

        node_output_features = self.linear_output(node_features, node_attr)

        node_output = node_output_features

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        mask = self.linear_mask.output_mask
        c_x = (1 - mask) + c_x * mask
        return c_s * node_mask + c_x * node_output



class GraphNetwork(torch.nn.Module):
    def __init__(self,
                 mul,
                 irreps_out,
                 lmax,
                 nlayers,
                 number_of_basis,
                 radial_layers,
                 radial_neurons,
                 node_dim,
                 node_embed_dim,
                 input_dim,
                 input_embed_dim):
        super().__init__()
        
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim)+'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim)+'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        act = {1: torch.nn.functional.silu,
               -1: torch.tanh}
        act_gates = {1: torch.sigmoid,
                     -1: torch.tanh}

        self.layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps_in, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in,
                                    self.irreps_node_attr,
                                    self.irreps_edge_attr,
                                    gate.irreps_in,
                                    number_of_basis,
                                    radial_layers,
                                    radial_neurons)

            irreps_in = gate.irreps_out

            self.layers.append(CustomCompose(conv, gate))
        #last layer: conv
        self.layers.append(GraphConvolution(irreps_in,
                        self.irreps_node_attr,
                        self.irreps_edge_attr,
                        self.irreps_out,
                        number_of_basis,
                        radial_layers,
                        radial_neurons,)
                        )

        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype = torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype = torch.float64)

    def forward(self, data):
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        # edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis = 'gaussian', cutoff = False)
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'][0], self.number_of_basis, basis = 'gaussian', cutoff = False)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization = 'component')
        edge_attr = edge_sh
        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        n=None
        count = 0
        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, n)
            count += 1
        # x = x.reshape((1, -1))
        # print('x.shape: ', x.shape)
        x = x.mean(dim=0)#.reshape(-1, len(self.irreps_out))
        return x



class GraphNetworkClassifier(torch.nn.Module):
    def __init__(self,
                 mul,
                 irreps_out,
                 lmax,
                 nlayers,
                 number_of_basis,
                 radial_layers,
                 radial_neurons,
                 node_dim,
                 node_embed_dim,
                 input_dim,
                 input_embed_dim):
        super().__init__()
        
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim)+'x0e')
        self.irreps_node_attr = Irreps(str(node_embed_dim)+'x0e')
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps([(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        act = {1: torch.nn.functional.silu,
               -1: torch.tanh}
        act_gates = {1: torch.sigmoid,
                     -1: torch.tanh}

        self.layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps_in, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in,
                                    self.irreps_node_attr,
                                    self.irreps_edge_attr,
                                    gate.irreps_in,
                                    number_of_basis,
                                    radial_layers,
                                    radial_neurons)

            irreps_in = gate.irreps_out

            self.layers.append(CustomCompose(conv, gate))
        #last layer: conv
        self.layers.append(GraphConvolution(irreps_in,
                        self.irreps_node_attr,
                        self.irreps_edge_attr,
                        self.irreps_out,
                        number_of_basis,
                        radial_layers,
                        radial_neurons,)
                        )

        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype = torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype = torch.float64)
        self.binary_classifier = torch.nn.Linear(self.irreps_out.dim, 1, dtype=torch.float64)   #!


    def forward(self, data):
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        # edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis = 'gaussian', cutoff = False)
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'][0], self.number_of_basis, basis = 'gaussian', cutoff = False) #!
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization = 'component')
        edge_attr = edge_sh
        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        n=None
        count = 0
        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded, numb, n)
            count += 1
        # x = x.reshape((1, -1))
        # print('x.shape: ', x.shape)
        #! x = x.mean(dim=0)#.reshape(-1, len(self.irreps_out))
        x = scatter_mean(x, data.batch, dim=0)
        x = self.binary_classifier(x)
        x = torch.sigmoid(x)    #.unsqueeze(0)  # Apply sigmoid activation
        return x


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))


def get_epoch_lists(total_epochs, num_folds=5, ep_per_fold=1):
    # Initialize lists to store performance metrics for each fold
    epset = total_epochs//(num_folds*ep_per_fold)
    epset_frac = total_epochs%(num_folds*ep_per_fold)
    ep_lists = []
    for i in range(epset):
        ep_list = []
        for j in range(num_folds):
            ep_list.append(ep_per_fold)
        ep_lists.append(ep_list)
    total_sum = sum([sum(sublist) for sublist in ep_lists])
    if total_sum < total_epochs:
        ep_list = []
        while sum(ep_list)<epset_frac:
            ep_list.append(ep_per_fold)
        ep_lists.append(ep_list)

    print(ep_lists)
    total_sum = sum([sum(sublist) for sublist in ep_lists])
    print("Total epochs:", total_sum)
    return ep_lists



import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model_classifier(model, train_loader, criterion, device, optimizer):
    model.train()
    total_loss, total_accuracy = 0, 0
    for data in train_loader:
        optimizer.zero_grad()
        outputs = model(data.to(device))  # Ensure outputs are logits
        y_true = data.y.to(device).float()  # Ensure targets are the correct shape and type
        y_pred = (outputs >= 0.5).float()
        # print("outputs: ", outputs)
        # print('y_true: ', y_true)
        # print('y_pred: ', y_pred)
        # loss = criterion(outputs, y_true)
        loss = criterion(outputs, y_true).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Convert logits to binary predictions
        # preds = torch.sigmoid(outputs).round()
        # total_accuracy += accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())
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
            # print("outputs: ", outputs)
            # print('y_true: ', y_true)
            # loss = criterion(outputs, y_true)
            loss = criterion(outputs, y_true).mean()
            total_loss += loss.item()
            # preds = torch.sigmoid(outputs).round()
            # total_accuracy += accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())
            total_accuracy += accuracy_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    return avg_loss, avg_accuracy

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_classifier(model,
          optimizer,
          tr_set,
          tr_nums,
          te_set,
          loss_fn,
          run_name,
          max_iter, #!
        #   ep_lists,   #!
          k_fold,   #!
          scheduler,
          device,
          batch_size,
          ):
    model.to(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    te_loader = DataLoader(te_set, batch_size = batch_size)
    # Perform k-fold cross validation
    # fold_performance = []
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
    
    for step in range(max_iter):
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

            df_tr = generate_dataframe(model, tr_loader, loss_fn, None, device)
            df_te = generate_dataframe(model, te_loader, loss_fn, None, device)
            dfs = {'train': df_tr, 'test': df_te}
            num_dfs = len(dfs)
            fig, axs = plt.subplots(1, num_dfs, figsize=(6*num_dfs, 6))  # Adjust figure size as needed
            for i, (k, df_) in enumerate(dfs.items()):
                reals, preds = list(df_['real']), list(df_['pred'])
                # print('reals: ', type(reals[0]), reals)
                # print('preds: ', type(preds[0]), preds)
                cm = confusion_matrix(reals, preds)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[i])
                # Setting titles and labels
                axs[i].set_title(f'{k} Confusion Matrix')
                axs[i].set_xlabel('Predicted labels')
                axs[i].set_ylabel('True labels')
                # Optionally, include metrics like accuracy or F1-score in the subplot
                accuracy = accuracy_score(reals, preds)
                axs[i].annotate(f'Accuracy = {accuracy:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
            fig.suptitle(f"{run_name} Confusion Matrices")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the main title
            fig.savefig(join('./models', run_name + '_cm_subplot.png'))
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


def generate_dataframe(model, dataloader, loss_fn, scaler, device, classification=True):
    # model.eval()  # Ensure model is in evaluation mode
    start_time = time.time()
    with torch.no_grad():
        # Initialize an empty dataframe with columns
        df = pd.DataFrame(columns=['id', 'name', 'loss', 'real', 'pred', 'output'])
        for d in dataloader:
            d.to(device)
            outputs = model(d).cpu()
            y_trues = d.y.cpu()  # Assuming your dataloader provides a 'target' key

            losses = loss_fn(outputs, y_trues).cpu().numpy()  # Compute loss for the batch
            # Scaling and classification adjustments for the whole batch
            reals = y_trues.numpy()
            outs = outputs.numpy()
            preds = outputs.numpy()
            if scaler is not None:
                reals = scaler.inverse_transform(reals)
                preds = scaler.inverse_transform(preds)
            if classification:
                preds = (preds >= 0.5)

            # Assuming 'id' and 'symbol' are lists of ids and symbols for each instance in the batch
            batch_ids = d['id']
            batch_names = d['symbol']
            # Create a temporary dataframe for the current batch
            for i in range(len(batch_ids)):
                temp_df = pd.DataFrame({
                    'id': [batch_ids[i]],
                    'name': [batch_names[i]],
                    'loss': [losses[i]],
                    'real': [reals[i]],
                    'pred': [preds[i]],
                    'output': [outs[i]]
                })
                df = pd.concat([df, temp_df], ignore_index=True)
    return df

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
