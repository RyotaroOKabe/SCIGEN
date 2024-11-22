#%%
import os
from os.path import join
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import json
import wandb

from utils.data import Dataset_Cls, augment_data_diffuse
from utils.data_mp import mp_struct_prop_stable_filter
from utils.model_class_mag import GraphNetworkClassifierMag
from utils.model_train import train_classifier
from utils.output import generate_dataframe
from utils.plot_data import plot_confusion_matrices
from utils.record import log_buffer, logger
from utils.common import make_dict, magnetic_atoms
from config_eval import api_key, model_dir, data_dir, seedn

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
file_name = os.path.basename(__file__)


#%% params (data)
binary_classification = True
mp_file = './data/mp_full_prop.pkl'
r_max = 4
tr_ratio = 0.9
batch_size = 16
nearest_neighbor = True
cut_data = None
epsilon=1e-3
scaler = None #LogShiftScaler(epsilon, -3.1, 2.2)
target = 'mag'    # ['ehull', 'f_energy']
descriptor = 'ie'   # ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']
stable_threshold = 0.1    # ehull threshold [eV]
natm_max = 40
change_cells=True
num_diff = 2
diff_factor = {'lattice': 0.05, 'frac': 0.05}
num_diff_small = 1
diff_factor_small = {'lattice': 0.01, 'frac': 0.01}

#%% params (model)
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
max_iter = 150    # total eppochs for training 
k_fold = 5
lmax = 2
mul = 8
nlayers = 1
number_of_basis = 10
radial_layers = 1
radial_neurons = 100
node_dim = 118
node_embed_dim = 32
input_dim = 118
input_embed_dim = 32
out_dim = 64
irreps_out = f'{out_dim}x0e'
if binary_classification: 
    num_classes = 2
    loss_fn = nn.BCEWithLogitsLoss(reduce=False)    #nn.MSELoss()  #MSELoss_Norm()    #nn.MSELoss()
else: 
    num_classes = 3
    loss_fn = nn.CrossEntropyLoss(reduce=False)
loss_fn_name = loss_fn.__class__.__name__
lr = 0.005 
weight_decay = 0.05 
schedule_gamma = 0.96 

conf_dict = make_dict([file_name,run_name,model_dir,data_dir,
                    r_max, tr_ratio,cut_data,nearest_neighbor,batch_size,target,descriptor,
                    stable_threshold,natm_max,change_cells,diff_factor,num_diff, diff_factor_small, num_diff_small, max_iter, 
                    k_fold, lmax, mul, nlayers, number_of_basis,radial_layers, radial_neurons, node_dim,
                    node_embed_dim,input_dim,input_embed_dim,out_dim,irreps_out,loss_fn_name, binary_classification, 
                    lr,weight_decay,schedule_gamma,seedn])

logger.info('Configuration')
for k, v in conf_dict.items():
    logger.info(f'{k}: {v}')

#%%
# load data
data = pd.read_pickle('./data/mp_full_prop.pkl')  
data['natm'] = data['structure'].apply(lambda x: len(x))
data['formula'] = data['structure'].apply(lambda x: x.get_chemical_formula())
data['elements'] = data['structure'].apply(lambda x: list(set(x.get_chemical_symbols())))
data['nelem'] = data['elements'].apply(lambda x: len(x))
logger.info(f"Loaded data from {mp_file}, {len(data)} samples")
    
# remove materials which does not contain magnetic atoms
data = data[data['elements'].apply(lambda x: any([a in magnetic_atoms for a in x]))].reset_index(drop=True)
data = data[data['ehull'] <= stable_threshold].reset_index(drop=True)
data = data[data['natm'] <= natm_max].reset_index(drop=True)
data = data[data['mag'].isin(['NM', 'AFM', 'FM', 'FiM'])].reset_index(drop=True)
logger.info(f"Filtered data: {len(data)} samples")

if cut_data is not None and len(data) > cut_data:
    data = data.sample(n=cut_data, random_state=seedn)  # You can adjust the random_state for reproducibility
    data = data.reset_index(drop=True)
    logger.info(f"Cut the data to {cut_data} samples")

if binary_classification: 
    ORDER_ENCODE = {"NM": 0, "AFM": 1, "FM": 1, "FiM": 1}
else:     
    ORDER_ENCODE = {"NM": 0, "AFM": 1, "FM": 2, "FiM": 2}
data['label'] = data['mag'].apply(lambda x: ORDER_ENCODE[x])
logger.info(f"Label encoding: {ORDER_ENCODE}")

#%% 
# process data
dataset = Dataset_Cls(data, r_max, 'label', descriptor, scaler, nearest_neighbor)  # dataset
num = len(dataset)
print('dataset: ', num)
tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num - sum(tr_nums)
idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seedn)
tr_set, te_set = torch.utils.data.Subset(dataset, idx_tr), torch.utils.data.Subset(dataset, idx_te)
np.savetxt(join(model_dir, f'{run_name}_idx_tr.txt'), idx_tr, fmt='%d')
np.savetxt(join(model_dir, f'{run_name}_idx_te.txt'), idx_te, fmt='%d')
logger.info(f"Data split into {len(tr_set)} training and {len(te_set)} testing samples")

#%% 
# model
model = GraphNetworkClassifierMag(mul,
                     irreps_out,
                     lmax,
                     nlayers,
                     number_of_basis,
                     radial_layers,
                     radial_neurons,
                     node_dim,
                     node_embed_dim,
                     input_dim,
                     input_embed_dim,
                     num_classes)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print('number of parameters: ', num_params)

optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = schedule_gamma)

#%% 
# WandB initialization
wandb.init(project='gnn_eval', entity='ryotarookabe', name=run_name + '_mag')
wandb.config.update(conf_dict)
wandb.watch(model, log='all')   

# save the config file as json
logger.info(f"Save the config file as json: {run_name}")
with open(join(model_dir, f'{run_name}_config.json'), 'w') as f:
    json.dump(conf_dict, f)

# save the logger
logger.info(f"Start training {run_name}")
with open(join(model_dir, f"{run_name}.log"), 'w') as f:
    f.write(log_buffer.getvalue())

# Train the GNN model
train_classifier(model,
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
          num_classes,
          )
logger.info(f"Finished training {run_name}")

#%% 
# Set to evaluation mode if using for inference
model = model.eval()  
# Generate Data Loader
tr_loader = DataLoader(tr_set, batch_size = batch_size)
te1_loader = DataLoader(te_set, batch_size = batch_size)
# Generate Data Frame
df_tr = generate_dataframe(model, tr_loader, loss_fn, scaler, device, num_classes)
df_te = generate_dataframe(model, te1_loader, loss_fn, scaler, device, num_classes)


#%%
# Plot confusion matrix
dfs = {'train': df_tr, 'test': df_te}
fig = plot_confusion_matrices(dfs, run_name, save_path=join('./models', run_name + '_cm_subplot.png'))
logger.info(f"Plotted confusion matrices for {run_name}")
wandb.log({"confusion_matrix_final": wandb.Image(fig)})
wandb.finish()

# save the logger
logger.info(f"Finished training {run_name}")
with open(join(model_dir, f"{run_name}.log"), 'w') as f:
    f.write(log_buffer.getvalue())