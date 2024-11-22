#%%
import os
from os.path import join
import time
import json
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
import wandb

from utils.data_mp import mp_struct_prop_stable_filter
from utils.data import Dataset_Cls, augment_data_diffuse, count_elements
from utils.model_class import GraphNetworkClassifier
from utils.model_train import train_classifier
from utils.output import generate_dataframe
from utils.plot_data import plot_confusion_matrices
from utils.record import log_buffer, logger
from utils.common import make_dict
from config_eval import api_key, matbench_data_dir, model_dir, data_dir, seedn

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
file_name = os.path.basename(__file__)

#%% params (data)
load_saved_data = True
save_file = join(data_dir, 'mbd_mp.pkl')
r_max = 4
tr_ratio = 0.9
batch_size = 16
nearest_neighbor = True
cut_data = None
epsilon=1e-3
scaler = None #LogShiftScaler(epsilon, -3.1, 2.2)
target = 'ehull'    # ['ehull', 'f_energy']
descriptor = 'ie'   # ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']
stable_threshold = 0.1   # ehull threshold [eV]
change_cells=False
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
loss_fn = nn.BCEWithLogitsLoss(reduce=False) 
loss_fn_name = loss_fn.__class__.__name__
lr = 0.005 
weight_decay = 0.05 
schedule_gamma = 0.96 

conf_dict = make_dict([file_name,run_name,model_dir,data_dir,
                    r_max, tr_ratio,cut_data,nearest_neighbor,batch_size,target,descriptor,save_file,
                    stable_threshold,change_cells,diff_factor,num_diff, diff_factor_small, num_diff_small, max_iter, 
                    k_fold, lmax, mul, nlayers, number_of_basis,radial_layers, radial_neurons, node_dim,
                    node_embed_dim,input_dim,input_embed_dim,out_dim,irreps_out,loss_fn_name, 
                    lr,weight_decay,schedule_gamma,seedn])

logger.info('Configuration')
for k, v in conf_dict.items():
    logger.info(f'{k}: {v}')

#%%
# load data
if load_saved_data:
    data = pd.read_pickle(save_file)  
    logger.info(f"Loaded data from {save_file}")
else: 
    mbd_data_dir = matbench_data_dir 
    mbd_mp_path = join(mbd_data_dir, 'mp')
    mbd_wbm_path = join(mbd_data_dir, 'wbm')
    if 'wb' in save_file:
        print('download and process matbench-discovert WBM dataset!')
        wbm_cse_path = join(mbd_wbm_path, '2022-10-19-wbm-computed-structure-entries.json.bz2')
        df_wbm = pd.read_json(wbm_cse_path) 
        df_wbm['n_elements'] = df_wbm.apply(count_elements, axis=1)
        wbm_summary_path = join(mbd_wbm_path, "2023-12-13-wbm-summary.csv.gz")
        df_wbm_summary = pd.read_csv(wbm_summary_path) 
        df_wbm['structure'] = df_wbm['computed_structure_entry'].progress_map(lambda x: Atoms(AseAtomsAdaptor().get_atoms(ComputedStructureEntry.from_dict(x).structure)))
        df_wbm['ehull'] = df_wbm['material_id'].progress_map(lambda x: df_wbm_summary[df_wbm_summary['material_id']==x]['e_above_hull_wbm'].item())
        df_wbm.rename(columns={'material_id': 'mpid'}, inplace=True)
        data = df_wbm
        logger.info(f"Downloaded and processed matbench-discovert WBM dataset!")
    elif 'mp' in save_file:
        print('download and process matbench-discovert MP dataset!')
        data_path = join(mbd_mp_path, "2023-02-07-mp-computed-structure-entries.json.gz")
        df_mp = pd.read_json(data_path)
        mp_summary_path = join(mbd_mp_path, '2023-01-10-mp-energies.csv.gz')
        df_mp_summary = pd.read_csv(mp_summary_path, na_filter=False)
        df_mp['structure'] = df_mp['entry'].progress_map(lambda x: Atoms(AseAtomsAdaptor().get_atoms(ComputedStructureEntry.from_dict(x).structure)))
        df_mp['ehull'] = df_mp['material_id'].progress_map(lambda x: df_mp_summary[df_mp_summary['material_id']==x]['energy_above_hull'].item())
        df_mp.rename(columns={'material_id': 'mpid'}, inplace=True)
        data = df_mp
        logger.info(f"Downloaded and processed matbench-discovert MP dataset!")
    with open(save_file, 'wb') as file:
        pkl.dump(data, file)
    logger.info(f"Saved data to {save_file}")
    
data['stable'] = np.where(data[target] < stable_threshold, 1, 0)
logger.info(f"Data loaded: {len(data)}")
logger.info(f"Stable: {len(data[data['stable']==True])}, Unstable: {len(data[data['stable']==False])}")

data = data[data['stable']==True]
if cut_data is not None:
    data = data.sample(n=cut_data, random_state=seedn)  # You can adjust the random_state for reproducibility
    data = data.reset_index(drop=True)
    logger.info(f"Cut the data to {cut_data} samples")
# Define scale factors
if change_cells:
    data = augment_data_diffuse(data, num_diff, diff_factor, num_diff_small, diff_factor_small)
    logger.info(f"Augmented data with {num_diff} large and {num_diff_small} small cell changes")


#%% 
# process data
dataset = Dataset_Cls(data, r_max, target, descriptor, scaler, nearest_neighbor)  # dataset
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
model = GraphNetworkClassifier(mul,
                     irreps_out,
                     lmax,
                     nlayers,
                     number_of_basis,
                     radial_layers,
                     radial_neurons,
                     node_dim,
                     node_embed_dim,
                     input_dim,
                     input_embed_dim)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print('number of parameters: ', num_params)

optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = schedule_gamma)

#%% 
# WandB initialization
wandb.init(project='gnn_eval', entity='ryotarookabe', name=run_name + f'_{target}')
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
          )
logger.info(f"Finished training {run_name}")

#%% 
# Set to evaluation mode if using for inference
model = model.eval()  
# Generate Data Loader
tr_loader = DataLoader(tr_set, batch_size = batch_size)
te1_loader = DataLoader(te_set, batch_size = batch_size)
# Generate Data Frame
df_tr = generate_dataframe(model, tr_loader, loss_fn, scaler, device)
df_te = generate_dataframe(model, te1_loader, loss_fn, scaler, device)
logger.info(f"Generated dataframes for {run_name}: {len(df_tr)} training and {len(df_te)} testing samples")

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