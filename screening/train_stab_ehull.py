#%%
"""
https://www.notion.so/240402-use-Matbench-dataset-for-Ehull-classification-and-regression-09c1714ac7a5419998c4c7c9b26926e0
"""
# modules
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import time
import pickle as pkl
import matplotlib.pyplot as plt
import os
from os.path import join
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
palette = ['#43AA8B', '#F8961E', '#F94144']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
seedn=42

# files
from utils.data import *
from utils.model_class import *
from utils.plot_data import *
from config_file import *
file_name = os.path.basename(__file__)
print("File Name:", file_name)
#%%
# params (data)
load_saved_data = True
save_file = join(data_dir, 'mbd_mp.pkl')
# save_file = join(data_dir, 'mpdata_phdb.pkl')
print(save_file)
r_max = 4
tr_ratio = 0.9
batch_size = 16
cut_data = 90000
# k_folds = 5
# kfold = KFold(n_splits=k_folds, shuffle=True)
epsilon=1e-3
mu, sigma = -3.1, 2.2
scaler = None #LogShiftScaler(epsilon, mu, sigma)
target = 'ehull'    # ['ehull', 'f_energy']
descriptor = 'ie'   # ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']
stable_threshold = 0.1   #0.1
change_cells=False
num_diff = 2
diff_factor = {'lattice': 0.05, 'frac': 0.05}
num_diff0 = 1
diff_factor0 = {'lattice': 0.01, 'frac': 0.01}
print(target, descriptor)

#%% params (model)
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
max_iter = 200    # total eppochs for training 
k_fold = 5
# num_folds = 10
# ep_per_fold = max_iter//num_folds
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=seedn)
# ep_lists = get_epoch_lists(max_iter, num_folds, ep_per_fold)
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
loss_fn = nn.BCEWithLogitsLoss(reduce=False)    #nn.MSELoss()  #MSELoss_Norm()    #nn.MSELoss()
lr = 0.005 # random.uniform(0.001, 0.05) #0.005
weight_decay = 0.05 # random.uniform(0.01, 0.5) #0.05
schedule_gamma = 0.96 # random.uniform(0.85, 0.99) #0.96

print('\nmodel parameters')
print('run_name: ', run_name)
print('max iteration: ', max_iter)
print('max l: ', lmax)
print('multiplicity: ', mul)
print('convolution layer: ', nlayers)
print('cut off radius for neighbors: ', r_max)
print('radial distance bases: ', number_of_basis)
print('radial embedding layers: ', radial_layers)
print('radial embedding neurons per layer: ', radial_neurons)
print('node attribute dimension: ', node_dim)
print('node attribute embedding dimension: ', node_embed_dim)
print('input dimension: ', input_dim)
print('input embedding dimension: ', input_embed_dim)
print('irreduceble output representation: ', irreps_out)
print('loss function: ', loss_fn)
print('learning rate: ', lr)
print('weight decay: ', weight_decay)
print('learning rate scheduler: exponentialLR')
print('schedule factor: ', schedule_gamma)

#%%
import wandb
wandb.init(project='ehull_prediction', entity='ryotarookabe', name=run_name + '_'+ target)
wandb.config.update({
    'file_name':file_name,
    'r_max': r_max,
    'tr_ratio': tr_ratio,
    'cut_data': cut_data,
    'batch_size': batch_size,
    'target': target,
    'descriptor': descriptor,
    # 'lattice_change': lattice_change,
    'save_file': save_file,
    'stable_threshold': stable_threshold,
    'change_cell': change_cells,
    'diff_factor': diff_factor,
    'num_diff': num_diff,
    'diff_factor0': diff_factor0,
    'num_diff0': num_diff0,
    'max_iter': max_iter,
    'k_fold': k_fold,
    # 'num_folds': num_folds,
    # 'ep_per_fold': ep_per_fold,
    'lmax': lmax,
    'mul': mul,
    'nlayers': nlayers,
    'number_of_basis': number_of_basis,
    'radial_layers': radial_layers,
    'radial_neurons': radial_neurons,
    'node_dim': node_dim,
    'node_embed_dim': node_embed_dim,
    'input_dim': input_dim,
    'input_embed_dim': input_embed_dim,
    'out_dim': out_dim,
    'irreps_out': irreps_out,
    'learning_rate': lr,
    'weight_decay': weight_decay,
    'schedule_gamma': schedule_gamma,
})

#%%
# load data

if load_saved_data:
    mpdata = pd.read_pickle(save_file)  
else: 
    # from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase import Atoms
    mbd_data_dir = '/home/rokabe/data2/gnn/matbench-discovery/data'
    mbd_mp_path = join(mbd_data_dir, 'mp')
    mbd_wbm_path = join(mbd_data_dir, 'wbm')
    if 'wb' in save_file:
        print('download and process matbench-discovert WBM dataset!')
        wbm_cse_path = join(mbd_wbm_path, '2022-10-19-wbm-computed-structure-entries.json.bz2')
        df_wbm = pd.read_json(wbm_cse_path) # .set_index('material_id')

        def count_elements(row):
            composition = row['computed_structure_entry']['composition']
            return len(composition.keys())

        df_wbm['n_elements'] = df_wbm.apply(count_elements, axis=1)

        wbm_summary_path = join(mbd_wbm_path, "2023-12-13-wbm-summary.csv.gz")
        df_wbm_summary = pd.read_csv(wbm_summary_path) # .set_index('material_id')
        df_wbm['structure'] = df_wbm['computed_structure_entry'].progress_map(lambda x: Atoms(AseAtomsAdaptor().get_atoms(ComputedStructureEntry.from_dict(x).structure)))
        df_wbm['ehull'] = df_wbm['material_id'].progress_map(lambda x: df_wbm_summary[df_wbm_summary['material_id']==x]['e_above_hull_wbm'].item())
        df_wbm.rename(columns={'material_id': 'mpid'}, inplace=True)
        mpdata = df_wbm
    elif 'mp' in save_file:
        print('download and process matbench-discovert MP dataset!')
        data_path = join(mbd_mp_path, "2023-02-07-mp-computed-structure-entries.json.gz")
        df_mp = pd.read_json(data_path)
        mp_summary_path = join(mbd_mp_path, '2023-01-10-mp-energies.csv.gz')
        df_mp_summary = pd.read_csv(mp_summary_path, na_filter=False)
        df_mp['structure'] = df_mp['entry'].progress_map(lambda x: Atoms(AseAtomsAdaptor().get_atoms(ComputedStructureEntry.from_dict(x).structure)))
        df_mp['ehull'] = df_mp['material_id'].progress_map(lambda x: df_mp_summary[df_mp_summary['material_id']==x]['energy_above_hull'].item())
        df_mp.rename(columns={'material_id': 'mpid'}, inplace=True)
        mpdata = df_mp
    with open(save_file, 'wb') as file:
        pkl.dump(mpdata, file)
    
mpdata['stable'] = np.where(mpdata[target] < stable_threshold, 1, 0)
# mpdata['stable'] = mpdata[target].apply(lambda x: 1 if x < stable_threshold else 0)

# mpdata = mpdata[mpdata['stable']==True]
if cut_data is not None:
    mpdata = mpdata.sample(n=cut_data, random_state=seedn)  # You can adjust the random_state for reproducibility
    mpdata = mpdata.reset_index(drop=True)
# Define scale factors
if change_cells:
    new_rows = []  # To collect new rows
    for index, row in mpdata.iterrows():
        new_rows.append(row)
        if row['stable']:
            for j in range(num_diff):
                original_structure = row['structure']
                # Shrink and expand the structure
                diff_structure = diffuse_structure(original_structure, diff_factor)
                
                # Append new rows for the shrunk and expanded structures
                # Copy the original row and update the structure
                diff_row = row.copy()
                diff_row['structure'] = diff_structure
                diff_row['mpid'] = row['mpid'] + f'-d{j}'
                diff_row['stable'] = 0
                new_rows.append(diff_row)

            if diff_factor0 is not None:    #! add stable materiials with slight diffusion
                for j in range(num_diff0):
                    original_structure = row['structure']
                    # Shrink and expand the structure
                    diff_structure = diffuse_structure(original_structure, diff_factor0)
                    
                    # Append new rows for the shrunk and expanded structures
                    # Copy the original row and update the structure
                    diff_row = row.copy()
                    diff_row['structure'] = diff_structure
                    diff_row['mpid'] = row['mpid'] + f'-s{j}'
                    diff_row['stable'] = 1
                    new_rows.append(diff_row)

    # Create a new DataFrame from the list of new rows
    mpdata = pd.DataFrame(new_rows).reset_index(drop=True)
print('mpdata: ', len(mpdata))

# Now `new_mpdata` contains the original data but with two rows per original structure,
# one with the cell shrunk and one with the cell expanded.

# scalar = mpdata['ehull'].mean()
plt.hist(mpdata['ehull'], bins = 20)
plt.title(f'mpdata: {len(mpdata)}')


#%% 
# process data
dataset = Dataset_InputStability(mpdata, r_max, target, descriptor, scaler)  # dataset
num = len(dataset)
print('dataset: ', num)
k_fold = 5
tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num - sum(tr_nums)
# te_num = int(te_ratio*num)
idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seedn)

tr_set, te_set = torch.utils.data.Subset(dataset, idx_tr), torch.utils.data.Subset(dataset, idx_te)



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

optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = schedule_gamma)

print(model)
#%% 
# train
train_classifier(model,
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
          )

#%% plot results

model_name = run_name      # pre-trained model. Rename if you want to use the model you trained in the tabs above. 
model_file = join(model_dir, f'{model_name}.torch')
# model = torch.load(model_file)
model.load_state_dict(torch.load(model_file)['state'])
model = model.to(device)
model = model.eval()  # Set to evaluation mode if using for inference

#%%
# Generate Data Loader
tr_loader = DataLoader(tr_set, batch_size = batch_size)
te1_loader = DataLoader(te_set, batch_size = batch_size)

# Generate Data Frame
df_tr = generate_dataframe(model, tr_loader, loss_fn, scaler, device)
df_te = generate_dataframe(model, te1_loader, loss_fn, scaler, device)


#%%
# plot
dfs = {'train': df_tr, 'test': df_te}
num_dfs = len(dfs)
fig, axs = plt.subplots(1, num_dfs, figsize=(6*num_dfs, 6))  # Adjust figure size as needed
for i, (k, df_) in enumerate(dfs.items()):
    reals, preds = list(df_['real']), list(df_['pred'])
    print('reals: ', type(reals[0]), reals)
    print('preds: ', type(preds[0]), preds)
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

wandb.log({"confusion_matrix_final": wandb.Image(fig)})
wandb.finish()
