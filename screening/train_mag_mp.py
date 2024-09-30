#%%
"""
https://www.notion.so/240812-magnetic-ordering-check-a8eee7b24f5b421d97c905eb57542ac6
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
from utils.mp_data import *
from utils.model_class import *
from utils.plot_data import *
from config_file import *
file_name = os.path.basename(__file__)
print("File Name:", file_name)
#%%
# params (data)
binary_classification = False
mp_file = '/home/rokabe/data1/application/common_data/mp_full_prop.pkl'
ehull_max = 0.1
natm_max = 40
r_max = 4.0
tr_ratio = 0.9
batch_size = 16
nearest_neighbor = False
cut_data = None
epsilon=1e-3
scaler = None
target = 'mag'    # ['ehull', 'f_energy']
descriptor = 'ie'   # ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']
print(target, descriptor)

#%% params (model)
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
max_iter = 200    # total eppochs for training 
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
wandb.init(project='mag_order', entity='ryotarookabe', name=run_name + '_'+ target)
wandb.config.update({
    'file_name':file_name,
    'binary_classification': binary_classification,
    'mp_file': mp_file,
    'ehull_max': ehull_max,
    'natm_max': natm_max,
    'r_max': r_max,
    'tr_ratio': tr_ratio,
    'cut_data': cut_data,
    'nearest_neighbor': nearest_neighbor,
    'batch_size': batch_size,
    'target': target,
    'descriptor': descriptor,
    # 'lattice_change': lattice_change,
    # 'save_file': save_file,
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
mpdata = pd.read_pickle('./data/mp_full_prop.pkl')  
mpdata['natm'] = mpdata['structure'].apply(lambda x: len(x))
mpdata['formula'] = mpdata['structure'].apply(lambda x: x.get_chemical_formula())
mpdata['elements'] = mpdata['structure'].apply(lambda x: list(set(x.get_chemical_symbols())))
mpdata['nelem'] = mpdata['elements'].apply(lambda x: len(x))

df = mpdata.copy()
MAGNETIC_ATOMS = ['Ga', 'Tm', 'Y', 'Dy', 'Nb', 'Pu', 'Th', 'Er', 'U',
                  'Cr', 'Sc', 'Pr', 'Re', 'Ni', 'Np', 'Nd', 'Yb', 'Ce',
                  'Ti', 'Mo', 'Cu', 'Fe', 'Sm', 'Gd', 'V', 'Co', 'Eu',
                  'Ho', 'Mn', 'Os', 'Tb', 'Ir', 'Pt', 'Rh', 'Ru']

# remove materials which does not contain magnetic atoms
df = df[df['elements'].apply(lambda x: any([a in MAGNETIC_ATOMS for a in x]))].reset_index(drop=True)
df = df[df['ehull'] <= ehull_max].reset_index(drop=True)
df = df[df['natm'] <= natm_max].reset_index(drop=True)
df = df[df['mag'].isin(['NM', 'AFM', 'FM', 'FiM'])].reset_index(drop=True)

if binary_classification: 
    ORDER_ENCODE = {"NM": 0, "AFM": 1, "FM": 1, "FiM": 1}
else:     
    ORDER_ENCODE = {"NM": 0, "AFM": 1, "FM": 2, "FiM": 2}
df['label'] = df['mag'].apply(lambda x: ORDER_ENCODE[x])

print('len(df): ', len(df))


#%%
# df['structure'] = df['structure'].apply(pmg2ase)
dataset = Dataset_Cls(df, r_max, target, descriptor, scaler, nearest_neighbor)  # dataset
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
                     input_embed_dim,
                     num_classes,)

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
          num_classes,
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
df_tr = generate_dataframe(model, tr_loader, loss_fn, scaler, num_classes, device)
df_te = generate_dataframe(model, te1_loader, loss_fn, scaler, num_classes, device)


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

