#%%
"""
https://www.notion.so/240316-ehull-evaluation-regression-classification-c9af94f384f34ee6a0dba4ee443cc198
"""
# modules
import torch
from torch import nn
import pandas as pd
import time
import numpy as np
from matplotlib import pyplot as plt
import os
from os.path import join
from sklearn.model_selection import train_test_split
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

from utils.utils_data import *
from utils.utils_model import GraphNetwork_Classifier
from utils.utils_train import train
from utils.helpers import make_dict
from config_scigen import *
file_name = os.path.basename(__file__)
print("File Name:", file_name)
#%%
run_name = time.strftime('%y%m%d-%H%M%S', time.localtime())
model_dir = './models'
data_dir = './data'
tr_ratio = 0.9
batch_size = 1
k_fold = 5

#%%
load_saved_data = True
save_file = join(data_dir, 'mpdata_mp20_test.pkl')
stable_threshold = 0.1

#%%


max_iter = 200 
lmax = 2 
mul = 4 
nlayers = 2 
r_max = 4 
number_of_basis = 10 
radial_layers = 1
radial_neurons = 100 
node_dim = 118
node_embed_dim = 32 
input_dim = 118
input_embed_dim = 32 
irreps_out = '64x0e'
descriptor = 'ie'

loss_fn = nn.BCEWithLogitsLoss(reduce=False) 
loss_fn_name = loss_fn.__class__.__name__
lr = 0.005
weight_decay = 0.05 
schedule_gamma = 0.96 

conf_dict = make_dict([run_name, model_dir, data_dir, raw_dir, data_file, tr_ratio, batch_size, k_fold, 
                       max_iter, lmax, mul, nlayers, r_max, number_of_basis, radial_layers, radial_neurons, 
                       node_dim, node_embed_dim, input_dim, input_embed_dim, irreps_out, descriptor, 
                       loss_fn_name, lr, weight_decay, schedule_gamma, device, seedn])

for k, v in conf_dict.items():
    print(f'{k}: {v}')


#%%
# load data

if load_saved_data:
    mpdata = pd.read_pickle(save_file)  
else: 
    # mpdata = mp_struct_ehull(api_key, 100, True, save_file)
    # with open('./data/mpids_phonon_db.txt', 'r') as f:
    #     mpids_phdb = [line.rstrip('\n') for line in f]
    # mpdata = mp_struct_ehull(api_key, mpids_phdb, True, save_file)
    mpdata = mp_struct_ehull_stable_filter(api_key, use_ase=True, save_file=save_file, s_u_nums=[2000, 1000], natm_max=20)
    
mpdata['stable'] = np.where(mpdata['ehull'] < stable_threshold, 1, 0)
# mpdata['stable'] = mpdata[target].apply(lambda x: 1 if x < stable_threshold else 0)

mpdata = mpdata[mpdata['stable']==True]
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
data_dict = generate_data_dict(data_dir=data_dir, run_name=run_name, data=data, r_max=r_max, descriptor=descriptor, option=option, factor=factor)

#%%
num = len(data_dict)
tr_nums = [int((num * tr_ratio)//k_fold)] * k_fold
te_num = num - sum(tr_nums)
idx_tr, idx_te = train_test_split(range(num), test_size=te_num, random_state=seedn)
with open(f'./data/idx_{run_name}_tr.txt', 'w') as f: 
    for idx in idx_tr: f.write(f"{idx}\n")
with open(f'./data/idx_{run_name}_te.txt', 'w') as f: 
    for idx in idx_te: f.write(f"{idx}\n")

#%%
data_set = torch.utils.data.Subset(list(data_dict.values()), range(len(data_dict)))
tr_set, te_set = torch.utils.data.Subset(data_set, idx_tr), torch.utils.data.Subset(data_set, idx_te)



#%% 
# model
model = GraphNetwork_kMVN(mul,
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
print(model)

#%%
opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = schedule_gamma)

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



