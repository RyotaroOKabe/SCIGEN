#%%
"""
https://www.notion.so/230408-visualization-of-the-generative-process-84753ea722e14a358cf61832902bb127
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from ase import Atom, Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize.plot import plot_atoms
import matplotlib as mpl
import torch
from torch import nn
from torch_geometric.loader import DataLoader
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import os, sys
sys.path.append('../')
from os.path import join
import imageio
from dirs import *
sys.path.append(ehull_pred_path)
from inpaint.mat_utils import vis_structure, get_pstruct_list, get_traj_pstruct_list, output_gen, str2pmg, pmg2ase, lattice_params_to_matrix_torch
from ehull_prediction.utils.data import Dataset_InputStability
from ehull_prediction.utils.model_class import GraphNetworkClassifier, generate_dataframe
import warnings
# from m3gnet.models import Relaxer
for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="torch")
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])


#%%
# model setting 
model_name = '240325-200817'      # pre-trained model. Rename if you want to use the model you trained in the tabs above. 
r_max = 4.
tr_ratio = 0.0
batch_size = 16
scaler = None #LogShiftScaler(epsilon, mu, sigma)
target = 'ehull'    # ['ehull', 'f_energy']
descriptor = 'ie'   # ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']
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

#%%
job = job_folder # "2023-06-10/mp_20_2"   #!
task = 'gen'
label ='demo8.3'
add = None if label is None else '_' + label
jobdir = join(hydradir, job)
use_name = task + add
use_path = join(jobdir, f'eval_{use_name}.pt') #!

frac_coords, atom_types, lengths, angles, num_atoms, run_time, \
        all_frac_coords, all_atom_types, all_lengths, all_angles = output_gen(use_path)
lattices = lattice_params_to_matrix_torch(lengths, angles).to(dtype=torch.float32)
all_lattices = torch.stack([lattice_params_to_matrix_torch(all_lengths[i], all_angles[i]) for i in range(len(all_lengths))])
num = len(lattices)
print("jobdir: ", jobdir)

#%%
#[1] load structure
pstruct_list = get_pstruct_list(num_atoms, frac_coords, atom_types, lattices, atom_type_prob=True)
# traj_pstruct_list = get_traj_pstruct_list(num_atoms, all_frac_coords, all_atom_types, all_lattices, atom_type_prob=False)
astruct_list = [Atoms(AseAtomsAdaptor().get_atoms(pstruct)) for pstruct in pstruct_list]

#%%
# check structure
idx = 21
pstruct = pstruct_list[idx]
astruct = astruct_list[idx]
vis_structure(pstruct, supercell=np.diag([1,1,1]), title='pstruct')
vis_structure(astruct, supercell=np.diag([1,1,1]), title='astruct')

#%%
new_rows = []  # To collect new rows
for i, astruct in enumerate(astruct_list):
    row1 = {}  # Initialize the dictionary here
    row1['mpid'] = format(i, '05')
    row1['structure'] = astruct
    row1['f_energy'] = 0.
    row1['ehull'] = 0.
    row1['stable'] = 1
    new_rows.append(row1)

mpdata = pd.DataFrame(new_rows).reset_index(drop=True)
dataset = Dataset_InputStability(mpdata, r_max, target, descriptor, scaler)  # dataset
num = len(dataset)
idx_te = range(num)
te_set = torch.utils.data.Subset(dataset, idx_te)


#%%
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
model_file = f'./ehull_prediction/models/{model_name}.torch'
# model = torch.load(model_file)
model.load_state_dict(torch.load(model_file)['state'])
model = model.to(device)
model = model.eval()  # Set to evaluation mode if using for inference
#%%
# Generate Data Loader
loss_fn = nn.BCEWithLogitsLoss(reduce=False) 

te_loader = DataLoader(te_set, batch_size = batch_size)
df_te = generate_dataframe(model, te_loader, loss_fn, scaler, device)

num_stable = (df_te['pred'] == 1).sum()
print(f"num stable: {num_stable}/{len(df_te)}")
print('Stable materials: ', list(df_te['mpid']))

# %%