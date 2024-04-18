#%%
"""
https://www.notion.so/230408-visualization-of-the-generative-process-84753ea722e14a358cf61832902bb127
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import pandas as pd
import sys
import argparse
sys.path.append('../')
from InpaintCrysDiff.dirs_template import *
sys.path.append(ehull_pred_path)
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
from os.path import join
import imageio
from inpaint.mat_utils import vis_structure, get_pstruct_list, get_traj_pstruct_list, output_gen, str2pmg, pmg2ase, ase2pmg, lattice_params_to_matrix_torch, movie_structs, convert_seconds_short, chemical_symbols, vol_density, get_composition, smact_validity
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
savedir = join(homedir, 'figures')

#%%
# model setting 
model_name = ehull_pred_name      # pre-trained model. Rename if you want to use the model you trained in the tabs above. 
model_name2 = ehull_pred_name2
r_max = 4.
tr_ratio = 0.0
batch_size = 16
nearest_neighbor = False
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
loss_fn = nn.BCEWithLogitsLoss(reduce=False) 

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--out_name', default=out_name, type=str)
args = parser.parse_args()
job = job_folder # "2023-06-10/mp_20_2"   
task = 'gen'
label = args.out_name
add = "" if label is None else '_' + label
jobdir = join(hydradir, job)
use_name = task + add
use_path = join(jobdir, f'eval_{use_name}.pt') 

frac_coords, atom_types, lengths, angles, num_atoms, run_time, \
        all_frac_coords, all_atom_types, all_lengths, all_angles = output_gen(use_path)
lattices = lattice_params_to_matrix_torch(lengths, angles).to(dtype=torch.float32)
get_traj = True if all([len(a)>0 for a in [all_frac_coords, all_atom_types, all_lengths, all_angles]]) else False
if get_traj:
    print('We have access to traj data!')
    all_lattices = torch.stack([lattice_params_to_matrix_torch(all_lengths[i], all_angles[i]) for i in range(len(all_lengths))])
num = len(lattices)
print("use_path: ", use_path)

#%%
#[1] load structure
print(f'[1.1] Load structure data')
start_time1 = time.time()
pstruct_list = get_pstruct_list(num_atoms, frac_coords, atom_types, lattices, atom_type_prob=True)
# traj_pstruct_list = get_traj_pstruct_list(num_atoms, all_frac_coords, all_atom_types, all_lattices, atom_type_prob=False)
astruct_list = [Atoms(AseAtomsAdaptor().get_atoms(pstruct)) for pstruct in pstruct_list]
run_time1 = time.time() - start_time1
total_num = len(astruct_list)
print(f'Total outputs:{total_num} materials')
print(f'run time: {run_time1} sec = {convert_seconds_short(run_time1)}')
print(f'{run_time1/total_num} sec/material')

# check structure
# idx = 21
# pstruct = pstruct_list[idx]
# astruct = astruct_list[idx]
# vis_structure(pstruct, supercell=np.diag([1,1,1]), title='pstruct')
# vis_structure(astruct, supercell=np.diag([1,1,1]), title='astruct')

#%%
print(f'[1.1.2] Filter by SMACT composition validity')
use_smact_comp_valid = True

if use_smact_comp_valid:
    idx_cvalids = []
    for idx, astruct in enumerate(astruct_list):
        atom_types = np.array([chemical_symbols.index(s) for s in astruct.get_chemical_symbols()])
        elems, comps = get_composition(atom_types)
        cvalid = smact_validity(elems, comps)
        if cvalid:
            idx_cvalids.append(idx)

    astruct_list = [astruct_list[idx] for idx in idx_cvalids]
    pstruct_list = [pstruct_list[idx] for idx in idx_cvalids]
    
    print(f"Filter materials by SMACT's composition validity: {len(astruct_list)}/{total_num}")

#%%
print(f'[1.2] Generate initial dataset')
start_time1 = time.time()
new_rows = []  # To collect new rows
for i, astruct in enumerate(astruct_list):
    row1 = {}  # Initialize the dictionary here
    row1['mpid'] = format(i, '05')
    row1['structure'] = astruct
    row1['f_energy'] = 0.
    row1['ehull'] = 0.
    row1['stable'] = 1
    new_rows.append(row1)

# del astruct_list, frac_coords, atom_types, lattices, lengths, angles, all_lengths, all_angles

mpdata = pd.DataFrame(new_rows).reset_index(drop=True)    
mpdata['occupy_ratio'] = mpdata['structure'].map(vol_density)
mpdata = mpdata[mpdata['occupy_ratio']<1.7].reset_index(drop=True)
print(f'Filter materials by space occupation ratio: {len(mpdata)}/{num}')
# mpdata_file = join('./ehull_prediction/data/mpdata_mp20_test.pkl')  
# mpdata = pd.read_pickle(mpdata_file)   
dataset = Dataset_InputStability(mpdata, r_max, target, descriptor, scaler, nearest_neighbor)  # dataset
num = len(dataset)
idx_te = range(num)
te_set = torch.utils.data.Subset(dataset, idx_te)
run_time1 = time.time() - start_time1
total_num = len(astruct_list)
print(f'Total outputs:{total_num} materials')
print(f'run time: {run_time1} sec = {convert_seconds_short(run_time1)}')
print(f'{run_time1/total_num} sec/material')

#%%
print(f'[2.0] Load GNN models')
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
model_file = join(ehull_pred_path, 'models', model_name + '.torch') 
# model = torch.load(model_file)
model.load_state_dict(torch.load(model_file)['state'])
model = model.to(device)
model = model.eval()  # Set to evaluation mode if using for inference

model2 = GraphNetworkClassifier(mul,
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
model_file2 = join(ehull_pred_path, 'models', model_name2 + '.torch')
# model = torch.load(model_file)
model2.load_state_dict(torch.load(model_file2)['state'])
model2 = model2.to(device)
model2 = model2.eval()  # Set to evaluation mode if using for inference
#%%
# Generate Data Loader
print(f'[2.1] Stability classification (1)')
start_time2 = time.time()
te_loader = DataLoader(te_set, batch_size = batch_size)
# Evaluate with GNN classifier
df_te = generate_dataframe(model, te_loader, loss_fn, scaler, device)

df_stable = df_te[df_te['pred'] == 1]
id_stable = list(df_stable['id'])
num_stable = (df_te['pred'] == 1).sum()

run_time2 = time.time() - start_time2
print(f'Total outputs:{total_num} materials')
print(f'run time: {run_time2} sec = {convert_seconds_short(run_time2)}')
print(f'{run_time2/total_num} sec/material')

print(f"num stable: {num_stable}/{len(df_te)}")
print('Stable materials: ', id_stable)

#%%
# Generate Data Loader
print(f'[2.2] Stability classification (2)')
start_time2 = time.time()
mpdata2 = mpdata[mpdata['mpid'].isin(id_stable)].reset_index(drop=True) # Evaluate onlyt the down-sampled materials by the early GNN model. 
dataset2 = Dataset_InputStability(mpdata2, r_max, target, descriptor, scaler, nearest_neighbor)
num2 = len(dataset2)
idx_te2 = range(num2)
te_set2 = torch.utils.data.Subset(dataset2, idx_te2)
te_loader2 = DataLoader(te_set2, batch_size = batch_size)
# Evaluate with GNN classifier
df_te2 = generate_dataframe(model2, te_loader2, loss_fn, scaler, device)
df_stable2 = df_te2[df_te2['pred'] == 1]
id_stable2 = list(df_stable2['id'])
num_stable2 = (df_te2['pred'] == 1).sum()
run_time2 = time.time() - start_time2
print(f'Total outputs:{len(df_te2)} materials')
print(f'run time: {run_time2} sec = {convert_seconds_short(run_time2)}')
print(f'{run_time2/total_num} sec/material')
print(f"num stable: {num_stable2}/{len(df_te2)}")
print('Stable materials: ', id_stable2)

#%%
gen_cif = True
if gen_cif:
    cif_dir= join(jobdir, use_name + '_filtered')
    os.system(f'rm -r {cif_dir}')
    os.makedirs(cif_dir)
    # pstruct_list_filtered = [pstruct_list[int(idx)] for idx in id_stable2]
    # for i, struct in enumerate(pstruct_list_filtered):
    for idx in id_stable2:
        struct = pstruct_list[int(idx)]
        try: 
            # Construct a filename for each structure
            # filename = f"{format(i, '05')}.cif"
            filename = f"{idx}.cif"
            
            # Specify the directory where you want to save the CIF files
            cif_path = os.path.join(cif_dir, filename)
            
            # Save the structure as a CIF file
            struct.to(fmt="cif", filename=cif_path)

            print(f"Saved: {cif_path}")
        except Exception as e:
            print(f'Got an error when generating material ({cif_path})', e)


#%%
gen_movie = False
t_step = 1  # can be improved later.
if gen_movie:
    print(f'[3] Generate images and gifs')
    start_time3 = time.time()
    if get_traj:
        traj_pstruct_list, t_list = get_traj_pstruct_list(num_atoms, all_frac_coords, all_atom_types, all_lattices, t_step, atom_type_prob=False)
    # for _idx in id_stable:
    for i in range(num):
        idx = format(i, '05')
        unstable_dir = join(savedir, job, use_name, 'unstable')
        gif_name = f"0000_{i}"
        try:
            # idx = int(_idx)
            print(gif_name)
            if idx in id_stable:
                print(f'[Stable material!!] {idx}')
                structdir = join(savedir, job, use_name, idx)
                movie_structs(traj_pstruct_list[i], t_interval=10, name=gif_name, savedir=structdir, supercell=np.diag([3,3,1]))
                print("Succeed in saving movie of stable structure in: ", structdir)
            else: 
                print(f'[Unstable...] {idx}')
                pstruct = pstruct_list[i]
                vis_structure(pstruct, supercell=np.diag([3,3,1]), title=idx, savedir=unstable_dir)
        except Exception as e:
            print(f'Got an error when generating material ({idx})', e)
    
    run_time3 = time.time() - start_time3
    print(f'Total outputs:{total_num} materials')
    print(f'run time: {run_time3} sec = {convert_seconds_short(run_time3)}')
    print(f'{run_time3/total_num} sec/material')
            


# %%
