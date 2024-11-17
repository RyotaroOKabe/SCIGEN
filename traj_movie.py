#%%
"""
Save the movie of the generated structures in time series
"""
import numpy as np
import time
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import matplotlib as mpl
import torch
from os.path import join
import warnings
from script.sc_utils import lattice_params_to_matrix_xy_torch
from script.mat_utils import vis_structure, get_pstruct_list, get_traj_pstruct_list, output_gen, movie_structs, convert_seconds_short
from config_scigen import home_dir, hydra_dir, job_dir, out_name
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="torch")
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
save_dir = join(home_dir, 'figures')

#%%
label = out_name
job = job_dir 
task = 'gen'
add = "" if label is None else '_' + label
jobdir = join(hydra_dir, job)
use_name = task + add
use_path = join(jobdir, f'eval_{use_name}.pt') 

frac_coords, atom_types, lengths, angles, num_atoms, run_time, \
        all_frac_coords, all_atom_types, all_lengths, all_angles = output_gen(use_path)
lattices = lattice_params_to_matrix_xy_torch(lengths, angles).to(dtype=torch.float32)
get_traj = True if all([len(a)>0 for a in [all_frac_coords, all_atom_types, all_lengths, all_angles]]) else False
if get_traj:
    print('We have access to traj data!')
    all_lattices = torch.stack([lattice_params_to_matrix_xy_torch(all_lengths[i], all_angles[i]) for i in range(len(all_lengths))])
num = len(lattices)
print("use_path: ", use_path)

#%%
#[1] load structure
print(f'[1.1] Load structure data')
start_time1 = time.time()
pstruct_list = get_pstruct_list(num_atoms, frac_coords, atom_types, lattices, atom_type_prob=True)
astruct_list = [Atoms(AseAtomsAdaptor().get_atoms(pstruct)) for pstruct in pstruct_list]
run_time1 = time.time() - start_time1
total_num = len(astruct_list)
print(f'Total outputs:{total_num} materials')
print(f'run time: {run_time1} sec = {convert_seconds_short(run_time1)}')
print(f'{run_time1/total_num} sec/material')

# check structure
idx = 21
pstruct = pstruct_list[idx]
astruct = astruct_list[idx]
vis_structure(pstruct, supercell=np.diag([1,1,1]), title='pstruct')
# vis_structure(astruct, supercell=np.diag([1,1,1]), title='astruct')

#%%
t_step = 1  # Set the time step for the trajectory
idx_list = [0,1,2]  #TODO: change this to a list of indices you want to generate movie for
print(f'[3] Generate images and gifs')
start_time3 = time.time()
if get_traj:
    traj_pstruct_list, t_list = get_traj_pstruct_list(num_atoms, all_frac_coords, all_atom_types, all_lattices, t_step, atom_type_prob=False)
# for _idx in id_stable:
for i in idx_list:  #range(num):
    idx = format(i, '05')
    unstable_dir = join(save_dir, job, use_name, 'unstable')
    gif_name = f"0000_{i}"
    try:
        # idx = int(_idx)
        print(gif_name)
        # print(f'[Stable material!!] {idx}')
        struct_dir = join(save_dir, job, use_name, idx)
        movie_structs(traj_pstruct_list[i], t_interval=10, name=gif_name, save_dir=struct_dir, supercell=np.diag([3,3,1]))
        print("Succeed in saving movie of stable structure in: ", struct_dir)
    except Exception as e:
        print(f'Got an error when generating material ({idx})', e)

run_time3 = time.time() - start_time3
print(f'Total outputs:{total_num} materials')
print(f'run time: {run_time3} sec = {convert_seconds_short(run_time3)}')
print(f'{run_time3/total_num} sec/material')
        


# %%
