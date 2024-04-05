#%%
"""
https://www.notion.so/230408-visualization-of-the-generative-process-84753ea722e14a358cf61832902bb127
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import torch
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import os, sys
sys.path.append('../')
from os.path import join
from dirs import *
from inpaint.mat_utils import get_pstruct_list, output_gen, lattice_params_to_matrix_torch
import warnings
# from m3gnet.models import Relaxer
for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="torch")
savedir = join(homedir, 'figures')

#%%
job = job_folder # "2023-06-10/mp_20_2"   #!
task = 'gen'
label = out_name
add = None if label is None else '_' + label
jobdir = join(hydradir, job)
use_name = task + add
use_path = join(jobdir, f'eval_{use_name}.pt') #!

frac_coords, atom_types, lengths, angles, num_atoms, run_time, \
        all_frac_coords, all_atom_types, all_lengths, all_angles = output_gen(use_path)
lattices = lattice_params_to_matrix_torch(lengths, angles).to(dtype=torch.float32)
get_traj = True if all([len(a)>0 for a in [all_frac_coords, all_atom_types, all_lengths, all_angles]]) else False
if get_traj:
    print('We have access to traj data!')
    all_lattices = torch.stack([lattice_params_to_matrix_torch(all_lengths[i], all_angles[i]) for i in range(len(all_lengths))])
num = len(lattices)
print("jobdir: ", jobdir)

#%%
#[1] load structure
pstruct_list = get_pstruct_list(num_atoms, frac_coords, atom_types, lattices, atom_type_prob=True)
cif_dir= join(jobdir, use_name)
os.system(f'rm -r {cif_dir}')
os.makedirs(cif_dir)

# Assuming pstruct_list is your list of Structure objects
for i, struct in enumerate(pstruct_list):
    # Construct a filename for each structure
    filename = f"{format(i, '05')}.cif"
    
    # Specify the directory where you want to save the CIF files
    cif_path = os.path.join(cif_dir, filename)
    
    # Save the structure as a CIF file
    struct.to(fmt="cif", filename=cif_path)

    print(f"Saved: {cif_path}")
