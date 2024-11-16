#%%
"""
Script for visualizing the generative process of crystal structures.
Reference: https://www.notion.so/230408-visualization-of-the-generative-process-84753ea722e14a358cf61832902bb127
"""

import os
import sys
import torch
import warnings
import argparse

# Set default tensor data type and device
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import custom modules
sys.path.append('../')  # Adjust path for custom modules
from os.path import join
from config_scigen import home_dir, hydra_dir, job_dir, out_name
from script.mat_utils import (
    get_pstruct_list, output_gen, lattice_params_to_matrix_torch, save_combined_cif
)

# Suppress specific warnings for cleaner output
for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="torch")

# Directory for saving figures
savedir = join(home_dir, 'figures')

#%%
# Configuration
combine_cif = False  # Option to combine all structures into a single CIF file
task = 'gen'
# label = out_name

parser = argparse.ArgumentParser()
parser.add_argument('--job_dir', default=job_dir, type=str)
parser.add_argument('--label', default=out_name, type=str)
args = parser.parse_args()

job_dir = args.job_dir
label = args.label

#%%
add = f'_{label}' if label else ''
job_dir = join(hydra_dir, job_dir)
use_name = f'eval_{task}{add}'
use_path = join(job_dir, f'{use_name}.pt')

# Load data from generated output
frac_coords, atom_types, lengths, angles, num_atoms, run_time, \
    all_frac_coords, all_atom_types, all_lengths, all_angles = output_gen(use_path)

# Convert lattice parameters to matrix format
lattices = lattice_params_to_matrix_torch(lengths, angles).to(dtype=torch.float32)

# Check if trajectory data is available
get_traj = all(len(a) > 0 for a in [all_frac_coords, all_atom_types, all_lengths, all_angles])
if get_traj:
    print('We have access to trajectory data!')
    all_lattices = torch.stack([lattice_params_to_matrix_torch(all_lengths[i], all_angles[i]) for i in range(len(all_lengths))])

# Number of structures to process
num = len(lattices)
print("job_dir: ", job_dir)

#%%
# [1] Load structure data and save CIF files
pstruct_list = get_pstruct_list(num_atoms, frac_coords, atom_types, lattices, atom_type_prob=True)
cif_dir = join(job_dir, use_name)

# Remove and recreate the CIF directory
os.system(f'rm -r {cif_dir}')
os.makedirs(cif_dir)

# Save structures in CIF format
if combine_cif:
    # Save all structures in a single CIF file
    cifs_file = join(cif_dir, 'cifs.cif')
    save_combined_cif(pstruct_list, cifs_file)
else:
    # Save each structure as an individual CIF file
    for i, struct in enumerate(pstruct_list):
        filename = f"{format(i, '05')}.cif"
        cif_path = join(cif_dir, filename)
        struct.to(fmt="cif", filename=cif_path)
        print(f"Saved: {cif_path}")
