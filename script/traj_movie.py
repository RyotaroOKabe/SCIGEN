#%%
"""
Save the movie of the generated structures in time series
"""
import sys
import numpy as np
import time
import torch
from os.path import join
import warnings
from sc_utils import lattice_params_to_matrix_xy_torch
from mat_utils import get_traj_pstruct_list, output_gen, movie_structs, convert_seconds_short
from eval_funcs import parse_arguments
sys.path.append('../')
from config_scigen import home_dir, hydra_dir, job_dir, out_name
torch.set_default_dtype(torch.float64)
for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="torch")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
args = parse_arguments(job_dir, out_name)
save_dir = join(home_dir, 'figures')
label = out_name
job = job_dir 
jobdir = join(hydra_dir, args.job_dir)
use_name = f"gen_{args.label}" if args.label else "gen"
use_path = join(jobdir, f'eval_{use_name}.pt') 
idx_list = [int(idx) for idx in args.idx_list]
traj_movie_config = {'supercell': np.diag([int(i) for i in args.supercell]), 'rot': '5x,5y,90z'} #TODO: adjust the orientation of the structure if necessary
frac_coords, atom_types, lengths, angles, num_atoms, run_time, \
        all_frac_coords, all_atom_types, all_lengths, all_angles = output_gen(use_path)

assert all([len(a)>0 for a in [all_frac_coords, all_atom_types, all_lengths, all_angles]]), f"Trajectory data is not available for {use_path}"
all_lattices = torch.stack([lattice_params_to_matrix_xy_torch(all_lengths[i], all_angles[i]) for i in range(len(all_lengths))])
total_num = all_lattices.shape[1]

#%%
print(f'Generate images and gifs for idx list {args.idx_list} materials in {use_path}')
start_time = time.time()
traj_pstruct_list, t_list = get_traj_pstruct_list(num_atoms, all_frac_coords, all_atom_types, all_lattices, t_step=1, atom_type_prob=False)
print(f'Loaded {len(traj_pstruct_list)} structures ({len(t_list)} time steps): {time.time() - start_time} sec')
for i in idx_list:  
    name = format(i, '05')
    unstable_dir = join(save_dir, job, use_name, 'unstable')
    gif_name = f"0000_{i}"
    try:
        print(gif_name)
        struct_dir = join(save_dir, job, use_name, name)
        print(f'Generating {struct_dir}')
        movie_structs(traj_pstruct_list[i], t_interval=10, name=gif_name, save_dir=struct_dir, **traj_movie_config)
        print("Succeeded in saving movie of stable structure in: ", struct_dir)
    except Exception as e:
        print(f'Got an error when generating material ({name})', e)

print(f'Total outputs:{time.time() - start_time} materials')
print(f'run time: {run_time} sec = {convert_seconds_short(run_time)}')
        


# %%
