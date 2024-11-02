import os
from os.path import join
from config_scigen import hydra_dir, job_dir

############
model_path = join(hydra_dir, job_dir)
dataset = 'mp_20'
batch_size = 10
num_batches_to_samples = 20
num_materials = batch_size * num_batches_to_samples
save_traj_idx = [0]  # List of indices to save trajectory
num_run = 1
idx_start = 0
c_scale = None
c_vert = True
header = 'sc'
sc_list = ['tri']
atom_list = ['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Tb', 'Dy', 'Yb']
###################

sc_natm_range = {   # max number of atoms in the unit cell
    'tri': [1, 4],    # 1 atom for triangle
    'hon': [1, 8],      # 2 atom for honeycomb
    'kag': [1, 12],    # 3 atom for kagome
    'sqr': [1, 4],    # 1 atom for square
    'elt': [1, 8],    # 2 atom for elongated triangle
    'sns': [1, 16],    # 4 atom for snub square
    'tsq': [1, 16],    # 4 atom for truncated square
    'srt': [1, 20],    # 6 atom for small rhombitrihexagonal
    'snh': [1, 20],    # 6 atom for snub hexagonal
    'trh': [1, 20],    # 6 atom for 
    'grt': [1, 20],    # 12 atom
    'lieb': [1, 12],    # 3 atom
    'van': [1, 20],   
}

# Handle c_scale argument conditionally
c_scale_arg = f'--c_scale {c_scale}' if c_scale is not None else ''
c_vert_arg = '--c_vert' if c_vert else '--no_c_vert'

for i, sc in enumerate(sc_list):
    for j in range(idx_start, idx_start + num_run):
        tag = format(j, '03d')
        
        # Handle save_traj argument: add `--save_traj` if the current index is in save_traj_idx
        save_traj_arg = '--save_traj' if j in save_traj_idx else '--no_save_traj'

        label = f"{header+'_' if len(header) > 0 else ''}{sc}{num_materials}_{tag}"
        natm_range = [str(i) for i in sc_natm_range[sc]]

        # Construct the command string
        job_command = f'python script/generation.py --model_path {model_path} \
                    --dataset {dataset} --label {label} --sc {sc} \
                    --batch_size {batch_size} --num_batches_to_samples {num_batches_to_samples}   \
                    --natm_range {" ".join(natm_range)} {save_traj_arg}   \
                    --known_species {" ".join(atom_list)}   \
                    {c_scale_arg} {c_vert_arg}'

        print([i, j], job_command)
        os.system(job_command)
        print([i, j], label, 'done')
        print()
