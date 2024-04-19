import os
# from inpaint.inpaint_utils import arch_nickname

############
model_path = '/home/rokabe/data2/generative/hydra/singlerun/2023-12-25/diff_mp20_ab_0'
dataset = 'mp_20'
batch_size=500
num_batches_to_samples=100
num_materials = batch_size * num_batches_to_samples
# save_traj = False
num_run =6
idx_start = 4
arch_list = ['honeycomb']
atom_list = ['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Tb', 'Dy', 'Yb']
###################

arch_nickname = {
'triangular': 'tri',    # 1 atom
'honeycomb': 'hon',     # 2 atom
'kagome': 'kag',    # 3 atom
'square': 'sqr',    # 1 atom
'4_6_12':	'grt',    # 12 atom
'3p3_4p2':	'elt',    # 2 atom
'3p2_4_3_4':	'sns',    # 4 atom
'3p4_6':	'snh',    # 6 atom
'3_4_6_4':	'srt',    # 6 atom
'3_12p2':	'trh',    # 6 atom
'4_8p2':	'tsq',    # 4 atom
'lieb':	'lieb',    # 3 atom
}

arch_natm_max = {   # max number of atoms in the unit cell
'triangular': 4,    # 1 atom
'honeycomb': 8,      # 2 atom
'kagome': 12,    # 3 atom
'square': 4,    # 1 atom
'4_6_12': 20,    # 12 atom
'3p3_4p2':	8,    # 2 atom
'3p2_4_3_4': 16,    # 4 atom
'3p4_6':	20,    # 6 atom
'3_4_6_4':	20,    # 6 atom
'3_12p2':	20,    # 6 atom
'4_8p2':	16,    # 4 atom
'lieb': 12 # 3 atom
}

for j in range(idx_start, idx_start+num_run):
    for i, arch in enumerate(arch_list):
        tag = format(j, '03d')
        save_traj_arg = '--save_traj True' if j == 0 else ''  # save traj only for the first run
        label = f'inp_{arch_nickname[arch]}{num_materials}_{tag}'
        max_atom = arch_natm_max[arch]
        job_command = f'python inpaint/generation.py --model_path {model_path} \
                    --dataset {dataset} --label {label} --arch {arch} \
                    --batch_size {batch_size} --num_batches_to_samples {num_batches_to_samples}   \
                    --max_atom {max_atom} {save_traj_arg}'
        print([i, j], job_command)
        os.system(job_command)
        print([i, j], label, 'done')
        print()
