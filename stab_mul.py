import os
from config_scigen import *
# from inpaint.inpaint_utils import arch_nickname

num_materials = 1000
num_run =1
idx_start = 0
header = 'sc'   # 'inp'
arch_list = ['kag']  # ['van', 'tri', 'hon', 'kag', 'sqr'] 
gen_cif = True
gen_movie = False

for j in range(idx_start, idx_start+num_run):
    for i, arch in enumerate(arch_list):
        tag = format(j, '03d')
        label = f'{header}_{arch}{num_materials}_{tag}'
        job_command = f'python script/eval_stab.py  --label {label} --gen_cif {gen_cif} --gen_movie {gen_movie}'
        print([i, j], job_command)
        os.system(job_command)
        print([i, j], label, 'done')
        print()
