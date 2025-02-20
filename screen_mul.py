import os
from config_scigen import *

num_materials = 50000
num_run =2
idx_start = 0
header = 'sc'  
sc_list = ['tri', 'hon', 'kag', 'sqr', 'elt', 'sns', 'tsq', 'srt', 'snh', 'trh', 'lieb']  
gen_cif = True
screen_mag = False

for j in range(idx_start, idx_start+num_run):
    for i, sc in enumerate(sc_list):
        tag = format(j, '03d')
        label = f'{header}_{sc}{num_materials}_{tag}'
        job_command = f'python script/eval_screen.py  --label {label} --gen_cif {gen_cif} --screen_mag {screen_mag}'
        print([i, j], job_command)
        os.system(job_command)
        print([i, j], label, 'done')
        print()
