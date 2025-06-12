import os
from os.path import join
import sys
import json
import torch
from torch import nn
# Local imports
sys.path.append('../')
print(sys.path)  # Print all directories Python searches for modules
from config_scigen import gnn_eval_path, home_dir, hydra_dir, job_dir, out_name, stab_pred_name_A, stab_pred_name_B, mag_pred_name
sys.path.append(gnn_eval_path)
from gnn_eval.utils.data import Dataset_Cls
from gnn_eval.utils.record import log_buffer, logger

from eval_funcs import load_model, load_model_mag, parse_arguments, process_data, load_df, classify_stability, generate_cif_files

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = os.path.join(home_dir, 'figures')

def main():
    args = parse_arguments(job_dir, out_name)
    model_names = [stab_pred_name_A, stab_pred_name_B]  #TODO: Adjust the model names if needed
    if args.screen_mag:
        model_names.append(mag_pred_name)
    jobdir = join(hydra_dir, args.job_dir)
    use_name = f"gen_{args.label}" if args.label else "gen"
    use_path = join(jobdir, f"eval_{use_name}.pt")
    folder_tag = 'filtered' if not args.screen_mag else 'mag'
    loss_fn = nn.BCEWithLogitsLoss(reduce=False) 

    model_list = [{} for _ in range(len(model_names))]
    for i, model_name in enumerate(model_names):
        param_dict_path = join(gnn_eval_path, 'models', f'{model_name}_config.json')
        with open(param_dict_path, 'r') as f:
            param_dict = json.load(f)
        mag_model = False
        try:     
            model = load_model(model_name, param_dict, device, logger)
            # logger.info(f"Loaded model {model_name}.")
        except: 
            model = load_model_mag(model_name, param_dict, device, logger)
            mag_model = True
            # logger.info(f"Loaded model {model_name} for magnetic materials.")
        model_list[i]['model'] = model
        model_list[i]['param'] = {k: param_dict[k] for k in ['r_max', 'descriptor', 'scaler']}
        model_list[i]['param']['nearest'] = param_dict['nearest_neighbor']
        model_list[i]['param']['target'] = 'label' if mag_model else param_dict['target']
        model_list[i]['batch_size'] = param_dict['batch_size']

    # Process data
    astruct_list_full = process_data(use_path, logger)
    df = load_df(astruct_list_full, logger)
    
    # [1] Filter by SMACT validity
    df1 = df[df['smact_valid']].reset_index(drop=True)
    logger.info(f"[1] Filtered {len(df1)}/{len(df)} materials by SMACT validity.")
    
    # [2] Filter by space occupation ratio
    df2 = df1[df1['occupy_ratio'] < 1.7].reset_index(drop=True)
    logger.info(f"[2] Filtered {len(df2)}/{len(df1)} materials by space occupation ratio.")
    
    dfs = [df2]
    for i, model_dict in enumerate(model_list):
        model = model_dict['model']
        param_dict, batch_size = model_dict['param'], model_dict['batch_size']
        dataset = Dataset_Cls(dfs[-1], **param_dict)
        df = classify_stability(model, dataset, loss_fn, param_dict['scaler'], batch_size, device, logger)
        id_stable = df[df['pred'] == 1]['id'].values
        logger.info(f"[{i+3}] Model {model_names[i]} classified {len(id_stable)}/{len(dfs[-1])} materials as stable.")
        df_ = dfs[-1][dfs[-1]['mpid'].isin(id_stable)].reset_index(drop=True)
        dfs.append(df_)

    cif_dir= join(jobdir, use_name + '_' + folder_tag)
    if args.gen_cif:
        generate_cif_files(dfs[-1], cif_dir, logger)

    # Write logs from memory to the file
    log_file = join(cif_dir, f"{use_name}.log")
    logger.info(f"Save log to {log_file}: {len(dfs[-1])} materials.")
    with open(log_file, 'w') as f:
        f.write(log_buffer.getvalue())

if __name__ == "__main__":
    main()
