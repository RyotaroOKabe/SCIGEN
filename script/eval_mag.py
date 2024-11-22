import os
from os.path import join
import sys
import json
import torch
from torch import nn
# Local imports
sys.path.append('../')
from config_scigen import gnn_eval_path, home_dir, hydra_dir, job_dir, out_name, stab_pred_name_A, stab_pred_name_B, mag_pred_name
sys.path.append(gnn_eval_path)
from gnn_eval.utils.data import Dataset_Cls
from gnn_eval.utils.record import log_buffer, logger

from eval_funcs import load_model, load_model_mag, parse_arguments, process_data, load_df, classify_stability, generate_cif_files

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = os.path.join(home_dir, 'figures')
t_step = 1  # Time step for trajectory processing


def main():
    args = parse_arguments()
    jobdir = join(hydra_dir, args.job_dir)
    use_name = f"gen_{args.label}" if args.label else "gen"
    use_path = join(jobdir, f"eval_{use_name}.pt")
    loss_fn = nn.BCEWithLogitsLoss(reduce=False) 

    # Load models
    param_dict_path_A = join(gnn_eval_path, 'models', f'{stab_pred_name_A}_config.json')
    param_dict_path_B = join(gnn_eval_path, 'models', f'{stab_pred_name_B}_config.json')
    param_dict_path_C = join(gnn_eval_path, 'models', f'{mag_pred_name}_config.json')
    with open(param_dict_path_A, 'r') as f:
        param_dict_A = json.load(f)
    with open(param_dict_path_B, 'r') as f:
        param_dict_B = json.load(f)
    with open(param_dict_path_C, 'r') as f:
        param_dict_C = json.load(f)

    model_A = load_model(stab_pred_name_A, param_dict_A, device, logger)
    model_B = load_model(stab_pred_name_B, param_dict_B, device, logger)
    model_C = load_model_mag(mag_pred_name, param_dict_C, device, logger)
    
    r_max_A, target_A, descriptor_A, scaler_A, nearest_neighbor_A, batch_size_A = [param_dict_A[k] for k in ['r_max', 'target', 'descriptor', 'scaler', 'nearest_neighbor', 'batch_size']]
    r_max_B, target_B, descriptor_B, scaler_B, nearest_neighbor_B, batch_size_B = [param_dict_B[k] for k in ['r_max', 'target', 'descriptor', 'scaler', 'nearest_neighbor', 'batch_size']]
    r_max_C, target_C, descriptor_C, scaler_C, nearest_neighbor_C, batch_size_C = [param_dict_B[k] for k in ['r_max', 'target', 'descriptor', 'scaler', 'nearest_neighbor', 'batch_size']]

    # Process data
    astruct_list_full = process_data(use_path, logger)
    df = load_df(astruct_list_full, logger)
    
    # [1] Filter by SMACT validity
    df1 = df[df['smact_valid']].reset_index(drop=True)
    logger.info(f"[1] Filtered {len(df1)}/{len(df)} materials by SMACT validity.")
    
    # [2] Filter by space occupation ratio
    df2 = df1[df1['occupy_ratio'] < 1.7].reset_index(drop=True)
    logger.info(f"[2] Filtered {len(df2)}/{len(df1)} materials by space occupation ratio.")

    # [3] Classify stability by model A
    dataset_A = Dataset_Cls(df2, r_max_A, target_A, descriptor_A, scaler_A, nearest_neighbor_A)
    df_A = classify_stability(model_A, dataset_A, loss_fn, scaler_A, batch_size_A, device, logger)
    id_stable_A = df_A[df_A['pred'] == 1]['id'].values
    logger.info(f"[3] Model A classified {len(id_stable_A)}/{len(df2)} materials as stable.")
    print('id_stable_A: ', id_stable_A)
    df3 = df2[df2['mpid'].isin(id_stable_A)].reset_index(drop=True)
    
    # [4] Classify stability by model B
    dataset_B = Dataset_Cls(df3, r_max_B, target_B, descriptor_B, scaler_B, nearest_neighbor_B)
    df_B = classify_stability(model_B, dataset_B, loss_fn, scaler_B, batch_size_B, device, logger)
    id_stable_B = df_B[df_B['pred'] == 1]['id'].values
    logger.info(f"[4] Model B classified {len(id_stable_B)}/{len(df3)} materials as stable.")
    print('id_stable_B: ', id_stable_B)
    df4 = df3[df3['mpid'].isin(id_stable_B)].reset_index(drop=True)
    
    # [4] Classify stability by model B
    dataset_C = Dataset_Cls(df4, r_max_C, 'label', descriptor_C, scaler_C, nearest_neighbor_C)
    df_C = classify_stability(model_C, dataset_C, loss_fn, scaler_C, batch_size_C, device, logger)
    id_stable_C = df_C[df_C['pred'] == 1]['id'].values
    logger.info(f"[5] Model C classified {len(id_stable_C)}/{len(df4)} materials as stable.")
    print('id_stable_B: ', id_stable_C)
    
    df5 = df4[df4['mpid'].isin(id_stable_C)].reset_index(drop=True)
    
    cif_dir= join(jobdir, use_name + '_mag')
    if args.gen_cif:
        generate_cif_files(df5, cif_dir, logger)


    # Write logs from memory to the file
    log_file = join(cif_dir, f"{use_name}.log")
    logger.info(f"Save log to {log_file}")
    with open(log_file, 'w') as f:
        f.write(log_buffer.getvalue())

    logger.info(f"Logs saved to {log_file}")


if __name__ == "__main__":
    main()
