import os
from os.path import join
import sys
import time
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import logging
from io import StringIO

# Local imports
sys.path.append('../')
from config_scigen import gnn_eval_path, home_dir, hydra_dir, job_dir, out_name, stab_pred_name_A, stab_pred_name_B, mag_pred_name
sys.path.append(gnn_eval_path)
from script.mat_utils import (
    vis_structure, get_pstruct_list, get_traj_pstruct_list, output_gen, 
    lattice_params_to_matrix_torch, movie_structs, convert_seconds_short, ase2pmg,
    chemical_symbols, vol_density, get_composition, smact_validity, charge_neutrality
)
from gnn_eval.utils.data import Dataset_Cls
from gnn_eval.utils.model_class import GraphNetworkClassifier
from gnn_eval.utils.model_class_mag import GraphNetworkClassifierMag
from gnn_eval.utils.output import generate_dataframe
from gnn_eval.utils.record import log_buffer, logger

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = os.path.join(home_dir, 'figures')
t_step = 1  # Time step for trajectory processing


def load_model(model_name, param_dict):
    """Loads and prepares a GNN model."""
    logger.info(f"Loading model: {model_name}")
    model = GraphNetworkClassifier(
        mul=param_dict['mul'],
        irreps_out=param_dict['irreps_out'],
        lmax=param_dict['lmax'],
        nlayers=param_dict['nlayers'],
        number_of_basis=param_dict['number_of_basis'],
        radial_layers=param_dict['radial_layers'],
        radial_neurons=param_dict['radial_neurons'],
        node_dim=param_dict['node_dim'],
        node_embed_dim=param_dict['node_embed_dim'],
        input_dim=param_dict['input_dim'],
        input_embed_dim=param_dict['input_embed_dim']
    )
    model_file = join(gnn_eval_path, 'models', f"{model_name}.torch")
    model.load_state_dict(torch.load(model_file)['state'])
    return model.to(device).eval()


def load_model_mag(model_name, param_dict):
    """Loads and prepares a GNN model."""
    logger.info(f"Loading model: {model_name}")
    model = GraphNetworkClassifierMag(
        mul=param_dict['mul'],
        irreps_out=param_dict['irreps_out'],
        lmax=param_dict['lmax'],
        nlayers=param_dict['nlayers'],
        number_of_basis=param_dict['number_of_basis'],
        radial_layers=param_dict['radial_layers'],
        radial_neurons=param_dict['radial_neurons'],
        node_dim=param_dict['node_dim'],
        node_embed_dim=param_dict['node_embed_dim'],
        input_dim=param_dict['input_dim'],
        input_embed_dim=param_dict['input_embed_dim'],
        num_classes=param_dict['num_classes']
    )
    model_file = join(gnn_eval_path, 'models', f"{model_name}.torch")
    model.load_state_dict(torch.load(model_file)['state'])
    return model.to(device).eval()


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Material Stability Classification Pipeline")
    parser.add_argument('--label', default=out_name, type=str, help="Output label name")
    parser.add_argument('--job_dir', default=job_dir, type=str, help="Job directory")
    parser.add_argument('--gen_cif', type=lambda x: x.lower() == 'true', default=True, help="Generate CIF files")
    parser.add_argument('--gen_movie', type=lambda x: x.lower() == 'true', default=False, help="Generate movies of structures")
    return parser.parse_args()


def process_data(output_path):
    """Processes structure data from output."""
    logger.info("Processing structure data...")
    frac_coords, atom_types, lengths, angles, num_atoms, _, all_frac_coords, all_atom_types, all_lengths, all_angles = output_gen(output_path)
    lattices = lattice_params_to_matrix_torch(lengths, angles).to(dtype=torch.float32)
    get_traj = all(len(a) > 0 for a in [all_frac_coords, all_atom_types, all_lengths, all_angles])
    
    if get_traj:
        logger.info("Trajectory data is available.")
        all_lattices = torch.stack([lattice_params_to_matrix_torch(all_lengths[i], all_angles[i]) for i in range(len(all_lengths))])

    pstruct_list = get_pstruct_list(num_atoms, frac_coords, atom_types, lattices, atom_type_prob=True)
    astruct_list = [Atoms(AseAtomsAdaptor().get_atoms(pstruct)) for pstruct in pstruct_list]

    assert len(pstruct_list) == len(astruct_list), "Mismatch between generated and adapted structures"
    return astruct_list


def load_df(astruct_list):
    logger.info("Loading structures into DataFrame...")
    rows = []
    for i, struct in enumerate(astruct_list):
        row = {
            'mpid': f"{i:05}",
            'structure': struct,
            'f_energy': 0.0,
            'ehull': 0.0,
            'label': 0,
            # 'stable': 1,
            'smact_valid': charge_neutrality(struct),
            'occupy_ratio': vol_density(struct)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} structures into DataFrame.")
    return df


def classify_stability(model, dataset, loss_fn, scaler, batch_size):
    """Classifies stability using the given GNN model."""
    logger.info("Classifying material stability...")
    te_set = torch.utils.data.Subset(dataset, range(len(dataset)))
    loader = DataLoader(te_set, batch_size=batch_size)
    print(loader)
    return generate_dataframe(model, loader, loss_fn, scaler, device)


def generate_cif_files(df, cif_dir):
    """Generates CIF files for filtered structures."""
    logger.info("Generating CIF files...")
    os.makedirs(cif_dir, exist_ok=True)
    for i, row in df.iterrows():
        mpid, astruct = row['mpid'], row['structure']
        pstruct = ase2pmg(astruct)
        filename = join(cif_dir, f"{mpid}.cif")
        try:
            pstruct.to(fmt="cif", filename=filename)
        except Exception as e:
            logger.error(f"Error generating CIF for {mpid}: {e}")
    os.system(f'zip -r {cif_dir}.zip {cif_dir}')
    logger.info("CIF generation completed.")


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

    model_A = load_model(stab_pred_name_A, param_dict_A)
    model_B = load_model(stab_pred_name_B, param_dict_B)
    model_C = load_model_mag(mag_pred_name, param_dict_C)
    
    r_max_A, target_A, descriptor_A, scaler_A, nearest_neighbor_A, batch_size_A = [param_dict_A[k] for k in ['r_max', 'target', 'descriptor', 'scaler', 'nearest_neighbor', 'batch_size']]
    r_max_B, target_B, descriptor_B, scaler_B, nearest_neighbor_B, batch_size_B = [param_dict_B[k] for k in ['r_max', 'target', 'descriptor', 'scaler', 'nearest_neighbor', 'batch_size']]
    r_max_C, target_C, descriptor_C, scaler_C, nearest_neighbor_C, batch_size_C = [param_dict_B[k] for k in ['r_max', 'target', 'descriptor', 'scaler', 'nearest_neighbor', 'batch_size']]

    # Process data
    astruct_list_full = process_data(use_path)
    df = load_df(astruct_list_full)
    
    # [1] Filter by SMACT validity
    df1 = df[df['smact_valid']].reset_index(drop=True)
    logger.info(f"[1] Filtered {len(df1)}/{len(df)} materials by SMACT validity.")
    
    # [2] Filter by space occupation ratio
    df2 = df1[df1['occupy_ratio'] < 1.7].reset_index(drop=True)
    logger.info(f"[2] Filtered {len(df2)}/{len(df1)} materials by space occupation ratio.")

    # [3] Classify stability by model A
    dataset_A = Dataset_Cls(df2, r_max_A, target_A, descriptor_A, scaler_A, nearest_neighbor_A)
    df_A = classify_stability(model_A, dataset_A, loss_fn, scaler_A, batch_size_A)
    id_stable_A = df_A[df_A['pred'] == 1]['id'].values
    logger.info(f"[3] Model A classified {len(id_stable_A)}/{len(df2)} materials as stable.")
    print('id_stable_A: ', id_stable_A)
    df3 = df2[df2['mpid'].isin(id_stable_A)].reset_index(drop=True)
    
    # [4] Classify stability by model B
    dataset_B = Dataset_Cls(df3, r_max_B, target_B, descriptor_B, scaler_B, nearest_neighbor_B)
    df_B = classify_stability(model_B, dataset_B, loss_fn, scaler_B, batch_size_B)
    id_stable_B = df_B[df_B['pred'] == 1]['id'].values
    logger.info(f"[4] Model B classified {len(id_stable_B)}/{len(df3)} materials as stable.")
    print('id_stable_B: ', id_stable_B)
    df4 = df3[df3['mpid'].isin(id_stable_B)].reset_index(drop=True)
    
    # [4] Classify stability by model B
    dataset_C = Dataset_Cls(df4, r_max_C, 'label', descriptor_C, scaler_C, nearest_neighbor_C)
    df_C = classify_stability(model_C, dataset_C, loss_fn, scaler_C, batch_size_C)
    id_stable_C = df_C[df_C['pred'] == 1]['id'].values
    logger.info(f"[5] Model C classified {len(id_stable_C)}/{len(df4)} materials as stable.")
    print('id_stable_B: ', id_stable_C)
    
    df5 = df4[df4['mpid'].isin(id_stable_C)].reset_index(drop=True)
    
    cif_dir= join(jobdir, use_name + '_mag')
    if args.gen_cif:
        generate_cif_files(df5, cif_dir)


    # Write logs from memory to the file
    log_file = join(cif_dir, f"{use_name}.log")
    with open(log_file, 'w') as f:
        f.write(log_buffer.getvalue())

    logger.info(f"Logs saved to {log_file}")


if __name__ == "__main__":
    main()
