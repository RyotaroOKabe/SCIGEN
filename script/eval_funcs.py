import os
from os.path import join
import sys
import argparse
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
# Local imports
sys.path.append('../')
from config_scigen import gnn_eval_path
sys.path.append(gnn_eval_path)
from script.mat_utils import (
    get_pstruct_list, output_gen, lattice_params_to_matrix_torch, ase2pmg,
    chemical_symbols, vol_density, charge_neutrality
)
from gnn_eval.utils.model_class import GraphNetworkClassifier
from gnn_eval.utils.model_class_mag import GraphNetworkClassifierMag
from gnn_eval.utils.output import generate_dataframe

def parse_arguments(job_dir, out_name):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Material Stability Classification Pipeline")
    parser.add_argument('--label', default=out_name, type=str, help="Output label name")
    parser.add_argument('--job_dir', default=job_dir, type=str, help="Job directory")
    parser.add_argument('--gen_cif', type=lambda x: x.lower() == 'true', default=True, help="Generate CIF files")
    parser.add_argument('--gen_movie', type=lambda x: x.lower() == 'true', default=False, help="Generate movies of structures")
    parser.add_argument('--screen_mag', type=lambda x: x.lower() == 'true', default=False, help="Screen magnetic materials")
    return parser.parse_args()


def load_model(model_name, param_dict, device, logger):
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


def load_model_mag(model_name, param_dict, device, logger):
    """Loads and prepares a GNN model."""
    logger.info(f"Loading model (mag): {model_name}")
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


def process_data(output_path, logger):
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


def load_df(astruct_list, logger):
    # logger.info("Loading structures into DataFrame...")
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


def classify_stability(model, dataset, loss_fn, scaler, batch_size, device, logger):
    """Classifies stability using the given GNN model."""
    logger.info("Classifying material stability...")
    te_set = torch.utils.data.Subset(dataset, range(len(dataset)))
    loader = DataLoader(te_set, batch_size=batch_size)
    print(loader)
    return generate_dataframe(model, loader, loss_fn, scaler, device)


def generate_cif_files(df, cif_dir, logger):
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
