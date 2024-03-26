#%%
import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list, recommand_step_lr

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
import math 
import random   
from sample import chemical_symbols
from p_tqdm import p_map

import pdb

import os

from inpcrysdiff.pl_modules.diffusion_w_type import sample_inpaint, MAX_ATOMIC_NUM  
from inpaint_utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


def diffusion(loader, model, step_lr, save_traj):

    frac_coords = []
    num_atoms = []
    num_known = []  
    atom_types = []
    lattices = []
    all_frac_coords = []
    all_atom_types = []
    all_lattices = []
    all_lengths, all_angles = [], []    # ignore if we save traj
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample_inpaint(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        num_known.append(outputs['num_known'].detach().cpu()) 
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        if save_traj:
            all_frac_coords.append(traj['all_frac_coords'].detach().cpu())
            all_atom_types.append(traj['atom_types'].detach().cpu())
            all_lattices.append(traj['all_lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    num_known = torch.cat(num_known, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)
    if save_traj: 
        all_frac_coords = torch.cat(all_frac_coords, dim=1)
        all_atom_types = torch.cat(all_atom_types, dim=1)
        all_lattices = torch.cat(all_lattices, dim=1)
        all_lengths, all_angles = lattices_to_params_shape(all_lattices)    # works for all-time outputs

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, num_known, all_frac_coords, all_atom_types, all_lattices, all_lengths, all_angles 
    )


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)

    if torch.cuda.is_available():
        model.to('cuda')

    model.sample_inpaint = sample_inpaint.__get__(model)    

    print('Evaluate the diffusion model.')

    test_set = SampleDataset(args.dataset, args.max_atom, args.batch_size * args.num_batches_to_samples, args.bond_sigma_per_mu, args.known_species, args.arch, device)     
    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['gen'][args.dataset]

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, num_known, \
        all_frac_coords, all_atom_types, all_lattices, all_lengths, all_angles) = diffusion(test_loader, model, step_lr, args.save_traj)    

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'

    run_time = time.time() - start_time
    total_num = args.batch_size * args.num_batches_to_samples
    print(f'Total outputs: {args.num_batches_to_samples} samples x {args.batch_size} batches = {total_num} materials')
    print(f'run time: {run_time} sec = {convert_seconds_short(run_time)}')
    print(f'{run_time/args.num_batches_to_samples} sec/sample')
    print(f'{run_time/total_num} sec/material')
    
    torch.save({
        'eval_setting': args,
        'num_atoms': num_atoms,
        'num_known': num_known,
        'frac_coords': frac_coords,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'all_frac_coords': all_frac_coords,
        'all_atom_types': all_atom_types,
        'all_lengths': all_lengths,
        'all_angles': all_angles,
        'time': run_time,
    }, model_path / gen_out_name)
      
#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--label', default='')


    # parser.add_argument('--n_step_each', default=100, type=int) #?
    # parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)

    parser.add_argument('--max_atom', default=20, type=int)
    parser.add_argument('--bond_sigma_per_mu', default=0.1)  
    parser.add_argument('--known_species', default=['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Dy'])  
    parser.add_argument('--arch', default=['kagome']) # 'kagome', 'honeycomb', 'triangular' 
    
    args = parser.parse_args()


    main(args)
