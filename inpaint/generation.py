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
import math #!
import random   #!
from sample import chemical_symbols
from p_tqdm import p_map

import pdb

import os

from inpcrysdiff.pl_modules.diffusion_w_type import sample_inpaint  #!


train_dist = {
    'perov_5' : [0, 0, 0, 0, 0, 1],
    'carbon_24' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    'mp_20' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301]
}


def diffusion(loader, model, step_lr):      #TODO

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample_inpaint(batch, step_lr = step_lr)      #TODO
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )

class SampleDataset(Dataset):      #TODO

    def __init__(self, dataset, total_num, bond_mu_sigma, known_species, arch_type):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_known_arch = {'kagome': 3, 'honeycomb': 2, 'triangular': 1}    #!
        self.num_known_options = [self.num_known_arch[k] for k in self.arch_type]   #!
        self.bond_mu, self.bond_sigma = bond_mu_sigma
        self.lattice_a_mu_sigma = {1:[self.bond_mu, self.bond_sigma], 2:[math.sqrt(3)*self.bond_mu, math.sqrt(3)*self.bond_sigma], 
                                   3:[2*self.bond_mu, 2*self.bond_sigma]}
        self.num_known = np.random.choice(self.num_known_options, self.total_num) #, p = None)  #!
        # self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)   #!
        self.is_carbon = dataset == 'carbon_24'
        
        self.bond_mu_sigma = bond_mu_sigma  #!
        self.known_species = known_species  #!
        self.known_nelem = []
        self.arch_type = arch_type  #!
        self.frac_coords_archs = {1: torch.tensor([[0.0, 0.0, 0.0]]), 2:torch.tensor([[1/3, 2/3, 0.0], [2/3, 1/3, 0.0]]), 
                                  3: torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])}

        self.get_num_atoms()    #!
        self.get_known_fcoords()    #!
        self.get_known_lattice()    #!
        self.get_known_atypes() #!
        
    def get_num_atoms(self):
        # self.distributions = {} #!
        # self.num_atomss = {}
        self.num_atoms = np.zeros(self.total_num)
        for n in self.num_known_options:
            # Make a copy of self.distribution for each key
            type_distribution = self.distribution.copy()
            # Set the first n elements to 0
            type_distribution[:n+1] = [0] * (n + 1)
            num_atoms_ = np.random.choice(len(type_distribution), self.total_num, p = type_distribution)
            mask = self.num_known == n
            self.num_atoms += mask * num_atoms_
            # self.distributions[n] = type_distribution
            # self.num_atomss[n] = np.random.choice(len(type_distribution), total_num, p = type_distribution) 

    def get_known_fcoords(self):    #!
        # use arch_type, num_atoms
        # return frac_coords_known, mask_x
        self.num_unk = self.num_atoms - self.num_known
        self.frac_coords_list_known = [self.frac_coords_archs[n] for n in self.num_known]   # function of num_known, num_atoms, (num_known_arch?)  #?
        self.frac_coords_list_zeros = [torch.zeros(natm, 3) for natm in self.num_atoms]
        self.frac_coords_list = []
        self.mask_x_list = []
        for i, (fcoords_kn, fcoords_z) in enumerate(zip(self.frac_coords_list_known, self.frac_coords_list_zeros)):
            n_kn = fcoords_kn.shape[0]
            fcoords, mask = fcoords_z.copy(), fcoords_z.copy()
            fcoords[:n_kn, :] = fcoords_kn
            mask[:n_kn, :] = torch.ones_like(fcoords_kn) 
            self.frac_coords_list.append(fcoords)
            self.mask_x_list.append(mask[:, 0].flatten())
    
    def get_known_lattice(self):    #!
        # use arch_type, bond_mu_sigma
        # return lattice_known, mask_l
        self.lattice_list = [] # Lattice matrices (batch, 3, 3). vec{a} and vec{b} are determined, but vec{c} are random.    #?
        for i, n_kn in enumerate(self.num_known):
            mu, sigma = self.lattice_a_mu_sigma[n_kn]
            a, c = abs(random.gauss(mu, sigma)), mu
            lattice = self.get_hexagonal_cell(a,c)
            self.lattice_list.append(lattice)
        self.mask_l_list = [torch.tensor([1,1,0]) for _ in range(self.total_nu)]   # (batch, 3, 3) stack torch.tensor([[1,1,1],[1,1,1],[0,0,0]]).

    def get_known_atypes(self):    #!
        # use arch_type, known_species, num_atoms
        # run this after get_known_fcoords(), so that we then already know return frac_coords_known, mask_x, num_known. 
        # return atom_types_known, mask_t
        self.atom_types_list = [] # function of num_known, num_atoms, (num_known_arch?)
        for i, (n_kn, natm) in enumerate(zip(self.num_known, self.num_atoms)):
            type_known = random.choice(self.known_species)
            types_idx_known = [chemical_symbols.index(type_known)] * n_kn
            types_unk = random.choices(chemical_symbols, k=natm-n_kn)
            types_idx_unk = [chemical_symbols.index(elem) for elem in types_unk]
            types = types_idx_known.extend((types_idx_unk))
            self.atom_types_list.append(torch.tensor(types))
        self.mask_t_list = self.mask_x_list   # function of self.mask_x. 
    
    def get_hexagonal_cell(self, a, c):
        a1 = torch.tensor([a, 0, 0], dtype=torch.float)
        a2 = torch.tensor([a * -0.5, a * math.sqrt(3) / 2, 0], dtype=torch.float)
        a3 = torch.tensor([0, 0, c], dtype=torch.float)
        lattice_matrix = torch.stack([a1, a2, a3], dim=0)
        return lattice_matrix

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        num_atom = self.num_atoms[index]
        num_known_ = self.num_known[index]
        frac_coords_known_=self.frac_coords_list[index]
        lattice_known_=self.lattice_list[index]
        atom_types_known_=self.atom_types_list[index]
        mask_x_, mask_l_, mask_t_ = [a[index] for a in [self.mask_x_list, self.mask_l_list, self.mask_t_list]]
        
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
            num_known=num_known_,    #!
            frac_coords_known=frac_coords_known_,    #!
            lattice_known=lattice_known_,    #!
            atom_types_known=atom_types_known_,    #!
            mask_x=mask_x_,    #!
            mask_l=mask_l_,    #!
            mask_t=mask_t_,    #!
            
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)

    if torch.cuda.is_available():
        model.to('cuda')

    model.sample_inpaint = sample_inpaint.__get__(model)    #!

    print('Evaluate the diffusion model.')

    test_set = SampleDataset(args.dataset, args.batch_size * args.num_batches_to_samples, args.bond_mu_sigma, args.known_species, args.arch)     #!
    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['gen'][args.dataset]

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(test_loader, model, step_lr)

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
    }, model_path / gen_out_name)
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--label', default='')
    
    parser.add_argument('--bond_mu_sigma', default=[5, 2])  #!
    parser.add_argument('--known_species', default=['Fe', 'Co', 'Ni', 'Gd', 'Dy', 'Nd', 'Ru'])  #!
    parser.add_argument('--arch', default=['kagome']) # 'kagome', 'honeycomb', 'triangular' #!
    
    args = parser.parse_args()


    main(args)
