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
import pickle as pkl

import pdb

import os
from inpcrysdiff.pl_modules.diffusion_w_type import MAX_ATOMIC_NUM  


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

# metallic radius reference (tempolary): https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)#Note_b
metallic_radius = {'Mn': 1.27, 'Fe': 1.26, 'Co': 1.25, 'Ni': 1.24, 'Ru': 1.34, 'Nd': 1.814, 'Gd': 1.804, 'Dy': 1.781}
with open('./data/kde_bond.pkl', 'rb') as file:
    kde_dict = pkl.load(file)

lattice_types = {'kagome': 'hexagonal', 'honeycomb': 'hexagonal', 'triangular': 'hexagonal'}

def convert_seconds_short(sec):
    minutes, seconds = divmod(int(sec), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"


def hexagonal_cell(a, c, device):
    l1 = torch.tensor([a, 0, 0], dtype=torch.float, device=device)
    l2 = torch.tensor([a * -0.5, a * math.sqrt(3) / 2, 0], dtype=torch.float, device=device)
    l3 = torch.tensor([0, 0, c], dtype=torch.float, device=device)
    lattice_matrix = torch.stack([l1, l2, l3], dim=0)
    return lattice_matrix

def square_cell(a, c, device):
    l1 = torch.tensor([a, 0, 0], dtype=torch.float, device=device)
    l2 = torch.tensor([0, a, 0], dtype=torch.float, device=device)
    l3 = torch.tensor([0, 0, c], dtype=torch.float, device=device)
    lattice_matrix = torch.stack([l1, l2, l3], dim=0)
    return lattice_matrix

def frac_coords_all(n_atm, frac_known):
    fcoords_zero = torch.zeros(int(n_atm), 3)
    n_kn = frac_known.shape[0]
    fcoords, mask = fcoords_zero.clone(), fcoords_zero.clone()
    fcoords[:n_kn, :] = frac_known
    mask[:n_kn, :] = torch.ones_like(frac_known) 
    mask = mask[:, 0].flatten()
    return fcoords, mask

def atm_types_all(n_atm, n_kn, type_known):
    types_idx_known = [chemical_symbols.index(type_known)] * n_kn
    types_unk = random.choices(chemical_symbols[1:MAX_ATOMIC_NUM+1], k=int(n_atm-n_kn))
    types_idx_unk = [chemical_symbols.index(elem) for elem in types_unk]
    types = torch.tensor(types_idx_known + types_idx_unk)
    mask = torch.zeros_like(types)
    mask[:n_kn] = 1
    return types, mask

class AL_Template():
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        self.bond_len = bond_len
        self.num_atom = num_atom
        self.type_known = type_known
        self.frac_z = frac_z
        self.device = device
        self.mask_l = torch.tensor([[1,1,0]]) 


class AL_Triangular(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 1
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]])
        self.a_len = 1 * self.bond_len
        self.c_len = 1 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)


class AL_Honeycomb(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 2
        self.frac_known = torch.tensor([[1/3, 2/3, self.frac_z], [2/3, 1/3, self.frac_z]])
        self.a_len = math.sqrt(3) * self.bond_len
        self.c_len = math.sqrt(3) * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)

class AL_Kagome(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 2
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z], [0.5, 0.0, self.frac_z], [0.0, 0.5, self.frac_z]])
        self.a_len = 2 * self.bond_len
        self.c_len = 2 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = hexagonal_cell(self.a_len, self.c_len, self.device)

class AL_Square(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 1
        self.frac_known = torch.tensor([[0.0, 0.0, self.frac_z]])
        self.a_len = 1 * self.bond_len
        self.c_len = 1 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = square_cell(self.a_len, self.c_len, self.device)

class AL_4_8_2_Square(AL_Template):
    def __init__(self, bond_len, num_atom, type_known, frac_z, device):
        super().__init__(bond_len, num_atom, type_known, frac_z, device)
        self.num_known = 4
        x = 1/(2+math.sqrt(2))
        self.frac_known = torch.tensor([[x, 0.0, self.frac_z],
                                        [1-x, 0.0, self.frac_z],
                                        [0.0, x, self.frac_z],
                                        [0.0, 1-x, self.frac_z]])
        self.a_len = (1+math.sqrt(2))*self.bond_len
        self.c_len = 1 * self.bond_len
        self.frac_coords, self.mask_x = frac_coords_all(self.num_atom, self.frac_known)
        self.atom_types, self.mask_t = atm_types_all(self.num_atom, self.num_known, self.type_known)
        self.cell = square_cell(self.a_len, self.c_len, self.device)

al_dict = {'triangular': AL_Triangular, 'honeycomb': AL_Honeycomb, 'kagome': AL_Kagome, 'square': AL_Square, '4_8_2_square': AL_4_8_2_Square}
num_known_dict = {'triangular': 1, 'honeycomb': 2, 'kagome': 3, 'square': 1, '4_8_2_square': 4}

class SampleDataset(Dataset):      
    def __init__(self, dataset, max_atom, total_num, bond_sigma_per_mu, known_species, arch_type, device):
        super().__init__()
        self.total_num = total_num  #!
        self.distribution = train_dist[dataset][:max_atom+1] #! modify to chhange the samplingg range options. 
        # self.num_known_arch = {'kagome': 3, 'honeycomb': 2, 'triangular': 1}    
        self.bond_sigma_per_mu = bond_sigma_per_mu      #!
        self.known_species = known_species      #!
        self.arch_options = arch_type      #!
        self.device = device
        # self.frac_coords_archs = {1: torch.tensor([[0.0, 0.0, 0.0]]), 2:torch.tensor([[1/3, 2/3, 0.0], [2/3, 1/3, 0.0]]), 
        #                           3: torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])}
        self.arch_list = random.choices(self.arch_options, k=self.total_num)
        self.type_known_list = random.choices(self.known_species, k=self.total_num)
        self.num_known_list =[num_known_dict[arch] for arch in self.arch_list]
        if self.bond_sigma_per_mu is not None:  #!
            print('Sample bond length from Gaussian')
            self.radii_list = [metallic_radius[s] for s in self.type_known_list]
            self.bond_mu_list = [2*r for r in self.radii_list]
            self.bond_sigma_list = [b*self.bond_sigma_per_mu for b in self.bond_mu_list]
            self.bond_len_list = [np.random.normal(self.bond_mu_list[i], self.bond_sigma_list[i]) for i in range(self.total_num)]
        else:
            print('Sample bond length from KDE')
            self.bond_len_list = [kde_dict[s].resample(1).item() for s in self.type_known_list]
        self.frac_z_known_list = [random.uniform(0, 1) for _ in range(self.total_num)]
        # self.num_known_options = [self.num_known_arch[k] for k in self.arch_type]   #! first sample AL types, then go to AL class
        # self.num_known = np.random.choice(self.num_known_options, self.total_num) #, p = None)      #!
        self.is_carbon = dataset == 'carbon_24' #!

        self.frac_coords_list = []
        self.atom_types_list = []
        self.lattice_list = []
        self.mask_x_list, self.mask_t_list, self.mask_l_list =  [], [], [] 
        self.get_num_atoms()
        for i, (num_atom, arch, type_known, bond_len, frac_z_known) in enumerate(zip(self.num_atom_list, self.arch_list, self.type_known_list, self.bond_len_list, self.frac_z_known_list)):
            al_obj = al_dict[arch]
            material = al_obj(bond_len, num_atom, type_known, frac_z_known, self.device)
            self.frac_coords_list.append(material.frac_coords)
            self.atom_types_list.append(material.atom_types)
            self.lattice_list.append(material.cell)
            self.mask_x_list.append(material.mask_x)
            self.mask_t_list.append(material.mask_t)
            self.mask_l_list.append(material.mask_l)
        self.generate_dataset()
   
        
    def get_num_atoms(self):
        self.num_atom_list = []
        for n_kn in self.num_known_list:
            # Make a copy of self.distribution for each key
            type_distribution = self.distribution.copy()
            # Set the first n elements to 0
            type_distribution[:n_kn+1] = [0] * (n_kn + 1)
            sum_p = sum(type_distribution)
            type_distribution_norm = [p / sum_p for p in type_distribution] 
            num_atom = np.random.choice(len(type_distribution_norm), 1, p = type_distribution_norm)[0]
            self.num_atom_list.append(num_atom)

    def generate_dataset(self):
        self.data_list = []
        for index in tqdm(range(self.total_num)):
            num_atom = np.round(self.num_atom_list[index]).astype(np.int64)
            num_known_ = np.round(self.num_known_list[index]).astype(np.int64)
            frac_coords_known_=self.frac_coords_list[index]
            lattice_known_=self.lattice_list[index].unsqueeze(0)
            atom_types_known_=self.atom_types_list[index]
            mask_x_, mask_l_, mask_t_ = [a[index] for a in [self.mask_x_list, self.mask_l_list, self.mask_t_list]]
            
            data = Data(
                num_atoms=torch.LongTensor([num_atom]),
                num_nodes=num_atom,
                num_known=num_known_,    
                frac_coords_known=frac_coords_known_,    
                lattice_known=lattice_known_,    
                atom_types_known=atom_types_known_,    
                mask_x=mask_x_,    
                mask_l=mask_l_,    
                mask_t=mask_t_,    
                
            )
            if self.is_carbon:
                data.atom_types = torch.LongTensor([6] * num_atom)
            self.data_list.append(data) 


    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        return self.data_list[index]
    
