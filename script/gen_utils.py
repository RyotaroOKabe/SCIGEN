import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data import Dataset
from sc_utils import *    
import numpy as np
import random   
import pickle as pkl
from sc_utils import *  

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
            0.08995430424528301],
    'uniform' : [0.0] + [1.0]*30
}
# combine two dictionary into one
train_dist = {**train_dist, **train_dist_sc}

# reference of metallic radii: https://www.sciencedirect.com/science/article/pii/0022190273803574
metallic_radius = {'Mn': 1.292, 'Fe': 1.277, 'Co': 1.250, 'Ni': 1.246, 'Ru': 1.339, 'Nd': 1.821, 'Gd': 1.802, 'Tb': 1.781, 'Dy': 1.773, 'Yb': 1.940}
with open('./data/kde_bond.pkl', 'rb') as file:
    kde_dict = pkl.load(file)

def convert_seconds_short(sec):
    minutes, seconds = divmod(int(sec), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"

# I am trying to pass None keyword as a command line parameter to a script as follows, if I explicity mention Category=None it works but the moment I switch to sys.argv[1] it fails, any pointers on how to fix this? Please wriite a function that takes a list of strings and returns a dictionary with the key as the first letter of the string and the value as the string itself. If the first letter is already a key, add the string to the list of values.
def parse_none_or_value(argument, obj=float):
    if argument == 'None':
        return None
    return obj(argument)

class SampleDataset(Dataset):      
    def __init__(self, dataset, natm_range, total_num, bond_sigma_per_mu, use_min_bond_len, known_species, sc_list, c_vec_cons, reduced_mask, seed, device):
        super().__init__()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # get self.natm_min, self.natm_max by first convert natm_range into a list of integers, and get min and max
        self.natm_range = sorted([int(i) for i in natm_range])
        self.natm_min, self.natm_max = self.natm_range[0], self.natm_range[-1]
        print('natm_range: ', [self.natm_min, self.natm_max])
        self.total_num = total_num 
        self.bond_sigma_per_mu = bond_sigma_per_mu
        self.use_min_bond_len = use_min_bond_len
        self.known_species = known_species   
        self.sc_options = sc_list   
        self.sc_list = random.choices(self.sc_options, k=self.total_num)  
        self.c_vec_cons = c_vec_cons
        self.reduced_mask = reduced_mask
        self.device = device

        if dataset == 'uniform':  
            self.distributions_dict = {sc: train_dist[dataset][:self.natm_max+1] for sc in self.sc_options} 
        else:
            self.distributions_dict = {sc: train_dist[sc][:self.natm_max+1] for sc in self.sc_options}  
        
        # print('natm_range: ', [self.natm_min, self.natm_max])
        # print('distributions_dict: ', self.distributions_dict)
        self.type_known_list = random.choices(self.known_species, k=self.total_num)
        self.num_known_list =[num_known_dict[sc] for sc in self.sc_list]
        if self.bond_sigma_per_mu is not None:  
            # print('Sample bond length from Gaussian')
            self.radii_list = [metallic_radius[s] for s in self.type_known_list]
            self.bond_mu_list = [2*r for r in self.radii_list]
            self.bond_sigma_list = [b*self.bond_sigma_per_mu for b in self.bond_mu_list]
            self.bond_len_list = [np.random.normal(self.bond_mu_list[i], self.bond_sigma_list[i]) for i in range(self.total_num)]
        else:
            # print('Sample bond length from KDE')
            # self.bond_len_list = [kde_dict[s].resample(1).item() for s in self.type_known_list]
            self.min_bond_len_dict = {elem: 0 for elem in chemical_symbols}
            if use_min_bond_len:
                for k, v in metallic_radius.items():
                    self.min_bond_len_dict[k] = 2*v
            self.bond_len_list = [max(kde_dict[s].resample(1).item(), self.min_bond_len_dict[s]) for s in self.type_known_list]
        self.frac_z_known_list = [random.uniform(0, 1) for _ in range(self.total_num)]
        self.is_carbon = dataset == 'carbon_24' 

        self.frac_coords_list = []
        self.atom_types_list = []
        self.lattice_list = []
        self.mask_x_list, self.mask_t_list, self.mask_l_list =  [], [], [] 
        self.get_num_atoms()
        for i, (num_atom, sc, type_known, bond_len, frac_z_known) in enumerate(zip(self.num_atom_list, self.sc_list, self.type_known_list, self.bond_len_list, self.frac_z_known_list)):
            sc_obj = al_dict[sc] 
            material = sc_obj(bond_len, num_atom, type_known, frac_z_known, self.c_vec_cons, self.reduced_mask, self.device)    #TODO: add option about c_len scale and mask_l
            self.frac_coords_list.append(material.frac_coords)
            self.atom_types_list.append(material.atom_types)
            self.lattice_list.append(material.cell)
            self.mask_x_list.append(material.mask_x)
            self.mask_t_list.append(material.mask_t)
            self.mask_l_list.append(material.mask_l)
        self.generate_dataset()
   
        
    def get_num_atoms(self):
        self.num_atom_list = []
        for (n_kn, sc) in zip(self.num_known_list, self.sc_list):    
            type_distribution = self.distributions_dict[sc].copy()
            # Set the first n elements to 0
            type_distribution[:n_kn+1] = [0] * (n_kn + 1)
            type_distribution[:self.natm_min] = [0] * self.natm_min
            sum_p = sum(type_distribution)
            assert sum_p > 0.0, f"sum_p is {sum_p}, type_distribution: {type_distribution}"
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
    
