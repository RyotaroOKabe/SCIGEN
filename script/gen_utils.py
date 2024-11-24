import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
import random   
import pickle as pkl
from sc_utils import *  
from sc_natm import natm_dist


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
    def __init__(self, dataset, 
                 natm_range, 
                 total_num, 
                 bond_sigma_per_mu, 
                 use_min_bond_len, 
                 known_species, 
                 sc_list, 
                 frac_z,
                 c_vec_cons, 
                 reduced_mask, 
                 seed, 
                 device):
        super().__init__()
        self.seed = seed
        set_seeds(self.seed)
        self.dataset = dataset
        self.natm_range = sorted([int(i) for i in natm_range])
        self.natm_min, self.natm_max = self.natm_range[0], self.natm_range[-1]
        print('natm_range: ', [self.natm_min, self.natm_max])
        self.total_num = total_num 
        self.bond_sigma_per_mu = bond_sigma_per_mu
        self.use_min_bond_len = use_min_bond_len
        self.known_species = known_species   
        self.sc_options = sc_list   
        self.sc_list = random.choices(self.sc_options, k=self.total_num)  
        self.frac_z = frac_z
        self.c_vec_cons = c_vec_cons
        self.reduced_mask = reduced_mask
        self.device = device
        self.num_atom_distribution()
        self.process()
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


    def num_atom_distribution(self):
        if self.dataset == 'uniform':  
            self.distributions_dict = {sc: natm_dist[self.dataset][:self.natm_max+1] for sc in self.sc_options} 
        else:
            self.distributions_dict = {sc: (natm_dist[sc][:self.natm_max+1] if sc in natm_dist.keys() 
                                            else natm_dist[self.dataset][:self.natm_max+1]) for sc in self.sc_options}

    def process(self):
        self.type_known_list = random.choices(self.known_species, k=self.total_num)
        # self.num_known_list =[num_known_dict[sc] for sc in self.sc_list]
        if self.bond_sigma_per_mu is not None:  
            # print('Sample bond length from Gaussian')
            self.radii_list = [metallic_radius[s] for s in self.type_known_list]
            self.bond_mu_list = [2*r for r in self.radii_list]
            self.bond_sigma_list = [b*self.bond_sigma_per_mu for b in self.bond_mu_list]
            self.bond_len_list = [np.random.normal(self.bond_mu_list[i], self.bond_sigma_list[i]) for i in range(self.total_num)]
        else:
            # print('Sample bond length from KDE')
            self.min_bond_len_dict = {elem: 0 for elem in chemical_symbols}
            if self.use_min_bond_len:
                for k, v in metallic_radius.items():
                    self.min_bond_len_dict[k] = 2*v
            self.bond_len_list = [max(kde_dict[s].resample(1).item(), self.min_bond_len_dict[s]) for s in self.type_known_list]
        self.frac_z_known_list = [random.uniform(0, 1) for _ in range(self.total_num)]
        self.is_carbon = self.dataset == 'carbon_24' 

        self.num_known_list = []
        self.num_atom_list = []
        self.frac_coords_list = []
        self.atom_types_list = []
        self.lattice_list = []
        self.mask_x_list, self.mask_t_list, self.mask_l_list =  [], [], [] 
        # self.get_num_atoms()
        for i, (sc, type_known, bond_len, frac_z_known) in enumerate(zip(self.sc_list, self.type_known_list, self.bond_len_list, self.frac_z_known_list)):
            sc_obj = sc_dict[sc] 
            sc_material = sc_obj(bond_len=bond_len, 
                              num_atom=None, 
                              type_known=type_known, 
                              frac_z=frac_z_known, 
                              c_vec_cons=self.c_vec_cons, 
                              reduced_mask=self.reduced_mask, 
                              device=self.device)    
            num_known = sc_material.num_known
            type_distribution = self.distributions_dict[sc].copy()
            type_distribution[:num_known+1] = [0] * (num_known + 1)
            type_distribution[:self.natm_min] = [0] * self.natm_min
            sum_p = sum(type_distribution)
            assert sum_p > 0.0, f"sum_p is {sum_p}, type_distribution: {type_distribution}"
            type_distribution_norm = [p / sum_p for p in type_distribution] 
            num_atom = np.random.choice(len(type_distribution_norm), 1, p = type_distribution_norm)[0]
            sc_material.num_atom = num_atom
            sc_material.frac_coords_all()
            sc_material.atm_types_all()

            self.num_known_list.append(num_known)
            self.num_atom_list.append(num_atom)
            self.frac_coords_list.append(sc_material.frac_coords)
            self.atom_types_list.append(sc_material.atom_types)
            self.lattice_list.append(sc_material.cell)
            self.mask_x_list.append(sc_material.mask_x)
            self.mask_t_list.append(sc_material.mask_t)
            self.mask_l_list.append(sc_material.mask_l)
            
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
    
def set_seeds(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)