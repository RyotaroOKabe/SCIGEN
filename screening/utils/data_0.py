import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
import pickle as pkl
import pandas as pd
from tqdm import tqdm
try:
    from mp_api.client import MPRester
except:
    print("Error: from mp_api.client import MPRester")
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
import mendeleev as md  # replace with common
# from utils.common import *
from abc import ABC, abstractmethod
from ase import Atoms
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.local_env import CrystalNN
from tqdm import tqdm
import time


chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

class MD():
    def __init__(self):
        self.radius, self.pauling, self.ie, self.dip = {}, {}, {}, {}
        for atomic_number in range(1, 119):
            ele = md.element(atomic_number)
            # print(str(ele))
            self.radius[atomic_number] = ele.atomic_radius
            self.pauling[atomic_number] = ele.en_pauling
            ie_dict = ele.ionenergies
            self.ie[atomic_number] = ie_dict[min(list(ie_dict.keys()))] if len(ie_dict)>0 else 0
            self.dip[atomic_number] = ele.dipole_polarizability

md_class = MD()

def pm2ase(pstruct):
    return Atoms(list(map(lambda x: x.symbol, pstruct.species)),
                    positions = pstruct.cart_coords.copy(),
                    cell = pstruct.lattice.matrix.copy(), 
                    pbc=True)

# Initialize the MPRester
def mp_struct_ehull(api_key, mat_input=None, use_ase=True, save_file=None):
    with MPRester(api_key) as mpr:
        mp_dict = {}

        # Determine whether to use a list of material IDs or fetch a number of materials based on `material_input`
        if isinstance(mat_input, list):  # List of material IDs provided
            materials_to_fetch = mat_input
            fetch_by_id = True
        elif isinstance(mat_input, int):  # Maximum number of materials provided
            materials_to_fetch = range(mat_input)  # Use range to generate indices
            fetch_by_id = False
            summaries = mpr.summary.search(fields=["material_id", "energy_above_hull", "formation_energy_per_atom"], all_fields=False)
        else:
            raise ValueError("mat_input must be either a list of material IDs or an integer representing the number of materials to fetch.")
        for i, identifier in tqdm(enumerate(materials_to_fetch), desc="Processing"):
            try:
                if fetch_by_id:
                    # Fetching by material ID
                    mpid = identifier
                    pstruct = mpr.get_structure_by_material_id(mpid)
                    summary = mpr.summary.get_data_by_id(mpid, fields=["material_id", "energy_above_hull", "formation_energy_per_atom"])
                else:
                    # Fetching by index, only for demo purposes; replace with actual logic to fetch materials without IDs
                    # This branch assumes you're fetching the first `material_input` materials, adjust according to your API's capabilities
                    # summary = mpr.summary.search(fields=["material_id", "energy_above_hull", "formation_energy_per_atom"], all_fields=False)[i]
                    summary = summaries[i]
                    mpid = summary.material_id
                    pstruct = mpr.get_structure_by_material_id(mpid)

                e_above_hull = summary.energy_above_hull
                f_energy_per_atom = summary.formation_energy_per_atom

                if pd.isna(e_above_hull) or pd.isna(f_energy_per_atom):
                    continue

                if use_ase:
                    struct = pm2ase(pstruct)
                else:
                    struct = pstruct

                mp_dict[mpid] = {"structure": struct, "ehull": e_above_hull, 'f_energy': f_energy_per_atom}
                print(f'[{i}] mpid:', mpid, 'ehull:', e_above_hull, 'f_energy: ', f_energy_per_atom)
            except Exception as e:
                print(f'Error loading material with ID {mpid}: {e}')

    data_for_df = []
    for mpid, values in mp_dict.items():
        # Assuming you want to keep the full mpid string, otherwise extract the numeric part
        # For the structure, you might want to use something like values['structure'].formula if it's an actual object
        row = {
            "mpid": mpid,
            "structure": values["structure"],  # This is simplified for the example
            "ehull": values["ehull"],
            "f_energy": values["f_energy"]
        }
        data_for_df.append(row)
    # Step 3: Convert to DataFrame
    mpdata = pd.DataFrame(data_for_df)
    
    if isinstance(save_file, str):
        with open(save_file, 'wb') as file:
            pkl.dump(mpdata, file)

    return mpdata


def mp_struct_ehull_stable_filter(api_key, use_ase=True, save_file=None, s_u_nums=[2000, 1000], natm_max=20):
    num_stable, num_unstable = s_u_nums
    with MPRester(api_key) as mpr:
        mp_dict = {}
        stable_materials = []
        unstable_materials = []
        summaries =  mpr.summary.search(fields=["material_id", "energy_above_hull", "formation_energy_per_atom", "nsites"], all_fields=False)

        max_attempts = 10000  # Set a maximum number of attempts to prevent infinite loops
        attempts = 0
        while len(stable_materials) < num_stable or len(unstable_materials) < num_unstable:
            if attempts >= max_attempts:
                print("Reached maximum number of attempts")
                break
            try:
                # Adjust fetching logic here. This is a placeholder for fetching a single material at a time
                # You might want to implement a more efficient batch-fetching and filtering strategy
                identifier = attempts  # Placeholder for actual material ID or index
                # Fetch material data by ID or another identifier
                # summary = mpr.summary.search(fields=["material_id", "energy_above_hull", "formation_energy_per_atom", "nsites"], all_fields=False)[identifier]
                summary = summaries[identifier]
                mpid = summary.material_id
                e_above_hull = summary.energy_above_hull
                nsites = summary.nsites  # Number of atoms per unit cell

                if pd.isna(e_above_hull) or pd.isna(nsites):
                    continue

                # Filtering based on stability and number of atoms
                if nsites < natm_max:
                    if e_above_hull < 0.1 and len(stable_materials) < num_stable:
                        stable_materials.append(mpid)
                        print(f'[{attempts}] stable :', mpid, 'ehull:', e_above_hull)
                    elif e_above_hull >= 0.1 and len(unstable_materials) < num_unstable:
                        unstable_materials.append(mpid)
                        print(f'[{attempts}] unstable :', mpid, 'ehull:', e_above_hull)
                else: 
                    print(f'XX [{attempts}] not applicable :', mpid, 'ehull:', e_above_hull)   

            except Exception as e:
                print(f'Error loading material with ID {mpid}: {e}')
            
            attempts += 1

        # Now fetch the detailed data for each selected material
        for i, mpid in tqdm(enumerate(stable_materials + unstable_materials), desc="Fetching detailed data"):
            try:
                pstruct = mpr.get_structure_by_material_id(mpid)
                summary = mpr.get_data_by_id(mpid, fields=["material_id", "energy_above_hull", "formation_energy_per_atom"])
                e_above_hull = summary.energy_above_hull
                f_energy_per_atom = summary.formation_energy_per_atom

                if use_ase:
                    struct = pm2ase(pstruct)
                else:
                    struct = pstruct

                mp_dict[mpid] = {"structure": struct, "ehull": e_above_hull, 'f_energy': f_energy_per_atom}
                print(f'[{i}] mpid:', mpid, 'ehull:', e_above_hull, 'f_energy: ', f_energy_per_atom)
            except Exception as e:
                print(f'Error loading material with ID {mpid}: {e}')

        data_for_df = []
        for mpid, values in mp_dict.items():
            row = {
                "mpid": mpid,
                "structure": values["structure"],
                "ehull": values["ehull"],
                "f_energy": values["f_energy"]
            }
            data_for_df.append(row)

        mpdata = pd.DataFrame(data_for_df)
    
    if save_file:
        with open(save_file, 'wb') as file:
            pkl.dump(mpdata, file)

    return mpdata


def get_node_attr(atomic_numbers):
    z = []
    for atomic_number in atomic_numbers:
        node_attr = [0.0] * 118
        node_attr[atomic_number - 1] = 1
        z.append(node_attr)
    return torch.from_numpy(np.array(z, dtype = np.float64))

def init_node_feature(atomic_numbers, descriptor='mass'):
    x = []
    for atomic_number in atomic_numbers:
        node_feature = [0.0] * 118
        node_feature[atomic_number - 1] = atom_feature(int(atomic_number), descriptor)
        x.append(node_feature)
    return torch.from_numpy(np.array(x, dtype = np.float64))

def get_node_deg(edge_dst, n):
    node_deg = np.zeros((n, 1), dtype = np.float64)
    for dst in edge_dst:
        node_deg[dst] += 1
    node_deg += node_deg == 0
    return torch.from_numpy(node_deg)

# def atom_feature(atomic_number: int, descriptor):
#     """_summary_

#     Args:
#         atomic_number (_int_): atomic number 
#         descriptor (_'str'_): descriptor type. select from ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']

#     Returns:
#         _type_: descriptor
#     """
#     if descriptor=='mass':  # Atomic Mass (amu)
#         feature = Atom(atomic_number).mass
#     elif descriptor=='number':  # atomic number
#         feature = atomic_number
#     else:
#         ele = md.element(atomic_number) # use mendeleev
#         if descriptor=='radius':    # Atomic Radius (pm)
#             feature = ele.atomic_radius
#         elif descriptor=='en': # Electronegativity (Pauling)
#             feature = ele.en_pauling
#         elif descriptor=='ie':  # Ionization Energy (eV)
#             feature = ele.ionenergies[1]
#         elif descriptor=='dp':  # Dipole Polarizability (Å^3)
#             feature = ele.dipole_polarizability
#         else:   # no feature
#             feature = 1
#     return feature

def atom_feature(atomic_number: int, descriptor):
    """_summary_

    Args:
        atomic_number (_int_): atomic number 
        descriptor (_'str'_): descriptor type. select from ['mass', 'number', 'radius', 'en', 'ie', 'dp', 'non']

    Returns:
        _type_: descriptor
    """
    if descriptor=='mass':  # Atomic Mass (amu)
        feature = Atom(atomic_number).mass
    elif descriptor=='number':  # atomic number
        feature = atomic_number
    else:
        # ele = md.element(atomic_number) # use mendeleev
        if descriptor=='radius':    # Atomic Radius (pm)
            feature = md_class.radius[atomic_number]
        elif descriptor=='en': # Electronegativity (Pauling)
            feature = md_class.pauling[atomic_number]
        elif descriptor=='ie':  # Ionization Energy (eV)
            feature = md_class.ie[atomic_number]
        elif descriptor=='dp':  # Dipole Polarizability (Å^3)
            feature = md_class.dip[atomic_number]
        else:   # no feature
            feature = 1
    return feature

class Dataset_Input(Dataset):
    def __init__(self, df, r_max, target='ehull', descriptor='mass', scaler=None):
        super().__init__()
        self.df = df
        self.r_max = r_max
        self.target = target
        self.descriptor=descriptor
        self.scaler = scaler
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        target  = row[self.target]
        astruct = row['structure']
        mpid = row['mpid']
        symbols = list(astruct.symbols).copy()
        positions = torch.from_numpy(astruct.get_positions().copy())
        numb = len(positions)
        lattice = torch.from_numpy(astruct.cell.array.copy()).unsqueeze(0)
        edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = astruct, cutoff = self.r_max, self_interaction = True)
        z = get_node_attr(astruct.arrays['numbers'])
        x =  init_node_feature(astruct.arrays['numbers'], self.descriptor)  
        node_deg = get_node_deg(edge_dst, len(x)) 
        if self.scaler is not None:
            y = torch.tensor([self.scaler.forward(target)]).unsqueeze(0)
        else: 
            y = torch.tensor([target]).unsqueeze(0)
        data = Data(id = mpid,
                    pos = positions,
                    lattice = lattice,
                    symbol = symbols,
                    r_max = self.r_max,
                    z = z,
                    x = x,
                    y = y,
                    node_deg = node_deg,
                    edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim = 0),
                    edge_shift = torch.tensor(edge_shift, dtype = torch.float64),
                    edge_vec = torch.tensor(edge_vec, dtype = torch.float64),
                    edge_len = torch.tensor(edge_len, dtype = torch.float64),
                    numb = numb)
        return data

class Dataset_InputStability(Dataset):  #!
    def __init__(self, df, r_max, target='ehull', descriptor='mass', scaler=None, nearest=False, adjust_r_max=False):
        super().__init__()
        self.df = df
        self.r_max = r_max
        self.target = target
        self.descriptor=descriptor
        self.scaler = scaler
        self.nearest = nearest
        self.num_data = len(self.df)
        self.adjust_r_max = adjust_r_max
        self.generate_dataset()
    
    def generate_dataset(self):
        self.data_list = []
        for index in tqdm(range(self.num_data)):
            # start_time = time.time()
            row = self.df.iloc[index]
            target  = row[self.target]
            astruct = row['structure']
            mpid = row['mpid']
            symbols = list(astruct.symbols).copy()
            positions = torch.from_numpy(astruct.get_positions().copy())
            numb = len(positions)
            lattice = torch.from_numpy(astruct.cell.array.copy()).unsqueeze(0)
            # print(f'[index {index}] pos: {positions.shape}, lat; {lattice.shape}')
            # print('check1.1: ', time.time()-start_time)
            if self.nearest:
                edge_src, edge_dst, edge_shift, edge_vec, edge_len = nearest_neighbor_list(a = astruct, weight_cn = True, self_intraction = False)
            else: 
                if self.adjust_r_max:
                    r_max_ = self.r_max
                    lat_m_mul = 2*lattice.norm(dim=-1).min().item()
                    if lat_m_mul < r_max_:
                        r_max_ = lat_m_mul
                    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = astruct, cutoff = r_max_, self_interaction = True)
                else:
                    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = astruct, cutoff = self.r_max, self_interaction = True)
            z = get_node_attr(astruct.arrays['numbers'])
            x =  init_node_feature(astruct.arrays['numbers'], self.descriptor)  
            node_deg = get_node_deg(edge_dst, len(x)) 
            # print('check1.2: ', time.time()-start_time)
            if self.scaler is not None:
                y = torch.tensor([self.scaler.forward(target)]).unsqueeze(0)
            else: 
                y = torch.tensor([target]).unsqueeze(0)
            if 'stable' in row.keys():
                y = torch.tensor([float(row['stable'])], dtype=torch.float32).unsqueeze(0) #y= 
            else:
                y = torch.tensor([float(target <= 0.1)], dtype=torch.float32).unsqueeze(0) #
            # print('check1.3: ', time.time()-start_time)
            data = Data(id = mpid,
                        pos = positions,
                        lattice = lattice,
                        symbol = symbols,
                        r_max = self.r_max,
                        z = z,
                        x = x,
                        y = y,
                        node_deg = node_deg,
                        edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim = 0),
                        edge_shift = torch.tensor(edge_shift, dtype = torch.float64),
                        edge_vec = torch.tensor(edge_vec, dtype = torch.float64),
                        edge_len = torch.tensor(edge_len, dtype = torch.float64),
                        numb = numb)
            # print('check1.4: ', time.time()-start_time)
            self.data_list.append(data)     
    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        return self.data_list[index]


def adjust_cell(structure, scale_factor):
    """
    Adjusts the cell dimensions of an ASE Atoms object by a scale factor.
    
    Parameters:
    - structure: ASE Atoms object.
    - scale_factor: float, the factor by which to scale the cell dimensions.
    
    Returns:
    - ASE Atoms object with adjusted cell dimensions.
    """
    new_cell = structure.cell * scale_factor
    # Create a new structure with the same positions and scaled cell
    new_structure = structure.copy()
    new_structure.set_cell(new_cell, scale_atoms=True)
    return new_structure


def diffuse_structure(structure: Atoms, diff_factors: dict) -> Atoms:
    """
    Diffuses the structure of the input material.

    Args:
        original_structure (ase.Atoms): The original structure of the input material.
        lfa_factors (dict): A dictionary with keys 'lattice', 'frac', and 'atype' representing
                            the diffusion factors for the lattice matrix, fractional coordinates,
                            and atom types, respectively.

    Returns:
        ase.Atoms: The diffused structure.
    """
    # Diffuse the lattice if required
    astruct = structure.copy()
    if diff_factors.get('lattice') is not None:
        perturbation_scale = diff_factors['lattice']
        current_cell = np.array(structure.get_cell())
        l_norm = np.linalg.norm(current_cell, axis=-1, keepdims=True)
        rand_l = np.random.normal(loc=0, scale=1, size=current_cell.shape)
        new_cell = current_cell + rand_l * l_norm * perturbation_scale
        astruct.set_cell(new_cell, scale_atoms=False)  # Set scale_atoms to False to only change the cell



    # Diffuse the fractional coordinates if required
    if diff_factors.get('frac') is not None:
        frac_coords = astruct.get_scaled_positions()
        noise = np.random.normal(loc=0, scale=diff_factors['frac'], size=frac_coords.shape)
        new_frac_coords = (frac_coords + noise)%1
        astruct.set_scaled_positions(new_frac_coords)

    # Diffuse the atom types if required. This part is a bit tricky without a specific mechanism to 'diffuse' types.
    # An example approach could involve randomly swapping some atom types based on the 'atype' factor, but
    # this would require a clear definition of how atom types should be diffused or altered.
    # if lfa_factors.get('atype') is not None:
    #     # Implement atom type diffusion based on your specific requirements.
    #     pass

    return astruct


def nearest_neighbor_list(a, weight_cn = True, self_intraction = False):
    cnn = CrystalNN(weighted_cn=weight_cn)
    if isinstance(a, Structure):
        pstruct = a 
    elif isinstance(a, Atoms):
        pstruct = ase2pmg(a)
    pcell = np.array(pstruct.lattice.matrix)
    species = [str(s) for s in pstruct.species]
    cart, frac = pstruct.cart_coords, pstruct.frac_coords
    count = 0
    cnn_edge_src, cnn_edge_dst, cnn_edge_shift, cnn_edge_vec, cnn_edge_len = [], [], [], [], []
    cnn_elem_src, cnn_elem_dst = [], []
    for i, site in enumerate(pstruct):
        nn_info = cnn.get_nn_info(pstruct, i)
        for nn_ in nn_info:
            j = nn_['site_index']
            i_elem, j_elem = species[i], species[j]
            jimage = nn_['image']
            cart_shift = np.einsum('ij, i->j', pcell, jimage)
            e_vec = cart[j] - cart[i] + cart_shift
            e_len = np.linalg.norm(e_vec)
            cnn_edge_src.append(i)
            cnn_edge_dst.append(j)
            cnn_edge_shift.append(jimage)
            cnn_edge_vec.append(e_vec)
            cnn_edge_len.append(e_len)
            cnn_elem_src.append(i_elem)
            cnn_elem_dst.append(j_elem)
            # print(f"[{i}] Distance to neighbor {nn_['site'].species_string} (index: {j}, image: {jimage}): {nn_['weight']} Å")
            # print(f"{[i, j]} edge vec", e_vec)
            # print(f"{[i, j]} edge len", e_len)
            count +=  1
    try:
        cnn_edge_src, cnn_edge_dst, cnn_edge_len = [np.array(a) for a in [cnn_edge_src, cnn_edge_dst, cnn_edge_len]]
        cnn_edge_shift, cnn_edge_vec = [np.stack(a) for a in [cnn_edge_shift, cnn_edge_vec]]
        return cnn_edge_src, cnn_edge_dst, cnn_edge_shift, cnn_edge_vec, cnn_edge_len
    except:
        print('Skip: ', pstruct.formula)
        return cnn_edge_src, cnn_edge_dst, cnn_edge_shift, cnn_edge_vec, cnn_edge_len

def ase2pmg(astruct):
    lattice = Lattice(astruct.cell)  # Convert the cell to a Pymatgen Lattice
    species = astruct.get_chemical_symbols()  # Get the list of element symbols
    positions = astruct.get_scaled_positions()  # Get the atomic positions
    pstruct = Structure(lattice, species, positions)
    return pstruct

class Scaler(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, X):
        pass

class DivisionScaler(Scaler):
    def __init__(self, value):
        self.value = value

    def forward(self, X):
        return X/self.value

    def backward(self, X):
        return X*self.value

class LogScaler(Scaler):
    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon

    def forward(self, X):
        return np.log(X + self.epsilon)

    def backward(self, X):
        return np.exp(X) - self.epsilon

class LogShiftScaler(Scaler):
    def __init__(self, epsilon=1e-3, mu=None, sigma=None):
        self.epsilon = epsilon
        self.mu = mu
        self.sigma = sigma

    def forward(self, X):
        Y = np.log(X + self.epsilon)
        return (Y - self.mu) / self.sigma

    def backward(self, X):
        if self.mu is None or self.sigma is None:
            raise ValueError("Mean (mu) and standard deviation (sigma) must be provided during initialization.")

        Y = X * self.sigma + self.mu  # Reverse the normalization
        return np.exp(Y) - self.epsilon
    
import random
def remove_random_atoms(astruct, num_atoms_to_remove):
    """
    Remove a specified number of atoms randomly from an Atoms object.
    
    Args:
    - atoms (Atoms): The ASE Atoms object.
    - num_atoms_to_remove (int): Number of atoms to remove.
    
    Returns:
    - Atoms: A new Atoms object with atoms removed.
    """
    astruct_ = astruct.copy()
    assert 0 < num_atoms_to_remove < len(astruct_), "Invalid number of atoms to remove."
    
    # Generate a list of indices for atoms to be removed
    indices_to_remove = random.sample(range(len(astruct_)), num_atoms_to_remove)
    
    # Delete the atoms at the selected indices
    del astruct_[indices_to_remove]
    
    return astruct_
