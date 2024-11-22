import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ase import Atoms, Atom
from ase.neighborlist import neighbor_list
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.local_env import CrystalNN
import mendeleev as md
from tqdm import tqdm
from abc import ABC, abstractmethod


class MD():
    """
    Manages atomic features using the Mendeleev library.
    Precomputes and caches properties like radius, electronegativity, etc.
    """
    def __init__(self):
        self.radius, self.pauling, self.ie, self.dip = {}, {}, {}, {}
        for atomic_number in range(1, 119):
            ele = md.element(atomic_number)
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

def ase2pmg(astruct):
    lattice = Lattice(astruct.cell)  # Convert the cell to a Pymatgen Lattice
    species = astruct.get_chemical_symbols()  # Get the list of element symbols
    positions = astruct.get_scaled_positions()  # Get the atomic positions
    pstruct = Structure(lattice, species, positions)
    return pstruct

class Dataset_Cls(Dataset): 
    """
    PyTorch Dataset for generating graph representations of atomic structures.
    """
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
        self._process_dataset()
    
    def _process_dataset(self):
        """Generates a list of graph data for each sample in the dataset."""
        self.data_list = []
        self.error_dict = {}
        for i, row in tqdm(self.df.iterrows(), total=self.num_data, desc="Processing dataset"):
            try: 
                target  = row[self.target]
                astruct = row['structure']
                mpid = row['mpid']
                symbols = list(astruct.symbols).copy()
                positions = torch.from_numpy(astruct.get_positions().copy())
                numb = len(positions)
                lattice = torch.from_numpy(astruct.cell.array.copy()).unsqueeze(0)

                if self.nearest:
                    edge_src, edge_dst, edge_shift, edge_vec, edge_len = nearest_neighbor_list(a = astruct, weight_cn = True, self_intraction = False)
                else: 
                    if self.adjust_r_max:
                        r_max_ = self.r_max
                        lat_m_mul = 2*lattice.norm(dim=-1).min().item()
                        if lat_m_mul < r_max_:
                            r_max_ = lat_m_mul
                    else:
                        r_max_ = self.r_max
                    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = astruct, cutoff = r_max_, self_interaction = True)
                    
                z = create_node_input(astruct.arrays['numbers'], descriptor='one_hot')   # node attribute
                x = create_node_input(astruct.arrays['numbers'], descriptor=self.descriptor)  # init node feature
                node_deg = get_node_deg(edge_dst, len(x)) 
                
                if self.scaler is not None:
                    y = torch.tensor([self.scaler.forward(target)]).unsqueeze(0)
                else: 
                    y = torch.tensor([target]).unsqueeze(0)
                if self.target == 'ehull':
                    if 'stable' in row.keys():
                        y = torch.tensor([float(row['stable'])], dtype=torch.float32).unsqueeze(0) 
                    else:
                        y = torch.tensor([float(target <= 0.1)], dtype=torch.float32).unsqueeze(0) 
                else:   # mag
                    y = torch.tensor([float(row['label'])], dtype=torch.float32).unsqueeze(0)   #TODO: need the add the case for multi-class classification

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
                self.data_list.append(data)    
            except Exception as e:
                print(f"[{i}] Error in processing {row['mpid']}: {e}") 
                self.error_dict[row['mpid']] = e

    def __len__(self):
        # return self.num_data
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]


def get_node_deg(edge_dst, n):
    """
    Compute node degrees from the destination edges.
    Args:
        edge_dst (np.ndarray): Destination edges.
        n (int): Number of nodes.
    Returns:
        torch.Tensor: Node degrees.
    """
    node_deg = np.zeros((n, 1), dtype = np.float64)
    for dst in edge_dst:
        node_deg[dst] += 1
    node_deg += node_deg == 0
    return torch.from_numpy(node_deg)


def atom_feature(atomic_number: int, descriptor):
    """
    Get atomic features based on the descriptor.
    Args:
        atomic_number (int): Atomic number of the element.
        descriptor (str): Type of descriptor. Can be 'mass', 'number', 'radius', 'en', 'ie', 'dp', 'non'.
    Returns:
        float: The atomic feature value.
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

def create_node_input(atomic_numbers, descriptor='mass'):
    """
    Create node input features for a list of atomic numbers.
    Args:
        atomic_numbers (list): List of atomic numbers.
        descriptor (str, optional): Descriptor for the node features. Defaults to 'mass'.
    Returns:
        torch.Tensor: Tensor of node input features.
    """
    x = []
    for atomic_number in atomic_numbers:
        atomic = [0.0] * 118         
        atomic[atomic_number - 1] = atom_feature(int(atomic_number), descriptor)
        x.append(atomic)
    return torch.from_numpy(np.array(x, dtype = np.float64))


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
    return astruct


def augment_data_diffuse(data, num_diff, diff_factor, num_diff_small, diff_factor_small):
    """Augment dataset with diffused structures."""
    augmented_data = []
    for _, row in data.iterrows():
        augmented_data.append(row)

        if row["stable"]:
            # Add diffused unstable structures
            for j in range(num_diff):
                diff_row = row.copy()
                diff_row["structure"] = diffuse_structure(row["structure"], diff_factor)
                diff_row["mpid"] += f"-d{j}"
                diff_row["stable"] = 0
                augmented_data.append(diff_row)

            # Add slightly diffused stable structures as acceptable structures
            for j in range(num_diff_small):
                diff_row = row.copy()
                diff_row["structure"] = diffuse_structure(row["structure"], diff_factor_small)
                diff_row["mpid"] += f"-s{j}"
                diff_row["stable"] = 1
                augmented_data.append(diff_row)

    augmented_df = pd.DataFrame(augmented_data).reset_index(drop=True)
    print(f"Augmented dataset size: {len(augmented_df)}")
    return augmented_df

def count_elements(row):
    """
    Count the number of elements in a material in ComputedStructureEntry format.
    """
    composition = row['computed_structure_entry']['composition']
    return len(composition.keys())

# We do not need the functions below in the current implementation
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
            count +=  1
    try:
        cnn_edge_src, cnn_edge_dst, cnn_edge_len = [np.array(a) for a in [cnn_edge_src, cnn_edge_dst, cnn_edge_len]]
        cnn_edge_shift, cnn_edge_vec = [np.stack(a) for a in [cnn_edge_shift, cnn_edge_vec]]
        return cnn_edge_src, cnn_edge_dst, cnn_edge_shift, cnn_edge_vec, cnn_edge_len
    except:
        print('Skip: ', pstruct.formula)
        return cnn_edge_src, cnn_edge_dst, cnn_edge_shift, cnn_edge_vec, cnn_edge_len


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
    
