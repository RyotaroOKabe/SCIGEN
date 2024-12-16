import os
import math
import itertools
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from ase.visualize.plot import plot_atoms
from ase.build import make_supercell
import imageio
from ase import Atoms
from ase.data import covalent_radii
from pymatgen.core import Structure, Lattice
import smact
from smact.screening import pauling_test
import periodictable
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)
plt.rcParams.update({
    'font.family': 'lato',
    'axes.linewidth': 1,
    'mathtext.default': 'regular',
    'xtick.bottom': True,
    'ytick.left': True,
    'font.size': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
})
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
colors = dict(zip(['train', 'valid', 'test'], palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', palette)
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
# import mendeleev as md 

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


def vis_structure(struct_in, ax=None, supercell=np.diag([1,1,1]), title=None, rot='5x,5y,90z', save_dir=None, palette=palette):
    if type(struct_in)==Structure:
        struct = Atoms(list(map(lambda x: x.symbol, struct_in.species)) , # list of symbols got from pymatgen
                positions=struct_in.cart_coords.copy(),
                cell=struct_in.lattice.matrix.copy(), pbc=True) 
    elif type(struct_in)==Atoms:
        struct=struct_in
    struct = make_supercell(struct, supercell)
    symbols = np.unique(list(struct.symbols))
    z = dict(zip(symbols, range(len(symbols))))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
        fig.patch.set_facecolor('white')
    norm = plt.Normalize(vmin=0, vmax=len(symbols)-1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', palette)
    color = [mpl.colors.to_hex(k) for k in cmap(norm([z[j] for j in list(struct.symbols)]))]
    plot_atoms(struct, ax, radii=0.25, colors=color, rotation=(rot))

    ax.set_xlabel(r'$x_1\ (\AA)$')
    ax.set_ylabel(r'$x_2\ (\AA)$')
    if title is None:
        ftitle = f"{struct.get_chemical_formula().translate(sub)}"
        fname =  struct.get_chemical_formula()
    else: 
        ftitle = f"{title} / {struct.get_chemical_formula().translate(sub)}"
        fname = f"{title}_{struct.get_chemical_formula()}"
    ax.set_title(ftitle, fontsize=15)
    if save_dir is not None:
        path = save_dir
        if not os.path.isdir(f'{path}'):
            os.mkdir(path)
        fig.savefig(f'{path}/{fname}.png')
    if ax is not None:
        return ax


def movie_structs(astruct_list, name, t_interval=1, save_dir=None, supercell=np.diag([1,1,1]), rot='5x,5y,90z'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, struct_in in tqdm(enumerate(astruct_list), desc='Generating Images', total=len(astruct_list)):
        if i<len(astruct_list):
            if i%t_interval==0:
                vis_structure(struct_in,  supercell=supercell, title=f"{{0:04d}}".format(i), rot=rot, save_dir=save_dir)
        else: 
            vis_structure(struct_in,  supercell=supercell, title=f"{{0:04d}}".format(i), rot=rot, save_dir=save_dir)
    
    with imageio.get_writer(os.path.join(save_dir, f'{name}.gif'), mode='I') as writer:
        image_files = sorted(os.listdir(save_dir))
        for figurename in tqdm(image_files, desc='Generating GIF', total=len(image_files)):
            if figurename.endswith('png'):
                image = imageio.imread(os.path.join(save_dir, figurename))
                writer.append_data(image)



# material data conversion
def str2pmg(cif):
    pstruct=Structure.from_str(cif, "CIF")
    return pstruct

# pymatgen > ase.Atom
def pmg2ase(pstruct):
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

def get_atomic_number(element_symbol):
    try:
        element = periodictable.elements.symbol(element_symbol)
        return element.number
    except ValueError:
        return -1  # Return None for invalid element symbols
    
def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def lattice_matrix_to_params_torch(lattice_matrix):
    """Batched torch version to compute lattice parameters from matrix.

    lattice_matrix: torch.Tensor of shape (N, 3, 3), unit A
    """
    vector_a = lattice_matrix[:, 0]
    vector_b = lattice_matrix[:, 1]
    vector_c = lattice_matrix[:, 2]

    # Compute lattice lengths
    lengths = torch.norm(torch.stack([vector_a, vector_b, vector_c], dim=1), dim=2)

    # Compute lattice angles
    dot_ab = torch.sum(vector_a * vector_b, dim=1)
    dot_ac = torch.sum(vector_a * vector_c, dim=1)
    dot_bc = torch.sum(vector_b * vector_c, dim=1)

    cos_alpha = dot_bc / (lengths[:, 1] * lengths[:, 2])
    cos_beta = dot_ac / (lengths[:, 0] * lengths[:, 2])
    cos_gamma = dot_ab / (lengths[:, 0] * lengths[:, 1])

    alphas_rad, betas_rad, gammas_rad = [torch.acos(torch.clamp(cos_ang, -1.0, 1.0)) for cos_ang in [cos_alpha, cos_beta, cos_gamma]]
    angles_deg = torch.rad2deg(torch.stack([alphas_rad, betas_rad, gammas_rad]).T)

    return lengths, angles_deg

def output_gen(data_path):
    # for recon, gen, opt
    data = torch.load(data_path, map_location='cpu')
    keys = list(data.keys())
    lengths = data['lengths']
    angles = data['angles']
    num_atoms = data['num_atoms']
    frac_coords = data['frac_coords']
    atom_types = data['atom_types']
    try: 
        eval_setting = data['eval_setting']
        eval_setting = eval_setting.__dict__
    except:
        eval_setting = ''
    if 'time' in keys:
        time = data['time']
    else: 
        time = ''
    if 'all_frac_coords' in keys:
        all_frac_coords = data['all_frac_coords']
        all_atom_types =data['all_atom_types']
        all_lengths =data['all_lengths']
        all_angles =data['all_angles']
    else: 
        all_frac_coords, all_atom_types, all_lengths, all_angles = None, None, None, None
    return frac_coords, atom_types, lengths, angles, num_atoms, time, \
        all_frac_coords, all_atom_types, all_lengths, all_angles, eval_setting


def get_pstruct_list(num_atoms, frac_coords, atom_types, lattices, atom_type_prob=True):
    pstruct_list = []
    n = len(lattices)
    for i in tqdm(range(n)):
        sum_idx_bef = num_atoms[:i].sum()
        sum_idx_aft = num_atoms[:i+1].sum()
        frac = frac_coords[sum_idx_bef:sum_idx_aft, :].to('cpu').to(dtype=torch.float32)
        lattice = lattices[i].to('cpu')
        cart = frac@lattice.T
        atypes = atom_types[sum_idx_bef:sum_idx_aft].to('cpu').to(dtype=torch.float32)
        if atom_type_prob:
            atom_types_ = torch.argmax(atypes, dim=1) +1
        else: 
            atom_types_ = atypes
        # print('atoms: ', atoms.shape)
        # print('cart: ', cart.shape)
        # print('lattice: ', lattice.shape)
        pstruct = Structure(lattice, atom_types_, frac)
        # Atoms(symbols=atoms, positions = cart, cell = lattice, pbc=True) 
        pstruct_list.append(pstruct)
    return pstruct_list

def get_traj_pstruct_list(num_atoms, all_frac_coords, all_atom_types, all_lattices, t_step, atom_type_prob=True):
    """
    Function to get a list of pymatgen structure objects from a trajectory
    Args:
    num_atoms (torch.tensor): number of atoms in each frame of a trajectory
    all_frac_coords (torch.tensor): fractional coordinates of all atoms in a trajectory
    all_atom_types (torch.tensor): atom type of all atoms in a trajectory
    all_lattices (torch.tensor): lattice vectors of all frames in a trajectory
    t_step (int): time step to consider in the trajectory
    atom_type_prob (bool): if True, atom type probabilities are given, if False, atom type indices are given
    Returns:
    pstruct_lists (list): list of list of pymatgen structure objects
    """
    pstruct_lists = []
    T, n = all_lattices.shape[:2]
    if not isinstance(t_step, int):
        t_step = 1
    t_list = [t for t in range(T) if t%t_step==0]
    for i in range(n):
        pstruct_list = []
        for t in range(T):
            if t%t_step==0:
                sum_idx_bef = num_atoms[:i].sum()
                sum_idx_aft = num_atoms[:i+1].sum()
                frac = all_frac_coords[t, sum_idx_bef:sum_idx_aft, :].to('cpu').to(dtype=torch.float32)
                lattice = all_lattices[t, i].to('cpu').to(dtype=torch.float32)
                cart = frac@lattice.T
                atypes = all_atom_types[t, sum_idx_bef:sum_idx_aft].to('cpu')
                if atom_type_prob:
                    atom_types_ = torch.argmax(atypes, dim=1) +1
                else: 
                    atom_types_ = atypes
                # print('atoms: ', atoms.shape)
                # print('cart: ', cart.shape)
                # print('lattice: ', lattice.shape)
                pstruct = Structure(lattice, atom_types_, frac)
                # Atoms(symbols=atoms, positions = cart, cell = lattice, pbc=True) 
                pstruct_list.append(pstruct)
        pstruct_lists.append(pstruct_list)
    return pstruct_lists, t_list



def structures_to_cif_string(structures):
    """
    Concatenates multiple Structure objects into a single CIF string.

    Args:
    - structures (list): A list of pymatgen.core.structure.Structure objects.

    Returns:
    - str: A single string in CIF format containing all structures.
    """
    cif_strings = []

    for i, struct in enumerate(structures):
        # Get the CIF string for the structure
        cif_str = struct.to(fmt="cif")
        # Optionally, you can add a comment before each structure for clarity
        cif_strings.append(f"# Structure {i}\n{cif_str}")

    # Join all CIF strings into a single string, separated by newlines
    combined_cif_str = "\n".join(cif_strings)
    return combined_cif_str

def save_combined_cif(structures, output_file_path):
    """
    Saves multiple Structure objects to a single CIF file.

    Args:
    - structures (list): A list of pymatgen.core.structure.Structure objects.
    - output_file_path (str): The file path where the combined CIF data will be saved.
    """
    # Use the function defined earlier to get the combined CIF string
    combined_cif_str = structures_to_cif_string(structures)

    # Write the combined CIF string to the specified file
    with open(output_file_path, 'w') as file:
        file.write(combined_cif_str)
    print(f"Combined CIF file saved to: {output_file_path}")

import glob
from ase.io import read

def load_cif_files_from_folder(folder_path):
    """
    Load all CIF files in the specified folder as a list of ase.Atoms objects.

    Args:
    - folder_path (str): Path to the folder containing CIF files.

    Returns:
    - List[ase.Atoms]: A list of ase.Atoms objects loaded from the CIF files.
    """
    # Construct the pattern to match all CIF files in the folder
    cif_pattern = os.path.join(folder_path, '*.cif')
    
    # Find all files matching the pattern
    cif_files = sorted(glob.glob(cif_pattern))
    # print(cif_files)
    
    atoms_list = []
    for i, cif_file in enumerate(cif_files):
        # print(cif_file)
        try:
            # Load each CIF file as an ase.Atoms object
            # atoms_list = [read(cif_file) for cif_file in cif_files]
            atoms_list.append(read(cif_file))
        except Exception as error:
            print(f"Failed to process {cif_file}: {error}")
    return atoms_list


def convert_seconds_short(sec):
    minutes, seconds = divmod(int(sec), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"


from pymatgen.analysis.local_env import CrystalNN
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


def atom_volume(element_symbol):
    elem_no = chemical_symbols.index(element_symbol)
    radius = covalent_radii[elem_no]  # This is in angstroms  
    volume = (4/3) * math.pi * (radius ** 3)
    return volume

def vol_density(astruct):
    lvol = astruct.cell.volume
    species = astruct.get_chemical_symbols()
    atoms_volume = 0
    for s in species:
        atoms_volume += atom_volume(s)
    return atoms_volume/(abs(lvol)+1e-4)


def get_composition(atom_types):    
    elem_counter = Counter(atom_types)
    composition = [(elem, elem_counter[elem])
                    for elem in sorted(elem_counter.keys())]
    elems, counts = list(zip(*composition))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    comps = tuple(counts.astype('int').tolist())
    return elems, comps


def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):    #
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False

def charge_neutrality(astruct, use_pauling_test=True, include_alloys=True):
    atom_types = np.array([chemical_symbols.index(s) for s in astruct.get_chemical_symbols()])
    elems, comps = get_composition(atom_types)
    return smact_validity(elems, comps, use_pauling_test, include_alloys)
