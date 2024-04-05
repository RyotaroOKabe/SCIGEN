import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from os.path import join as opj
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from ase import Atoms
from copy import copy
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
api_key = "PvfnzQv5PLh4Lzxz1pScKnAtcmmWVaeU"
from ase import Atoms
from ase.visualize.plot import plot_atoms
from ase.build import make_supercell
from pymatgen.core import Structure, Lattice
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import copy
import imageio

# utilities
from tqdm import tqdm

# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

# standard formatting for plots
fontsize = 16
textsize = 14
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
plt.rcParams['font.family'] = 'lato'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize


# colors for datasets
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
datasets = ['train', 'valid', 'test']
datasets2 = ['train', 'test']
colors = dict(zip(datasets, palette[:-1]))
colors2 = dict(zip(datasets2, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', palette)
# cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
from pymatgen.core.structure import Structure

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

def vis_structure(struct_in, ax=None, supercell=np.diag([1,1,1]), title=None, rot='5x,5y,90z', savedir=None, palette=palette):
    if type(struct_in)==Structure:
        struct = Atoms(list(map(lambda x: x.symbol, struct_in.species)) , # list of symbols got from pymatgen
                positions=struct_in.cart_coords.copy(),
                cell=struct_in.lattice.matrix.copy(), pbc=True) 
    elif type(struct_in)==Atoms:
        struct=struct_in
    struct = make_supercell(struct, supercell)
    symbols = np.unique(list(struct.symbols))
    len_symbs = len(list(struct.symbols))
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
    if savedir is not None:
        path = savedir
        if not os.path.isdir(f'{path}'):
            os.mkdir(path)
        fig.savefig(f'{path}/{fname}.png')
    if ax is not None:
        return ax
    plt.show()
    plt.close()


def movie_structs(astruct_list, name, t_interval=1, savedir=None, supercell=np.diag([1,1,1])):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, struct_in in enumerate(astruct_list):
        if i<len(astruct_list):
            if i%t_interval==0:
                vis_structure(struct_in,  supercell=supercell, title=f"{{0:04d}}".format(i), savedir=savedir)
        else: 
            vis_structure(struct_in,  supercell=supercell, title=f"{{0:04d}}".format(i), savedir=savedir)
    
    with imageio.get_writer(os.path.join(savedir, f'{name}.gif'), mode='I') as writer:
        for figurename in sorted(os.listdir(savedir)):
            if figurename.endswith('png'):
                image = imageio.imread(os.path.join(savedir, figurename))
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
    if 'eval_setting' in keys:
        eval_setting = data['eval_setting']
    else: 
        eval_setting =  ''
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
        all_frac_coords, all_atom_types, all_lengths, all_angles


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

def get_traj_pstruct_list(num_atoms, all_frac_coords, all_atom_types, all_lattices, atom_type_prob=True):
    pstruct_lists = []
    T, n = all_lattices.shape[:2]
    for i in range(n):
        pstruct_list = []
        for t in range(T):
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
    return pstruct_lists
