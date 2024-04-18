#%%
"""
https://www.notion.so/240403-check-the-statistics-profile-of-the-dataset-c45bd31a14b84697a9df792f524cdd04
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import matplotlib as mpl
import torch
from tqdm import tqdm
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import os, sys
from os.path import join
import imageio
import pickle as pkl
from scipy.stats import gaussian_kde
from pymatgen.analysis.local_env import CrystalNN
sys.path.append('../')
from InpaintCrysDiff.dirs_template import *
sys.path.append(ehull_pred_path)
from inpaint.mat_utils import vis_structure, ase2pmg, chemical_symbols
import warnings
# from m3gnet.models import Relaxer
for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="torch")
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
datasets = ['g', 'y', 'r']
colors = dict(zip(datasets, palette))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
savedir = join(homedir, 'figures')


#%%
mpdata_file01 = join(ehull_pred_path, 'data/mpdata_mp20_test.pkl')  #!
mpdata01 = pd.read_pickle(mpdata_file01)    #!
mpdata_file02 = join(ehull_pred_path, 'data/mpdata_mp20_train.pkl')  #!
mpdata02 = pd.read_pickle(mpdata_file02)    #!
mpdata_file03 = join(ehull_pred_path, 'data/mpdata_mp20_val.pkl')  #!
mpdata03 = pd.read_pickle(mpdata_file03)    #!
mpdata0 = pd.concat([mpdata01, mpdata02, mpdata03], ignore_index=True)


#%%
dist_profile = {el:[] for el in chemical_symbols}
astruct_list0 = list(mpdata0['structure'])
pstruct_list0 = [ase2pmg(astruct) for astruct in astruct_list0]
cnn = CrystalNN(weighted_cn=True)
for h, pstruct in tqdm(enumerate(pstruct_list0)):
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
            if i_elem == j_elem:
                dist_profile[i_elem].append(e_len)
            count +=  1
    try:
        cnn_edge_src, cnn_edge_dst, cnn_edge_len = [np.array(a) for a in [cnn_edge_src, cnn_edge_dst, cnn_edge_len]]
        cnn_edge_shift, cnn_edge_vec = [np.stack(a) for a in [cnn_edge_shift, cnn_edge_vec]]
    except:
        print('Skip: ', h, pstruct.formula)
    # print('count: ', count)
dprof = {k:v for k, v in dist_profile.items() if len(v)>0}

elems = ['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Dy']
for elem in elems:
    list_len = dprof[elem]
    plt.hist(list_len, bins=50, label=f"{elem}: {round(np.mean(list_len), 3)}")
    plt.xlabel('Bond lengths (Angstrom)')
    plt.ylabel('Num samples')
plt.legend()

#%%
elems = ['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Tb', 'Dy', 'Yb']
kde_dict = {}
for elem in dprof.keys():
    len_data = dprof[elem]
    counts, bin_edges = np.histogram(len_data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit KDE to data
    kde = gaussian_kde(len_data)
    kde_values = kde(bin_centers)
    # plt.hist(len_data, bins=50, label=f"{elem} (mean: {round(np.mean(len_data), 3)} A)")
    kde_dict[elem] = kde
    if elem in elems:
    # if True:
        plt.figure(figsize=(6, 4))
        plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), color='#56b3e9', alpha=0.8, label="Histogram")
        plt.plot(bin_centers, kde_values, color='#d55c00', label='KDE', linewidth=2)
        plt.legend()
        plt.xlabel(f'Bond lengths ($\AA$)')
        plt.ylabel('Density')
        plt.title(f"{elem} (mean: {round(np.mean(len_data), 3)} $\AA$)")
        plt.show()
        plt.savefig(join(savedir, f'{elem}_bond_mp20.pdf'))
        plt.close()
# with open('./data/kde_bond.pkl', 'wb') as file:
#     pkl.dump(kde_dict, file)

#%%
with open('./data/kde_bond.pkl', 'rb') as file:
    kde_dict = pkl.load(file)
    
for elem in dprof.keys():
    if elem in elems:
        kde = kde_dict[elem]
        resampled_data = kde.resample(1000)  # Resampling 1000 points
        # Use the resampled data to define the range
        x_range = np.linspace(resampled_data.min(), resampled_data.max(), 1000)
        # Compute KDE values over this range
        kde_values = kde(x_range)
        # Plot histogram of the resampled data
        plt.hist(resampled_data[0], bins=100, density=True, alpha=0.5, color='blue', label='Resampled Histogram')
        # Plot the KDE curve based on the x_range derived from resampled data
        plt.plot(x_range, kde_values, color='red', label='KDE')
        plt.legend()
        plt.xlabel('Bond lengths (Angstrom)')
        plt.ylabel('Prob.')
        plt.title(f"{elem} (mean: {round(np.mean(len_data), 3)} A)")
        plt.show()
        plt.close()

#%%
