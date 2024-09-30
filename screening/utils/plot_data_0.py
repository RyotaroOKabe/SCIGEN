import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter
from ase import Atoms, Atom
from copy import copy
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
palette = ['#90BE6D', '#277DA1', '#F8961E', '#F94144']
datasets = ['train', 'valid', 'test']
colors = dict(zip(datasets, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', [palette[k] for k in [0,2,1]])
import sklearn


def loss_plot(model_file, device, fig_file):
    history = torch.load(model_file + '.torch', map_location = device)['history']
    steps = [d['step'] + 1 for d in history]
    loss_train = [d['train']['loss'] for d in history]
    loss_valid = [d['valid']['loss'] for d in history]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(steps, loss_train, 'o-', label='Training')
    ax.plot(steps, loss_valid, 'o-', label='Validation')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(fig_file  + '_loss_train_valid.png')
    plt.close()

def loss_test_plot(model, device, fig_file, dataloader, loss_fn):
    loss_test = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            output = model(d)
            y_true = d.y
            loss = loss_fn(output, y_true).cpu()
            loss_test.append(loss)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.array(loss_test), label = 'testing loss: ' + str(np.mean(loss_test)))
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(fig_file + '_loss_test.png')
    plt.close()
    

import numpy as np
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