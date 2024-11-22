import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from ase import Atoms
from ase.visualize.plot import plot_atoms
from ase.build import make_supercell
from pymatgen.core import Structure
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Utility settings
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)

# Formatting for plots
fontsize = 16
textsize = 14
plt.rcParams.update({
    'font.family': 'lato',
    'axes.linewidth': 1,
    'mathtext.default': 'regular',
    'xtick.bottom': True,
    'ytick.left': True,
    'font.size': fontsize,
    'axes.labelsize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': textsize,
})

# Color palettes
palette = ['#43AA8B', '#F8961E', '#F94144', '#277DA1']
datasets = ['train', 'valid', 'test']
datasets2 = ['train', 'test']
colors = dict(zip(datasets, palette[:-1]))
colors2 = dict(zip(datasets2, palette[:-1]))
cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', palette)

# Subscript formatting for plot titles
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


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


def plot_confusion_matrices(dfs, run_name, save_path=None):
    """
    Plots confusion matrices for given datasets.

    Args:
        dfs (dict): A dictionary where keys are dataset names (e.g., 'train', 'test') 
                    and values are DataFrames containing 'real' and 'pred' columns.
        run_name (str): Name of the current run, used in the figure title.
        save_path (str, optional): Path to save the figure. If None, the figure is not saved.

    Returns:
        matplotlib.figure.Figure: The figure object containing the confusion matrices.
    """
    num_dfs = len(dfs)
    fig, axs = plt.subplots(1, num_dfs, figsize=(6 * num_dfs, 6))  # Adjust figure size as needed
    for i, (dataset, df) in enumerate(dfs.items()):
        reals, preds = list(df['real']), list(df['pred'])
        cm = confusion_matrix(reals, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[i])
        axs[i].set_title(f'{dataset.capitalize()} Confusion Matrix')
        axs[i].set_xlabel('Predicted labels')
        axs[i].set_ylabel('True labels')
        accuracy = accuracy_score(reals, preds)
        axs[i].annotate(f'Accuracy = {accuracy:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')

    fig.suptitle(f"{run_name} Confusion Matrices")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the main title

    if save_path:
        fig.savefig(save_path)
        print(f"Confusion matrices saved to {save_path}")

    return fig



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