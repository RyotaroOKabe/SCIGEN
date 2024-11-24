#%%
import time
import argparse
import torch
from pathlib import Path
from torch_geometric.data import DataLoader
from eval_utils import load_model, lattices_to_params_shape, recommand_step_lr
import random   
from scigen.pl_modules.diffusion_w_type import sample_scigen
from gen_utils import SampleDataset, convert_seconds_short, parse_none_or_value
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


def diffusion(loader, model, step_lr, save_traj):
    
    frac_coords = []
    num_atoms = []
    num_known = []  
    atom_types = []
    lattices = []
    all_frac_coords = []
    all_atom_types = []
    all_lattices = []
    all_lengths, all_angles = [], []    # ignore if we save traj
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample_scigen(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        num_known.append(outputs['num_known'].detach().cpu()) 
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        if save_traj:
            all_frac_coords.append(traj['all_frac_coords'].detach().cpu())
            all_atom_types.append(traj['atom_types'].detach().cpu())
            all_lattices.append(traj['all_lattices'].detach().cpu())
        print(f'batch {idx+1}/{len(loader)} done') 

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    num_known = torch.cat(num_known, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)
    if save_traj: 
        all_frac_coords = torch.cat(all_frac_coords, dim=1)
        all_atom_types = torch.cat(all_atom_types, dim=1)
        all_lattices = torch.cat(all_lattices, dim=1)
        all_lengths, all_angles = lattices_to_params_shape(all_lattices)    # works for all-time outputs

    print('num_atoms: ', num_atoms.shape, num_atoms)
    print('num_known: ', num_known.shape, num_known)
    
    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, num_known, all_frac_coords, all_atom_types, all_lattices, all_lengths, all_angles 
    )



def main(args):
    # randomly generate seed for each run
    # Generate a random seed by combining time and randomness
    seed = int(time.time() * random.random())
    print(f"Generated seed: {seed}")
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    print('args: ', args)   
    model, _, cfg = load_model(
        model_path, load_data=False)
    if torch.cuda.is_available():
        model.to('cuda')
    model.sample_scigen = sample_scigen.__get__(model)    
    print('Evaluate the diffusion model.')
    c_vec_cons = {'scale': args.c_scale, 'vert': args.c_vert}
    # print('c_vec_cons: ', c_vec_cons)
    test_set = SampleDataset(dataset=args.dataset, 
                            natm_range=args.natm_range,
                            total_num=args.batch_size * args.num_batches_to_samples, 
                            bond_sigma_per_mu=args.bond_sigma_per_mu,
                            use_min_bond_len=args.use_min_bond_len,
                            known_species=args.known_species, 
                            sc_list=args.sc, 
                            frac_z=args.frac_z,
                            c_vec_cons=c_vec_cons,
                            reduced_mask=args.reduced_mask,
                            seed = seed,
                            device=device)     
    test_loader = DataLoader(test_set, batch_size = args.batch_size)
    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['gen'][args.dataset]
    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, num_known, \
        all_frac_coords, all_atom_types, all_lattices, all_lengths, all_angles) = diffusion(test_loader, model, step_lr, args.save_traj)    

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'
    print(f'gen_out_name: {gen_out_name}')

    run_time = time.time() - start_time
    total_num = args.batch_size * args.num_batches_to_samples
    print('args: ', args)   
    print(f'Total outputs: {args.num_batches_to_samples} samples x {args.batch_size} batches = {total_num} materials')
    print(f'run time: {run_time} sec = {convert_seconds_short(run_time)}')
    print(f'{run_time/args.num_batches_to_samples} sec/sample')
    print(f'{run_time/total_num} sec/material')
    
    config = {
        'eval_setting': args,
        'num_atoms': num_atoms,
        'num_known': num_known,
        'frac_coords': frac_coords,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'all_frac_coords': all_frac_coords,
        'all_atom_types': all_atom_types,
        'all_lengths': all_lengths,
        'all_angles': all_angles,
        'c_vec_cons': c_vec_cons,
        'seed': seed,
        'time': run_time,
    }
    torch.save(config, model_path / gen_out_name)

      
#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--save_traj', type=lambda x: x.lower() == 'true', default=False, help="Save trajectory during generation")
    parser.add_argument('--natm_range', nargs='+', default=[1, 20])
    parser.add_argument('--bond_sigma_per_mu', default=None)   
    parser.add_argument('--use_min_bond_len', type=lambda x: x.lower() == 'true', default=False, help="Use minimum bond length with metallic radius")
    parser.add_argument('--known_species', nargs='+', default=['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Tb', 'Dy', 'Yb']) 
    parser.add_argument('--sc', nargs='+', default=['kag', 'hon', 'tri', 'sqr']) 
    parser.add_argument('--c_scale', default=None, help="Scale to multiply bond length for LZ constraint") 
    parser.add_argument('--c_vert', type=lambda x: x.lower() == 'true', default=False, help="Use LZ constraint for vertical bonds")
    parser.add_argument('--frac_z', default=None, type=parse_none_or_value, help="Fraction of z-coordinate of the 2D geometric pattern. If None, return random frac_z in [0, 1).")
    parser.add_argument('--reduced_mask', type=lambda x: x.lower() == 'true', default=False, help="Use reduced mask")
    args = parser.parse_args()
    main(args)
