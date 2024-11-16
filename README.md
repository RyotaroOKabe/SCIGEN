
# Structural Constraint Integration in Generative Model for Discovery of Quantum Material Candidates

Implementation codes for crystal structure prediction by Joint equivariant diffusion with structural constraints.  

<p align="center">
  <img src="assets/scigen_logo.png" width="250">
</p>

<p align="center">
  <img src="assets/figure1.png" width="600">
</p>

## Table of Contents
- [Dependencies](#dependencies)
- [Config Setting](#config-setting)
- [Training](#training)
- [Evaluation Task](#evaluation-task)
- [Metastable Structure Generation](#metastable-structure-generation)
- [Convert Output to CIF Files](#convert-output-to-cif-files)
- [Generate Movie of Material Generation](#generate-movie-of-material-generation)
- [References](#references)

---

## Dependencies

### Version-dependent libraries
```bash
python==3.9.20   
torch==2.0.1+cu118   
torch-geometric==2.3.0   
pytorch_lightning==1.3.8   
pymatgen==2023.9.25   
hydra-core==1.1.0
hydra-joblib-launcher==1.1.5
e3nn==0.5.1
```

### Other libraries    
```bash
pandas
smact
wandb
imageio
...
```

---

## Config Setting   
Set the configuration files:
- Duplicate `.env.template` file and rename it as `.env`. 
- Set the following paths: 
  - `PROJECT_ROOT`, `HYDRA_JOBS`, `WANDB_DIR`

---

## Training
Use DiffCSP for training:

```bash
python scigen/run.py data=mp_20 model=diffusion_w_type expname=<expname>
```

---

## Evaluation Task

### Configurations for evaluation
`config_scigen.py`:
1. Use the pre-trained model:
   - Download the pre-trained model files.
   - Place the zip folder in the home directory (`PROJECT_ROOT`) and unzip it.

2. Use the model you trained:
   - Set the configuration file.
   - Choose to use either the pre-trained model or your own model.
     - Download the pre-trained model and store it in `scigen/prop_models/mp_20`.

---

## Metastable Structure Generation

### Configuration

| Parameter              | Description                                                                                 | Default Value                           |
|------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------|
| `batch_size`           | Number of materials to generate per batch.                                                 | `10`                                    |
| `num_batches_to_samples` | Number of batches to sample during generation.                                             | `20`                                    |
| `num_materials`        | Total number of materials to generate (`batch_size * num_batches_to_samples`).              | `batch_size * num_batches_to_samples`   |
| `save_traj_idx`        | Indices for which the generation trajectory will be saved.                                  | `[]`                                    |
| `num_run`              | Number of independent runs to perform.                                                     | `1`                                     |
| `idx_start`            | Starting index for labeling generated materials.                                            | `0`                                     |
| `c_scale`              | Scaling factor for the c-axis; `None` means no constraint.                                  | `None`                                  |
| `c_vert`               | Whether to constrain the c-axis to be vertical.                                             | `False`                                 |
| `header`               | Prefix for labeling the generated materials.                                                | `'sc'`                                  |
| `sc_list`              | List of structural constraints (e.g., triangular lattice).                                  | `['tri']`                               |
| `atom_list`            | Atomic species to include in the generated materials.                                       | `['Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Nd', 'Gd', 'Tb', 'Dy', 'Yb']` |
| `generate_cif`         | Whether to save the generated materials as CIF files.                                       | `True`                                  |


### Lattice Types
Select from the following lattice types: 
- **Triangular (tri)**, **Honeycomb (hon)**, **Kagome (kag)**, **Square (square)**, **Elongated (elt)**
- **Snub square (sns)**, **Truncated square (tsq)**, **Small rhombitrihexagonal (srt)**
- **Snub hexagonal (snh)**, **Truncated hexagonal (trh)**, **Lieb (lieb)**

<p align="center">
  <img src="assets/SI_arch_lattice_unit_bk.png" width="500">
</p>

Run the following to generate structures:
```bash
python gen_mul.py
```

---

## Convert Output to CIF Files

Convert generated outputs into CIF files:
```bash
python script/save_cif.py --label <out_name>
```
*If `out_name` is set in `config_scigen.py`, you do not need to set `--label`.

---

## Generate Movie of Material Generation

Set `out_name` in `config_scigen.py`, then run:
```bash
python gen_movie.py
```

---

## References

### Publication
```plaintext
@article{okabe2024structural,
  title={Structural Constraint Integration in Generative Model for Discovery of Quantum Material Candidates},
  author={Okabe, Ryotaro and Cheng, Mouyang and Chotrattanapituk, Abhijatmedhi and Hung, Nguyen Tuan and Fu, Xiang and Han, Bowen and Wang, Yao and Xie, Weiwei and Cava, Robert J and Jaakkola, Tommi S and others},
  journal={arXiv preprint arXiv:2407.04557},
  year={2024}
}
```
[https://arxiv.org/abs/2407.04557](https://arxiv.org/abs/2407.04557)

### Dataset
Generated material dataset:  
[https://doi.org/10.6084/m9.figshare.c.7283062.v1](https://doi.org/10.6084/m9.figshare.c.7283062.v1)
