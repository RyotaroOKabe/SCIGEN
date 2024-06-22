# SCIGEN

Implementation codes for crystal structure prediction by Joint equivariant diffusion with structural constraints.  

### Dependencies

```
python==3.9.16   
torch==2.0.1+cu118   
torch-geometric==2.3.0   
pytorch_lightning==1.3.8   
pymatgen==2023.9.25   
```

### Training

```
python inpcdiff/run.py data=mp_20 model=diffusion_w_type expname=<expname>   
```



### Metastable structure generation

```
python inpaint/generation.py --model_path <model_path> --label <label> --max_atom <max_atom>   
```


### Evaluation

#### Down-sample stable material structures, and save as CIF files. 
python inpaint/eval_stability.py    


