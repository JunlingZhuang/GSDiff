# DiGress: Discrete Denoising diffusion models for graph generation

Update (Nov 20th, 2023): Working with large graphs (more than 100-200 nodes)? Consider using SparseDiff, a sparse version of DiGress: https://github.com/qym7/SparseDiff

Update (July 11th, 2023): the code now supports multi-gpu. Please update all libraries according to the instructions. 
All datasets should now download automatically

  - For the conditional generation experiments, check the `guidance` branch.
  - If you are training new models from scratch, we recommand to use the `fixed_bug` branch in which some neural
network layers have been fixed. The `fixed_bug` branch has not been evaluated, but should normally perform better.
If you train the `fixed_bug` branch on datasets provided in this code, we would be happy to know the results.

## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9```
  - `conda activate digress`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```

  - Navigate to the ./src/analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```

Note: graph_tool and torch_geometric currently seem to conflict on MacOS, I have not solved this issue yet.

## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - To run the continuous model: `python3 main.py model=continuous`
  - To run the discrete model: `python3 main.py`
  - You can specify the dataset with `python3 main.py dataset=guacamol`. Look at `configs/dataset` for the list
of datasets that are currently available

## MSD phase-1 floorplan graph baseline

This repository includes a vanilla unconditional DiGress baseline for Modified Swiss Dwellings (MSD). It uses only the access graph (`graph_out`) and does not condition on geometry or partial layouts.

### Quick start: wall-augmented MSD

Authenticate Kaggle first. The script uses `kaggle` if it is installed, otherwise it falls back to `uvx kaggle`.

```powershell
cd D:\Github\GSDiff\digress

# Prepare the wall-augmented graph dataset.
# Defaults: wall_contact_eps=0.01, wall_min_contact_length=0.02, wall_segment_gap=0.45.
.\.venv\Scripts\python.exe scripts\download_msd_dataset.py --prepare --add-wall-edges

# Train the wall-augmented unconditional DiGress graph model.
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall.yaml

# Resume training from a checkpoint if needed.
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall.yaml general.resume="outputs/<run>/checkpoints/msd_wall/last.ckpt"

# Generate wall-augmented test graphs and the combined sample PNG.
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd_wall
```

The preprocessing command overwrites `data/msd_wall/*` and clears its processed cache. It does not delete `../datasets/msd/raw/*`.

### Download and prepare

For the access-only baseline, use the original MSD access graph:

```powershell
cd digress

# Download raw Kaggle data to ../datasets/msd/raw.
.\.venv\Scripts\python.exe scripts\download_msd_dataset.py

# Or download and immediately build DiGress graphs plus sample visualizations.
.\.venv\Scripts\python.exe scripts\download_msd_dataset.py --prepare

```

The prepared files are:
- `data/msd/graphs.p`: NetworkX graphs consumed by `dataset=msd`
- `data/msd/dataset_stats.json`: node/edge counts, type proportions, and room area statistics
- `data/msd/sample_graphs.png`: combined quick-look graph grid
- `data/msd/vis/sample_*.png`: RPLAN-style 2x3 dataset diagnostics

If the raw Kaggle data already exists, `--prepare` skips re-downloading and rebuilds the DiGress graph pickle from `train/graph_out`.
For `--add-wall-edges`, the script uses the raw MSD CSV because `WALL` polygons are not present in `graph_out`. Wall edges are inferred from local wall-room contact segments: each room must have enough boundary contact with the same `WALL` polygon, the contact segments must be locally close on that wall, and the rooms must lie on opposite sides of the wall.

### Train and test

```powershell
cd digress

# Train the unconditional MSD graph model.
.\.venv\Scripts\python.exe src\main.py dataset=msd +experiment=msd.yaml

# Train the wall-augmented unconditional MSD graph model.
.\.venv\Scripts\python.exe src\main.py dataset=msd_wall +experiment=msd_wall.yaml

# Generate samples using the checkpoint from the same config family.
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd

# Generate samples from the wall-augmented checkpoint.
.\.venv\Scripts\python.exe scripts\test_graph_generation.py msd_wall
```

The same generic graph test entrypoint works for RPLAN:

```powershell
.\.venv\Scripts\python.exe scripts\test_graph_generation.py rplan
```

Compatibility wrappers are kept, so `scripts/test_rplan.py` and `scripts/test_msd.py` still work.

Useful overrides:
- `train.batch_size=4`: lower memory use on larger MSD graphs
- `dataset.max_nodes=32`: filter to smaller graphs for smoke tests
- `train.n_epochs=2`: short pipeline validation run
    
## Checkpoints

**My drive account has unfortunately been deleted, and I have lost access to the checkpoints. If you happen to have a downloaded checkpoint stored locally, I would be glad if you could send me an email at vignac.clement@gmail.com or raise a Github issue.**

The following checkpoints should work with the latest commit:

  - [QM9 (heavy atoms only)](https://drive.switch.ch/index.php/s/8IhyGE4giIW1wV3) \\
  
  - [Planar](https://drive.switch.ch/index.php/s/8IhyGE4giIW1wV3) \\

  - MOSES (the model in the paper was trained a bit longer than this one): https://drive.google.com/file/d/1LUVzdZQRwyZWWHJFKLsovG9jqkehcHYq/view?usp=sharing -- This checkpoint has been sent to me, but I have not tested it. \\

  - SBM: ~~https://drive.switch.ch/index.php/s/rxWFVQX4Cu4Vq5j~~ \\
    Performance of this checkpoint:
    - Test NLL: 4757.903
    - `{'spectre': 0.0060240439382095445, 'clustering': 0.05020166160905111, 'orbit': 0.04615866844490847, 'sbm_acc': 0.675, 'sampling/frac_unique': 1.0, 'sampling/frac_unique_non_iso': 1.0, 'sampling/frac_unic_non_iso_valid': 0.625, 'sampling/frac_non_iso': 1.0}`

  - Guacamol: https://drive.google.com/file/d/1KHNCnPJmPjIlmhnJh1RAvhmVBssKPqF4/view?usp=sharing -- This checkpoint has been sent to me, but I have not tested it.

## Generated samples

We provide the generated samples for some of the models. If you have retrained a model from scratch for which the samples are
not available yet, we would be very happy if you could send them to us!


## Troubleshooting 

`PermissionError: [Errno 13] Permission denied: '/home/vignac/DiGress/src/analysis/orca/orca'`: You probably did not compile orca.
    

## Use DiGress on a new dataset

To implement a new dataset, you will need to create a new file in the `src/datasets` folder. Depending on whether you are considering
molecules or abstract graphs, you can base this file on `moses_dataset.py` or `spectre_datasets.py`, for example. 
This file should implement a `Dataset` class to process the data (check [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)), 
as well as a `DatasetInfos` class that is used to define the noise model and some metrics.

For molecular datasets, you'll need to specify several things in the DatasetInfos:
  - The atom_encoder, which defines the one-hot encoding of the atom types in your dataset
  - The atom_decoder, which is simply the inverse mapping of the atom encoder
  - The atomic weight for each atom atype
  - The most common valency for each atom type

The node counts and the distribution of node types and edge types can be computed automatically using functions from `AbstractDataModule`.

Once the dataset file is written, the code in main.py can be adapted to handle the new dataset, and a new file can be added in `configs/dataset`.


## Cite the paper

```
@inproceedings{
vignac2023digress,
title={DiGress: Discrete Denoising diffusion for graph generation},
author={Clement Vignac and Igor Krawczuk and Antoine Siraudin and Bohan Wang and Volkan Cevher and Pascal Frossard},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=UaAD-Nu86WX}
}
```
