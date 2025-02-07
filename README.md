# Omnicell

Omnicell is a comprehensive benchmarking and generalization framework for single-cell perturbation response prediction methods. It provides a unified pipeline for training, evaluating, and comparing various single-cell methods.

## Overview

The framework standardizes the process of:
- Loading and preprocessing single-cell data
- Training different model architectures
- Evaluating predictions across multiple metrics
- Comparing performance across methods

## Project Structure

```
omnicell/
├── configs/
│ ├── ETL/ # Data preprocessing configs
│ ├── models/ # Model architecture configs
│ └── {dataset}/ # Dataset-specific configs
│ └── random_splits/ # Train/test split configurations
├── jobs/ # SLURM job submission scripts
├── omnicell/
│ ├── models/ # Model implementations
│ │ ├── sclambda/ # SCLambda model
│ │ ├── VAE/ # Variational autoencoders
│ │ ├── llm/ # Language model-based approaches
│ │ └── ...
│ ├── data/ # Data loading utilities
│ └── processing/ # Data processing utilities
└── train.py # Main training script
```

## Supported Models

The framework currently supports multiple model architectures:
- Nearest Neighbor approaches
- Flow-based models
- Language model-based approaches
- scGen
- scVIDR
- SCLambda (with and without gradient clipping)
- GEARS

With more model architectures coming!

## Configuration System

The framework uses a hierarchical YAML-based configuration system with four main components:

1. **ETL Config**: Data preprocessing and feature extraction settings
2. **Model Config**: Model architecture and training hyperparameters
3. **Split Config**: Dataset splitting strategy
4. **Eval Config**: Evaluation metrics and settings

Example model config (SCLambda):

```yaml
name: sclambda_large_no_clip
latent_dim: 30
hidden_dim: 512
training_epochs: 200
batch_size: 500
lambda_MI: 200
eps: 0.001
seed: 1234
validation_frac: 0.2
large: True
clip: False
```


## Running Experiments

### Local Execution


#### Scripts
```bash
python train.py \
--etl_config configs/ETL/your_etl_config.yaml \
--datasplit_config configs/your_dataset/splits/split_config.yaml \
--eval_config configs/your_dataset/splits/eval_config.yaml \
--model_config configs/models/your_model_config.yaml \
-l DEBUG
```

#### Notebook Use

Define the environment variable `OMNICELL_ROOT` in your `~/.bashrc` file like such: 

```bash
export OMNICELL_ROOT='/orcd/data/omarabu/001/opitcho/omnicell'```
```

### Cluster Execution (SLURM)

The repository includes SLURM job scripts for running experiments on HPC clusters:


```bash
sbatch jobs/sc_lambda_large_repogle_no_clip.sh
```

## Key Features

- **Modular Design**: Easy integration of new models and methods
- **Unified Interface**: Consistent training and evaluation pipeline across methods
- **Reproducibility**: Configuration-based experimentation
- **HPC Support**: Built-in SLURM job submission scripts
- **Flexible Data Loading**: Support for various single-cell data formats
- **Automatic Logging**: Comprehensive experiment tracking

## Dependencies

- PyTorch
- scanpy
- numpy
- scipy
- PyYAML
- logging

## Logging

The framework provides comprehensive logging with configurable levels:
- DEBUG: Detailed debugging information
- INFO: General execution information
- WARNING: Warning messages
- ERROR: Error messages
- CRITICAL: Critical issues

Logs are saved as `output_{slurm_id}_{model_name}_{split_config_name}.log`

## Results

Results are automatically saved in a structured format:
```
results/
└── {dataset}/
    └── {etl_config}/
        └── {model}/
            ├── predictions.npz
            ├── control.npz
            └── ground_truth.npz
```

## Contributing

To add a new model:
1. Create a new model class in `omnicell/models/`
2. Implement the required interface methods (`train`, `predict`, `save`, `load`)
3. Add model configuration in `configs/models/`
4. Register the model in the `get_model` function in `train.py`
