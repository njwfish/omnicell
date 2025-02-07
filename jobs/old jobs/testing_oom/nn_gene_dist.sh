#!/bin/bash
#SBATCH -t 48:00:00          # walltime = 48 hours
#SBATCH -n 4                 # 4 CPU cores
#SBATCH --gres=gpu:1 --constraint=high-capacity  # 1 non-A100 GPU 
#SBATCH --mem=128G           # memory per node
hostname                     # Print the hostname of the compute node

# Define arrays for holdouts and targets



source ~/.bashrc
mamba activate sandbox

# Run the training script

echo "Running training"

python train.py --etl_config configs/satija_IFNB_raw/ETL/preprocess_and_UCE_embedding.yaml \
 --datasplit_config configs/satija_IFNB_raw/splits/ho_IFNAR2.yaml \
 --eval_config configs/satija_IFNB_raw/evals/ev_IFNAR2_A549.yaml \
 --model_config configs/models/nearest-neighbor/nearest-neighbor_gene_dist_substitute.yaml -l DEBUG


echo "Training Done - Generating Evaluations"

python generate_evaluations.py --root_dir ./results/ho_IFNAR2

echo "Job Finished"