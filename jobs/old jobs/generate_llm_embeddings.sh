#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512GB
#SBATCH -p ou_bcs_low
#SBATCH --array=0-15        # 4 models x 4 datasets = 16 combinations
#SBATCH --gres=gpu:h100:1  # 1 h100 GPU
hostname

source ~/.bashrc
conda activate huggingface

# Define arrays of models and datasets
MODELS=("MMedllama-3-8B" "llamaPMC-13B" "llamaPMC-7B" "bioBERT")
DATASETS=("satija_IFNB_raw" "repogle_k562_essential_raw" "kang" "essential_gene_knockouts_raw")

# Calculate indices
model_idx=$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))
dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))

# Get current model and dataset
MODEL=${MODELS[$model_idx]}
DATASET=${DATASETS[$dataset_idx]}

echo "Processing model: $MODEL, dataset: $DATASET"

# Run Python script with arguments
python -m scripts.generate_llm_gene_embeddings \
    --model_name "$MODEL" \
    --dataset_name "$DATASET"