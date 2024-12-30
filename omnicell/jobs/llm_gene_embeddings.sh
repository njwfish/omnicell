#!/bin/bash
#SBATCH -t 12:00:00          
#SBATCH --ntasks-per-node=4  
#SBATCH -p ou_bcs_low 
#SBATCH --mem=400GB          
#SBATCH --array=0-11          # 4 models x 3 datasets = 12 combinations
#SBATCH --gres=gpu:h100:1 

hostname

mamba activate huggingface

# Define arrays
MODELS=("MMedllama-3-8B" "llamaPMC-13B" "llamaPMC-7B" "bioBERT")
DATASETS=("satija_IFNB_raw" "essential_gene_knockouts_raw" "kang")

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