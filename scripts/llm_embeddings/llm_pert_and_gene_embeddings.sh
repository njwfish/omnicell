#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -n 4
#SBATCH --mem=256GB
#SBATCH -p ou_bcs_low
#SBATCH --array=0-15       # 4 datasets × 4 LLMs = 16 combinations
#SBATCH --gres=gpu:h100:1

hostname

# Define datasets and models
DATASETS=("essential_gene_knockouts_raw" "repogle_k562_essential_raw" "kang" "satija_IFNB_raw")
LLMS=("MMedllama-3-8B" "llamaPMC-13B" "llamaPMC-7B"  "bioBERT")

# Calculate indices for current task
dataset_idx=$((SLURM_ARRAY_TASK_ID / ${#LLMS[@]}))
llm_idx=$((SLURM_ARRAY_TASK_ID % ${#LLMS[@]}))

# Get current dataset and LLM
DATASET=${DATASETS[$dataset_idx]}
LLM=${LLMS[$llm_idx]}

source ~/.bashrc
conda activate huggingface

echo "Current git branch" 
git branch

echo "Generating pert and gene llm embeddings for dataset ${DATASET} with model ${LLM}"

# Generate gene embeddings
python -m scripts.llm_embeddings.generate_llm_gene_embeddings \
    --dataset_name ${DATASET} \
    --model_name ${LLM}

# Generate perturbation embeddings
python -m scripts.llm_embeddings.generate_llm_pert_embeddings \
    --dataset_name ${DATASET} \
    --model_name ${LLM}

echo "Finished embeddings for dataset ${DATASET} with model ${LLM}"