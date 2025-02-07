#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -n 4
#SBATCH --mem=256GB
#SBATCH -p newnodes
#SBATCH --array=0-2       # 3 datasets only

hostname

# Define datasets
DATASETS=("essential_gene_knockouts_raw" "repogle_k562_essential_raw" "satija_IFNB_raw")

# Get current dataset (direct index since no LLM dimension)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
conda activate huggingface

echo "Generating GenePT embeddings for dataset ${DATASET}"

# Generate embeddings
python -m scripts.genePT_embeddings.generate_genePT_pert_embeddings \
    --dataset_name ${DATASET}

echo "Finished embeddings for dataset ${DATASET} with GenePT"