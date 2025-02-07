#!/bin/bash
#SBATCH -t 12:00:00          # walltime = 48 hours
#SBATCH --ntasks-per-node=4  # 4 CPU cores
#SBATCH --mem=150GB          # memory per node
#SBATCH --array=0-5          # Job array with indices 0-5 (6 cell lines)

hostname                     # Print the hostname of the compute node

# ===== CONFIGURATION =====
# Set your config paths here
PREPROCESSING_CONFIG="configs/ETL/preprocess_no_embedding.yaml"
MODEL_CONFIG="configs/models/cell.yaml"
# =======================

source ~/.bashrc
conda activate omnicell

# Define the base directory for configs
CONFIG_BASE_DIR="configs"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/satija_IFNB_raw/random_splits/acrossC_ood_ss:10"

# Array of cell lines in order (matching array indices)
CELL_LINES=("A549" "BXPC3" "HAP1" "HT29" "K562" "MCF7")

# Use the SLURM_ARRAY_TASK_ID to select the split
CURRENT_CELL_LINE=${CELL_LINES[$SLURM_ARRAY_TASK_ID]}
SPLIT_DIR="split_${CURRENT_CELL_LINE}"

echo "Processing cell line: ${CURRENT_CELL_LINE}"
echo "Using preprocessing config: ${PREPROCESSING_CONFIG}"
echo "Using model config: ${MODEL_CONFIG}"

# Generate evaluations
python generate_evaluations.py \
    --root_dir ./results/satija_IFNB_raw/cell_emb/rs-across-perts-ood-split/rs-across-perts-ood-split-ss:10_split_${CURRENT_CELL_LINE}

echo "All jobs finished for ${CURRENT_CELL_LINE}"