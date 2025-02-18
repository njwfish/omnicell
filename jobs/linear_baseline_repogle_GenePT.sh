#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=96GB
#SBATCH -p ou_bcs_low
#SBATCH --array=0-1       # 1 Gene Embeddings x 2 Splits = 2 combinations

hostname

CONFIG_BASE_DIR="configs"
ETL_BASE_DIR="configs/ETL"
EMBEDIDNG_BASE_DIR="configs/embeddings"


# ===== CONFIGURATION =====
DATASET="repogle_k562_essential_raw"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/splits/${DATASET}/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1"
MODEL_CONFIG="${CONFIG_BASE_DIR}/models/linear_mean_model.yaml"
MODEL_NAME="linear_mean_model"

# Define embeddings
EMBEDDING="pemb_GenePT"
EMBEDDING_CONFIG="${EMBEDIDNG_BASE_DIR}/${EMBEDDING}.yaml"

# Define configs and splits
ETL_CONFIGS=("no_preproc_drop_unmatched")
SPLITS=(0 1)

# Calculate indices
etl_config_idx=$((SLURM_ARRAY_TASK_ID / ${#SPLITS[@]}))
split_idx=$((SLURM_ARRAY_TASK_ID % ${#SPLITS[@]}))

# Get current config and split
ETL_CONFIG="${ETL_BASE_DIR}/${ETL_CONFIGS[$etl_config_idx]}.yaml"
SPLIT_DIR="split_${split_idx}"

source ~/.bashrc
conda activate omnicell

echo "Processing ETL config: ${ETL_CONFIG}"
echo "Processing split: ${SPLIT_DIR}"

# Run training
python train.py \
    --etl_config ${ETL_CONFIG} \
    --datasplit_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/split_config.yaml \
    --eval_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/eval_config.yaml \
    --model_config ${MODEL_CONFIG} \
    --slurm_id ${SLURM_ARRAY_JOB_ID} \
    --slurm_array_task_id ${SLURM_ARRAY_TASK_ID} \
    --embedding_config ${EMBEDDING_CONFIG} \
    -l DEBUG

# Generate evaluations

echo "Generating evaluations for ./results/${DATASET}/${ETL}/${MODEL}"

python generate_evaluations.py \
    --root_dir ./results/${DATASET}/${ETL}/${MODEL}

echo "All jobs finished for ${SPLIT_DIR}"