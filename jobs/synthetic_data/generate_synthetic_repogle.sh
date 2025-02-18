#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -n 4 #4 CPUS
#SBATCH --mem=100GB
#SBATCH -p newnodes
#SBATCH --array=0-3 # CHANGE HERE TO MATCH SIZE OF CROSS PRODUCT: 2 ETL Configs x 2 Splits = 4 combinations

hostname

CONFIG_BASE_DIR="./configs"
ETL_BASE_DIR="./configs/ETL"

DATASET="repogle_k562_essential_raw"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/splits/${DATASET}/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1"

### CHANGE HERE FOR THE CORRECT MODEL CONFIG ###
MODEL_CONFIG="${CONFIG_BASE_DIR}/models/scot/proportional_scot.yaml"
MODEL_NAME="proportional_scot"

### CHANGE HERE TO SELECT ONLY THE RELEVANT ETL CONFIGS UNDER ${ETL_BASE_DIR} ###
ETL_CONFIGS=("no_preproc_drop_unmatched")

### CHANGE HERE TO ONLY SELECT ONE OF THE RANDOM SPLITS ###
SPLITS=(0 1) # 2 splits (0) or (1)



# ===== CONFIGURATION =====


# Calculate indices for 2 dimensions
total_splits=${#SPLITS[@]}
total_etl=${#ETL_CONFIGS[@]}

# Calculate indices properly
etl_config_idx=$((SLURM_ARRAY_TASK_ID / total_splits))
split_idx=$((SLURM_ARRAY_TASK_ID % total_splits))

# Get current configs and split
ETL_CONFIG="${ETL_BASE_DIR}/${ETL_CONFIGS[$etl_config_idx]}.yaml"
SPLIT_DIR="split_${SPLITS[$split_idx]}"

source ~/.bashrc
conda activate omnicell

echo "Processing ETL config: ${ETL_CONFIG}"
echo "Processing split: ${SPLIT_DIR}"

# Run synthetic data generation with logging
python generate_synthetic_data.py \
    --dataset ${DATASET} \
    --model_config ${MODEL_CONFIG} \
    --etl_config ${ETL_CONFIG} \
    --datasplit_config ${SPLIT_BASE_DIR}/${SPLIT_DIR}/split_config.yaml \
    --slurm_id ${SLURM_ARRAY_JOB_ID} \
    --slurm_array_task_id ${SLURM_ARRAY_TASK_ID} \
    -l DEBUG

echo "Synthetic data generation completed for ${SPLIT_DIR}"