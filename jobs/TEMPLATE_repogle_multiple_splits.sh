#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -n 4      #4 CPUS
#SBATCH --mem=512GB
#SBATCH -p ou_bcs_low
#SBATCH --gres=gpu:h100:1  # 1 h100 GPU
#SBATCH --array=0-5        # CHANGE HERE TO MATCH SIZE OF CROSS PRODUCT: 3 Gene Embeddings x 2 Splits = 6 combinations 


hostname


### CHANGE HERE FOR THE CORRECT MODEL CONFIG ###
MODEL_CONFIG="${CONFIG_BASE_DIR}/models/sclambda_normal.yaml"
MODEL_NAME="sclambda_normal"

### CHANGE HERE TO SELECT ONLY THE RELEVANT ETL CONFIGS UNDER ${ETL_BASE_DIR} ###
ETL_CONFIGS=("norm_log_drop_unmatched")

### CHANGE HERE TO SELECT ONLY THE RELEVANT EMBEDDING CONFIGS UNDER ${EMB_BASE_DIR} ###
EMB_CONFIGS=("pemb_GenePT" "pemb_llamaPMC7B")

### CHANGE HERE TO ONLY SELECT ONE OF THE RANDOM SPLITS ###
SPLITS=(0 1) # 2 splits (0) or (1)


CONFIG_BASE_DIR="configs"
ETL_BASE_DIR="configs/ETL"
EMB_BASE_DIR="configs/embeddings"


# ===== CONFIGURATION =====
DATASET="repogle_k562_essential_raw"
SPLIT_BASE_DIR="${CONFIG_BASE_DIR}/splits/${DATASET}/random_splits/rs_accP_k562_ood_ss:ns_20_2_most_pert_0.1"



# Calculate indices for 3 dimensions
total_splits=${#SPLITS[@]}
total_emb=${#EMB_CONFIGS[@]}
total_etl=${#ETL_CONFIGS[@]}

# Calculate indices properly
etl_config_idx=$((SLURM_ARRAY_TASK_ID / (total_emb * total_splits)))
remaining=$((SLURM_ARRAY_TASK_ID % (total_emb * total_splits)))
emb_config_idx=$((remaining / total_splits))
split_idx=$((remaining % total_splits))


# Get current configs and split
ETL_CONFIG="${ETL_BASE_DIR}/${ETL_CONFIGS[$etl_config_idx]}.yaml"
EMB_CONFIG="${EMB_BASE_DIR}/${EMB_CONFIGS[$emb_config_idx]}.yaml"
SPLIT_DIR="split_${SPLITS[$split_idx]}"


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
    -l DEBUG

echo "Generating evaluations for ./results/${DATASET}/${ETL}/${MODEL}"

# Generate evaluations
python generate_evaluations.py \
    --root_dir ./results/${DATASET}/${ETL}/${MODEL}
    
echo "All jobs finished for ${SPLIT_DIR}"