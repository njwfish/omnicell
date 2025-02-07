#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -p ou_bcs_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=500GB
echo hostname

source ~/.bashrc
conda activate omnicell


python -m notebooks.gears.test_script

echo "Job finished"