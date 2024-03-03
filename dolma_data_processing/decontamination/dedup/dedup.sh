#!/bin/bash
#SBATCH --job-name="dedup"
#SBATCH --output=dedup.out
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

python dedup.py --contaminant_path="/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/dolma_data_processing/decontamination/dedup/contaminant.txt" --base_dir="/data/tir/projects/tir7/user_data/mchen5/dolma_100B" --output_dir="/data/tir/projects/tir5/users/mengyan3/dolma_data_processed/dolma_100B_deduped" --num_processes=8