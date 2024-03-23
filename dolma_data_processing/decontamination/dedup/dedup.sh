#!/bin/bash
#SBATCH --job-name="dedup3-stack"
#SBATCH --output=dedup3-stack.out
#SBATCH --mem=600G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

python dedup.py --contaminant_path="/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/dolma_data_processing/decontamination/dedup/contaminant.txt" --base_dir="/data/tir/projects/tir7/user_data/mchen5/dolma_1T" --output_dir="/data/tir/projects/tir7/user_data/mchen5/dolma_1T_deduped" --domain="stack-code" --num_processes=30