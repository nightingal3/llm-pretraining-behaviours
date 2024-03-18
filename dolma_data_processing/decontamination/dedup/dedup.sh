#!/bin/bash
#SBATCH --job-name="dedup-wiki"
#SBATCH --output=dedup-wiki.out
#SBATCH --mem=400G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

python dedup.py --contaminant_path="/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/dolma_data_processing/decontamination/dedup/contaminant.txt" --base_dir="/data/tir/projects/tir7/user_data/mchen5/dolma_1T" --output_dir="/data/tir/projects/tir7/user_data/mchen5/dolma_1T_deduped" --domain="wiki-en-simple" --num_processes=30