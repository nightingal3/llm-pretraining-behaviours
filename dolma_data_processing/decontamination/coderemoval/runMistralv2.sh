#!/bin/bash
#SBATCH --job-name=testbeforepush
#SBATCH --output=testbeforepush.out
#SBATCH --error=testbeforepush.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END

source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm1

export HF_HOME="/data/tir/projects/tir7/user_data/ssingha2/hf_cache"

set -euo pipefail

python3 /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/dolma_data_processing/decontamination/coderemoval/Mistralv2-sample.py
