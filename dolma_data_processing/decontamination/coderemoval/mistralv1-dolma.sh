#!/bin/bash
#SBATCH --job-name=fetchinputs
#SBATCH --output=fetchinputs.out
#SBATCH --error=fetchinputs.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END


source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm1

set -euo pipefail

python3 /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/dolma_data_processing/decontamination/coderemoval/batched_dolma.py
