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

if [ -f /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/configs/.env ]; then
    set -a 
    source /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/configs/.env
    set +a 
fi

set -euo pipefail

python3 /data/tir/projects/tir7/user_data/ssingha2/llm_pretraining-behaviours2/llm-pretraining-behaviours/dolma_data_processing/decontamination/code_removal/Mistralv2-sample.py
