#!/bin/bash
#SBATCH --job-name="collect_models_test"
#SBATCH --output=collect_models_test_1.out
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

python collect_openllm_models_data.py --num_models=10 --openllm_dir="open-llm-leaderboard-results-2024-03-25"