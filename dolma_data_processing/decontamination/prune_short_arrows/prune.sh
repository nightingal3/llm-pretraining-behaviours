#!/bin/bash
#SBATCH --job-name="prune_c4"
#SBATCH --output=prune_c4.out
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

domain="c4"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

python prune.py --domain $domain
