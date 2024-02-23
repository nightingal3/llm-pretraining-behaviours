#!/bin/bash
#SBATCH --job-name="dedup"
#SBATCH --output=dedup.out
#SBATCH --mem=1024G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

python dedup.py
