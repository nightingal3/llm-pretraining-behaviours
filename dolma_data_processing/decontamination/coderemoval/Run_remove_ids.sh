#!/bin/bash
#SBATCH --job-name=removewiki
#SBATCH --output=removewiki.out
#SBATCH --error=removewiki.err
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END


source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

cpu_workers=${SLURM_CPUS_PER_TASK:-30}

python ./remove_ids.py 




