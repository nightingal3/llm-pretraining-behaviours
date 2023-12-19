#!/bin/bash
#SBATCH --job-name=get_tokens
#SBATCH --output=get_tokens.out
#SBATCH --cpus-per-task=30
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

# delete intermediate files immediately so we don't run out of space
python my_scripts/fetch_tokens_from_dolma.py --domain gutenberg-books
python my_scripts/fetch_tokens_from_dolma.py --domain peS2o
#python my_scripts/fetch_tokens_from_dolma.py --domain c4
#python my_scripts/fetch_tokens_from_dolma.py --domain stack-code
#python my_scripts/fetch_tokens_from_dolma.py --domain common-crawl
