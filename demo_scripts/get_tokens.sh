#!/bin/bash
#SBATCH --job-name="get_common_crawl"
#SBATCH --output=get_common_crawl.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

# domains=("peS2o" "common-crawl" "stack-code" "wiki-en-simple" "c4" "gutenberg-books")

domain=${1:-common-crawl}
num_total_tokens=${2:-1T}
base_dir=${3:-/data/tir/projects/tir5/mengyan3/dolma_data_processed/dolma_1T}

output="${base_dir}/${domain}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

# delete intermediate files immediately so we don't run out of space
python demo_scripts/fetch_tokens_from_dolma.py --num_total_tokens $num_total_tokens --output $output --domain $domain
