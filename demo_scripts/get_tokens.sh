#!/bin/bash
#SBATCH --job-name="get_stack-code"
#SBATCH --output=get_stack-code.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

# domains=("peS2o" "common-crawl" "stack-code" "wiki-en-simple" "c4" "gutenberg-books")

domain="stack-code"
num_total_tokens=${2:-100B}
base_dir=${3:-/data/tir/projects/tir7/user_data/mchen5/dolma_100B}

output="${base_dir}/${domain}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

# delete intermediate files immediately so we don't run out of space
python demo_scripts/fetch_tokens_from_dolma.py --num_total_tokens $num_total_tokens --output $output --domain $domain
