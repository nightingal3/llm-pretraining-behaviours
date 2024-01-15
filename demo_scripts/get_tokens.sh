#!/bin/bash
#SBATCH --job-name=get_tokens_wiki
#SBATCH --output=get_tokens_wiki.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=75G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu

domain=${1:-gutenberg-books}
num_total_tokens=${2:-100B}
base_dir=${3:-/data/tir/projects/tir3/users/mengyan3/dolma_100B}

output="${base_dir}/${domain}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

# delete intermediate files immediately so we don't run out of space
python demo_scripts/fetch_tokens_from_dolma.py --domain $domain --num_total_tokens $num_total_tokens --output $output
