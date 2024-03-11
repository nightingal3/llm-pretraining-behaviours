#!/bin/bash
#SBATCH --job-name="get_c4_parallel"
#SBATCH --output=get_c4_parallel.out
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=END
#SBATCH --mail-user=mchen5@andrew.cmu.edu

# domains=("peS2o" "common-crawl" "stack-code" "wiki-en-simple" "c4" "gutenberg-books")

domain=${1:-c4}
num_total_tokens=${2:-1T}
base_dir=${3:-/data/tir/projects/tir5/users/mengyan3/dolma_data_processed/dolma_1T}
file_lsts_dir=${4:-/data/tir/projects/tir7/user_data/mchen5/dolma_file_splits}

output="${base_dir}/${domain}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

# delete intermediate files immediately so we don't run out of space
python demo_scripts/fetch_tokens_from_dolma_parallel.py --num_total_tokens $num_total_tokens --output $output --domain $domain --file_lsts_dir $file_lsts_dir
