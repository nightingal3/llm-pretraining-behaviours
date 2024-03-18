#!/bin/bash
#SBATCH --job-name=preprocess_data
#SBATCH --output=preprocess_data.out
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END

# Usage: sbatch demo_scripts/preprocess_data.sh <arrow_dir (needs to exist)> <dataset_bin (to be created)> \
#   <dataset_json (to be created)> <path_to_llama_tok (needs to exist)>

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

set -euo pipefail

input_dir="/data/tir/projects/tir7/user_data/mchen5/dolma_100B/wiki-en-simple"
output_dir="/data/tir/projects/tir7/user_data/ssingha2/dolma_100B_json/wiki-en-simple"

cpu_workers=${SLURM_CPUS_PER_TASK:-30}

# Loop through all .arrow files in the input directory
for arrow_file in "$input_dir"/*.arrow; do
    # Generate the corresponding output file name
    base_name=$(basename -- "$arrow_file")
    output_file="$output_dir/${base_name%.arrow}.jsonl"

    # Check if the output file already exists
    if [[ ! -f $output_file ]]; then
        # Convert pyarrow to jsonl
        python ./convert_pyarrow_to_jsonl.py \
            --input "$arrow_file" \
            --output "$output_file"
    fi
done



