#!/bin/bash
#SBATCH --job-name=gutenberg-books
#SBATCH --output=gutenberg-books.out
#SBATCH --error=gutenberg-books.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END


source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm1

if [ -f /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/configs/.env ]; then
    set -a  # Automatically export all variables
    source /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/configs/.env
    set +a  # Stop automatically exporting
fi

set -euo pipefail

input_dir="/data/tir/projects/tir7/user_data/ssingha2/dolma_100B_json_p2/gutenberg-books"
output_dir="/data/tir/projects/tir7/user_data/ssingha2/dolma_100B_decisions/gutenberg-books"


# Loop over all .jsonl files in the input directory
for jsonl_file in "$input_dir"/*.jsonl; do
    echo "Processing file: $jsonl_file"
    
    # Calculate the line count for the current .jsonl file
    line_count=$(wc -l "$jsonl_file" | awk '{print $1}')
    echo "Total lines in $jsonl_file: $line_count"

    base_name=$(basename -- "$jsonl_file")
    output_csv="$output_dir/${base_name%.jsonl}.csv"
    
    for (( i=0; i<=line_count; i+=1000 )); do
        echo "Processing chunk starting at line: $i"
        python3 /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/dolma_data_processing/decontamination/coderemoval/Mistralv2-dolma.py  --input "$jsonl_file" --output "$output_csv" --idx $i
    done
done