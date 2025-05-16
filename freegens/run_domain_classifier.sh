#!/bin/bash
#SBATCH --job-name=domain_classification
#SBATCH --output=domain_classification_%A_%a.log
#SBATCH --error=domain_classification_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Set up environment - adjust as needed for your system
# source /path/to/your/virtualenv/bin/activate
set -a 
source /data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretrain_lms/configs/.env
set +a

source ${MINICONDA_PATH}
conda activate ${TOWERLLM_ENV_NAME}

export OPENAI_API_KEY=<key>
# Base directories
BASE_DIR="/data/tir/projects/tir3/users/mengyan3/subsets_for_tagging"
OUTPUT_DIR="domain_classification_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Define target files to process
declare -a TARGETS=(
    "HuggingFaceFW/fineweb/sample.jsonl"
    "allenai/c4/sample.jsonl"
    "bigcode/starcoderdata/sample.jsonl"
    "monology/pile-uncopyrighted/sample.jsonl"
)

# Add RedPajama if it exists
if [ -f "${BASE_DIR}/togethercomputer/RedPajama-Data-1T/sample.jsonl" ]; then
    TARGETS+=("togethercomputer/RedPajama-Data-1T/sample.jsonl")
fi

echo "Starting domain classification jobs at $(date)"

# Process each target file
for target in "${TARGETS[@]}"; do
    # Extract dataset name for output file naming
    dataset_name=$(echo $target | sed 's/\//_/g' | sed 's/\.jsonl$//')
    
    echo "Processing $target..."
    
    # Run the domain classification script with a limit of 10,000 samples
    python /data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/freegens/classify_domains_multistage.py \
        --input "${BASE_DIR}/${target}" \
        --output "${OUTPUT_DIR}/${dataset_name}_predicted_domains.csv" \
        --limit 1000
    
    echo "Completed processing $target at $(date)"
done

echo "All domain classification tasks completed at $(date)"