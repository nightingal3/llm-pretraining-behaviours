#!/bin/bash
#SBATCH --job-name=subset_data
#SBATCH --output=logs/subset_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=array
#SBATCH --array=8-9
#SBATCH --mail-type END
#SBATCH --mail-user emmy@cmu.edu

# Array of input directories
declare -a dirs=(
    "/data/datasets/huggingface/huggingfacetb/cosmopedia/auto_math_text"
    "/data/datasets/huggingface/huggingfacetb/cosmopedia/web_samples_v1"
    "/data/datasets/huggingface/huggingfacetb/cosmopedia/web_samples_v2"
    "/data/datasets/huggingface/hug\gingfacetb/cosmopedia/stanford"
    "/data/datasets/huggingface/huggingfacetb/cosmopedia/stories"
    "/data/datasets/huggingface/bigcode/the-stack"
    "/data/datasets/huggingface/bigcode/starcoderdata"
    "/data/datasets/huggingface/huggingfacetb/smollm-corpus/cosmopedia-v2"
    "/data/datasets/shared/fineweb/sample/100BT"
    "/data/datasets/shared/fineweb-edu/sample/100BT"
)

declare -a out_names=(
    "auto_math_text"
    "web_samples_v1"
    "web_samples_v2"
    "stanford"
    "stories"
    "the-stack"
    "starcoderdata"
    "cosmopedia-v2"
    "fineweb_100bt"
    "fineweb_edu_100bt"
)


# Get current directory from array
INPUT_DIR=${dirs[$SLURM_ARRAY_TASK_ID]}
OUT_NAME=${out_names[$SLURM_ARRAY_TASK_ID]}

# Get parent and dataset name
if [[ $INPUT_DIR == */smollm-corpus/* ]]; then
    PARENT_DIR="smollm-corpus"
    DATASET_NAME=$(basename $INPUT_DIR)
elif [[ $INPUT_DIR == */cosmopedia/* ]]; then
    PARENT_DIR="cosmopedia"
    DATASET_NAME=$(basename $INPUT_DIR)
elif [[ $INPUT_DIR == */bigcode/* ]]; then
    PARENT_DIR="bigcode"
    DATASET_NAME=$(basename $INPUT_DIR)
else
    PARENT_DIR=$(basename $(dirname $INPUT_DIR))
    DATASET_NAME=$(basename $INPUT_DIR)
fi

# Set output directory and create it
OUTPUT_DIR="/data/tir/projects/tir3/users/mengyan3/subsets_for_tagging/${OUT_NAME}"
mkdir -p $OUTPUT_DIR

# Create logs directory
mkdir -p logs

# Run the script

# if the output file already exists, skip
if [ -f "${OUTPUT_DIR}/${DATASET_NAME}_subset.jsonl" ]; then
    echo "Output file already exists, skipping"
    exit 0
fi

python subset_data.py \
    --input-dir $INPUT_DIR \
    --output-file "${OUTPUT_DIR}/${DATASET_NAME}_subset.jsonl" \
    --subset-size 1000000 \
    --batch-size 10000 \
    --seed 42

echo "Wrote to ${OUTPUT_DIR}/${DATASET_NAME}_subset.jsonl"