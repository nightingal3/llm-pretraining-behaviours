#!/bin/bash
#SBATCH --job-name=preprocess_data
#SBATCH --output=preprocess_data.out
#SBATCH --cpus-per-task=30
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --partition=babel-shared-long
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END

# Usage: sbatch demo_scripts/preprocess_data.sh <arrow_dir (needs to exist)> <dataset_bin (to be created)> \
#   <dataset_json (to be created)> <path_to_llama_tok (needs to exist)>

set -euo pipefail

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    echo "Usage: sbatch demo_scripts/preprocess_data.sh [dataset_bin]"
    exit 0
fi
arrow_file=${1:-/data/tir/projects/tir6/general/mengyan3/tower-llm-training/wiki-en-simple_200000000/part_1.arrow}
dataset_bin=${2:-./wiki-en-simple_200000000-bin}
dataset_json=${3:-/data/tir/projects/tir6/general/mengyan3/tower-llm-training/wiki-en-simple_200000000/part1.jsonl}
external_tokenizer=${4:-/data/datasets/models/huggingface/meta-llama/Llama-2-70b-hf/}
cpu_workers=${SLURM_CPUS_PER_TASK:-30}

if [[ ! -f $dataset_json ]]; then
    # convert pyarrow to jsonl
    python ./demo_scripts/convert_pyarrow_to_jsonl.py \
        --input $arrow_file \
        --output $dataset_json
fi

# preprocess data
mkdir -p $dataset_bin
python ./Megatron-DeepSpeed/tools/preprocess_data.py \
    --input $dataset_json \
    --output-prefix $dataset_bin/data \
    --dataset-impl mmap \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $external_tokenizer \
    --append-eod \
    --workers $cpu_workers 


