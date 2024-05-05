#!/bin/bash
#SBATCH --job-name=preprocess_data_%A_%a
#SBATCH --output=preprocess_data_%A_%a.out
#SBATCH --cpus-per-task=30
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --partition=babel-shared-long
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END

set -euo pipefail

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    echo "Usage: sbatch demo_scripts/preprocess_data.sh [dataset_bin]"
    exit 0
fi

EXP_CONFIG=$1

IFS=',' read -r task_id arrow_file dataset_bin dataset_json external_tokenizer <<< $(sed "${SLURM_ARRAY_TASK_ID}q;d" $CSV_FILE_PATH)

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
    --workers 30 


