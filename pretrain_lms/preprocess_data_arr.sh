#!/bin/bash
#SBATCH --job-name=preprocess_data_%A_%a
#SBATCH --output=preprocess_data_%A_%a.out
#SBATCH --cpus-per-task=30
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --partition=general
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --array=5%5

set -eo pipefail

set -a 
source /data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretrain_lms/configs/.env
set +a

source ${MINICONDA_PATH}
conda activate ${TOWERLLM_ENV_NAME}


if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    echo "Usage: sbatch demo_scripts/preprocess_data.sh [dataset_bin]"
    exit 0
fi

EXP_CONFIG=$1
external_tokenizer=/data/models/huggingface/meta-llama/Llama-2-70b-hf/

IFS=',' read -r task_id arrow_file dataset_bin dataset_json cmd <<< $(sed "${SLURM_ARRAY_TASK_ID}q;d" $EXP_CONFIG)
echo '=== JOB INFO ==='
echo "exp_config: ${EXP_CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Running on: ${HOSTNAME}"
echo "arrow_file: ${arrow_file}"
echo "dataset_bin: ${dataset_bin}"
echo "dataset_json: ${dataset_json}"
echo "external_tokenizer: ${external_tokenizer}"
echo '=== END JOB INFO ==='

if [[ ! -f $dataset_json ]]; then
    # convert pyarrow to jsonl
    python /data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretrain_lms/convert_pyarrow_to_jsonl.py \
        --input $arrow_file \
        --output $dataset_json
fi
echo "Created dataset jsonl file: $dataset_json"

# preprocess data
mkdir -p $dataset_bin
python /data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/Megatron-DeepSpeed/tools/preprocess_data.py \
    --input $dataset_json \
    --output-prefix $dataset_bin/data \
    --dataset-impl mmap \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $external_tokenizer \
    --append-eod \
    --workers 30 

echo "Finished preprocessing data"


