#!/bin/bash
#SBATCH --job-name=convert_to_hf_try
#SBATCH --output=convert_to_hf_try.out
#SBATCH --mem=30G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --mail-user=pfernand@cs.cmu.edu
#SBATCH --mail-type=END

source ~/conda/etc/profile.d/conda.sh
conda activate llm-pretraining-env

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Usage: sbatch demo_scripts/convert_to_hf.sh <checkpoint_path> <model_config> <external_tokenizer> <output_dir>

set -euo pipefail
set -x

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
   echo "Usage: sbatch demo_scripts/convert_to_hf.sh [checkpoint_path] [model_config] [external_tokenizer] [output_dir]"
   exit 0
fi

CHECKPOINT_PATH=${1:-/data/tir/projects/tir6/general/mengyan3/tower-llm-training/llama_mini_try_1B}
model_config=${2:-./demo_scripts/configs/Llama2_220M.yaml}
external_tokenizer=${3:-meta-llama/Llama-2-7b-hf}
output_dir=${4:-/data/tir/projects/tir5/users/mengyan3/dolma_checkpts/llama_mini_try_1B_hf}
repo=/home/pfernand/repos/llm-pretraining-behaviours/

num_layers=$(yq '.training.num_layers' $model_config)
num_attention_heads=$(yq '.training.num_attention_heads' $model_config)
seq_length=$(yq '.training.seq_length' $model_config)
num_kv_heads=$(yq '.training.num_kv_heads' $model_config)
hidden_size=$(yq '.training.hidden_size' $model_config)
ffn_hidden_size=$(yq '.training.ffn_hidden_size' $model_config)


tune_steps=$(yq '.training.tune_steps' $model_config)
lr=$(yq '.training.lr' $model_config)
min_lr=$(yq '.training.min_lr' $model_config)
weight_decay=$(yq '.training.weight_decay' $model_config)
grad_clip=$(yq '.training.grad_clip' $model_config)
lr_warmup_steps=$(yq '.training.lr_warmup_steps' $model_config)
save_interval=$(yq '.training.save_interval' $model_config)
eval_interval=$(yq '.training.eval_interval' $model_config)
train_steps=$(yq '.training.train_steps' $model_config)
tp=1 # don't use this
seed=42

echo -n '{
        "train_batch_size" : 2,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": 0
        },
        "bf16": {
            "enabled": true
        }
}' | jq . > /tmp/convert_ds_config.json

distributed_args="--num_nodes=1 --num_gpus=1 --master_port 12345"
ds_args="--zero-stage=0 --deepspeed --deepspeed_config /tmp/convert_ds_config.json"
deepspeed $distributed_args \
       $repo/scripts/deepspeed_to_hf_llama.py \
       --output-dir $output_dir \
       --tensor-model-parallel-size $tp \
       --no-pipeline-parallel \
       --num-layers $num_layers \
       --hidden-size $hidden_size \
       --ffn-hidden-size $ffn_hidden_size \
       --num-attention-heads $num_attention_heads \
       --num-key-value-heads $num_kv_heads \
       --seq-length $seq_length \
       --max-position-embeddings $seq_length \
       --no-query-key-layer-scaling \
       --disable-bias-linear \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --micro-batch-size 1 \
       --load $CHECKPOINT_PATH \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $external_tokenizer \
       --fp16 \
       --distributed-timeout-minutes 60 \
        --no-save-optim \
        --no-save-rng \
        --no-load-optim \
        --no-load-rng \
        --seed $seed \
        $ds_args 