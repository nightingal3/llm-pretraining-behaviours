#!/bin/bash
#SBATCH --job-name=train_model_try
#SBATCH --output=train_model_try.out
#SBATCH --mem=30G
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=1-00:00:00
#SBATCH --partition=babel-shared-long
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Usage: sbatch demo_scripts/train_model.sh <checkpoint_path> <dataset_bin> <external_tokenizer>

set -euo pipefail

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
   echo "Usage: sbatch demo_scripts/train_model.sh [checkpoint_path] [dataset_bin] [external_tokenizer]"
   exit 0
fi

CHECKPOINT_PATH=${1:-./llama_mini_try}
dataset_bin=${2:-wiki-en-simple_200000000-bin/data_text_document}
external_tokenizer=${3:-meta-llama/Llama-2-7b-hf}
repo=/data/tir/projects/tir6/general/mengyan3/tower-llm-training
data_path="${repo}/${dataset_bin}"

# TODO - just read this from config
# llama mini
num_layers=12
num_attention_heads=8
seq_length=2048
num_kv_heads=8
hidden_size=1024
ffn_hidden_size=4096


tune_steps=1000
lr=0.00015
min_lr=1.0e-5
weight_decay=1e-2
grad_clip=1.0
lr_warmup_steps=100
save_interval=1000
eval_interval=1000
train_steps=100000

tp=1 # don't use this
micro_batch_size=1
seed=42

distributed_args="--num_nodes=1 --num_gpus=4 --master_port 12345"
ds_args="--zero-stage=2 --deepspeed --deepspeed_config /data/tir/projects/tir6/general/mengyan3/tower-llm-training/demo_scripts/ds_config.json"
deepspeed $distributed_args \
       $repo/Megatron-DeepSpeed/pretrain_gpt.py \
       --tensor-model-parallel-size $tp \
       --no-pipeline-parallel \
       --num-layers $num_layers \
       --hidden-size $hidden_size \
       --ffn-hidden-size $ffn_hidden_size \
       --num-attention-heads $num_attention_heads \
       --num-key-value-heads $num_kv_heads \
       --micro-batch-size $micro_batch_size \
       --seq-length $seq_length \
       --max-position-embeddings $seq_length \
       --train-iters $train_steps \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $data_path \
       --data-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $external_tokenizer \
       --split 989,10,1 \
       --distributed-backend nccl \
       --lr $lr \
       --lr-decay-style cosine \
       --min-lr $min_lr \
       --weight-decay $weight_decay \
       --clip-grad $grad_clip \
       --lr-warmup-iters $lr_warmup_steps \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval $save_interval \
       --eval-interval $eval_interval \
       --eval-iters 10 \
       --fp16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --use-flash-attn \
       --distributed-timeout-minutes 60 \
       --seed $seed \
       $ds_args 


    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \ # don't need these for pretraining