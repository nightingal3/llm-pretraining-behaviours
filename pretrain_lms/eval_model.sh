#!/bin/bash
#SBATCH --job-name=eval_model_try
#SBATCH --output=eval_model_try.out
#SBATCH --mem=30G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=long
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END

CHECKPOINT_PATH=${1:-./llama_mini_example}
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

distributed_args="--num_nodes=1 --num_gpus=1 --master_port 12345"
ds_args="--zero-stage=2 --deepspeed --deepspeed_config /data/tir/projects/tir6/general/mengyan3/tower-llm-training/demo_scripts/ds_config.json"


deepspeed $distributed_args \
    $repo/Megatron-DeepSpeed/tasks/main.py \
    --task "EVAL-HARNESS" \
    --tensor-model-parallel-size $tp \
    --no-pipeline-parallel \
    --num-layers $num_layers \
    --hidden-size $hidden_size \
    --ffn-hidden-size $ffn_hidden_size \
    --num-attention-heads $num_attention_heads \
    --num-key-value-heads $num_kv_heads \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --num-key-value-heads $num_kv_heads \
    --use-flash-attn-v2 \
    --seq-length $seq_length \
    --max-position-embeddings $seq_length \
    --fp16 \
    --valid-data $eval_set \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $external_tokenizer \
    --load $CHECKPOINT_PATH \
    --micro-batch-size ${micro_batch_size} \
    --log-interval 10 \
    --no-load-optim \
    --no-load-rng \
    --distributed-timeout-minutes 60 \
    $ds_args 
