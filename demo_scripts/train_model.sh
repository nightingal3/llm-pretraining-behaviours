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

# Usage: sbatch demo_scripts/train_model.sh <checkpoint_path> <model config (see ./configs)> <dataset_bin> <external_tokenizer>
# to use the wandb logger: --wandb_logger --wandb_entity <your username> --wandb_id <some id> --wandb_api_key <your api key>
set -euo pipefail

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
   echo "Usage: sbatch demo_scripts/train_model.sh [checkpoint_path] [dataset_bin] [external_tokenizer]"
   exit 0
fi

CHECKPOINT_PATH=${1:-./llama_mini_try}
model_config=${2:-./demo_scripts/configs/Llama2_220M.yaml}
dataset_bin=${3:-wiki-en-simple_200000000-bin/data_text_document}
external_tokenizer=${4:-meta-llama/Llama-2-7b-hf}
repo=/data/tir/projects/tir6/general/mengyan3/tower-llm-training
data_path="${repo}/${dataset_bin}"

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
micro_batch_size=$(yq '.training.micro_batch_size' $model_config)
seed=$(yq '.training.seed' $model_config)

NUM_GPUS=$(nvidia-smi -L | wc -l)

distributed_args="--num_nodes=1 --num_gpus=${NUM_GPUS} --master_port 12345"
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
       --use-flash-attn-v2 \
       --distributed-timeout-minutes 60 \
       --seed $seed \
       --wandb_logger \
       --wandb_entity nightingal3 \
      --wandb_id llama_mini_460M \
      --wandb_api_key $WANDB_API_KEY \
       $ds_args 


    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \ # don't need these for pretraining