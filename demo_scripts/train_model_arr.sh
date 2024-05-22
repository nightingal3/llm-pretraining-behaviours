#!/bin/bash
#SBATCH --job-name=train_model_%A_%a
#SBATCH --output=train_model_%A_%a.out
#SBATCH --mem=30G
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=1-00:00:00
#SBATCH --partition=babel-shared-long
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --array=1-10%3

set -a 
source ./demo_scripts/configs/.env
set +a

source ${MINICONDA_PATH}
conda activate ${TOWERLLM_ENV_NAME}

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Usage: sbatch demo_scripts/train_model.sh <checkpoint_path> <model config (see ./configs)> <dataset_bin> <external_tokenizer>
# to use the wandb logger: --wandb_logger --wandb_entity <your username> --wandb_id <some id> --wandb_api_key <your api key>
set -euo pipefail

EXP_CONFIG=$1

IFS=',' read -r task_id CHECKPOINT_PATH model_config dataset_bin external_tokenizer TOTAL_TRAIN_TOKENS <<< $(sed "${SLURM_ARRAY_TASK_ID}q;d" $CSV_FILE_PATH)

repo=${BASE_REPO}
data_path=${dataset_bin}

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
train_epochs=$(yq '.training.train_epochs' $model_config)

tp=1 # don't use this
micro_batch_size=$(yq '.training.micro_batch_size' $model_config)
seed=$(yq '.training.seed' $model_config)

NUM_GPUS=$(nvidia-smi -L | wc -l)

# build train step arguments - disregard steps if epochs are specified, else use steps, else fall back to 100k steps
if [ ! -z "$train_epochs" ]; then
   echo "train_epochs has been passed. If you specified the exact number of steps, this will be IGNORED in favour of epochs"
   # TODO: there's an argument --train-data-exact-num-epochs but it seems to be broken
   # manually calculate the number of steps for now: total tokens (rough) / (seq_len x batch size)
      echo "NOTE on epochs: This isn't implemented yet, using rough number for 1 epoch over 100B tokens..."
      total_seqs_rough=$((TOTAL_TRAIN_TOKENS / seq_length))
      batch_size=$((micro_batch_size * NUM_GPUS))
      rough_steps=$((total_seqs_rough / batch_size))

      train_steps_arg="--train-iters $rough_steps"
elif [ ! -z "$train_steps" ]; then
      train_steps_arg="--train-iters $train_steps"
   else
      train_steps_arg="--train-iters 100000"
fi
echo "train_steps_arg: $train_steps_arg"

# if no run id specified, then use unix timestamp as unique id
if [ -z "$WANDB_ID" ]; then
    WANDB_ID=$(date +%s)
fi

distributed_args="--num_nodes=1 --num_gpus=${NUM_GPUS} --master_port 12345"
ds_args="--zero-stage=2 --deepspeed --deepspeed_config ${repo}/demo_scripts/ds_config.json"
deepspeed $distributed_args \
       $repo/Megatron-DeepSpeed/pretrain_gpt.py \
       $train_steps_arg \
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
       --wandb_entity $WANDB_USER \
      --wandb_id $WANDB_ID \
      --wandb_api_key $WANDB_API_KEY \
      --shuffle_docs_before_split \
       $ds_args 
