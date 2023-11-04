#!/bin/bash

set -eo pipefail
set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

TOKENIZER="manu/tok-fr-en-code"
HF_MODEL_PATH=/mnt/data/duarte/tower-llm-training/croissllm-models-v2/small4_round_trip/hf_model
DS_MODEL_PATH=/mnt/data/duarte/tower-llm-training/croissllm-models-v2/small4_round_trip
MODEL_CONFIG_FILE=/home/duarte/tower-llm-training/configs/models/llama2_440m.yml

HIDDEN_SIZE=$(yq '.hidden_size' $MODEL_CONFIG_FILE)
FFN_HIDDEN_SIZE=$(yq '.ffn_hidden_size' $MODEL_CONFIG_FILE)
NUM_LAYERS=$(yq '.num_layers' $MODEL_CONFIG_FILE)
NUM_ATTENTION_HEADS=$(yq '.num_attention_heads' $MODEL_CONFIG_FILE)
SEQ_LENGTH=$(yq '.seq_length' $MODEL_CONFIG_FILE)
NUM_KV_HEADS=$(yq '.num_kv_heads' $MODEL_CONFIG_FILE)

ds_args="--deepspeed"
echo -n '{
    "train_batch_size" : 2,
    "steps_per_print": 1,
    "zero_optimization": {
        "stage": 0
    },
    "bf16": {
        "enabled": true
    }
}' | jq . > ds_config.json
ds_args="--zero-stage=0 ${ds_args} --deepspeed_config ds_config.json"
distributed_args="--include localhost:1,4 --master_port 6000"

echo "Using DeepSpeed for generation..."
export PYTHONPATH=$DIR/Megatron-DeepSpeed
export CUDA_DEVICE_MAX_CONNECTIONS=1
deepspeed $distributed_args $DIR/scripts/deepspeed_to_hf_llama.py \
    --output-dir $HF_MODEL_PATH \
    --tensor-model-parallel-size 2  \
    --pipeline-model-parallel-size 1  \
    --no-pipeline-parallel \
    --num-layers $NUM_LAYERS  \
    --hidden-size $HIDDEN_SIZE  \
    --max-position-embeddings $SEQ_LENGTH \
    --seq-length $SEQ_LENGTH  \
    --num-attention-heads $NUM_ATTENTION_HEADS  \
    --ffn-hidden-size $FFN_HIDDEN_SIZE  \
    --no-query-key-layer-scaling \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --bf16 \
    --micro-batch-size 1 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER}  \
    --load $DS_MODEL_PATH \
    --no-save-optim \
    --no-save-rng \
    --no-load-optim \
    --no-load-rng \
    $ds_args