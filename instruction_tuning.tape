task ConvertToDeepSpeed
    < base_model_dir=@
    > ds_model_dir=ds_model
    :: tokenizer=@
    :: tp=@
    :: pp=@
    :: repo=@
    :: model_config=@
    :: gpu_ids=@
{
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    
    export PYTHONPATH=$repo/Megatron-DeepSpeed

    # Read model config and parse to variables
    model_config_f="${repo}/configs/models/${model_config}.yml"
    hidden_size=$(yq '.hidden_size' $model_config_f)
    ffn_hidden_size=$(yq '.ffn_hidden_size' $model_config_f)
    num_layers=$(yq '.num_layers' $model_config_f)
    num_attention_heads=$(yq '.num_attention_heads' $model_config_f)
    seq_length=$(yq '.seq_length' $model_config_f)
    num_kv_heads=$(yq '.num_kv_heads' $model_config_f)

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
    distributed_args="--include localhost:$gpu_ids --master_port 6000"

    mkdir -p $ds_model_dir

    deepspeed $distributed_args $repo/scripts/hf_llama_to_deepspeed.py \
        --hf-model $base_model_dir \
        --tensor-model-parallel-size $tp \
        --pipeline-model-parallel-size $pp \
        --no-pipeline-parallel \
        --num-layers $num_layers  \
        --hidden-size $hidden_size  \
        --max-position-embeddings $seq_length \
        --seq-length $seq_length  \
        --num-attention-heads $num_attention_heads  \
        --ffn-hidden-size $ffn_hidden_size  \
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
        --tokenizer-name-or-path $tokenizer  \
        --save $ds_model_dir \
        --save-interval 1 \
        --no-save-optim \
        --no-save-rng \
        --no-load-optim \
        --no-load-rng \
        --lr 0.00001 \
        $ds_args

    rm $ds_model_dir/global_step0/*optim_states.pt
}

plan ConvertToDeepSpeed {
    reach ConvertToDeepSpeed via (BaseModel: croissant-small4)
}