task DumpHFDataset 
    >  dataset_json=dataset.json
    :: dataset_name=@
    :: repo=@
{
    python $repo/scripts/dump_hf_dataset.py \
        --dataset_name $dataset_name \
        --output $dataset_json
}

task PreprocessDataset
    < dataset_json=@DumpHFDataset
    > dataset_bin=data_bin
    :: repo=@
    :: tokenizer=@
    :: cpu_workers=@
{
    set -euo pipefail
    mkdir -p $dataset_bin
    python $repo/Megatron-DeepSpeed/tools/preprocess_data.py \
        --input $dataset_json \
        --output-prefix $dataset_bin/data \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $tokenizer \
        --append-eod \
        --workers $cpu_workers 
}

task GetDeepSpeedConfig
    < dataset_bin=@PreprocessDataset
    > ds_config=deepspeed.json
    :: repo=@
    :: tokenizer=@
    :: gpus=@
    :: zero_stage=@
    :: master_addr=@
    :: master_port=@
    :: batch_size=@
{
    micro_batch_size=$(($batch_size / $gpus))
    echo -n '{
        "train_batch_size" : '$batch_size',
        "train_micro_batch_size_per_gpu": '$micro_batch_size',
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": '$zero_stage'
        },
        "bf16": {
            "enabled": true
        }
    }' | jq . > $ds_config
}

task TuneDeepSpeedConfig
    < dataset_bin=@PreprocessDataset
    > ds_config=deepspeed.json
    :: repo=@
    :: tune_steps=10
    :: micro_batch_size=2
    :: tokenizer=@
    :: gpus=@
    :: tp=@
    :: pp=@
    :: zero_stage=@
    :: master_addr=@
    :: master_port=@
    :: hidden_size=@
    :: ffn_hidden_size=@
    :: num_layers=@
    :: num_attention_heads=@
    :: seq_length=@
    :: num_kv_heads=@
    :: batch_size=@
    :: lr=@
    :: min_lr=@
    :: lr_warmup_steps=4
    :: weight_decay=@
    :: grad_clip=@
    :: save_interval=@
    :: eval_interval=@
{
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    # NOTE: this is still not working
    set -euo pipefail
    echo -n '{
        "train_micro_batch_size_per_gpu": "auto",
        "fp16": {
            "enabled": true
        },
        "autotuning": {
            "enabled": true,
            "fast": false,
            "zero_optimization": {
                "stage": [0,1,2]
            },
            "arg_mappings": {
                "train_micro_batch_size_per_gpu": "--micro-batch-size"
            }
        }
    }' | jq . > ds_config.json

    ds_args="--deepspeed_config ds_config.json --deepspeed"
    distributed_args="--num_nodes 1 --num_gpus $gpus"

    # echo "0.0.0.0 slots=$gpus" > hostfile
    mkdir -p tmp_model

    deepspeed --autotuning tune $distributed_args \
        $repo/Megatron-DeepSpeed/pretrain_gpt.py \
        --tensor-model-parallel-size $tp \
        --no-pipeline-parallel \
        --num-layers $num_layers \
        --hidden-size $hidden_size \
        --ffn-hidden-size $ffn_hidden_size \
        --num-attention-heads $num_attention_heads \
        --seq-length $seq_length \
        --max-position-embeddings $seq_length \
        --train-iters $tune_steps \
        --data-path 1 $dataset_bin/data_text_document \
        --data-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $tokenizer \
        --split 949,50,1 \
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
        --num-key-value-heads $num_kv_heads \
        --use-flash-attn \
        $ds_args

        cp autotuning_results/profile_model_info/ds_config.json $ds_config
}

task Train
    < dataset_bin=@PreprocessDataset
    < ds_config=@TuneDeepSpeedConfig
    > model_dir=checkpoints
    :: repo=@
    :: external_model_dir=@
    :: external_restart=@
    :: tokenizer=@
    :: gpus=@
    :: tp=@
    :: pp=@
    :: zero_stage=@
    :: master_addr=@
    :: master_port=@
    :: hidden_size=@
    :: ffn_hidden_size=@
    :: num_layers=@
    :: num_attention_heads=@
    :: seq_length=@
    :: num_kv_heads=@
    :: train_steps=@
    :: batch_size=@
    :: lr=@
    :: min_lr=@
    :: lr_warmup_steps=@
    :: weight_decay=@
    :: grad_clip=@
    :: save_interval=@
    :: eval_interval=@
{
    cat $ds_config

    #ds_args="--deepspeed"
    #ds_args="--deepspeed_config=$ds_config ${ds_args}"
    #ds_args="--zero-stage=$zero_stage ${ds_args}"
    #ds_args="--deepspeed-activation-checkpointing ${ds_args}"

    #distributed_args="--nproc_per_node=$gpus"
    #distributed_args="--nnodes=1 ${distributed_args}"

    ds_args="--deepspeed_config=$ds_config --deepspeed"
    ds_args="--zero-stage=$zero_stage ${ds_args}"
    ds_args="--deepspeed-activation-checkpointing ${ds_args}"
    distributed_args="--num_nodes=1 --num_gpus=$gpus"    

    # if `save_external` is set, symlink it to the `model_dir`
    # and copy the config file to the `model_dir`
    if [ "$external_model_dir" != "" ]; then
        if [ "$external_restart" == false ]; then
            rm -rf $external_model_dir
        fi
        mkdir -p $external_model_dir
        ln -s $external_model_dir $model_dir
        cp $ds_config $model_dir
    fi

    micro_batch_size=$(($batch_size / $gpus))
    #torchrun $distributed_args \
    deepspeed $distributed_args \
       $repo/Megatron-DeepSpeed/pretrain_gpt.py \
       --tensor-model-parallel-size $tp \
       --no-pipeline-parallel \
       --num-layers $num_layers \
       --hidden-size $hidden_size \
       --ffn-hidden-size $ffn_hidden_size \
       --num-attention-heads $num_attention_heads \
       --micro-batch-size $micro_batch_size \
       --global-batch-size $batch_size \
       --seq-length $seq_length \
       --max-position-embeddings $seq_length \
       --train-iters $train_steps \
       --save $model_dir \
       --load $model_dir \
       --data-path 1 $dataset_bin/data_text_document \
       --data-impl mmap \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path $tokenizer \
       --split 949,50,1 \
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
       --num-key-value-heads $num_kv_heads \
       --use-flash-attn \
       $ds_args 
}

plan TrainLLM {
    reach Train
}