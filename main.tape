task DumpHFDataset 
    >  dataset_json=dataset.json
    :: dataset_name=@
    :: filter=@
    :: n_tokens=@
    :: repo=@
{
    python $repo/scripts/dump_hf_dataset.py \
        --dataset_name $dataset_name \
        --output $dataset_json \
        --filter $filter \
        --n_tokens $n_tokens
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

# WARNING: this is still not working! Please use (DSAutotuneConfig: false)
# and manually tune the config file
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
    :: model_config=@
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

    # Read model config and parse to variables
    model_config_f="${repo}/configs/${model_config}.yml"
    hidden_size=$(yq '.hidden_size' $model_config_f)
    ffn_hidden_size=$(yq '.ffn_hidden_size' $model_config_f)
    num_layers=$(yq '.num_layers' $model_config_f)
    num_attention_heads=$(yq '.num_attention_heads' $model_config_f)
    seq_length=$(yq '.seq_length' $model_config_f)
    num_kv_heads=$(yq '.num_kv_heads' $model_config_f)

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
    < ds_config=(
        UseDeepSpeed:
            false=/dev/null
            true=(
                DSAutotuneConfig:
                    false=$ds_config@GetDeepSpeedConfig
                    true=$ds_config@TuneDeepSpeedConfig
                )
        )
    > model_dir=checkpoints
    :: repo=@
    :: use_deepseed=(UseDeepSpeed: false true)
    :: external_model_dir=@
    :: external_resume=@
    :: tokenizer=@
    :: gpus=@
    :: tp=@
    :: pp=@
    :: zero_stage=@
    :: master_addr=@
    :: master_port=@
    :: rdzv_port=@
    :: model_config=@
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
    # Read model config and parse to variables
    model_config_f="${repo}/configs/models/${model_config}.yml"
    hidden_size=$(yq '.hidden_size' $model_config_f)
    ffn_hidden_size=$(yq '.ffn_hidden_size' $model_config_f)
    num_layers=$(yq '.num_layers' $model_config_f)
    num_attention_heads=$(yq '.num_attention_heads' $model_config_f)
    seq_length=$(yq '.seq_length' $model_config_f)
    num_kv_heads=$(yq '.num_kv_heads' $model_config_f)

    if [ "$use_deepseed" == true ]; then
        # read main optimization parameters from the DeepSpeed config file
        # (in case these were automatically tuned)
        zero_stage=$(jq -r '.zero_optimization.stage' $ds_config)
        micro_batch_size=$(jq -r '.train_micro_batch_size_per_gpu' $ds_config)
        
        ds_args="--deepspeed"
        ds_args="--deepspeed_config=$ds_config ${ds_args}"
        ds_args="--zero-stage=$zero_stage ${ds_args}"
        ds_args="--deepspeed-activation-checkpointing ${ds_args}"
        distributed_args="--num_nodes=1 --num_gpus=$gpus"
        launcher="deepspeed"

        echo "Using DeepSpeed with zero stage $zero_stage and micro batch size $micro_batch_size"
    else
        ds_args=""
        distributed_args="--nnodes=1 --nproc_per_node=$gpus --rdzv-endpoint localhost:$rdzv_port "
        launcher="torchrun"
        export  CUDA_DEVICE_MAX_CONNECTIONS=1
    fi

    # if `save_external` is set, symlink it to the `model_dir`
    # and copy the config file to the `model_dir`
    if [ "$external_model_dir" != "" ]; then
        if [ "$external_resume" == false ]; then
            rm -rf $external_model_dir
        fi
        mkdir -p $external_model_dir
        ln -s $external_model_dir $model_dir
        if [ "$use_deepseed" == true ]; then
            cp $ds_config $model_dir
        fi
    fi

    micro_batch_size=$(($batch_size / $gpus))
    $launcher $distributed_args \
       $repo/Megatron-DeepSpeed/pretrain_gpt.py \
       --tensor-model-parallel-size $tp \
       --no-pipeline-parallel \
       --num-layers $num_layers \
       --hidden-size $hidden_size \
       --ffn-hidden-size $ffn_hidden_size \
       --num-attention-heads $num_attention_heads \
       --micro-batch-size $micro_batch_size \
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
       --distributed-timeout-minutes 60 \
       $ds_args 
}

task ConvertHF
    < trained_model=$model_dir@Train
    > hf_model_dir
    :: repo=@
    :: tokenizer=@
    :: model_config=@
{
    # read last checkpoint
    last_checkpoint=$(cat $trained_model/latest_checkpointed_iteration.txt)
    # get checkpoint dir, with fixed number of digits (iter_0000100)
    checkpoint_n=$(printf "iter_%08d" $last_checkpoint)
    checkpoint=$trained_model/$checkpoint_n/mp_rank_00/model_optim_rng.pt

    # Convert to HF model using convert_megatron_gpt2_checkpoint.py
    python $repo/transformers/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py \
        $checkpoint
}


task Eval
    < train_model=$model_checkpoint
    < ds_config=(
        UseDeepSpeed:
            true=(
                DSAutotuneConfig:
                    false=$ds_config@GetDeepSpeedConfig
                    true=$ds_config@TuneDeepSpeedConfig
                )
            false=/dev/null
        )
    > eval_results 
    :: repo=@
    :: use_deepseed=(UseDeepSpeed: true false)
    :: gpus=@
    :: tp=@
    :: pp=@
    :: zero_stage=@
    :: master_addr=@
    :: master_port=@
    :: eval_set=@
    :: eval_metric=@
    :: tokenizer=@
    :: hidden_size=@
    :: ffn_hidden_size=@
    :: num_layers=@
    :: num_attention_heads=@
    :: num_kv_heads=@
    :: seq_length=@
    :: batch_size=@

{
    cat $ds_config

    if [ "$use_deepseed" == true ]; then
        # read main optimization parameters from the DeepSpeed config file
        # (in case these were automatically tuned)
        zero_stage=$(jq -r '.zero_optimization.stage' $ds_config)
        micro_batch_size=$(jq -r '.train_micro_batch_size_per_gpu' $ds_config)
        
        ds_args="--deepspeed"
        ds_args="--deepspeed_config=$ds_config ${ds_args}"
        ds_args="--zero-stage=$zero_stage ${ds_args}"
        ds_args="--deepspeed-activation-checkpointing ${ds_args}"
        distributed_args="--num_nodes=1 --num_gpus=$gpus"
        launcher="deepspeed"

        echo "Using DeepSpeed with zero stage $zero_stage and micro batch size $micro_batch_size"
        export  CUDA_DEVICE_MAX_CONNECTIONS=1
    else
        ds_args=""
        distributed_args="--nnodes=1 --nproc_per_node=$gpus"
        launcher="torchrun"
        export  CUDA_DEVICE_MAX_CONNECTIONS=1
    fi

    micro_batch_size=$(($batch_size / $gpus))
    $launcher $distributed_args \
        $repo/Megatron-DeepSpeed/tasks/main.py \
        --task "WIKITEXT103" \
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
        --use-flash-attn \
        --seq-length $seq_length \
        --max-position-embeddings $seq_length \
        --fp16 \
        --valid-data $eval_set \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $tokenizer \
        --load $train_model \
        --micro-batch-size $batch_size \
        --log-interval 10 \
        --no-load-optim \
        --no-load-rng
}

plan TrainLLM {
    reach Train via (UseDeepSpeed: false) * (DSAutotuneConfig: false)
}

plan TrainLLMTest {
    reach Train via (UseDeepSpeed: false) * (DSAutotuneConfig: false) * (Size: small1)
}

plan ScalingAnalysis {
    reach Train via (Size: small1 small2 small3)
}

plan EvalLLM {
    reach Eval via (UseDeepSpeed: false) * (DSAutotuneConfig: false)
}
