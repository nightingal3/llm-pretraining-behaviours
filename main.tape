import "submitters/scslurm.tape"

task DumpHFDataset 
    > dataset_json=dataset.json
    :: dataset_name=@
    :: dataset_path=@
    :: dataset_dirs=@
    :: dataset_stream=@
    :: filter=@
    :: percentile=@
    :: n_tokens=@
    :: n_tokens_test=@
    :: repo=@
    :: hf_dataset=@
    :: bilingual=@
    :: code=@
    :: .submitter=@
    :: .account=$dump_account
    :: .partition=$dump_partition
    :: .time=$dump_time 
    :: .cpus=$dump_cpus
    :: .mem=$dump_mem
{
    python $repo/scripts/dump_hf_dataset.py \
        --dataset_name $dataset_name \
        --output $dataset_json \
        $([ ! -z "$dataset_dirs" ] && echo "--dataset_dirs $dataset_dirs" || echo "") \
        $([ ! -z "$dataset_path" ] && echo "--dataset_path $dataset_path" || echo "") \
        $([ "$filter" == true ] && echo "--filter" || echo "") \
        $([ ! -z "$percentile" ] && echo "--percentile $percentile" || echo "") \
        $([ ! -z "$n_tokens" ] && echo "--n_tokens $n_tokens" || echo "") \
        $([ "$dataset_stream" == true ] && echo "--stream" || echo "") \
        $([ "$hf_dataset" == true ] && echo "--hf_dataset" || echo "") \
        $([ "$bilingual" == true ] && echo "--bilingual" || echo "") \
        $([ "$code" == true ] && echo "--code" || echo "") \
        $([ ! -z "$n_tokens_test" ] && echo "--n_tokens_test $n_tokens_test" || echo "") \
}

# Warning: This is working, but isnt passed to subsequent tasks
task TrainTokenizer
    < dataset_json=@DumpHFDataset[Dataset:*]
    > tokenizer
    :: repo=@
    :: pre_tokenizer=$pre_tokenizer[Dataset:*]
    :: vocab_size=@
    :: words_per_source=$tokenizer_words_per_source[Dataset:*]
    :: extra_tokens=@
    :: .submitter=@
    :: .account=$traintok_account
    :: .partition=$traintok_partition
    :: .time=$traintok_time
    :: .cpus=$traintok_cpus
    :: .mem=$traintok_mem
{
    mkdir tok_corpus
    mkdir tokenizer
    echo "Preparing tokenizer corpus..."
    python $repo/scripts/prepare_tokenizer_corpus.py \
        --data_paths $dataset_json \
        --words_per_source $words_per_source \
        --output tok_corpus/data \
        --pre_tokenizer 
    echo "Training tokenizer..."
    python $repo/scripts/train_tokenizer.py \
        --data_path $tokenizer_corpus \
        --vocab_size $vocab_size
        --extra_tokens $extra_tokens \
        --output $tokenizer
    echo "Analyzing tokenizer..."
    python $repo/scripts/analyze_tokenizer.py \
        --tokenizer_dir $tokenizer \
        --eval_sets $eval_sets \
}

task PreprocessDataset
    < dataset_json=@DumpHFDataset
    > dataset_bin=data_bin
    :: repo=@
    :: external_tokenizer=@
    :: cpu_workers=20
    :: .submitter=@
    :: .account=$preproc_account
    :: .partition=$preproc_partition
    :: .time=$preproc_time
    :: .cpus=$preproc_cpus
    :: .mem=$preproc_mem
{
    set -euo pipefail
    mkdir -p $dataset_bin
    python $repo/Megatron-DeepSpeed/tools/preprocess_data.py \
        --input $dataset_json \
        --output-prefix $dataset_bin/data \
        --dataset-impl mmap \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path $external_tokenizer \
        --append-eod \
        --workers $cpu_workers 
}

# WARNING: has not been tested recently
task GetDeepSpeedConfig
    < dataset_bin=@PreprocessDataset
    > ds_config=deepspeed.json
    :: repo=@
    :: gpus=@
    :: zero_stage=@
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
    :: external_tokenizer=@
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
        --tokenizer-name-or-path $external_tokenizer \
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


# NOTE: this task is needed to allow proper grafting of the datamix branches
# since it was not working when grafting directly in the Train task
task MakeDataMix 
    < dataset_bin=@PreprocessDataset
    > datamix_file
    :: datamix_weights=@
{
    # simply write datamix weight and path in dataset_bin to a file, separated by a space
    echo "$datamix_weights $dataset_bin/data_text_document" > $datamix_file
}

task Train
    < datamix_file=@MakeDataMix[Dataset:*]
    < ds_config=(
        UseDeepSpeed:
            true=(
                DSAutotuneConfig:
                    false=$ds_config@GetDeepSpeedConfig
                    true=$ds_config@TuneDeepSpeedConfig
                )
            false=/dev/null
        )
    > model_dir=checkpoints
    :: .submitter=@
    :: .C=$train_C
    :: .account=$train_account
    :: .partition=$train_partition
    :: .time=$train_time
    :: .cpus=$train_cpus
    :: .gres=$train_gres
    :: .mem=$train_mem
    :: .restart_on_timeout=true
    :: repo=@
    :: use_deepseed=(UseDeepSpeed: true false)
    :: external_model_dir=@
    :: external_resume=@
    :: external_tensorboard=@
    :: external_tokenizer=@
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
    :: grad_accum_steps=@
    :: lr=@
    :: min_lr=@
    :: lr_warmup_steps=@
    :: weight_decay=@
    :: grad_clip=@
    :: save_interval=@
    :: eval_interval=@
    :: seed=@
{
    data_path=""
    for f in $datamix_file; do
        # read file
        data_path="$data_path `cat $f`"
    done
    echo "Running with data_path=$data_path"

    # check if number folders passed in dataset_bin (space separated) is equal to 
    # the number of datamix weights passed (space separated)
    # n_weights=$(echo $datamix_weights | tr ' ' '\n' | wc -l)
    # n_folders=$(echo $dataset_bin | tr ' ' '\n' | wc -l)
    # if [ "$n_weights" != "$n_folders" ]; then
    #     echo "ERROR: number of datamix weights ($n_weights) is not equal to number of dataset folders ($n_folders)"
    #     exit 1
    # fi

    # # make dataset_path for mix for megatron
    # weights=($datamix_weights)
    # folders=($dataset_bin)
    # data_path=""
    # # iterate over weights and folders
    # i=0
    # while [ $i -lt ${#weights[@]} ]
    # do
    #     data_path+="${weights[$i]} ${folders[$i]}/data_text_document "
    #     i=$((i+1))
    # done

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
        distributed_args="--num_nodes=1 --num_gpus=$gpus --master_port $rdzv_port"
        launcher="deepspeed"

        echo "Using DeepSpeed with zero stage $zero_stage and micro batch size $micro_batch_size"
    else
        ds_args=""
        distributed_args="--nnodes=1 --nproc_per_node=$gpus --rdzv-endpoint localhost:$rdzv_port"
        launcher="torchrun"
        export  CUDA_DEVICE_MAX_CONNECTIONS=1
    fi
    distributed_args="$distributed_args"


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

    if [ "$external_tensorboard" != "" ]; then
        mkdir -p $external_tensorboard
        ln -s $external_tensorboard tensorboard
    else
        mkdir -p tensorboard
    fi

    tensorboard_args="--tensorboard-dir tensorboard/ --log-validation-ppl-to-tensorboard"
    micro_batch_size=$(($batch_size / ($gpus * $grad_accum_steps)))
    $launcher $distributed_args \
       $repo/Megatron-DeepSpeed/pretrain_gpt.py \
       --tensor-model-parallel-size $tp \
       --no-pipeline-parallel \
       --num-layers $num_layers \
       --hidden-size $hidden_size \
       --ffn-hidden-size $ffn_hidden_size \
       --num-attention-heads $num_attention_heads \
       --num-key-value-heads $num_kv_heads \
       --micro-batch-size $micro_batch_size \
       --global-batch-size $batch_size \
       --seq-length $seq_length \
       --max-position-embeddings $seq_length \
       --train-iters $train_steps \
       --save $model_dir \
       --load $model_dir \
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
       $tensorboard_args \
       $ds_args 
}


task Eval
    < trained_model=(
        UseExternal:
            false=$model_dir@Train
            true=$external_model_dir
        )
    < eval_set=@
    > eval_results 
    :: .submitter=@ 
    :: .C=$eval_C
    :: .account=$eval_account
    :: .partition=$eval_partition
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres
    :: .mem=$eval_mem
    :: repo=@
    :: use_deepseed=(UseDeepSpeed: true false)
    :: gpus=1
    :: batch_size=32
    :: tp=@
    :: pp=@
    :: master_addr=@
    :: master_port=@
    :: eval_metric=@
    :: eval_iteration=@
    :: external_tokenizer=@
    :: model_config=@
    :: rdzv_port=@
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
        ds_args="--deepspeed"
        echo -n '{
            "train_batch_size" : '$batch_size',
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": 0
            },
            "bf16": {
                "enabled": true
            }
        }' | jq . > ds_config.json
        ds_args="--zero-stage=0 ${ds_args} --deepspeed_config ds_config.json"
        distributed_args="--num_nodes=1 --num_gpus=$gpus --master_port $rdzv_port"
        launcher="deepspeed"

        echo "Using DeepSpeed for evaluation..."
    else
        ds_args=""
        distributed_args="--nnodes=1 --nproc_per_node=$gpus --rdzv-endpoint localhost:$rdzv_port "
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
        --tokenizer-name-or-path $external_tokenizer \
        --load $trained_model \
        $( [ "$eval_iteration" != "" ] && echo "--load-iteration $eval_iteration" || echo "") \
        --micro-batch-size $batch_size \
        --log-interval 10 \
        --no-load-optim \
        --no-load-rng \
        --distributed-timeout-minutes 60 \
        $ds_args 

}

task ConvertHF
    < trained_model=(
        UseExternal:
            false=$model_dir@Train
            true=$external_model_dir
        )
    > hf_model_dir=hf_model
    :: .submitter=@
    :: .C=$convert_C
    :: .account=$convert_account
    :: .partition=$convert_partition
    :: .time=$convert_time 
    :: .cpus=$convert_cpus
    :: .gres=$convert_gres
    :: .mem=$convert_mem
    :: repo=@
    :: external_tokenizer=@
    :: model_config=@
{
    export PYTHONPATH=$repo/Megatron-DeepSpeed
    
    mkdir -p $hf_model_dir

    # read last checkpoint
    #last_checkpoint=$(cat $trained_model/latest_checkpointed_iteration.txt)
    # get checkpoint dir, with fixed number of digits (iter_0000100)
    #checkpoint_dir="$trained_model/global_step${last_checkpoint}"

    # Convert from deepspeed to pytorch
    python $trained_model/zero_to_fp32.py $trained_model model.bin

    # Convert to HF model using convert_megatron_gpt2_checkpoint.py
    python $repo/scripts/convert_to_hf_model.py \
        model.bin $hf_model_dir \
        --config-yml=$repo/configs/models/$model_config.yml \
        --tokenizer-name=$external_tokenizer 

    # clean up
    rm model.bin
}

plan TrainTokenizer {
    reach TrainTokenizer
}

plan Preprocess {
    reach DumpHFDataset via (Dataset: *)
}

plan ScalingAnalysis {
    reach Train via (Size: small1 small2 small3 small4) * (DataMix: *)
}

plan TrainLLM {
    reach Eval via (Size: base)
}

plan MidTrainingEval {
    reach Eval via (Size: small1 small2 small3 small4) * (DataMix: *) * (UseExternal: true) * (EvalIteration: 340k) * (EvalSet: *)
}

plan MidTrainingConvert {
    reach ConvertHF via (Size: small4) * (DataMix: *) * (UseExternal: true) 
}

plan ConvertHF {
    reach ConvertHF via (Size: small1)
}
