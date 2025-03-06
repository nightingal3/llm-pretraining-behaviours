#!/bin/bash 
#SBATCH 
MODEL_LIST=(
    /data/tir/projects/tir5/users/mengyan3/dolma_checkpts_hf_final/llama2_220M_nl_40_code_60
)

# you can pass in the models to eval as a comma-separated str. If no arg passed, default to the model list above
if [ -n "$1" ]; then
    passed_lst=$1
    IFS="," read -r -a MODEL_LIST <<< "$passed_lst"
fi

for MODEL in ${MODEL_LIST[@]};do
    for TASK in reasoning-gen reasoning-mcq reasoning-ppl ;do
        accelerate launch --no_python lm-eval \
            --model hf \
            --model_args "pretrained=${MODEL}" \
            --tasks ${TASK} \
            --batch_size 4 \
            --include_path "eval_task_groups/" \
            --output "metadata/model_scores/${TASK}/" \
            --log_samples
    done
done
