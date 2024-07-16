#!/bin/bash 
MODEL_LIST=(
    # List models here
)

for MODEL in ${MODEL_LIST[@]};do
    for TASK in reasoning-gen reasoning-mcq reasoning-ppl ;do
        accelerate launch --no_python lm-eval \
            --model hf \
            --model_args "pretrained=${MODEL}" \
            --tasks ${TASK} \
            --batch_size 4 \
            --include_path "eval_task_groups/" \
            --output "metadata/scores/${TASK}/" \
            --log_samples
    done
done
