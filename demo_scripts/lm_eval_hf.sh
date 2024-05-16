#!/bin/bash

eval_tasks="arithmetic,asdiv,gsm8k,gsm8k_cot,gsm8k_cot_zeroshot,mathqa,minerva_math,minerva_math_algebra,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_intermediate_algebra,minerva_math_num_theory,minerva_math_prealgebra,minerva_math_precalc,logiqa2,fld,scrolls,lambada,anli,xnli"

lm_eval --model hf \
    --model_args pretrained=/data/tir/projects/tir6/general/mengyan3/tower-llm-training/llama-220m-hf,dtype="float" \
    --tasks asdiv \
    --device cuda:0 \
    --batch_size auto:4 