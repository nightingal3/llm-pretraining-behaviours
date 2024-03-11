MODELPATH=$1

lm_eval --model hf \
    --model_args pretrained=$MODELPATH \
    --tasks lambada_openai,lambada_standard,babi,gsm8k_cot_self_consistency,gsm8k_cot,fld,math_word_problems,asdiv,logiqa2,scrolls \
    --device cuda:0 \
    --batch_size auto
    
#note: `math_word_problems` encompasses Minerva Math, MathQA, GSM8k (no CoT) and a few others.

#I couldn't find ProScript in the list of potential tasks; we may need to add it with our other new tasks.