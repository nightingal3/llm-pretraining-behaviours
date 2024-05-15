declare -a model_list=(
    AbacusResearch/Jallabi-34B
    allenai/OLMo-7B
    EleutherAI/pythia-6.9b
    huggyllama/llama-7b
    jb723/cross_lingual_epoch2
    jb723/LLaMA2-en-ko-7B-model
    jisukim8873/falcon-7B-case-0
    jisukim8873/falcon-7B-case-1
    jisukim8873/falcon-7B-case-2
    jisukim8873/falcon-7B-case-3
    jisukim8873/falcon-7B-case-4
    jisukim8873/falcon-7B-case-5
    jisukim8873/falcon-7B-case-6
    jisukim8873/falcon-7B-case-8
    jisukim8873/falcon-7B-case-c
    kevin009/babyllama-v0.6
    kevin009/flyingllama-v2
    meta-llama/Llama-2-70b-chat-hf
    meta-llama/Llama-2-70b-hf
    meta-llama/Llama-2-7b
    mistralai/Mistral-7B-v0.1
    mistralai/Mixtral-8x7B-v0.1
    Monero/WizardLM-13b-OpenAssistant-Uncensored
    openai-community/gpt2
    openlm-research/open_llama_7b
    playdev7/theseed-v0.3
    Qwen/Qwen-7B
    tiiuae/falcon-7b
)

for MODEL in "${model_list[@]}"; do
    for TASK in gen mcq ppl; do
        echo "Evaluating ${MODEL} on reasoning-${TASK}"
        accelerate launch --no_python lm-eval \
            --task "reasoning-${TASK}" \
            --model_args "pretrained=${MODEL}" \
            --batch_size 4 \
            --log_samples \
            --include_path eval_task_groups/ \
            --output "metadata/scores/${MODEL}/${TASK}/" \
            --trust_remote_code
    done
done