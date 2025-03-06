#!/bin/bash
#SBATCH --job-name=new_eval
#SBATCH --output=new_eval%A_%a.out
#SBATCH --mem=50G
#SBATCH --array=1-3%8
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=long
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END

config=/data/tir/projects/tir6/general/mengyan3/tower-llm-training/demo_scripts/evals/mini_llama_models.csv

model_name=$(awk -F, -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $2}' $config)

echo "=== JOB INFO ==="
echo "Model name: ${model_name}"
echo "=== END JOB INFO ==="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

for TASK in reasoning-gen reasoning-mcq reasoning-ppl reasoning-all-orig ;do
    accelerate launch --no_python lm-eval \
        --model hf \
        --model_args "pretrained=${model_name}" \
        --tasks ${TASK} \
        --batch_size 4 \
        --include_path "eval_task_groups/" \
        --output "metadata/model_scores/${TASK}/" \
        --log_samples
done