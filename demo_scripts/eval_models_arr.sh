#!/bin/bash
#SBATCH --job-name=standard_eval_redo_boogaloo
#SBATCH --output=standard_eval_redo_boogaloo_%A-%a_BATCH1.out
#SBATCH --mem=50G
#SBATCH --array=4-30%8
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END

config=/data/tir/projects/tir6/general/mengyan3/tower-llm-training/demo_scripts/slurm_jobs/eval_models_commands.csv

model_name=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $2}' $config)

echo "=== JOB INFO ==="
echo "Model name: ${model_name}"
echo "=== END JOB INFO ==="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env

python metadata/collect_integrated_model_scores.py ${model_name} --overwrite --output_dir metadata/model_scores  --tasks_file metadata/eval_harness_tasks.txt