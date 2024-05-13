#!/bin/bash
#SBATCH --job-name=convert_to_hf_try
#SBATCH --output=convert_to_hf_try_%A_%a.out
#SBATCH --mem=30G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1:00:00
#SBATCH --array=1-3%3
#SBATCH --partition=general
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END

config=/data/tir/projects/tir6/general/mengyan3/tower-llm-training/demo_scripts/slurm_jobs/convert_to_hf.csv
input_path=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $2}' $config)
output_path=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $3}' $config)
model_config_path=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $4}' $config)

echo "=== JOB INFO ==="
echo "Starting task $SLURM_ARRAY_TASK_ID"
echo "Running on $HOSTNAME"
echo "Input path: $input_path"
echo "Output path: $output_path"
echo "Model config path: $model_config_path"
echo "==== END INFO ==="

./demo_scripts/convert_to_hf.sh $input_path $model_config_path NousResearch/Llama-2-7b-hf $output_path 
