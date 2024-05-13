#!/bin/bash
#SBATCH --job-name=fetch_tokens
#SBATCH --output=fetch_tokens-%A-%a.out
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=50
#SBATCH --array=1-6%6

config_file=./slurm_orchestrator_files/experiment_files/fetch_tokens.csv
domain=$(awk -F, -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $2}' $config_file)
num_tokens=$(awk -F, -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $3}' $config_file)
output=$(awk -F, -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $4}' $config_file)
command=$(awk -F, -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $5}' $config_file)
source /home/mengyan3/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env
echo '=== JOB INFO ==='
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Running on: ${HOSTNAME}"
echo "domain: ${domain}"
echo "num_tokens: ${num_tokens}"
echo "output: ${output}"
echo "command: ${command}"
echo '=== END JOB INFO ==='

python ./demo_scripts/fetch_tokens_from_dolma.py --num_tokens $num_tokens --output $output --domain $domain

