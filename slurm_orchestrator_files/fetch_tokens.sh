#/bin/bash
#SBATCH --job-name=fetch_tokens
#SBATCH --output=../slurm_logs/%x-%j-%a.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
#SBATCH --partition=general
#SBATCH --mail_user=emmy@cmu.edu
#SBATCH --mail_type=END
#SBATCH --array=1-6%8

config_file=./pretrain_llama/experiment_files/fetch_tokens.csv
domain=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $2}' $config_file)
num_tokens=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $3}' $config_file)
output=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $4}' $config_file)
command=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $5}' $config_file)
source /home/mengyan3/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env
echo '=== JOB INFO ==='
echo 'Job ID: $SLURM_JOB_ID'
echo 'Array ID: $SLURM_ARRAY_JOB_ID'
echo 'Running on: $HOSTNAME'
echo '=== END JOB INFO ==='

python demo_scripts/fetch_tokens_from_dolma.py --num_tokens {num_tokens} --output {output} --domain {domain}

