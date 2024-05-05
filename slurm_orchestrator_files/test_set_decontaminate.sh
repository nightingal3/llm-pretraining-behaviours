#/bin/bash
#SBATCH --job-name=test_set_decontaminate
#SBATCH --output=../slurm_logs/%x-%j-%a.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=50G
#SBATCH --partition=general
#SBATCH --mail_user=emmy@cmu.edu
#SBATCH --mail_type=END
#SBATCH --depend=afterok:fetch_tokens
#SBATCH --array=1-6%8

config_file=./pretrain_llama/experiment_files/test_set_decontaminate.csv
base_dir=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $2}' $config_file)
domain=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $3}' $config_file)
output_dir=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $4}' $config_file)
command=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID 'NR==TaskID+1 {print $5}' $config_file)
source /home/mengyan3/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env
echo '=== JOB INFO ==='
echo 'Job ID: $SLURM_JOB_ID'
echo 'Array ID: $SLURM_ARRAY_JOB_ID'
echo 'Running on: $HOSTNAME'
echo '=== END JOB INFO ==='

python dedup.py --contaminant_path=/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/dolma_data_processing/decontamination/dedup/contaminant.txt --base_dir={base_dir} --output_dir={output_dir} --domain={domain} --num_processes=30

