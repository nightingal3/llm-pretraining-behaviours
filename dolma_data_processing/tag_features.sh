#!/bin/bash
#SBATCH --array=25-32%8
#SBATCH --time=7-00:00:00
#SBATCH --partition=babel-shared-long
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --job-name=tag_dolma_features
#SBATCH --output=tag_dolma_features-%a.out

config=/data/tir/projects/tir6/general/mengyan3/tower-llm-training/tag_features_commands.csv

feature=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $2}' $config)
input_file=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $3}' $config)
output_file=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $4}' $config)
domain=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $5}' $config)

# add the feature to the output file name, replace .parquet with _<feature>.parquet
output_file="${output_file%.parquet}_${feature}_${domain}.parquet"

echo "=== JOB INFO ==="
echo "Starting task $SLURM_ARRAY_TASK_ID"
echo "Running on $HOSTNAME"
echo "Feature: $feature"
echo "Input file: $input_file"
echo "Output file: $output_file"
echo "Domain: $domain"
echo "==== END INFO ==="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env
cd dolma_data_processing
python get_dolma_features.py \
    --feature $feature \
    --input $input_file \
    --output $output_file