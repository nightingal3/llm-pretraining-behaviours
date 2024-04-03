#!/bin/bash
#SBATCH --job-name=first_15
#SBATCH --output=first_15-%A_%a.out
#SBATCH --error=first_15-%A_%a.err
#SBATCH --array=1-15%15
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --mail-type=END

source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm1

if [ -f /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/configs/.env ]; then
    set -a 
    source /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/configs/.env
    set +a 
fi

set -euo pipefail

CSV=/data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/dolma_data_processing/decontamination/code_removal/files.csv

# Calculate the line number this job should process
LINENUM=$(($SLURM_ARRAY_TASK_ID + 0))

# Read the file path from the CSV file on the calculated line
input_file=$(sed -n "${LINENUM}p" $CSV)

temp_path="${input_file/dolma_100B_json_p2/dolma_100B_decisions}"

#Replace the file extension from '.jsonl' to '.csv'
output_file="${temp_path%.jsonl}.csv"

# Check if FILE is not empty
if [ -z "$input_file" ]; then
  echo "The file path is empty or not found for line $LINENUM."
  exit 1
fi

echo "=== JOB INFO ==="
echo "Starting task $SLURM_ARRAY_TASK_ID"
echo "Running on $HOSTNAME"
echo "Input file: $input_file"
echo "Output file: $output_file"
echo "==== END INFO ==="

line_count=$(wc -l "$input_file" | awk '{print $1}')
echo "Total lines in $input_file: $line_count"

for (( i=0; i<=line_count; i+=1000 )); do
    echo "Processing chunk starting at line: $i"
    python3 /data/tir/projects/tir7/user_data/ssingha2/llm-pretraining-behaviours/dolma_data_processing/decontamination/coderemoval/Mistralv2-dolma.py  --input "$input_file" --output "$output_file" --idx $i
done