#!/bin/bash
#SBATCH --array=58,59,101,111,115,135
#SBATCH --time=2-00:00:00
#SBATCH --partition=array
#SBATCH --cpus-per-task=30
#SBATCH --mem=70G
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --job-name=tag_freegens_final_noparse
#SBATCH --output=/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretraining_doc_tagging/tagging_logs/final-%A-%a.out
#config=/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretraining_doc_tagging/tag_untagged_models.tsv
config=/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretraining_doc_tagging/tag_missing_tasks.tsv

# pythia testing
# 1 - 64

feature=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $2}' $config)
input_file=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $3}' $config)
output_file=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $4}' $config)
model=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $5}' $config)

# add the feature to the output file name, replace .parquet with _<feature>.parquet

if [ "$feature" != "domain_report" -a "$feature" != "entropy" ];
then
    output_file="${output_file%.parquet}_${feature}_${domain}.parquet"
fi


echo "=== JOB INFO ==="
echo "Starting task $SLURM_ARRAY_TASK_ID"
echo "Running on $HOSTNAME"
echo "Feature: $feature"
echo "Input file: $input_file"
echo "Output file: $output_file"
echo "Model: $model"
echo "==== END INFO ==="

# skip if output file already exists
if [ -f $output_file ]
then
    echo "Output file $output_file already exists. Skipping..."
    exit 0
fi

# skip instruct models
if [ $model == *instruct* ]
then
    echo "Skipping instruct model"
    exit 0
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env
cd pretraining_doc_tagging

# domain report
if [ $feature == "domain_report" ]
then
    export OPENAI_API_KEY=<API_KEY>
    echo "Getting domain report features"
    python ../freegens/classify_domains_multistage.py \
        --input $input_file \
        --output $output_file \
        --resps_format \
        --limit 2000
    exit 0
fi

# entropy
if [ $feature == "entropy" ]
then
    echo "Getting entropy features"
    python whole_corpus_measures/entropy.py \
        --input $input_file \
        --output $output_file \
        --text_column resps
    exit 0
fi

python get_dolma_features.py \
    --feature $feature \
    --input $input_file \
    --output $output_file