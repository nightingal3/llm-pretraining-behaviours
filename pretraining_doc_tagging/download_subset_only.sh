#!/bin/bash
#SBATCH --job-name=download_and_subset_sc
#SBATCH --output=pretraining_doc_tagging/logs/download_and_subset_sc_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=array
#SBATCH --array=6-7
#SBATCH --mail-type END
#SBATCH --mail-user emmy@cmu.edu

declare -a dsets=(
    "monology/pile-uncopyrighted"
    "allenai/c4"
    "togethercomputer/RedPajama-Data-1T"
    "bigcode/starcoderdata"
    "HuggingFaceFW/fineweb"
    "tiiuae/falcon-refinedweb"
    "LLM360/AmberDatasets"
    "LLM360/CrystalCoderDatasets"
)
# Note: already have samples from dolma, cosmopedia


source ~/miniconda3/etc/profile.d/conda.sh
conda activate towerllm-env
cd pretraining_doc_tagging


echo "=== JOB INFO ==="
echo "Starting task $SLURM_ARRAY_TASK_ID"
echo "Running on $HOSTNAME"
echo "Dataset: ${dsets[$SLURM_ARRAY_TASK_ID]}"
echo "==== END INFO ==="

# NOTE: sample from the first 100M. not quite random, but starcoder has some bad data around this pt so can't go further anyway
# TODO: after confirmed working run for full amount except for starcoder
python stream_and_sample.py \
    --dataset "${dsets[$SLURM_ARRAY_TASK_ID]}" \
    --subset_size 1000000 \
    --output /data/tir/projects/tir3/users/mengyan3/subsets_for_tagging/${dsets[$SLURM_ARRAY_TASK_ID]}/sample.json \
    --split train \
    --log-every 100000 \
    --max_docs 50000000 \
    --seed 42