#!/bin/bash
#SBATCH --array=13-15%8
#SBATCH --time=7-00:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --job-name=download_dset
#SBATCH --output=download_dset-%A-%a.out

set -euo pipefail

EXP_CONFIG=$1
IFS=',' read -r task_id dataset_name out_dir subset < <(sed "${SLURM_ARRAY_TASK_ID}q;d" <(tail -n +2 $EXP_CONFIG))

echo "=== JOB INFO ==="
echo "Task ID: $task_id"
echo "Running on: $(hostname)"
echo "Dataset Name: $dataset_name"
echo "Output Directory: $out_dir"
echo "Subset: $subset"
echo "================"

#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate towerllm-env

conda run -n towerllm-env python dolma_data_processing/download_hf_dataset.py ${dataset_name} ${out_dir} --subset ${subset}