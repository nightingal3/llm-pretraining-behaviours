#!/bin/bash
#SBATCH --job-name=sel_feats
#SBATCH --output=%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --array=9
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --partition=array

# List of tasks and their settings
declare -a tasks=(
    "arc_challenge,25-shot"
    "hellaswag,10-shot"
    "mmlu_0-shot,0-shot"
    "mmlu_5-shot,5-shot"
    "truthfulqa,0-shot"
    "winogrande,5-shot"
    "humaneval,0-shot"
    "gsm8k,5-shot",
    "lambada,0-shot",
    "mathqa,0-shot",
    "xnli,0-shot",
    "anli,0-shot",
    "logiqa2,0-shot",
)

# Get the current task from array
IFS=',' read -r TASK SETTING <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"

# Read initial features from CSV for this task
INITIAL_FEATURES=$(python -c "
import pandas as pd
df = pd.read_csv('/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/forward_selection_results_accuracy_test.csv')
task_row = df[(df['task'] == '$TASK') & (df['setting'] == '$SETTING')]
if not task_row.empty:
    features = eval(task_row['selected_features'].iloc[0])
    print(' '.join(features))
")



# Run the performance prediction script for this task
python performance_prediction/performance_predict_from_db_gradual.py \
    --metric accuracy \
    --db_path ./metadata/duckdb/2025_03_03.duckdb \
    --n_estimators 50 \
    --predictor_type all \
    --merge_arithmetic \
    --hyperparam_search \
    --merge_mmlu \
    --drop_instruction_tuned \
    --pseudo_feats_csv ./all_models_feature_stats_3_03_with_ratios.csv \
    --test_significance \
    --initial_features_csv /data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/forward_generation_results_303.csv
    #--selected_task "$TASK" \
    #--selected_setting "$SETTING"