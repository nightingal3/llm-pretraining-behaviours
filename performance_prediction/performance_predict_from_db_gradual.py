import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
import argparse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings
import os
import yaml
import shap
import matplotlib.pyplot as plt
import logging
import joblib
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from collections import defaultdict
from sklearn.inspection import PartialDependenceDisplay
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

from common_args import add_common_args, load_data
from metadata.duckdb.model_metadata_db import AnalysisStore
from performance_predict_from_db import train_and_evaluate_w_search

MIN_SAMPLES = 30
NUM_SEEDS = 50
BENCHMARK_DEFAULTS = {
    'arc_challenge': ['25-shot'],
    'hellaswag': ['10-shot'],
    'mmlu_0-shot': ['0-shot'],
    'mmlu_5-shot': ['5-shot'],
    'truthfulqa': ['0-shot'],
    'winogrande': ['5-shot'],
    'lambada': ['0-shot'],
    'gsm8k': ['5-shot'],
    #'arithmetic': ['5-shot'],
    'minerva': ['5-shot'],
    'humaneval': ['0-shot'],
    'mathqa': ['0-shot'],
    'xnli': ['0-shot'],
    'anli': ['0-shot'],
    'logiqa2': ['0-shot'],
    #'fld': '0-shot',
    #'asdiv': '5-shot'
}

def get_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--db_path",
        type=str,
        default="./metadata/duckdb/try.duckdb",
        help="Path to DuckDB database",
    )
    parser.add_argument(
        "--regressor",
        type=str,
        default="xgboost",
        choices=["xgboost", "linear", "svr"],
        help="Type of regressor to use",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="The learning rate for the XGBoost model",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="The maximum depth of the trees in the XGBoost model",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=10,
        help="The number of trees in the XGBoost model",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The path to save the predicted scores",
        default="./predicted_scores.csv",
    )
    parser.add_argument(
        "--missing_val",
        type=float,
        help="The value used for missing data in features",
        default=-1,
    )
    parser.add_argument(
        "--interpret_plot",
        type=str,
        choices=["shap"],
        default="shap",
        help="whether to plot feature importance using SHAP or other methods",
    )
    parser.add_argument(
        "--predictor_type", type=str, choices=["scaling_laws", "all", "non_scaling_laws"], default="all"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--hyperparam_search",
        action="store_true",
        help="Whether to perform hyperparameter search",
    )
    parser.add_argument(
        "--force_new_search",
        action="store_true",
        help="Whether to force a new hyperparameter search rather than using the best loaded model",
    )
    parser.add_argument(
        "--merge_mmlu",
        action="store_true",
        help="merge all the mmlu tasks into one when computing results",
    )
    parser.add_argument(
        "--merge_arithmetic",
        action="store_true",
        help="merge all the arithmetic tasks into one when computing results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--pseudo_feats_csv",
        type=str,
        help="The path to the CSV file containing the freegen features",
    )
    parser.add_argument(
        "--pseudo_feats_only",
        action="store_true",
        help="Whether to use only pseudo features",
    )
    parser.add_argument(
        "--measure_extrapolation",
        action="store_true",
        help="Whether to measure extrapolation performance (hold out largest 5 models)",
    )
    parser.add_argument(
        "--selected_task",
        type=str,
        help="Run analysis for this specific task only",
    )
    parser.add_argument(
        "--selected_setting",
        type=str,
        help="Run analysis for this specific setting only",
    )
    parser.add_argument(
        "--initial_features",
        type=str,
        nargs='+',
        help="List of initial features to start with"
    )
    parser.add_argument(
        "--initial_features_csv",
        type=str,
        help="Path to CSV containing initial features for each task"
    )
    parser.add_argument(
        "--test_significance",
        action="store_true",
        help="Run statistical significance testing instead of forward selection"
    )



    args = parser.parse_args()

    assert args.n_estimators > 0, "Number of trees must be greater than 0"
    assert args.lr > 0, "Learning rate must be greater than 0"
    assert args.max_depth > 0, "Max depth must be greater than 0"
    if not (args.model_feats or args.data_feats):
        raise ValueError("Please provide either model_feats or data_feats")
    assert not (args.initial_features and args.initial_features_csv), "Please provide initial features or initial features CSV, not both"
    return args

def get_initial_features(initial_features_csv, task, setting):
    """Read initial features for a task from CSV"""
    if not initial_features_csv:
        return None
        
    df = pd.read_csv(initial_features_csv)
    task_row = df[(df['task'] == task) & (df['setting'] == setting)]
    
    if len(task_row) == 0:
        print(f"No initial features found for {task} {setting}")
        return None
        
    features = eval(task_row['selected_features'].iloc[0])
    print(f"Loaded initial features for {task} {setting}: {features}")
    return features

def forward_feature_selection(df, labels, task, setting, args, base_features=['id'], num_seeds=10, threshold=0.0001):
    """
    Perform forward feature selection starting from given initial features.
    Args:
        initial_features: List of features to start with (already selected in previous runs)
        threshold: Minimum MAE improvement required to add a feature
    """
    # Initialize feature pools
    scaling_laws_features = ['total_params', 'pretraining_summary_total_tokens_billions']
    
    # Start with either initial features or scaling laws
    if args.initial_features_csv:
        selected_features = get_initial_features(args.initial_features_csv, task, setting)
        current_features = base_features + selected_features
    elif args.initial_features:
        selected_features = args.initial_features.copy()
        current_features = base_features + args.initial_features
    else:
        selected_features = scaling_laws_features.copy()
        current_features = base_features + scaling_laws_features
    
    # Get candidate features excluding already selected ones
    excluded_cols = base_features + selected_features + ['benchmark', 'setting', 'value', 'value_stderr']
    # if args.pseudo_feats_csv:
    #     excluded_cols += ["dimension", "num_heads", "mlp_ratio", "sequence_length", 'pretraining_summary_percentage_web', 'pretraining_summary_percentage_code', 'pretraining_summary_percentage_books', 'pretraining_summary_percentage_english', 'pretraining_summary_percentage_academic', 'pretraining_summary_percentage_reference']

    candidate_features = [col for col in df.columns if col not in excluded_cols]
    
    print(f"\nStarting with features: {selected_features}")
    print(f"Found {len(candidate_features)} additional features to evaluate:")
    print(candidate_features)
    

    improvement_scores = {}
    
    # Get baseline performance with scaling laws
    print("\nEvaluating baseline (scaling laws) features...")
    baseline_maes = []
    
    for seed in range(num_seeds):
        features_df = df[current_features].copy()
        fold_results = train_and_evaluate_w_search(
            features_df, labels, args,
            seed=seed,
            task=task,
            setting=setting,
            verbose=False
        )
        baseline_maes.append(np.mean(fold_results['mae']))
        print(f"Seed: {seed}, MAE: {baseline_maes[-1]:.4f}")
    
    baseline_mae = np.mean(baseline_maes)
    print(f"\nBaseline MAE with scaling laws: {baseline_mae:.4f}")
    
    # Iteratively add features
    while candidate_features:
        best_feature = None
        best_improvement = 0
        best_mae = float('inf')
        best_maes = None
        
        print(f"\nTesting {len(candidate_features)} remaining candidate features...")
        
        for feature in tqdm(candidate_features):
            test_features = current_features + [feature]
            feature_maes = []
            
            # Evaluate with current feature added
            for seed in range(num_seeds):
                features_df = df[test_features].copy()
                fold_results = train_and_evaluate_w_search(
                    features_df, labels, args,
                    seed=seed,
                    task=task,
                    setting=setting,
                    verbose=False
                )
                feature_maes.append(np.mean(fold_results['mae']))
            
            mean_mae = np.mean(feature_maes)
            improvement = baseline_mae - mean_mae
            
            print(f"{feature}: MAE = {mean_mae:.4f}, Improvement = {improvement:.4f}")
            
            # Check if this feature gives best improvement so far
            if improvement > best_improvement:
                best_improvement = improvement
                best_feature = feature
                best_mae = mean_mae
                best_maes = feature_maes
        
        # If we found a feature that might improve performance, test for significance
        if best_improvement > threshold:
            print(f"\nSelected feature {best_feature} with significant improvement:")
            print(f"Improvement: {best_improvement:.4f}")
            
            current_features.append(best_feature)
            selected_features.append(best_feature)
            improvement_scores[best_feature] = {
                'mae_improvement': best_improvement
            }
            candidate_features.remove(best_feature)
            baseline_mae = best_mae
            baseline_maes = best_maes
        else:
            print("\nNo more features improve performance. Stopping selection.")
            break
    
    # Create ordered selection history
    selection_history = []
    cumulative_mae = np.mean(baseline_maes)
    for feature in selected_features:
        if feature in improvement_scores:  # Skip scaling law features
            stats_improvement = improvement_scores[feature]
            selection_history.append({
                'feature': feature,
                'mae_before': cumulative_mae,
                'mae_after': cumulative_mae - stats_improvement['mae_improvement'],
                'improvement': stats_improvement['mae_improvement'],
            })
            cumulative_mae -= stats_improvement['mae_improvement']
    
    return {
        'selected_features': selected_features,
        'improvement_scores': improvement_scores,
        'selection_history': selection_history,
        'final_mae': baseline_mae,
        'baseline_mae': np.mean(baseline_maes),
        'task': task,
        'setting': setting
    }

def run_forward_selection_experiments(args):
    """Run forward selection experiments across all tasks and aggregate results."""
    # Load and preprocess data
    df = load_data_from_db(args.db_path, args.predictor_type, args.metric, args.drop_instruction_tuned)
    df = standardize_task_names(df)

    # Add pseudo features if specified
    if args.pseudo_feats_csv:
        pseudo_feats = pd.read_csv(args.pseudo_feats_csv)
        pseudo_feats.drop_duplicates(subset="id", inplace=True)
        df = df.merge(pseudo_feats, on="id", how="left")
        # Add this line to store column names
        args.pseudo_feats = [col for col in pseudo_feats.columns if col != "id"]
    
    # Preprocess data
    pseudo_feats_lst = [] if not args.pseudo_feats_csv else list(set(pseudo_feats.columns) - set(["id"]))
    df, enc = preprocess_data(df, args.predictor_type, pseudo_feats=pseudo_feats_lst, use_freegens_only=args.pseudo_feats_only)
    df = aggregate_multi_part_evals(df)
    df = df.astype({col: 'float64' for col in df.columns if col not in ['benchmark', 'setting', 'id']})
    
    # Get task settings
    task_settings = df.groupby(['benchmark', 'setting']).size().reset_index()
    
    # Track results for all tasks
    all_results = []
    debug_tasks = ["lambada", "gsm8k"]


    for _, row in task_settings.iterrows():
        task = row['benchmark']
        setting = row['setting']

        if args.selected_task and args.selected_setting:
            if task != args.selected_task or setting != args.selected_setting:
                continue
        
        root_task = None
        for task_base in BENCHMARK_DEFAULTS:
            if task.startswith(task_base):
                root_task = task_base
                break
        if not root_task:
            continue
        if setting not in BENCHMARK_DEFAULTS[root_task]:
            continue
            
        print(f"\n=== Running forward selection for {task}, {setting} ===")
        features, labels = prepare_task_data(df, task, setting)

        if len(features) < MIN_SAMPLES:
            print(f"Skipping {task}, {setting} due to insufficient samples")
            continue
        
        # Handle missing values
        mask = labels.notnull()
        features = features[mask]
        labels = labels[mask]
        
        feat_transform(features)

        results = forward_feature_selection(features, labels, task, setting, args)
        all_results.append(results)
        
        # Run forward selection
        output_name = f"forward_selection_results_303_{args.metric}"
        if args.selected_task and args.selected_setting:
            output_name += f"_{args.selected_task}_{args.selected_setting}"
        if args.pseudo_feats_csv:
            output_name += "_freegens"
        output_name += "_low_threshold_instructionadd.csv"
        
        pd.DataFrame(all_results).to_csv(output_name, index=False)
    
    return all_results

def analyze_forward_selection_results(all_results):
    """Analyze and print summary of forward selection results."""
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    print("\n=== Forward Selection Analysis ===")
    
    # Overall improvement statistics
    total_improvement = results_df['baseline_mae'] - results_df['final_mae']
    print("\nOverall Improvement Statistics:")
    print(f"Average improvement: {total_improvement.mean():.4f}")
    print(f"Median improvement: {total_improvement.median():.4f}")
    print(f"Max improvement: {total_improvement.max():.4f}")
    
    # Feature selection frequency
    all_selected_features = []
    for result in all_results:
        all_selected_features.extend(result['selected_features'])
    
    feature_counts = pd.Series(all_selected_features).value_counts()
    print("\nFeature Selection Frequency:")
    print(feature_counts)
    
    # Average improvement by feature
    feature_improvements = defaultdict(list)
    for result in all_results:
        for feature, improvement_dict in result['improvement_scores'].items():
            # Extract the mae_improvement value from the dictionary
            mae_improvement = improvement_dict['mae_improvement']
            feature_improvements[feature].append(mae_improvement)
    
    avg_improvements = {
        feature: np.mean(improvements) 
        for feature, improvements in feature_improvements.items()
    }
    
    print("\nAverage Improvement by Feature:")
    for feature, improvement in sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {improvement:.4f}")
    
    return results_df

def check_db_tables(db_path):
    store = AnalysisStore.from_existing(db_path)
    # List all tables
    tables = store.con.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table';
    """).fetchall()
    
    # Print table counts
    for table in tables:
        count = store.con.execute(f"""
            SELECT COUNT(*) FROM {table[0]}
        """).fetchone()[0]
        print(f"{table[0]}: {count} rows")
    
    store.con.close()

def get_agg_benchmark(row):
        benchmark = row['benchmark']
        setting = row['setting']
        
        # Mapping for known multi-part names.
        mapping = {
            'truthfulqa_mc1': 'truthfulqa',
            'truthfulqa_mc2': 'truthfulqa',
            'lambada_standard': 'lambada',
            'lambada_openai': 'lambada',
            'gsm8k': 'gsm8k',
            'gsm8k_cot': 'gsm8k'
        }
        
        # For benchmarks starting with "arithmetic_" or "xnli_", assign the proper mapped name.
        if benchmark.startswith("arithmetic_"):
            mapped = "arithmetic"
        elif benchmark.startswith("xnli_"):
            mapped = "xnli"
        elif benchmark.startswith("anli_"):
            mapped = "anli"
        elif benchmark.startswith("mmlu_"):
            mapped = f"mmlu_{setting}"
            return mapped
        else:
            mapped = mapping.get(benchmark, benchmark)
        
        # Check if the shot is one of the allowed defaults.
        # (If the mapped benchmark is not in BENCHMARK_DEFAULTS, skip it by returning None.)
        if mapped not in BENCHMARK_DEFAULTS:
            return None
        if setting in BENCHMARK_DEFAULTS[mapped]:
            return mapped
        else:
            return None

def aggregate_multi_part_evals(df: pd.DataFrame) -> pd.DataFrame:
    # Create a new column for the aggregated benchmark using get_agg_benchmark.
    df['aggregated_benchmark'] = df.apply(get_agg_benchmark, axis=1)
    
    # Drop rows that did not meet the default shot criteria.
    df = df[df['aggregated_benchmark'].notnull()].copy()
    
    # For downstream processing, rename 'aggregated_benchmark' to 'benchmark' if desired.
    df['benchmark'] = df['aggregated_benchmark']
    
    # Group by model id, benchmark, and setting to average the metric value.
    agg_dict = {'value': 'mean'}
    if 'metric_stderr' in df.columns:
        agg_dict['metric_stderr'] = 'mean'
    
    group_cols = ['id', 'benchmark', 'setting']
    other_cols = [col for col in df.columns if col not in group_cols + ['value', 'metric_stderr']]
    for col in other_cols:
        agg_dict[col] = 'first'
    
    # Group and aggregate while retaining all columns
    df_agg = df.groupby(group_cols, as_index=False).agg(agg_dict)

    df_agg = df_agg.drop(columns=['aggregated_benchmark'])
    return df_agg


def load_data_from_db(db_path: str, predictor_type: str, metric: str, drop_instruction_tuned: bool = False) -> pd.DataFrame:
    """Load and join data from DuckDB database"""
    try:
        store = AnalysisStore.from_existing(db_path)
        
        if predictor_type == "scaling_laws":
            query = """
                SELECT 
                    m.id,
                    m.total_params,
                    m.is_instruction_tuned,
                    d.pretraining_summary_total_tokens_billions,
                    e.benchmark,
                    e.setting,
                    e.metric_value as value,
                    e.metric_stderr as value_stderr
                FROM model_annotations m
                LEFT JOIN dataset_info d ON m.id = d.id
                LEFT JOIN evaluation_results e ON m.id = e.id
                WHERE m.total_params IS NOT NULL 
                AND d.pretraining_summary_total_tokens_billions IS NOT NULL
                AND e.metric = ?
            """
        else:
            query = """
                SELECT 
                    m.*,
                    d.*,
                    e.benchmark,
                    e.setting,
                    e.metric_value as value,
                    e.metric_stderr as value_stderr
                FROM model_annotations m
                LEFT JOIN dataset_info d ON m.id = d.id
                LEFT JOIN evaluation_results e ON m.id = e.id
                WHERE m.total_params IS NOT NULL 
                AND d.pretraining_summary_total_tokens_billions IS NOT NULL
                AND e.metric = ?
            """
        
        df = store.con.execute(query, [metric]).df()

        if predictor_type == "non_scaling_laws":
            df = df.drop(columns=["total_params", "pretraining_summary_total_tokens_billions"])
            # optional
            df = df.drop(columns=["dimension", "sequence_length", "num_heads"])

        if drop_instruction_tuned:
            df = df[df["is_instruction_tuned"] != True]
        df = df.drop(columns=["is_instruction_tuned"])
    finally:
        store.con.close()

    return df


def preprocess_data(df: pd.DataFrame, predictor_type: str, missing_val: int = -1, pseudo_feats: List = [], use_freegens_only: bool = False) -> pd.DataFrame:
    """Preprocess the data based on predictor type."""
    if use_freegens_only:
        # Use only pseudo (free generation) features
        freegen_columns = [col for col in pseudo_feats if col != "id"]
        #freegen_columns = ["edu_classifier_mean"]
        df = df[["id"] + freegen_columns + ["value", "benchmark", "setting"]]

        return df, None
    else:
        if predictor_type == "scaling_laws":
            # Keep only essential columns for scaling laws analysis
            cols_to_keep = [
                'total_params',
                'pretraining_summary_total_tokens_billions',
                'benchmark',
                'setting',
                'value',
                'id'
            ]
            df = df[cols_to_keep].copy()
            return df, None
        else:
            categorical_variables = [
                "activation",
                "attention_variant",
                "biases",
                "block_type",
                "layer_norm_type",
                "positional_embeddings"
            ]
            
            numeric_variables = [
                'total_params',
                'dimension',
                'num_heads',
                'mlp_ratio',
                'sequence_length',
                'pretraining_summary_total_tokens_billions',
                'pretraining_summary_percentage_web',
                'pretraining_summary_percentage_code',
                'pretraining_summary_percentage_books',
                'pretraining_summary_percentage_reference',
                'pretraining_summary_percentage_academic',
                'pretraining_summary_percentage_english'
            ]

            if predictor_type == "non_scaling_laws":
                # remove the total params and pretraining tokens
                numeric_variables.remove("total_params")
                numeric_variables.remove("pretraining_summary_total_tokens_billions")

                # optional
                numeric_variables.remove("dimension")
                numeric_variables.remove("sequence_length")
                numeric_variables.remove("num_heads")

            if len(pseudo_feats) > 0:
                numeric_variables.extend(pseudo_feats)
            
            df = df[numeric_variables + categorical_variables + ['benchmark', 'setting', 'value', 'id']].copy()

            for num in numeric_variables:
                df[num] = pd.to_numeric(df[num], errors='coerce').fillna(missing_val)
            
            encoders = {}
            for cat in categorical_variables:
                df[cat] = df[cat].fillna('missing')
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                df[cat] = enc.fit_transform(df[[cat]])
                encoders[cat] = enc

        
            df = df.astype({col: 'float32' for col in df.columns if col not in ['benchmark', 'setting', 'id']})
            
            return df, encoders

def prepare_task_data(df: pd.DataFrame, task: str, setting: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for a specific task and setting"""
    task_data = df[
        (df['benchmark'] == task) & 
        (df['setting'] == setting)
    ].copy()
    
    # Drop non-feature columns
    feature_df = task_data.drop(columns=[
        'benchmark', 'setting', 'value',
    ])
    
    labels = task_data['value']
    
    return feature_df, labels


def median_baseline(train_labels, test_feats):
    median = train_labels.median()
    return np.full(len(test_feats), median)

def standardize_task_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize duplicate task names and remove duplicates from name variations"""
    df = df.copy()
    
    # Convert hendrycksTest to mmlu
    hendrycks_mask = df['benchmark'].str.startswith('hendrycksTest-')
    if hendrycks_mask.any():
        df.loc[hendrycks_mask, 'benchmark'] = df.loc[hendrycks_mask, 'benchmark'].str.replace('hendrycksTest-', 'mmlu_')
    
    # Fix ARC challenge naming
    df.loc[df['benchmark'] == 'arc:challenge', 'benchmark'] = 'arc_challenge'
    
    # Remove duplicates, keeping first occurrence
    df = df.drop_duplicates(subset=['id', 'benchmark', 'setting'], keep='first')
    
    return df

def check_missing_values(df):
    for col in df.columns:
        n_nan = df[col].isna().sum()
        n_minus_one = (df[col] == -1).sum()
        if n_nan > 0 or n_minus_one > 0:
            print(f"{col}: {n_nan} NaN, {n_minus_one} -1 values")

# def train_and_evaluate_w_search(
#     features: pd.DataFrame,
#     labels: pd.Series,
#     args: argparse.Namespace,
#     task: str,
#     setting: str,
#     missing_val: float = -1,
#     seed: int = 42,
#     extrapolation_feats: Optional[pd.DataFrame] = None,
#     extrapolation_labels: Optional[pd.Series] = None,
#     **kwargs,
# ) -> Dict:
#     """Train model and evaluate performance with nested CV for hyperparameter tuning."""
#     #kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#     kf = KFold(n_splits=3, shuffle=True, random_state=seed)
#     results = {
#         'mae': [],
#         'predictions': [],
#         'true_values': [],
#         'feature_importance': [],
#         'all_mae_median_baseline': [],
#         'model_ids': [],
#         'all_shap_values': [],
#         'test_features': [],
#         'task_signed_errors': {},  # Store signed errors per model
#         'task_absolute_errors': {},  # Store absolute errors per model
#     }

#     # Parameter grid for tuning
#     param_grid = {
#         #'max_depth': [3, 6, 10],
#         'max_depth': [2, 3, 5],
#         'learning_rate': [0.01, 0.1, 0.3],
#         #'learning_rate': [0.01, 0.1],
#         'n_estimators': [50, 100],
#         #'reg_lambda': [0.1, 1, 10],
#         #'subsample': [0.8, 1]
#     }

#     # TODO: two different param grids for sl/all?
#     # param_grid = {
#     #     'max_depth': [3, 5, 7],
#     #     'learning_rate': [0.01, 0.1, 0.3],
#     #     'n_estimators': [500, 1000],
#     #     'subsample': [0.8, 1],
#     #     'min_child_weight': [1, 3],
#     # }


#     for train_idx, test_idx in kf.split(features):
#         # Split data
#         X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
#         y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
#         X_train_model_names, X_test_model_names = X_train['id'], X_test['id']

#         # Drop the 'id' column
#         #X_train = X_train.drop(columns=['id'])
#         #X_test = X_test.drop(columns=['id'])
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_train, y_train, test_size=0.2, random_state=seed
#         )   
#         X_train = X_train.drop(columns=['id'])
#         X_val = X_val.drop(columns=['id'])
#         X_test = X_test.drop(columns=['id'])

#         # Inner CV for hyperparameter tuning
#         inner_kf = KFold(n_splits=3, shuffle=True, random_state=seed)
#         model = xgb.XGBRegressor(
#             objective="reg:squarederror",
#             enable_categorical=True,
#             missing=missing_val,
#             random_state=seed,
#         )
#         grid_search = GridSearchCV(
#             estimator=model,
#             param_grid=param_grid,
#             cv=inner_kf,
#             scoring='neg_mean_absolute_error',
#             n_jobs=-1
#         )
#         grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
#         best_params = grid_search.best_params_
#         #print(f"Best params for task {task}, setting {setting}: {best_params}")

#         # Train with best hyperparameters
#         tuned_model = xgb.XGBRegressor(
#             objective="reg:squarederror",
#             enable_categorical=True,
#             missing=missing_val,
#             random_state=seed,
#             **best_params
#         )
#         tuned_model.fit(X_train, y_train)
#         #valid_idx = X_train["pretraining_summary_percentage_code"] > -1

#         #PartialDependenceDisplay.from_estimator(tuned_model, X_train[valid_idx], ["pretraining_summary_percentage_code"])
#         #plt.savefig("partial_dependence_code_lambada.png")
#         #assert False


#         # Evaluate predictions
#         preds = tuned_model.predict(X_test)
#         mae = mean_absolute_error(y_test, preds)
#         results['mae'].append(mae)
#         results['predictions'].extend(preds)
#         results['true_values'].extend(y_test)
#         results['model_ids'].extend(X_test_model_names)

#         # Feature importance
#         importance = tuned_model.feature_importances_
#         results['feature_importance'].append(importance)

#         # Median baseline
#         median_predictions = np.full(len(y_test), y_train.median())
#         mae_median = mean_absolute_error(y_test, median_predictions)
#         results['all_mae_median_baseline'].append(mae_median)

#         # SHAP values
#         if args.regressor == "xgboost":
#             explainer = shap.Explainer(tuned_model)
#             shap_values = explainer(X_test)
#             results['all_shap_values'].append(shap_values.values)
#             results['test_features'].append(X_test)

#         # Calculate errors per model
#         signed_errors = {name: pred - true for name, pred, true in zip(X_test_model_names, preds, y_test)}
#         absolute_errors = {name: abs(error) for name, error in signed_errors.items()}

#         # Store errors
#         results['task_signed_errors'].update(signed_errors)
#         results['task_absolute_errors'].update(absolute_errors)

#     # Save errors for analysis
#     error_df = pd.DataFrame([
#         {
#             "Model": model,
#             "SignedError": signed_error,
#             "AbsoluteError": abs_error
#         }
#         for model, signed_error, abs_error in zip(
#             results['task_signed_errors'].keys(),
#             results['task_signed_errors'].values(),
#             results['task_absolute_errors'].values()
#         )
#     ])
#     error_df = error_df.sort_values(by="SignedError", ascending=True)

#     synthetic_data = pd.DataFrame({
#         "total_params": np.log(np.linspace(1e8, 1e12, 100)),  # Log-transformed params
#         "pretraining_summary_total_tokens_billions": np.log(np.linspace(1e2, 1e5, 100)),  # Log-transformed tokens
#     })

#     # Predict on synthetic data
#     # TODO: create a better version of this check later
#     # breakpoint()
#     # synthetic_preds = tuned_model.predict(synthetic_data)

#     # # Plot predictions vs. num_params and num_tokens
#     # plt.figure(figsize=(12, 6))

#     # # Predictions vs. num_params
#     # plt.subplot(1, 2, 1)
#     # plt.plot(synthetic_data["total_params"], synthetic_preds, label="Predictions")
#     # plt.xlabel("Number of Parameters")
#     # plt.ylabel("Predicted Metric")
#     # plt.title("Predictions vs. Number of Parameters")
#     # plt.legend()

#     # # Predictions vs. num_tokens
#     # plt.subplot(1, 2, 2)
#     # plt.plot(synthetic_data["pretraining_summary_total_tokens_billions"], synthetic_preds, label="Predictions")
#     # plt.xlabel("Number of Tokens")
#     # plt.ylabel("Predicted Metric")
#     # plt.title("Predictions vs. Number of Tokens")
#     # plt.legend()

#     # plt.tight_layout()
#     # plt.savefig("overfitting_check.png")
#     # breakpoint()
#     return results

def quick_grid_search(features: pd.DataFrame, labels: pd.Series) -> Dict:
    """Quick grid search with caching per task"""
    cache_dir = Path("./hyperparam_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"sl_params.joblib"
    
    if cache_file.exists():
        return joblib.load(cache_file)
    
    param_grid = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100]
    }
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        enable_categorical=True,
        missing=-1,
        random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # Reduced from 5 for speed
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    grid_search.fit(features, labels)
    results = {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_
    }
    
    joblib.dump(results, cache_file)
    return results

def train_and_evaluate(
    features: pd.DataFrame,
    labels: pd.Series,
    args: argparse.Namespace,
    n_estimators: int = 10,
    lr: float = 0.1,
    max_depth: int = 10,
    missing_val: float = -1,
    seed: int = 42,
    **kwargs,
) -> Dict:
    """Train model and evaluate performance."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    results = {
        'mae': [],
        'predictions': [],
        'true_values': [],
        'feature_importance': [],
        'all_mae_median_baseline': [],
        'model_ids': [],
        'all_shap_values': [],
        'test_features': [],
        'task_signed_errors': {},  # Store signed errors per model
        'task_absolute_errors': {},  # Store absolute errors per model
    }

    # Check for missing values
    check_missing_values(features)
    
    for train_idx, test_idx in kf.split(features):
        # Split data
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        X_train_model_names, X_test_model_names = X_train['id'], X_test['id']

        # Drop the 'id' column
        X_train = X_train.drop(columns=['id'])
        X_test = X_test.drop(columns=['id'])

        # Train baseline (median) model for comparison
        median_predictions = median_baseline(y_train, y_test)
        mae_median = mean_absolute_error(y_test, median_predictions)
        results['all_mae_median_baseline'].append(mae_median)

        # Configure and train the main model
        kwargs = {
            "objective": "reg:squarederror",
            "learning_rate": lr,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "enable_categorical": True,
            "missing": missing_val,
            "random_state": seed,
        }
        if args.regressor == "xgboost":
            model = xgb.XGBRegressor(**kwargs)
        elif args.regressor == "linear":
            model = make_pipeline(SimpleImputer(), LinearRegression())
        else:
            model = make_pipeline(SimpleImputer(), SVR())

        model.fit(X_train, y_train)

        # Make predictions
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        # Store results
        results['mae'].append(mae)
        results['predictions'].extend(preds)
        results['true_values'].extend(y_test)
        results['model_ids'].extend(X_test_model_names)

        # Feature importance and SHAP values (if applicable)
        if args.regressor == "xgboost":
            importance = model.feature_importances_

            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            results['all_shap_values'].append(shap_values.values)
            results['test_features'].append(X_test)
        else:
            importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=args.seed
            ).importances_mean

        results['feature_importance'].append(importance)

        # Calculate per-model signed and absolute errors
        signed_errors = {name: pred - true for name, pred, true in zip(X_test_model_names, preds, y_test)}
        absolute_errors = {name: abs(error) for name, error in signed_errors.items()}

        # Store errors
        results['task_signed_errors'].update(signed_errors)
        results['task_absolute_errors'].update(absolute_errors)

    # Save errors for analysis
    error_df = pd.DataFrame([
        {
            "Model": model,
            "SignedError": signed_error,
            "AbsoluteError": abs_error
        }
        for model, signed_error, abs_error in zip(
            results['task_signed_errors'].keys(),
            results['task_signed_errors'].values(),
            results['task_absolute_errors'].values()
        )
    ])
    error_df = error_df.sort_values(by="SignedError", ascending=True)
    output_dir = Path(f"./performance_prediction/errors/{args.metric}")
    output_dir.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(output_dir / f"{args.regressor}_{args.predictor_type}.csv", index=False)
    print(f"Saved error details for metric {args.metric} to {output_dir}")
    return results


def postprocess_results(df_results: Dict, args) -> Dict:
    if args.merge_mmlu:
        mmlu_tasks = df_results[df_results["task"].str.startswith("hendrycksTest-") | df_results["task"].str.startswith("mmlu_")]
        df_results = pd.concat(
            [df_results, pd.DataFrame({"task": ["mmlu"], "mae": [mmlu_tasks["mae"].mean()], "std_mae": [mmlu_tasks["std_mae"].mean()], "med_baseline_mae": [mmlu_tasks["med_baseline_mae"].mean()]})]
        )
        # delete all the individual mmlu tasks
        df_results = df_results[~df_results["task"].str.startswith("hendrycksTest")& ~df_results["task"].str.startswith("mmlu_")]
    if args.merge_arithmetic:
        arithmetic_tasks = df_results[df_results["task"].str.startswith("arithmetic_")]
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame({"task": ["arithmetic"], 
                "mae": [arithmetic_tasks["mae"].mean()],
                "std_mae": [arithmetic_tasks["std_mae"].mean()],
                "med_baseline_mae": [arithmetic_tasks["med_baseline_mae"].mean()]}),
            ]
        )
        # delete all the individual arithmetic tasks
        df_results = df_results[~df_results["task"].str.startswith("arithmetic_")]

    # combine arc:challenge and arc_challenge
    if "arc:challenge" in df_results["task"].values:
        arc_challenge = df_results[df_results["task"] == "arc:challenge"]
        arc_challenge["task"] = "arc_challenge"
        df_results = pd.concat([df_results, arc_challenge])
        df_results = df_results[df_results["task"] != "arc:challenge"]
    # merge truthfulqa mc1 and mc2
    if "truthfulqa_mc1" in df_results["task"].values:
        truthfulqa_mc1 = df_results[df_results["task"] == "truthfulqa_mc1"]
        truthfulqa_mc2 = df_results[df_results["task"] == "truthfulqa_mc2"]
        mae_overall = (truthfulqa_mc1["mae"].iloc[0] + truthfulqa_mc2["mae"].iloc[0]) / 2
        df_results = pd.concat([df_results, pd.DataFrame({
            "task": ["truthfulqa_mc"],
            "mae": [mae_overall],
            "std_mae": [(truthfulqa_mc1["std_mae"].iloc[0] + truthfulqa_mc2["std_mae"].iloc[0]) / 2],
            "med_baseline_mae": [(truthfulqa_mc1["med_baseline_mae"].iloc[0] + truthfulqa_mc2["med_baseline_mae"].iloc[0]) / 2]
        })])
        df_results = df_results[df_results["task"] != "truthfulqa_mc2"]
        df_results = df_results[df_results["task"] != "truthfulqa_mc1"]
    # merge minerva math
    if "minerva_math_algebra" in df_results["task"].values:
        minerva_math_tasks = df_results[
            df_results["task"].str.startswith("minerva_math_")
        ]
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["minerva_math"],
                        "mae": [minerva_math_tasks["mae"].mean()],
                        "std_mae": [minerva_math_tasks["std_mae"].mean()],
                        "med_baseline_mae": [minerva_math_tasks["med_baseline_mae"].mean()],
                    }
                ),
            ]
        )
        # delete all the individual minerva math tasks
        df_results = df_results[
            ~df_results["task"].str.startswith("minerva_math_")
        ]
    # merge lambada_standard and lambada_openai
    if "lambada_standard" in df_results["task"].values:
        lambada_standard = df_results[df_results["task"] == "lambada_standard"]
        lambada_openai = df_results[df_results["task"] == "lambada_openai"]
        mae_overall = (lambada_standard["mae"].iloc[0] + lambada_openai["mae"].iloc[0]) / 2
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["lambada"],
                        "mae": [mae_overall],
                        "std_mae": [(lambada_standard["std_mae"].iloc[0] + lambada_openai["std_mae"].iloc[0]) / 2],
                        "med_baseline_mae": [(lambada_standard["med_baseline_mae"].iloc[0] + lambada_openai["med_baseline_mae"].iloc[0]) / 2],
                    }
                ),
            ]
        )
        df_results = df_results[df_results["task"] != "lambada_standard"]
        df_results = df_results[df_results["task"] != "lambada_openai"]
    if "gsm8k" in df_results["task"].values and "gsm8k_cot" in df_results["task"].values:
        gsm8k = df_results[df_results["task"] == "gsm8k"]
        gsm8k_cot = df_results[df_results["task"] == "gsm8k_cot"]
        mae_overall = (gsm8k["mae"].iloc[0] + gsm8k_cot["mae"].iloc[0]) / 2
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["gsm8k_all"],
                        "mae": [mae_overall],
                        "std_mae": [(gsm8k["std_mae"].iloc[0] + gsm8k_cot["std_mae"].iloc[0]) / 2],
                        "med_baseline_mae": [(gsm8k["med_baseline_mae"].iloc[0] + gsm8k_cot["med_baseline_mae"].iloc[0]) / 2],
                    }
                ),
            ]
        )
        df_results = df_results[df_results["task"] != "gsm8k"]
        df_results = df_results[df_results["task"] != "gsm8k_cot"]
    # delete fld default/fld star since they seem to be empty?
    if "fld_default" in df_results["task"].values:
        df_results = df_results[df_results["task"] != "fld_default"]
    if "fld_star" in df_results["task"].values:
        df_results = df_results[df_results["task"] != "fld_star"]
    # merge all the squad tasks
    print("RESULTS")
    print(df_results)
    return df_results


def feat_transform(dataset: pd.DataFrame):
    # transform total_params and pretraining_summary_total_tokens_billions to log scale
    (
        dataset["total_params"],
        dataset["pretraining_summary_total_tokens_billions"],
    ) = pd.to_numeric(dataset["total_params"], errors="coerce"), pd.to_numeric(
        dataset["pretraining_summary_total_tokens_billions"], errors="coerce"
    )

    dataset["total_params"] = np.log(dataset["total_params"])
    dataset["pretraining_summary_total_tokens_billions"] = np.log(
        dataset["pretraining_summary_total_tokens_billions"]
    )
    return dataset

def plot_task_group_shap_values(all_results: Dict, features: pd.DataFrame, args: argparse.Namespace, output_dir: Path):
    """Plot SHAP values for each major task group"""
    if args.regressor != "xgboost":
        return
    
    # Define task groups
    task_groups = {
        'arc_challenge': ['arc_challenge'],
        'drop': ['drop'],
        'gsm8k': ['gsm8k', 'gsm8k_cot'],
        'hellaswag': ['hellaswag'],
        'winogrande': ['winogrande'],
        'mmlu': [t for t in all_results.keys() if 'mmlu_' in t or 'hendrycksTest-' in t],
        'arithmetic': [t for t in all_results.keys() if 'arithmetic_' in t],
        'truthfulqa': ['truthfulqa_mc1', 'truthfulqa_mc2'],
        'minerva_math': [t for t in all_results.keys() if 'minerva_math_' in t],
        'lambada': ['lambada']
    }
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Get feature columns based on predictor type
    if args.predictor_type == "scaling_laws":
        feature_cols = ['total_params', 'pretraining_summary_total_tokens_billions']
    else:
        feature_cols = [col for col in features.columns if col not in ['id', 'benchmark', 'setting', 'value', 'value_stderr']]
    
    feature_matrix = features[feature_cols].copy()
    
    # Process each task group
    for group_name, task_list in task_groups.items():
        all_shap_values = []
        test_features_list = []
        
        # Collect SHAP values for all tasks in this group
        for task, results in all_results.items():
            if any(task.startswith(t) for t in task_list):
                if 'all_shap_values' in results:
                    all_shap_values.extend(results['all_shap_values'])
                    test_features_list.extend(results['test_features'])
        
        if not all_shap_values:
            logging.warning(f"No SHAP values found for task group: {group_name}")
            continue
            
        # Aggregate SHAP values for this group
        aggregated_shap_values = np.concatenate(all_shap_values, axis=0)
        aggregated_test_features = pd.concat(test_features_list, ignore_index=True)

        
        # Verify shapes match
        if aggregated_shap_values.shape[1] != len(feature_cols):
            logging.error(f"Shape mismatch for {group_name}: SHAP values shape {aggregated_shap_values.shape} vs features shape {feature_matrix.shape}")
            continue
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            aggregated_shap_values,
            aggregated_test_features,
            show=False,
            plot_size=(12, 8)
        )
        
        plt.title(f'SHAP Values for {group_name.upper()} Tasks')

        has_freegen_feats = args.pseudo_feats_csv is not None
        # Save plot
        plt.savefig(
            figures_dir / f'shap_values_{group_name}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
        # Save SHAP values summary
        shap_summary = pd.DataFrame({
            'feature': feature_cols,
            'mean_abs_shap': np.mean(np.abs(aggregated_shap_values), axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        shap_summary.to_csv(
            output_dir / f'shap_summary_{group_name}_{args.predictor_type}_{args.metric}.csv',
            index=False
        )
        
        logging.info(f"\nSHAP Summary for {group_name}:")
        logging.info(f"Features: {feature_cols}")
        # sort by mean abs shap
        for _, row in shap_summary.sort_values("mean_abs_shap", ascending=False).iterrows():
            logging.info(f"{row['feature']}: {row['mean_abs_shap']:.4f}")
            
def compile_per_model_predictions(all_results: Dict) -> pd.DataFrame:
    """
    Compiles predictions and true scores into a per-model format
    Returns pd.DataFrame with columns [y_col, Model, True, Predicted]
    """
    records = []
    
    for task, results in all_results.items():
        benchmark, setting = task.rsplit('_', 1)
        task_name = f"{benchmark}_{setting}"
        
        # Get predictions, true values, and model IDs from results
        predictions = results['predictions']
        true_values = results['true_values']
        model_ids = results['model_ids']
        
        for model_id, true_val, pred_val in zip(model_ids, true_values, predictions):
            record = {
                "y_col": task_name,
                "Model": model_id,
                "True": true_val,
                "Predicted": pred_val
            }
            records.append(record)
    
    return pd.DataFrame(records)

def remove_extrapolation_data(features: pd.DataFrame, labels: pd.Series) -> Tuple:
    # First ensure indices match by aligning the data
    features, labels = features.align(labels, join='inner', axis=0)
    
    # Now find largest models
    largest_idx = features["total_params"].nlargest(5).index
    
    largest_model_features = features.loc[largest_idx]
    largest_model_labels = labels.loc[largest_idx]

    features = features.drop(largest_idx)
    labels = labels.drop(largest_idx)

    return features, labels, largest_model_features, largest_model_labels

def merge_tasks(all_results_df, combine_method='average'):
    """
    Merge related tasks in results dataframe while preserving statistical information.
    
    Args:
        all_results_df: DataFrame with columns [task, feature_group, p_value, effect_size]
        combine_method: How to combine p-values within groups ('average' or 'fisher')
                      Note: Fisher's method assumes independence between tests
    
    Returns:
        DataFrame with merged tasks and corrected p-values
    
    Statistical Notes:
    - Independence: Tests may not be fully independent as they use same models
    - Multiple Testing: FDR correction applied globally after merging
    - Effect Sizes: Simple averaging used (could be weighted by sample size)
    """
    merged_df = all_results_df.copy()
    
    def combine_task_rows(task_rows, new_task_name):
        merged_stats = []
        for group in task_rows['feature_group'].unique():
            group_rows = task_rows[task_rows['feature_group'] == group]
            
            # Average effect sizes
            mean_effect = group_rows['effect_size'].mean()
            
            # Combine p-values based on method
            if combine_method == 'fisher':
                # Fisher's method: -2 * sum(ln(p))
                p_values = group_rows['p_value'].values
                fisher_stat = -2 * np.sum(np.log(p_values))
                combined_p = 1 - stats.chi2.cdf(fisher_stat, df=2*len(p_values))
                mean_p = combined_p
            else:  # average
                mean_p = group_rows['p_value'].mean()
            
            merged_stats.append({
                'task': new_task_name,
                'feature_group': group,
                'p_value': mean_p,
                'effect_size': mean_effect,
                'n_subtasks': len(group_rows),
                'combination_method': combine_method
            })
        return pd.DataFrame(merged_stats)
    
    # Merge MMLU tasks by shot count
    mmlu_mask = merged_df['task'].str.contains('mmlu_|hendrycksTest-', na=False)
    if mmlu_mask.any():
        mmlu_tasks = merged_df[mmlu_mask].copy()
        non_mmlu_tasks = merged_df[~mmlu_mask].copy()
        
        # Split by shot setting
        mmlu_0_shot = mmlu_tasks[mmlu_tasks['task'].str.contains('0-shot', na=False)]
        mmlu_5_shot = mmlu_tasks[mmlu_tasks['task'].str.contains('5-shot', na=False)]
        
        merged_rows = []
        if len(mmlu_0_shot) > 0:
            merged_rows.append(combine_task_rows(mmlu_0_shot, 'mmlu_0_shot'))
        if len(mmlu_5_shot) > 0:
            merged_rows.append(combine_task_rows(mmlu_5_shot, 'mmlu_5_shot'))
            
        merged_df = pd.concat([non_mmlu_tasks] + merged_rows, ignore_index=True)
    
    # Handle TruthfulQA variants
    truthfulqa_mask = merged_df['task'].str.contains('truthfulqa_mc[12]', na=False)
    if truthfulqa_mask.any():
        truthfulqa_tasks = merged_df[truthfulqa_mask]
        non_truthfulqa = merged_df[~truthfulqa_mask]
        merged_truthfulqa = combine_task_rows(truthfulqa_tasks, 'truthfulqa')
        merged_df = pd.concat([non_truthfulqa, merged_truthfulqa], ignore_index=True)
    
    # Handle Lambada variants
    lambada_mask = merged_df['task'].str.contains('lambada_', na=False)
    if lambada_mask.any():
        lambada_tasks = merged_df[lambada_mask]
        non_lambada = merged_df[~lambada_mask]
        merged_lambada = combine_task_rows(lambada_tasks, 'lambada')
        merged_df = pd.concat([non_lambada, merged_lambada], ignore_index=True)
    
    # Handle GSM8k variants
    gsm8k_mask = merged_df['task'].str.contains('gsm8k', na=False)
    if gsm8k_mask.any():
        gsm8k_tasks = merged_df[gsm8k_mask]
        non_gsm8k = merged_df[~gsm8k_mask]
        merged_gsm8k = combine_task_rows(gsm8k_tasks, 'gsm8k')
        merged_df = pd.concat([non_gsm8k, merged_gsm8k], ignore_index=True)
    
    # Fix arc:challenge naming
    merged_df['task'] = merged_df['task'].replace('arc:challenge', 'arc_challenge')
    
    # Apply FDR correction globally
    _, corrected_p = fdrcorrection(merged_df['p_value'].values)
    merged_df['p_value_corrected'] = corrected_p
    merged_df['significant'] = merged_df['p_value_corrected'] < 0.05
    
    return merged_df.sort_values(['task', 'feature_group']).reset_index(drop=True)

def print_merged_analysis(merged_df):
    """Print summary of merged statistical analysis with detailed statistical information."""
    print("\n=== Statistical Analysis Summary ===")
    print("Note: Results should be interpreted with caution due to:")
    print("- Potential dependence between tests (same models across tasks)")
    print("- Multiple comparison correction using FDR (less conservative than Bonferroni)")
    print("- Effect sizes averaged across tasks (may not be directly comparable)")
    
    # Print significant findings
    print("\nSignificant findings (q < 0.05):")
    sig_results = merged_df[merged_df['significant']]
    
    if len(sig_results) == 0:
        print("No significant results after FDR correction")
    else:
        for task in sig_results['task'].unique():
            print(f"\n{task}:")
            task_results = sig_results[sig_results['task'] == task]
            
            for _, row in task_results.iterrows():
                effect_dir = "improvement" if row['effect_size'] > 0 else "degradation"
                method = row.get('combination_method', 'none')
                print(f"  {row['feature_group']}:")
                print(f"    raw p = {row['p_value']:.4f} (combined using {method})")
                print(f"    FDR corrected p = {row['p_value_corrected']:.4f}")
                print(f"    effect size = {row['effect_size']:.4f} ({effect_dir})")
                if 'n_subtasks' in row:
                    print(f"    combined from {row['n_subtasks']} subtasks")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total tests performed: {len(merged_df)}")
    print(f"Significant results: {sum(merged_df['significant'])} ({sum(merged_df['significant'])/len(merged_df):.1%})")
    
    # Print detailed results by task
    print("\nDetailed results by task:")
    for task in merged_df['task'].unique():
        print(f"\n=== {task} ===")
        task_results = merged_df[merged_df['task'] == task]
        
        for _, row in task_results.iterrows():
            sig = "significant" if row['significant'] else "not significant"
            effect_dir = "improvement" if row['effect_size'] > 0 else "degradation"
            method = row.get('combination_method', 'none')
            print(f"{row['feature_group']}:")
            print(f"  raw p = {row['p_value']:.4f} (combined using {method})")
            print(f"  FDR corrected p = {row['p_value_corrected']:.4f} - {sig}")
            print(f"  effect size = {row['effect_size']:.4f} ({effect_dir})")
            if 'n_subtasks' in row:
                print(f"  combined from {row['n_subtasks']} subtasks")

def evaluate_feature_groups(df, labels, task, setting, args, base_features=['id'], initial_features=None):
    """
    Evaluate either feature groups or initial features against baseline.
    If initial_features provided, only test those vs scaling laws baseline.
    """
    scaling_laws = ['total_params', 'pretraining_summary_total_tokens_billions']
    baseline_features = scaling_laws + base_features
    print(f"\nEvaluating baseline (scaling laws) features...")
    
    # Get baseline performance
    baseline_maes = []
    for seed in tqdm(range(NUM_SEEDS)):
        features_df = df[baseline_features].copy()
        fold_results = train_and_evaluate_w_search(
            features_df, labels, args,
            seed=seed,
            task=task,
            setting=setting,
            verbose=False
        )
        baseline_maes.append(np.mean(fold_results['mae']))
        print(f"Seed: {seed}, MAE: {baseline_maes[-1]:.4f}")
    
    formatted_results = {
        'benchmark': task,
        'setting': setting,
        'scaling_laws_mae': np.mean(baseline_maes),
        'scaling_laws_ci': stats.t.interval(0.95, len(baseline_maes)-1,
                                          loc=np.mean(baseline_maes),
                                          scale=stats.sem(baseline_maes))[1] - np.mean(baseline_maes)
    }
    print(f"MAE for scaling laws: {formatted_results['scaling_laws_mae']:.4f} (CI: {formatted_results['scaling_laws_ci']:.4f})")

    # Test either initial features or all feature groups
    print("\nEvaluating feature set...")
    all_maes = []
    
    if initial_features:
        # Use provided features
        test_features = base_features + initial_features
        print(f"Testing initial features: {initial_features}")
    else:
        # Use all feature groups
        feature_groups = {
            'scaling_laws': scaling_laws,
            'data_composition': [
                'pretraining_summary_percentage_web',
                'pretraining_summary_percentage_books',
                'pretraining_summary_percentage_code',
                'pretraining_summary_percentage_reference',
                'pretraining_summary_percentage_academic',
                'pretraining_summary_percentage_english'
            ],
            'architecture': [
                'dimension',
                'num_heads',
                'sequence_length',
                'mlp_ratio',
                'activation',
                'attention_variant',
                'positional_embeddings',
                'biases'
            ],
        }
        test_features = base_features + [f for group in feature_groups.values() for f in group]
        print(f"Testing all feature groups")
    
    for seed in tqdm(range(NUM_SEEDS)):
        features_df = df[test_features].copy()
        fold_results = train_and_evaluate_w_search(
            features_df, labels, args,
            seed=seed,
            task=task,
            setting=setting,
            verbose=False
        )
        all_maes.append(np.mean(fold_results['mae']))
        print(f"Seed: {seed}, MAE: {all_maes[-1]:.4f}")

    mean_mae = np.mean(all_maes)
    ci = stats.t.interval(0.95, len(all_maes)-1,
                         loc=mean_mae,
                         scale=stats.sem(all_maes))[1] - mean_mae
    
    t_stat, p_value = stats.ttest_rel(baseline_maes, all_maes)
    
    formatted_results.update({
        'all_mae': mean_mae,
        'all_ci': ci,
        'all_p_raw': p_value,
        'features_tested': initial_features if initial_features else 'all_groups'
    })

    print(f"MAE for feature set: {mean_mae:.4f} (CI: {ci:.4f})")
    print(f"Raw p-value: {p_value:.4f}")
    
    return formatted_results

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    
    # Load data from DuckDB
    df = load_data_from_db(args.db_path, args.predictor_type, args.metric, args.drop_instruction_tuned)
    df = standardize_task_names(df)
    if args.pseudo_feats_csv:
        pseudo_feats = pd.read_csv(args.pseudo_feats_csv)
        pseudo_feats.drop_duplicates(subset="id", inplace=True)
        
        models_to_drop = [
            "Dampish/StellarX-4B-V0",
            "bigscience/bloom-1b7",
            "bigscience/bloom-7b1",
            "rinna/bilingual-gpt-neox-4b",
            "mosaicml/mpt-7b-storywriter",
            "openlm-research/open_llama_7b",
            "EleutherAI/pythia-14m",    
            "facebook/xglm-2.9B",
            "facebook/xglm-7.5B",
            "facebook/xglm-564M"
        ]
        #pseudo_feats = pseudo_feats[~pseudo_feats["id"].isin(models_to_drop)]
        cols_from_freegens = list(pseudo_feats.columns)
        # keep non instruct tuned
        non_instruct = df["id"].unique()
        #pseudo_feats = pseudo_feats[pseudo_feats["id"].isin(non_instruct)]
        #existing_ids = set(pseudo_feats["id"])
        #missing_ids = set(non_instruct) - set(existing_ids)
        #empty_rows = pd.DataFrame({
        #    "id": list(missing_ids),
        #    **{col: np.nan for col in df.columns if col != "id"}
        #})
        #pseudo_feats = pd.concat([pseudo_feats, empty_rows], ignore_index=True)

        #models_with_pseudo_feats = set(pseudo_feats["id"])

        df = df.merge(pseudo_feats, on="id", how="left")
        pseudo_feats = pseudo_feats[cols_from_freegens]

        #pseudo_feats.to_csv("/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/tagged_feats.csv", index=False)
        # TEMP: drop models without pseudo features
        #df = df.dropna(subset=[col for col in pseudo_feats.columns if col != "id"])
    
    # Preprocess data
    pseudo_feats_lst = [] if not args.pseudo_feats_csv else list(set(pseudo_feats.columns) - set(["id"]))
    df, enc = preprocess_data(df, args.predictor_type, pseudo_feats=pseudo_feats_lst, use_freegens_only=args.pseudo_feats_only)
    df = aggregate_multi_part_evals(df)
    df = df.astype({col: 'float64' for col in df.columns if col not in ['benchmark', 'setting', 'id']})


    #df = feat_transform(df)
    # Get unique task/setting combinations
    task_settings = df.groupby(['benchmark', 'setting']).size().reset_index()

    if args.test_significance:
        # Statistical significance testing pathway
        all_results = []
        all_p_values = []
        task_group_pairs = []

        for _, row in task_settings.iterrows():
            task = row['benchmark']
            setting = row['setting']


            if args.selected_task and task != args.selected_task:
                continue
            if args.selected_setting and setting != args.selected_setting:
                continue

            print(f"=== Evaluating {task}_{setting} ===")
            features, labels = prepare_task_data(df, task, setting)


            if len(features) < MIN_SAMPLES:
                logging.warning(f"Skipping {task}_{setting} due to insufficient samples")
                continue

            mask = labels.notnull()
            features = features[mask]
            labels = labels[mask]
            feat_transform(features)


            # Load task-specific features if provided
            if args.initial_features_csv:
                initial_features = get_initial_features(args.initial_features_csv, task, setting)
            else:
                initial_features = args.initial_features

            results = evaluate_feature_groups(features, labels, task, setting, args, initial_features=initial_features)
            all_results.append(results)
            
            # Collect p-values (only 'all' now since we're not doing ablations)
            all_p_values.append(results['all_p_raw'])
            task_group_pairs.append((task, 'all'))
        
        # Apply FDR correction
        _, corrected_p_values = fdrcorrection(all_p_values)
        for result, (task, _), p_corrected in zip(all_results, task_group_pairs, corrected_p_values):
            result['all_p_corrected'] = p_corrected
        
        results_df = pd.DataFrame(all_results)
        output_name = f"significance_test_results_{args.metric}_303"
        if args.selected_task and args.selected_setting:
            output_name += f"_{args.selected_task}_{args.selected_setting}"
        output_name += ".csv"
        results_df.to_csv(output_name, index=False)

    else:
        # Original forward selection pathway
        results = run_forward_selection_experiments(args)
        analysis = analyze_forward_selection_results(results)

if __name__ == "__main__":
    main()