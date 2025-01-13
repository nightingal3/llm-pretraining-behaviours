import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
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

from common_args import add_common_args, load_data
from metadata.duckdb.model_metadata_db import AnalysisStore

MIN_SAMPLES = 10

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
        "--predictor_type", type=str, choices=["scaling_laws", "all"], default="all"
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

    args = parser.parse_args()

    assert args.n_estimators > 0, "Number of trees must be greater than 0"
    assert args.lr > 0, "Learning rate must be greater than 0"
    assert args.max_depth > 0, "Max depth must be greater than 0"
    if not (args.model_feats or args.data_feats):
        raise ValueError("Please provide either model_feats or data_feats")

    return args

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

def load_data_from_db(db_path: str, predictor_type: str, metric: str) -> pd.DataFrame:
    """Load and join data from DuckDB database"""
    
    store = AnalysisStore.from_existing(db_path)
    
    if predictor_type == "scaling_laws":
        query = """
            SELECT 
                m.id,
                m.total_params,
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
    store.con.close()
    return df


def process_data(dataset: pd.DataFrame, args: argparse.Namespace):
    cols_to_drop = [
        "assigned person",
        "notes",
        "link to instruction/sft data",
        "instruction/sft data",
        "base model",
        "pretraining data",
        "is_preference_tuned",
        "merged",
        "link to pretraining data",
    ]

    categorical_variables = [
        "activation",
        "attention_variant",
        "batch_instances",
        "biases",
        "block_type",
        "layer_norm_type",
        "positional_embeddings",
    ]

    for col in cols_to_drop:
        if col in dataset.columns:
            dataset = dataset.drop(columns=[col])

    if args.drop_instruction_tuned:
        dataset = dataset[dataset["is_instruction_tuned"] != True]
        dataset = dataset.drop(columns=["is_instruction_tuned"])
    if args.predictor_type == "scaling_laws":
        # drop all but total params and num tokens
        dataset = dataset[
            [
                "total_params",
                "pretraining_summary_total_tokens_billions",
                "id",
            ]
            + list(cols_from_results)
        ]
        categorical_variables = []

    if args.predictor_type == "all":
        if "is_instruction_tuned" in dataset.columns:
            dataset["is_instruction_tuned"] = dataset["is_instruction_tuned"].map(
                {True: 1, False: 0, np.nan: -1}
            )

        for var in categorical_variables:
            dataset[var] = dataset[var].astype("category")

    return dataset



def preprocess_data(df: pd.DataFrame, predictor_type: str, missing_val: int = -1, pseudo_feats: List = []) -> pd.DataFrame:
    """Preprocess the data based on predictor type."""
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
            'batch_tokens',
            'pretraining_summary_total_tokens_billions',
            'pretraining_summary_percentage_web',
            'pretraining_summary_percentage_code',
            'pretraining_summary_percentage_books',
            'pretraining_summary_percentage_reference',
            'pretraining_summary_percentage_academic',
            'pretraining_summary_percentage_english'
        ]

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
    """Standardize duplicate task names while preserving domains"""
    df = df.copy()
    
    # Convert hendrycksTest to mmlu
    hendrycks_mask = df['benchmark'].str.startswith('hendrycksTest-')
    if hendrycks_mask.any():
        df.loc[hendrycks_mask, 'benchmark'] = df.loc[hendrycks_mask, 'benchmark'].str.replace('hendrycksTest-', 'mmlu_')
    
    # Fix ARC challenge naming
    df.loc[df['benchmark'] == 'arc:challenge', 'benchmark'] = 'arc_challenge'
    
    return df

def check_missing_values(df):
    for col in df.columns:
        n_nan = df[col].isna().sum()
        n_minus_one = (df[col] == -1).sum()
        if n_nan > 0 or n_minus_one > 0:
            print(f"{col}: {n_nan} NaN, {n_minus_one} -1 values")

def train_and_evaluate_w_search(
    features: pd.DataFrame,
    labels: pd.Series,
    args: argparse.Namespace,
    task: str,
    setting: str,
    missing_val: float = -1,
    seed: int = 42,
    **kwargs,
) -> Dict:
    """Train model and evaluate performance with nested CV for hyperparameter tuning"""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    results = {
        'mae': [],
        'predictions': [],
        'true_values': [],
        'feature_importance': [],
        'all_mae_median_baseline': [],
        'model_ids': []
    }

    param_grid = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200]
    }

    for train_idx, test_idx in kf.split(features):
        # Split data
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        X_train = X_train.drop(columns=['id'])
        X_test = X_test.drop(columns=['id'])

        # Inner CV for hyperparameter tuning
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=seed)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            enable_categorical=True,
            missing=missing_val,
            random_state=seed
        )
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_kf,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best params for task {task}, setting {setting}: {best_params}")
        # Train with best hyperparameters
        tuned_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            enable_categorical=True,
            missing=missing_val,
            random_state=seed,
            **best_params
        )
        tuned_model.fit(X_train, y_train)

        # Evaluate
        preds = tuned_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        results['mae'].append(mae)
        results['predictions'].extend(preds)
        results['true_values'].extend(y_test)

        # Feature importance
        importance = tuned_model.feature_importances_
        results['feature_importance'].append(importance)

        # Median baseline
        median_predictions = np.full(len(y_test), y_train.median())
        mae_median = mean_absolute_error(y_test, median_predictions)
        results['all_mae_median_baseline'].append(mae_median)

    return results

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
    lr: int = 0.1,
    max_depth: int = 10,
    missing_val: float = -1,
    seed: int = 42,
    **kwargs,
) -> Dict:
    """Train model and evaluate performance"""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    results = {
        'mae': [],
        'predictions': [],
        'true_values': [],
        'feature_importance': [],
        'all_mae_median_baseline': [],
        'model_ids': []
    }

    # check that we don't have both -1 and nan
    check_missing_values(features)
    
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        X_train_model_names, X_test_model_names = X_train['id'], X_test['id']
        all_model_ids = set(features['id'])
        logging.info(f"Total unique models for task: {len(all_model_ids)}")
        X_train = X_train.drop(columns=['id'])
        X_test = X_test.drop(columns=['id'])


        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        
        #results = quick_grid_search(X_train, y_train)
        #breakpoint()
        #selected_cols = ["total_params", "pretraining_summary_total_tokens_billions"
        #]
        #X_train = X_train[selected_cols]
        #X_test = X_test[selected_cols]
        #breakpoint()

        median_predictions = median_baseline(y_train, y_test)
        mae_median = mean_absolute_error(y_test, median_predictions)
        results['all_mae_median_baseline'].append(mae_median)

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
            model = xgb.XGBRegressor(
                **kwargs
            )
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

        # Get feature importance
        if args.regressor == "xgboost":
            importance = model.feature_importances_
        else:
            importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=args.seed
            ).importances_mean
        
        results['feature_importance'].append(importance)
    
    return results

def postprocess_results(df_results: Dict, args) -> Dict:
    if args.merge_mmlu:
        mmlu_tasks = df_results[df_results["task"].str.startswith("hendrycksTest-") | df_results["task"].str.startswith("mmlu_")]
        df_results = pd.concat(
            [df_results, pd.DataFrame({"task": ["mmlu"], "mae": [mmlu_tasks["mae"].mean()]})]
        )
        # delete all the individual mmlu tasks
        df_results = df_results[~df_results["task"].str.startswith("hendrycksTest")& ~df_results["task"].str.startswith("mmlu_")]
    if args.merge_arithmetic:
        arithmetic_tasks = df_results[df_results["task"].str.startswith("arithmetic_")]
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame({"task": ["arithmetic"], "mae": [arithmetic_tasks["mae"].mean()]}),
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
                    }
                ),
            ]
        )
        # delete all the individual minerva math tasks
        df_results = df_results[
            ~df_results["task"].str.startswith("minerva_math_")
        ]
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

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    
    # Load data from DuckDB
    df = load_data_from_db(args.db_path, args.predictor_type, args.metric)
    df = standardize_task_names(df)
    if args.pseudo_feats_csv:
        pseudo_feats = pd.read_csv(args.pseudo_feats_csv)
        df = df.merge(pseudo_feats, on="id", how="left")
    
    # Preprocess data
    pseudo_feats_lst = [] if not args.pseudo_feats_csv else list(set(pseudo_feats.columns) - set(["id"]))
    df, enc = preprocess_data(df, args.predictor_type, pseudo_feats=pseudo_feats_lst)

    #df = feat_transform(df)
    # Get unique task/setting combinations
    task_settings = df.groupby(['benchmark', 'setting']).size().reset_index()
    
    all_results = {}
    errs_per_model = defaultdict(dict)

    for _, row in task_settings.iterrows():
        task = row['benchmark']
        setting = row['setting']
        
        # Prepare data for this task/setting combination
        features, labels = prepare_task_data(df, task, setting)
        
        if len(features) < MIN_SAMPLES:
            continue
            
        # Train and evaluate
        if args.hyperparam_search:
            results = train_and_evaluate_w_search(features, labels, args, task=task, setting=setting, missing_val=args.missing_val)
        else:
            results = train_and_evaluate(features, labels, args, n_estimators=args.n_estimators, lr=args.lr, max_depth=args.max_depth, missing_val=args.missing_val)
        
        signed_errs = []
        for pred, true in zip(results['predictions'], results['true_values']):
            signed_err = pred - true
            signed_errs.append(signed_err)
        
        for i, model_id in enumerate(results['model_ids']):    
            if task not in errs_per_model[model_id]:
                errs_per_model[model_id][task] = []
            errs_per_model[model_id][task].append(signed_errs[i])

        # Store results
        result_key = f"{task}_{setting}"
        all_results[result_key] = {
            'mae': np.mean(results['mae']),
            'mean_signed_err': np.mean(signed_errs),
            'signed_errs_per_model': [],
            'std_mae': np.std(results['mae']),
            'predictions': results['predictions'],
            'true_values': results['true_values'],
            'feature_importance': np.mean(results['feature_importance'], axis=0),
            'median_baseline': np.mean(results['all_mae_median_baseline'])
        }
        
        logging.info(f"Task: {task}, Setting: {setting}, Metric: {args.metric}")
        logging.info(f"MAE: {np.mean(results['mae']):.4f} ± {np.std(results['mae']):.4f}")
        logging.info(f"Median Baseline MAE: {np.mean(results['all_mae_median_baseline']):.4f}")

        
    # save errors per model and rank models by signed err
    model_errs_dict = {
        "model_id": [],
        "mean_signed_err": [],
    }
    for model_id, task_dict in errs_per_model.items():
        mean_err = np.mean([np.mean(errs) for errs in task_dict.values()])
        model_errs_dict["model_id"].append(model_id)
        model_errs_dict["mean_signed_err"].append(mean_err)
    model_errs_df = pd.DataFrame(model_errs_dict)
    model_errs_df = model_errs_df.sort_values("mean_signed_err")

    model_errs_dir = Path("./performance_prediction/model_prediction_errs")
    model_errs_dir.mkdir(exist_ok=True)
    model_errs_df.to_csv(f"./performance_prediction/model_prediction_errs/{args.predictor_type}_{args.metric}_model_errs.csv", index=False)
        
    # Save results with metric in filename
    output_dir = Path("./performance_prediction/results")
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame([
        {
            'task': task.rsplit('_', 1)[0],
            'setting': task.split('_')[-1],
            'mae': results['mae'],
            'std_mae': results['std_mae'],
            'med_baseline_mae': results['median_baseline']
        }
        for task, results in all_results.items()
    ])

    results_df = postprocess_results(results_df, args)
    results_df.to_csv(output_dir / f"performance_prediction_{args.predictor_type}_{args.metric}.csv", index=False)

    logging.info(f"=== Average MAE across tasks: {results_df['mae'].mean():.4f} ===")
    logging.info(f"=== Median baseline MAE: {results_df['med_baseline_mae'].mean():.4f} ===")
    
    if args.regressor == "xgboost":
        # Print average across all tasks
        all_task_importances = []
        importance_data = {
            "task": [],
            "feature": [],
            "importance": []
        }
        for task, results in all_results.items():
            all_task_importances.append(results['feature_importance'])
            task_importance = results['feature_importance']
            for feat_name, imp in zip(features.columns, task_importance):
                importance_data["task"].append(task)
                importance_data["feature"].append(feat_name)
                importance_data["importance"].append(imp)

        print("\nOverall average feature importances:")
        avg_importances = np.mean(all_task_importances, axis=0)
        for fname, imp in zip(features.columns, avg_importances):
            print(f"{fname}: {imp:.4f}")

        importance_df = pd.DataFrame(importance_data)
        # sort by data importance
        tmp = importance_df.pivot(index="task", columns="feature", values="importance")
        sorted_importance = tmp.sort_values("total_params", ascending=False)
        sorted_importance.to_csv(output_dir / f"feature_importance_{args.predictor_type}_{args.metric}.csv")
    assert False
    # Plot feature importance
    if args.regressor == "xgboost":
        for task, results in all_results.items():
            plt.figure(figsize=(10, 6))
            feature_importance = pd.Series(
                results['feature_importance'],
                index=features.columns
            ).sort_values(ascending=True)
            
            feature_importance.plot(kind='barh')
            plt.title(f"Feature Importance for {task}")
            plt.tight_layout()
            plt.savefig(output_dir / f"feature_importance_{task}_{args.predictor_type}.png")
            plt.close()

if __name__ == "__main__":
    main()