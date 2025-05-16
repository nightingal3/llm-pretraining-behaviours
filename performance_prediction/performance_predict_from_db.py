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
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
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
import matplotlib.collections as mcoll
from matplotlib.patches import Rectangle
import logging
import joblib
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from collections import defaultdict
from sklearn.inspection import PartialDependenceDisplay
from shap.plots import colors

from common_args import add_common_args, load_data
from metadata.duckdb.model_metadata_db import AnalysisStore

MIN_SAMPLES = 30
BENCHMARK_DEFAULTS = {
    "arc_challenge": ["25-shot"],
    "hellaswag": ["10-shot"],
    "mmlu": ["0-shot", "5-shot"],
    "truthfulqa": ["0-shot"],
    "winogrande": ["5-shot"],
    "lambada": ["0-shot"],
    "gsm8k": ["5-shot"],
    "humaneval": ["0-shot"],
    #'arithmetic': ['5-shot'],
    #'minerva': ['5-shot'],
    "mathqa": ["0-shot"],
    "xnli": ["0-shot"],
    "anli": ["0-shot"],
    "logiqa2": ["0-shot"],
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
        choices=["xgboost", "lgbm", "rf"],
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
        "--predictor_type",
        type=str,
        choices=["scaling_laws", "all", "non_scaling_laws"],
        default="all",
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
        "--sel_features",
        nargs="+",
        help="The features to use for the model",
    )
    parser.add_argument(
        "--sel_tasks",
        nargs="+",
        help="The tasks to use for the model",
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
    tables = store.con.execute(
        """
        SELECT name FROM sqlite_master 
        WHERE type='table';
    """
    ).fetchall()

    # Print table counts
    for table in tables:
        count = store.con.execute(
            f"""
            SELECT COUNT(*) FROM {table[0]}
        """
        ).fetchone()[0]
        print(f"{table[0]}: {count} rows")

    store.con.close()


def load_data_from_db(
    db_path: str, predictor_type: str, metric: str, drop_instruction_tuned: bool = False
) -> pd.DataFrame:
    """Load and join data from DuckDB database"""

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
    store.con.close()

    if predictor_type == "non_scaling_laws":
        df = df.drop(
            columns=["total_params", "pretraining_summary_total_tokens_billions"]
        )
        # optional
        df = df.drop(columns=["dimension", "sequence_length", "num_heads"])

    if drop_instruction_tuned:
        df = df[df["is_instruction_tuned"] != True]
    df = df.drop(columns=["is_instruction_tuned"])
    return df


def preprocess_data(
    df: pd.DataFrame,
    predictor_type: str,
    missing_val: int = -1,
    pseudo_feats: List = [],
    use_freegens_only: bool = False,
) -> pd.DataFrame:
    """Preprocess the data based on predictor type."""
    if use_freegens_only:
        # Use only pseudo (free generation) features
        freegen_columns = [col for col in pseudo_feats if col != "id"]
        # freegen_columns = ["edu_classifier_mean"]
        df = df[["id"] + freegen_columns + ["value", "benchmark", "setting"]]

        return df, None
    else:
        if predictor_type == "scaling_laws":
            # Keep only essential columns for scaling laws analysis
            cols_to_keep = [
                "total_params",
                "pretraining_summary_total_tokens_billions",
                "benchmark",
                "setting",
                "value",
                "id",
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
                "positional_embeddings",
            ]

            numeric_variables = [
                "total_params",
                "dimension",
                "num_heads",
                "mlp_ratio",
                "sequence_length",
                "pretraining_summary_total_tokens_billions",
                "pretraining_summary_percentage_web",
                "pretraining_summary_percentage_code",
                "pretraining_summary_percentage_books",
                "pretraining_summary_percentage_reference",
                "pretraining_summary_percentage_academic",
                "pretraining_summary_percentage_english",
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

            df = df[
                numeric_variables
                + categorical_variables
                + ["benchmark", "setting", "value", "id"]
            ].copy()

            for num in numeric_variables:
                df[num] = pd.to_numeric(df[num], errors="coerce").fillna(missing_val)

            encoders = {}
            for cat in categorical_variables:
                df[cat] = df[cat].fillna("___MISSING___")
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                df[cat] = enc.fit_transform(df[[cat]])

                # Identify the encoded value for the placeholder and replace it with -1
                placeholder_code = np.where(enc.categories_[0] == "___MISSING___")[0][0]
                df[cat] = np.where(df[cat] == placeholder_code, -1, df[cat])
                encoders[cat] = enc

            df = df.astype(
                {
                    col: "float32"
                    for col in df.columns
                    if col not in ["benchmark", "setting", "id"]
                }
            )

            return df, encoders


def prepare_task_data(
    df: pd.DataFrame, task: str, setting: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for a specific task and setting"""
    task_data = df[(df["benchmark"] == task) & (df["setting"] == setting)].copy()

    # Drop non-feature columns
    feature_df = task_data.drop(
        columns=[
            "benchmark",
            "setting",
            "value",
        ]
    )

    labels = task_data["value"]

    return feature_df, labels


def median_baseline(train_labels, test_feats):
    median = train_labels.median()
    return np.full(len(test_feats), median)


def standardize_task_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize duplicate task names and remove duplicates from name variations"""
    df = df.copy()

    # Convert hendrycksTest to mmlu
    hendrycks_mask = df["benchmark"].str.startswith("hendrycksTest-")
    if hendrycks_mask.any():
        df.loc[hendrycks_mask, "benchmark"] = df.loc[
            hendrycks_mask, "benchmark"
        ].str.replace("hendrycksTest-", "mmlu_")

    # Fix ARC challenge naming
    df.loc[df["benchmark"] == "arc:challenge", "benchmark"] = "arc_challenge"

    # Remove duplicates, keeping first occurrence
    df = df.drop_duplicates(subset=["id", "benchmark", "setting"], keep="first")

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
    extrapolation_feats: Optional[pd.DataFrame] = None,
    extrapolation_labels: Optional[pd.Series] = None,
    verbose: bool = True,
    **kwargs,
) -> Dict:
    """Train model and evaluate performance with nested CV for hyperparameter tuning."""
    # kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    results = {
        "mae": [],
        "predictions": [],
        "true_values": [],
        "feature_importance": [],
        "all_mae_median_baseline": [],
        "all_mae_loglinear_baseline": [],
        "model_ids": [],
        "all_shap_values": [],
        "test_features": [],
        "task_signed_errors": {},  # Store signed errors per model
        "task_absolute_errors": {},  # Store absolute errors per model
    }

    param_grids = {
        "xgboost": {
            "max_depth": [2, 3, 5],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [50, 100],
        },
        "lgbm": {
            "max_depth": [2, 3, 5],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [50, 100, 200],
        },
        "rf": {
            "max_depth": [2, 3, 5],
            "n_estimators": [50, 100],
            "min_samples_split": [2, 5, 10],
            "max_features": ["auto", "sqrt"],
        },
    }
    # Parameter grid for tuning
    # TODO: two different param grids for sl/all?
    # param_grid = {
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.01, 0.1, 0.3],
    #     'n_estimators': [500, 1000],
    #     'subsample': [0.8, 1],
    #     'min_child_weight': [1, 3],
    # }

    if args.regressor == "lgbm":
        # doesn't allow missing vals, change back to nan
        features.replace(-1, np.nan, inplace=True)

    for train_idx, test_idx in kf.split(features):
        # Split data
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        X_train_model_names, X_test_model_names = X_train["id"], X_test["id"]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed
        )

        X_train_orig = X_train.copy()
        X_test_orig = X_test.copy()

        X_train = X_train.drop(columns=["id"])
        X_val = X_val.drop(columns=["id"])
        X_test = X_test.drop(columns=["id"])

        if (
            "total_params" in X_train_orig.columns
            and "pretraining_summary_total_tokens_billions" in X_train_orig.columns
        ):
            # Create log features
            log_features = ["total_params", "pretraining_summary_total_tokens_billions"]
            X_train_log = X_train_orig[log_features].copy()
            X_test_log = X_test_orig[log_features].copy()

            # Fit linear regression on log features
            log_linear_model = LinearRegression()
            log_linear_model.fit(X_train_log, y_train)

            # Make predictions
            log_linear_preds = log_linear_model.predict(X_test_log)
            log_linear_mae = mean_absolute_error(y_test, log_linear_preds)
            results["all_mae_loglinear_baseline"].append(log_linear_mae)

        # Inner CV for hyperparameter tuning
        param_grid = param_grids[args.regressor]
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=seed)

        if args.regressor == "xgboost":
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                enable_categorical=True,
                missing=missing_val,
                random_state=seed,
            )
        elif args.regressor == "lgbm":
            model = lgb.LGBMRegressor(
                objective="regression",
                random_state=seed,
                importance_type="gain",
                min_data_in_leaf=1,
                verbosity=-1,
            )
        elif args.regressor == "rf":
            model = RandomForestRegressor(random_state=seed, n_jobs=-1)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_kf,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )

        if args.regressor == "xgboost":
            grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        best_params = grid_search.best_params_

        if verbose:
            print(f"Best params for task {task}, setting {setting}: {best_params}")

        # Train with best hyperparameters

        if args.regressor == "xgboost":
            tuned_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                enable_categorical=True,
                missing=missing_val,
                random_state=seed,
                **best_params,
            )
        elif args.regressor == "lgbm":
            tuned_model = lgb.LGBMRegressor(
                objective="regression",
                random_state=seed,
                min_data_in_leaf=1,
                verbosity=-1,
                **best_params,
            )
        elif args.regressor == "rf":
            tuned_model = RandomForestRegressor(
                random_state=seed, n_jobs=-1, **best_params
            )

        tuned_model.fit(X_train, y_train)
        # valid_idx = X_train["pretraining_summary_percentage_code"] > -1

        # PartialDependenceDisplay.from_estimator(tuned_model, X_train[valid_idx], ["pretraining_summary_percentage_code"])
        # plt.savefig("partial_dependence_code_lambada.png")
        # assert False

        # Evaluate predictions
        preds = tuned_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        # print(f"mae in fold: {mae}")
        results["mae"].append(mae)
        results["predictions"].extend(preds)
        results["true_values"].extend(y_test)
        results["model_ids"].extend(X_test_model_names)

        # Feature importance
        importance = tuned_model.feature_importances_
        results["feature_importance"].append(importance)

        # Median baseline
        median_predictions = np.full(len(y_test), y_train.median())
        mae_median = mean_absolute_error(y_test, median_predictions)
        results["all_mae_median_baseline"].append(mae_median)

        # SHAP values
        if args.regressor == "xgboost":
            explainer = shap.Explainer(tuned_model)
            shap_values = explainer(X_test)
            results["all_shap_values"].append(shap_values.values)
            results["test_features"].append(X_test)
        elif args.regressor == "lgbm":
            explainer = shap.TreeExplainer(tuned_model)
            shap_values = explainer(X_test)
            results["all_shap_values"].append(shap_values.values)
            results["test_features"].append(X_test)

        # Calculate errors per model
        signed_errors = {
            name: pred - true
            for name, pred, true in zip(X_test_model_names, preds, y_test)
        }
        absolute_errors = {name: abs(error) for name, error in signed_errors.items()}

        # Store errors
        results["task_signed_errors"].update(signed_errors)
        results["task_absolute_errors"].update(absolute_errors)

        if verbose and "all_mae_loglinear_baseline" in results:
            print(
                f"Log-linear baseline MAE: {np.mean(results['all_mae_loglinear_baseline']):.4f}"
            )
            print(
                f"Median baseline MAE: {np.mean(results['all_mae_median_baseline']):.4f}"
            )
            print(f"Main model MAE: {np.mean(results['mae']):.4f}")

    # Save errors for analysis
    error_df = pd.DataFrame(
        [
            {"Model": model, "SignedError": signed_error, "AbsoluteError": abs_error}
            for model, signed_error, abs_error in zip(
                results["task_signed_errors"].keys(),
                results["task_signed_errors"].values(),
                results["task_absolute_errors"].values(),
            )
        ]
    )
    if verbose:
        error_df = error_df.sort_values(by="SignedError", ascending=True)
        output_dir = Path(f"./performance_prediction/errors/{args.metric}_searched")
        output_dir.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(
            output_dir / f"{args.regressor}_{args.predictor_type}.csv", index=False
        )
        print(f"Saved error details for metric {args.metric} to {output_dir}")

    synthetic_data = pd.DataFrame(
        {
            "total_params": np.log(
                np.linspace(1e8, 1e12, 100)
            ),  # Log-transformed params
            "pretraining_summary_total_tokens_billions": np.log(
                np.linspace(1e2, 1e5, 100)
            ),  # Log-transformed tokens
        }
    )

    # Predict on synthetic data
    # TODO: create a better version of this check later
    # breakpoint()
    # synthetic_preds = tuned_model.predict(synthetic_data)

    # # Plot predictions vs. num_params and num_tokens
    # plt.figure(figsize=(12, 6))

    # # Predictions vs. num_params
    # plt.subplot(1, 2, 1)
    # plt.plot(synthetic_data["total_params"], synthetic_preds, label="Predictions")
    # plt.xlabel("Number of Parameters")
    # plt.ylabel("Predicted Metric")
    # plt.title("Predictions vs. Number of Parameters")
    # plt.legend()

    # # Predictions vs. num_tokens
    # plt.subplot(1, 2, 2)
    # plt.plot(synthetic_data["pretraining_summary_total_tokens_billions"], synthetic_preds, label="Predictions")
    # plt.xlabel("Number of Tokens")
    # plt.ylabel("Predicted Metric")
    # plt.title("Predictions vs. Number of Tokens")
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig("overfitting_check.png")
    # breakpoint()
    return results


def quick_grid_search(features: pd.DataFrame, labels: pd.Series) -> Dict:
    """Quick grid search with caching per task"""
    cache_dir = Path("./hyperparam_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"sl_params.joblib"

    if cache_file.exists():
        return joblib.load(cache_file)

    param_grid = {
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [50, 100],
    }

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        enable_categorical=True,
        missing=-1,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # Reduced from 5 for speed
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    grid_search.fit(features, labels)
    results = {
        "best_params": grid_search.best_params_,
        "best_score": -grid_search.best_score_,
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
        "mae": [],
        "predictions": [],
        "true_values": [],
        "feature_importance": [],
        "all_mae_median_baseline": [],
        "model_ids": [],
        "all_shap_values": [],
        "test_features": [],
        "task_signed_errors": {},  # Store signed errors per model
        "task_absolute_errors": {},  # Store absolute errors per model
    }

    # Check for missing values
    check_missing_values(features)

    for train_idx, test_idx in kf.split(features):
        # Split data
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        X_train_model_names, X_test_model_names = X_train["id"], X_test["id"]

        # Drop the 'id' column
        X_train = X_train.drop(columns=["id"])
        X_test = X_test.drop(columns=["id"])

        # Train baseline (median) model for comparison
        median_predictions = median_baseline(y_train, y_test)
        mae_median = mean_absolute_error(y_test, median_predictions)
        results["all_mae_median_baseline"].append(mae_median)

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
        results["mae"].append(mae)
        results["predictions"].extend(preds)
        results["true_values"].extend(y_test)
        results["model_ids"].extend(X_test_model_names)

        # Feature importance and SHAP values (if applicable)
        if args.regressor == "xgboost":
            importance = model.feature_importances_

            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            results["all_shap_values"].append(shap_values.values)
            results["test_features"].append(X_test)
        else:
            importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=args.seed
            ).importances_mean

        results["feature_importance"].append(importance)

        # Calculate per-model signed and absolute errors
        signed_errors = {
            name: pred - true
            for name, pred, true in zip(X_test_model_names, preds, y_test)
        }
        absolute_errors = {name: abs(error) for name, error in signed_errors.items()}

        # Store errors
        results["task_signed_errors"].update(signed_errors)
        results["task_absolute_errors"].update(absolute_errors)

    # Save errors for analysis
    error_df = pd.DataFrame(
        [
            {"Model": model, "SignedError": signed_error, "AbsoluteError": abs_error}
            for model, signed_error, abs_error in zip(
                results["task_signed_errors"].keys(),
                results["task_signed_errors"].values(),
                results["task_absolute_errors"].values(),
            )
        ]
    )
    if verbose:
        error_df = error_df.sort_values(by="SignedError", ascending=True)
        output_dir = Path(f"./performance_prediction/errors/{args.metric}")
        output_dir.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(
            output_dir / f"{args.regressor}_{args.predictor_type}.csv", index=False
        )
        print(f"Saved error details for metric {args.metric} to {output_dir}")
        return results


def get_agg_benchmark(row):
    benchmark = row["benchmark"]
    setting = row["setting"]

    # Mapping for known multi-part names.
    mapping = {
        "truthfulqa_mc1": "truthfulqa",
        "truthfulqa_mc2": "truthfulqa",
        "lambada_standard": "lambada",
        "lambada_openai": "lambada",
        "gsm8k": "gsm8k",
        "gsm8k_cot": "gsm8k",
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
    df["aggregated_benchmark"] = df.apply(get_agg_benchmark, axis=1)

    # Drop rows that did not meet the default shot criteria.
    df = df[df["aggregated_benchmark"].notnull()].copy()

    # For downstream processing, rename 'aggregated_benchmark' to 'benchmark' if desired.
    df["benchmark"] = df["aggregated_benchmark"]

    # Group by model id, benchmark, and setting to average the metric value.
    agg_dict = {"value": "mean"}
    if "metric_stderr" in df.columns:
        agg_dict["metric_stderr"] = "mean"

    group_cols = ["id", "benchmark", "setting"]
    other_cols = [
        col for col in df.columns if col not in group_cols + ["value", "metric_stderr"]
    ]
    for col in other_cols:
        agg_dict[col] = "first"

    # Group and aggregate while retaining all columns
    df_agg = df.groupby(group_cols, as_index=False).agg(agg_dict)

    df_agg = df_agg.drop(columns=["aggregated_benchmark"])
    return df_agg


def postprocess_results(df_results: Dict, args) -> Dict:
    if args.merge_mmlu:
        mmlu_tasks = df_results[
            df_results["task"].str.startswith("hendrycksTest-")
            | df_results["task"].str.startswith("mmlu_")
        ]

        mmlu_0_shot = mmlu_tasks.loc[mmlu_tasks["setting"] == "0-shot"]
        mmlu_5_shot = mmlu_tasks.loc[mmlu_tasks["setting"] == "5-shot"]

        # delete all the individual mmlu tasks
        df_results = df_results[
            ~df_results["task"].str.startswith("hendrycksTest")
            & ~df_results["task"].str.startswith("mmlu_")
        ]

        # divide into 5 shot/0 shot
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["mmlu_0_shot"],
                        "mae": [mmlu_0_shot["mae"].mean()],
                        "std_mae": [mmlu_0_shot["std_mae"].mean()],
                        "med_baseline_mae": [mmlu_0_shot["med_baseline_mae"].mean()],
                        "loglinear_baseline_mae": [
                            mmlu_0_shot["loglinear_baseline_mae"].mean()
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["mmlu_5_shot"],
                        "mae": [mmlu_5_shot["mae"].mean()],
                        "std_mae": [mmlu_5_shot["std_mae"].mean()],
                        "med_baseline_mae": [mmlu_5_shot["med_baseline_mae"].mean()],
                        "loglinear_baseline_mae": [
                            mmlu_5_shot["loglinear_baseline_mae"].mean()
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

    if args.merge_arithmetic:
        arithmetic_tasks = df_results[df_results["task"].str.startswith("arithmetic_")]
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["arithmetic"],
                        "mae": [arithmetic_tasks["mae"].mean()],
                        "std_mae": [arithmetic_tasks["std_mae"].mean()],
                        "med_baseline_mae": [
                            arithmetic_tasks["med_baseline_mae"].mean()
                        ],
                    }
                ),
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
        mae_overall = (
            truthfulqa_mc1["mae"].iloc[0] + truthfulqa_mc2["mae"].iloc[0]
        ) / 2
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["truthfulqa_mc"],
                        "mae": [mae_overall],
                        "std_mae": [
                            (
                                truthfulqa_mc1["std_mae"].iloc[0]
                                + truthfulqa_mc2["std_mae"].iloc[0]
                            )
                            / 2
                        ],
                        "med_baseline_mae": [
                            (
                                truthfulqa_mc1["med_baseline_mae"].iloc[0]
                                + truthfulqa_mc2["med_baseline_mae"].iloc[0]
                            )
                            / 2
                        ],
                    }
                ),
            ]
        )
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
                        "med_baseline_mae": [
                            minerva_math_tasks["med_baseline_mae"].mean()
                        ],
                    }
                ),
            ]
        )
        # delete all the individual minerva math tasks
        df_results = df_results[~df_results["task"].str.startswith("minerva_math_")]
    # merge lambada_standard and lambada_openai
    if "lambada_standard" in df_results["task"].values:
        lambada_standard = df_results[df_results["task"] == "lambada_standard"]
        lambada_openai = df_results[df_results["task"] == "lambada_openai"]
        mae_overall = (
            lambada_standard["mae"].iloc[0] + lambada_openai["mae"].iloc[0]
        ) / 2
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["lambada"],
                        "mae": [mae_overall],
                        "std_mae": [
                            (
                                lambada_standard["std_mae"].iloc[0]
                                + lambada_openai["std_mae"].iloc[0]
                            )
                            / 2
                        ],
                        "med_baseline_mae": [
                            (
                                lambada_standard["med_baseline_mae"].iloc[0]
                                + lambada_openai["med_baseline_mae"].iloc[0]
                            )
                            / 2
                        ],
                    }
                ),
            ]
        )
        df_results = df_results[df_results["task"] != "lambada_standard"]
        df_results = df_results[df_results["task"] != "lambada_openai"]
    if (
        "gsm8k" in df_results["task"].values
        and "gsm8k_cot" in df_results["task"].values
    ):
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
                        "std_mae": [
                            (gsm8k["std_mae"].iloc[0] + gsm8k_cot["std_mae"].iloc[0])
                            / 2
                        ],
                        "med_baseline_mae": [
                            (
                                gsm8k["med_baseline_mae"].iloc[0]
                                + gsm8k_cot["med_baseline_mae"].iloc[0]
                            )
                            / 2
                        ],
                    }
                ),
            ]
        )
        df_results = df_results[df_results["task"] != "gsm8k"]
        df_results = df_results[df_results["task"] != "gsm8k_cot"]
    # merge ANLI
    if "anli_r1" in df_results["task"].values:
        anli_r1 = df_results[df_results["task"] == "anli_r1"]
        anli_r2 = df_results[df_results["task"] == "anli_r2"]
        anli_r3 = df_results[df_results["task"] == "anli_r3"]
        mae_overall = (
            anli_r1["mae"].iloc[0] + anli_r2["mae"].iloc[0] + anli_r3["mae"].iloc[0]
        ) / 3
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["anli"],
                        "mae": [mae_overall],
                        "std_mae": [
                            (
                                anli_r1["std_mae"].iloc[0]
                                + anli_r2["std_mae"].iloc[0]
                                + anli_r3["std_mae"].iloc[0]
                            )
                            / 3
                        ],
                        "med_baseline_mae": [
                            (
                                anli_r1["med_baseline_mae"].iloc[0]
                                + anli_r2["med_baseline_mae"].iloc[0]
                                + anli_r3["med_baseline_mae"].iloc[0]
                            )
                            / 3
                        ],
                    }
                ),
            ]
        )
        df_results = df_results[df_results["task"] != "anli_r1"]
        df_results = df_results[df_results["task"] != "anli_r2"]
        df_results = df_results[df_results["task"] != "anli_r3"]
    # merge xnli into english and other langs
    if "xnli_ar" in df_results["task"].values:
        xnli_en = df_results[df_results["task"] == "xnli_en"]
        xnli_non_en = df_results[
            (df_results["task"] != "xnli_en")
            & (df_results["task"].str.startswith("xnli_"))
        ]

        xnli_non_en_mae = xnli_non_en["mae"].mean()
        # delete all but xnli_en
        df_results = df_results[
            (~df_results["task"].str.startswith("xnli_"))
            | (df_results["task"] == "xnli_en")
        ]
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["xnli_non_en"],
                        "mae": [xnli_non_en_mae],
                        "std_mae": [xnli_non_en["std_mae"].mean()],
                        "med_baseline_mae": [xnli_non_en["med_baseline_mae"].mean()],
                    }
                ),
            ]
        )

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


def save_raw_shap_values(all_shap_values, test_features, group_name, output_dir, args):
    # Concatenate all SHAP values and features - no abs() here to preserve signs
    aggregated_shap = np.concatenate(all_shap_values, axis=0)
    aggregated_features = pd.concat(test_features, ignore_index=True)

    # Create DataFrame with features and their raw (signed) SHAP values
    shap_df = pd.DataFrame(
        aggregated_shap, columns=[f"shap_{col}" for col in aggregated_features.columns]
    )
    feature_df = pd.DataFrame(aggregated_features)

    combined_df = pd.concat([feature_df, shap_df], axis=1)

    # Maybe add some basic stats about negative vs positive values
    shap_stats = pd.DataFrame(
        {
            "feature": aggregated_features.columns,
            "mean_shap": np.mean(
                aggregated_shap, axis=0
            ),  # Regular mean to see direction
            "mean_abs_shap": np.mean(np.abs(aggregated_shap), axis=0),  # Magnitude
            "neg_count": (aggregated_shap < 0).sum(axis=0),  # Count of negative values
            "pos_count": (aggregated_shap > 0).sum(axis=0),  # Count of positive values
        }
    )

    # Save both raw values and summary stats
    has_freegen_feats = args.pseudo_feats_csv is not None
    combined_df.to_csv(
        output_dir
        / f"raw_shap_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}_303.csv",
        index=False,
    )
    shap_stats.to_csv(
        output_dir
        / f"shap_stats_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}_303.csv",
        index=False,
    )

    print(f"Saved raw SHAP values for {group_name} to {output_dir}")


def annotate_shap_summary(ax, encoder, threshold=0.05, fontsize=8, color="red"):
    # Print the encoder's categories
    print("Encoder categories:", encoder.categories_[0])

    # Reconstruct mapping: encoded integer -> original category
    mapping = {i: cat for i, cat in enumerate(encoder.categories_[0])}
    print("Mapping:", mapping)

    # Loop through scatter collections in the plot
    for coll in ax.collections:
        if isinstance(coll, mcoll.PathCollection):
            offsets = coll.get_offsets()  # array of [x, y] coordinates
            for x, y in offsets:
                try:
                    # Print what we're trying to map
                    encoded_val = int(round(x))
                    print(f"Trying to map x={x} (rounded to {encoded_val}) to category")
                    # Look up the original category label
                    label = mapping.get(encoded_val, str(encoded_val))
                    print(f"Got label: {label}")
                    if abs(y) > threshold:
                        ax.annotate(
                            label,
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 5),
                            ha="center",
                            fontsize=fontsize,
                            color=color,
                        )
                except Exception as e:
                    print(f"Error processing point ({x}, {y}): {e}")


def plot_task_group_shap_values_simple(
    all_results: dict, features: pd.DataFrame, args, output_dir, encoders: dict = None
):
    """Plot SHAP values for each major task group."""

    figures_dir = output_dir / "figures_02_13"
    figures_dir.mkdir(exist_ok=True)

    # Decide which columns to include
    if args.predictor_type == "scaling_laws":
        feature_cols = ["total_params", "pretraining_summary_total_tokens_billions"]
    else:
        exclude_cols = ["id", "benchmark", "setting", "value", "value_stderr"]
        feature_cols = [col for col in features.columns if col not in exclude_cols]

    READABLE_FEATURE_NAMES = {
        "pretraining_summary_total_tokens_billions": "[D] Total Tokens (B)",
        "total_params": "[A] Total Parameters",
        "pretraining_summary_percentage_code": "[D] % Code in Pretraining",
        "pretraining_summary_percentage_web": "[D] % Web in Pretraining",
        "question_words_ratio": "[F] Question Words Ratio",
        "dimension": "[A] Dimension",
        "layer_norm_type": "[A] LayerNorm",
        "pretraining_summary_percentage_academic": "[D] % Academic in Pretraining",
        "pretraining_summary_percentage_reference": "[D] % Reference in Pretraining",
        "positional_embeddings": "[A] Positional Embeddings",
        "pct_english_mean": "[F] % English Generated",
        "pretraining_summary_percentage_books": "[D] % Books in Pretraining",
        "domain_web_pct_mean": "[F] % Generated Weblike Text",
        "entropy_mean": "[F] Bigram Entropy",
        "dep_parse_dep_root_dist_max_mean": "[F] Max Dependency Length in generations (mean)",
        "pretraining_summary_percentage_english": "[D] % English in Pretraining",
        "imperative_verbs_ratio": "[F] Imperative Verbs Generated",
        "biases": "[A] Biases",
        "domain_reference_pct_mean": "[F] % Generated Reference Text",
        "instructions_words_ratio": "[F] Instruction-formatted Words Ratio",
        "sequence_length": "[A] Sequence Length",
    }
    # For each 'task group'
    for task_name, results in all_results.items():
        group_name = task_name

        if "all_shap_values" not in results:
            logging.warning(f"No SHAP values found for task group: {group_name}")
            continue

        all_shap_values = results["all_shap_values"]
        test_features_list = results["test_features"]

        if not all_shap_values:
            logging.warning(f"No SHAP values found for task group: {group_name}")
            continue

        # Aggregate
        aggregated_shap_values = np.concatenate(all_shap_values, axis=0)
        aggregated_test_features = pd.concat(test_features_list, ignore_index=True)

        if len(all_shap_values) > 0:
            save_raw_shap_values(
                all_shap_values, test_features_list, group_name, figures_dir, args
            )

        # Build a "valid" array (no -1 or NaNs)
        valid_data = {}
        valid_shap = {}
        for f_idx, f_name in enumerate(feature_cols):
            mask = (aggregated_test_features[f_name] != -1) & aggregated_test_features[
                f_name
            ].notna()
            if mask.any():
                valid_data[f_name] = aggregated_test_features.loc[mask, f_name].values
                valid_shap[f_name] = aggregated_shap_values[mask, f_idx]

        # Reconstruct arrays
        valid_features = np.zeros_like(
            aggregated_test_features[feature_cols].values, dtype=float
        )
        valid_shaps = np.zeros_like(aggregated_shap_values, dtype=float)
        for i, f_name in enumerate(feature_cols):
            if f_name in valid_data:
                mask = (
                    aggregated_test_features[f_name] != -1
                ) & aggregated_test_features[f_name].notna()
                valid_features[mask, i] = valid_data[f_name]
                valid_shaps[mask, i] = valid_shap[f_name]

        if valid_shaps.shape[1] != len(feature_cols):
            logging.error(f"Shape mismatch for {group_name}.")
            continue

        # Turn into a DataFrame for SHAP
        df_for_shap = pd.DataFrame(valid_features, columns=feature_cols)

        # Inverse-transform so categorical columns are 'object' -> SHAP colors them gray
        if encoders:
            for feat, encoder in encoders.items():
                if feat in feature_cols:
                    codes = df_for_shap[feat].values.reshape(-1, 1)
                    inverted_2d = encoder.inverse_transform(codes)
                    df_for_shap[feat] = inverted_2d[:, 0].astype(object)

        plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(12, 8))

        # 1) Let shap.summary_plot draw the standard beeswarm (numeric in red-blue, categorical in gray).
        readable_names = [READABLE_FEATURE_NAMES.get(col, col) for col in feature_cols]
        shap.summary_plot(
            valid_shaps,
            df_for_shap,
            feature_names=readable_names,
            plot_type="dot",  # ensure beeswarm
            show=False,
        )
        ax = plt.gca()

        ############################################################################
        # 2) Replicate summary_legacy's default "feature order": sorted by sum of abs SHAP
        ############################################################################
        # By default summary_legacy sorts ascending, so the bottom row is the
        # smallest mean(|SHAP|), top row is the largest. We replicate that:
        abs_means = np.sum(np.abs(valid_shaps), axis=0)  # sum across samples
        # Indices of features in ascending order of importance:
        feature_order = np.argsort(abs_means)

        # If you only want the top N features, mimic summary_legacy's max_display=20 behavior:
        max_display = min(20, len(feature_cols))
        feature_order = feature_order[-max_display:]  # keep the largest 20
        # summary_legacy draws them in ascending order from bottom->top
        # so the bottom row is feature_order[0], next row is feature_order[1], etc.

        ############################################################################
        # 3) For each categorical column, replicate the "vertical jitter" so we can
        #    overlay a second scatter with custom discrete colors.
        ############################################################################
        # Let's pick a distinct color map for categories:
        discrete_cmap = plt.cm.get_cmap("Set3")

        # We'll do one big dictionary: {feat_name: {category_label: color}}
        # so "feat_name" can have multiple categories each with a unique color.
        cat_color_map = {}

        if encoders:
            for feat, enc in encoders.items():
                if feat in feature_cols:
                    # Build a color dict for each category
                    categories = enc.categories_[0]
                    categories = [c for c in categories if c != "___MISSING___"]
                    feat_colors = {}
                    for i_cat, cat_str in enumerate(categories):
                        color = discrete_cmap(i_cat / max(1, len(categories) - 1))
                        feat_colors[cat_str] = color
                    cat_color_map[feat] = feat_colors

        # We'll define a helper function that does the same binning+jitter as summary_legacy:
        def compute_jitter(shaps_array, rng_seed=0):
            # same approach: random shuffle -> bin by percentile -> stack
            nbins = 100
            # We do a stable random shuffle so repeated calls match
            rng = np.random.RandomState(rng_seed)
            inds = np.arange(len(shaps_array))
            rng.shuffle(inds)
            shaps_array = shaps_array[inds]

            q = np.round(
                nbins
                * (shaps_array - shaps_array.min())
                / (shaps_array.max() - shaps_array.min() + 1e-8)
            )
            sort_order = np.argsort(q + rng.randn(len(q)) * 1e-6)

            layer = 0
            last_bin = -1
            ys = np.zeros_like(shaps_array)
            for idx in sort_order:
                if q[idx] != last_bin:
                    layer = 0
                ys[idx] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = q[idx]

            # scale to ~0.4 row height
            row_height = 0.4
            if np.max(np.abs(ys)) < 1e-8:
                return inds, ys  # no variation => all zero
            ys *= 0.9 * (row_height / np.max(np.abs(ys) + 1e-9))

            return inds, ys

        # Now overlay a second scatter for each "object-dtype" feature
        for pos_idx, f_idx in enumerate(feature_order):
            feat_name = feature_cols[f_idx]
            # is this feature truly categorical?
            if df_for_shap[feat_name].dtype != object:
                continue
            # get that column's shap + data
            col_shaps = valid_shaps[:, f_idx]
            col_data = df_for_shap[feat_name].values  # strings like "none","some"...

            # compute the jitter exactly like summary_legacy
            inds, ys = compute_jitter(col_shaps, rng_seed=42)
            col_shaps = col_shaps[inds]
            col_data = col_data[inds]

            # the row in the beeswarm for this feature is pos_idx (from the bottom)
            # so final y = pos_idx + ys[i]
            for i_samp in range(len(col_shaps)):
                sample_cat = col_data[i_samp]
                if sample_cat == "___MISSING___":
                    continue
                # pick color from cat_color_map if possible, else default
                cdict = cat_color_map.get(feat_name, {})
                color = cdict.get(sample_cat, "#333333")  # fallback gray
                ax.scatter(
                    col_shaps[i_samp],
                    pos_idx + ys[i_samp],
                    s=30,
                    color=color,
                    alpha=1.0,
                    zorder=10,  # on top
                    edgecolors="none",
                )

        ############################################################################
        # 4) Build a discrete legend for categories (optional)
        ############################################################################
        handles, labels = [], []
        for feat, feat_colors in cat_color_map.items():
            # "header" for that feature
            handles.append(Rectangle((0, 0), 0, 0, fill=False, ec="none", fc="none"))
            readable_name = READABLE_FEATURE_NAMES.get(feat, feat)
            labels.append(f"\n{readable_name}")
            # each category => color square
            for cat_str, c in feat_colors.items():
                handles.append(Rectangle((0, 0), 1, 1, fc=c, ec="none"))
                labels.append(cat_str)

        if handles:
            plt.legend(
                handles,
                labels,
                bbox_to_anchor=(1.35, 0.7),
                loc="upper left",
                borderaxespad=0.0,
                fontsize=12,
            )

        plt.title(group_name)
        # plt.tight_layout()

        # Save figure
        has_freegen_feats = args.pseudo_feats_csv is not None
        out_png = (
            figures_dir
            / f"shap_values_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}_303.png"
        )
        out_pdf = (
            figures_dir
            / f"shap_values_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}_303.pdf"
        )
        out_eps = (
            figures_dir
            / f"shap_values_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}_303.eps"
        )

        plt.savefig(out_png, bbox_inches="tight", dpi=300)
        plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
        plt.savefig(out_eps, bbox_inches="tight")
        plt.close()

        # Save a summary table
        shap_summary = pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_shap": np.mean(aggregated_shap_values, axis=0),
            }
        ).sort_values("mean_shap", ascending=False)

        shap_summary.to_csv(
            output_dir
            / f"shap_summary_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}.csv",
            index=False,
        )
        logging.info(f"\nSHAP Summary for {group_name}:")
        logging.info(f"Features: {feature_cols}")
        for _, row in shap_summary.iterrows():
            logging.info(f"{row['feature']}: {row['mean_shap']:.4f}")


def _plot_task_group_shap_values_simple_old(
    all_results: Dict,
    features: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    encoders: Dict[str, OrdinalEncoder] = None,
):
    """Plot SHAP values for each major task group"""

    # Define task groups
    task_groups = {
        "arc_challenge": ["arc_challenge"],
        "anli": ["anli"],
        "drop": ["drop"],
        "gsm8k": ["gsm8k", "gsm8k_cot"],
        "hellaswag": ["hellaswag"],
        "winogrande": ["winogrande"],
        #'mmlu_0-shot': [t for t in all_results.keys() if 'mmlu_' in t or 'hendrycksTest-' in t and '0-shot' in t],
        #'mmlu_5-shot': [t for t in all_results.keys() if 'mmlu_' in t or 'hendrycksTest-' in t and '5-shot' in t],
        "arithmetic": [t for t in all_results.keys() if "arithmetic_" in t],
        "truthfulqa": ["truthfulqa_mc1", "truthfulqa_mc2"],
        "minerva_math": [t for t in all_results.keys() if "minerva_math_" in t],
        "lambada": ["lambada"],
        "humaneval": ["humaneval"],
    }

    figures_dir = output_dir / "figures_02_13"
    figures_dir.mkdir(exist_ok=True)

    # Get feature columns based on predictor type
    if args.predictor_type == "scaling_laws":
        feature_cols = ["total_params", "pretraining_summary_total_tokens_billions"]
    else:
        feature_cols = [
            col
            for col in features.columns
            if col not in ["id", "benchmark", "setting", "value", "value_stderr"]
        ]

    feature_matrix = features[feature_cols].copy()

    # Process each task group
    # for group_name, task_list in task_groups.items():
    for task_name in all_results.keys():
        all_shap_values = []
        test_features_list = []
        task_list = [task_name]
        group_name = task_name

        # Collect SHAP values for all tasks in this group
        for task, results in all_results.items():
            if any(task.startswith(t) for t in task_list):
                if "all_shap_values" in results:
                    all_shap_values.extend(results["all_shap_values"])
                    test_features_list.extend(results["test_features"])

        if not all_shap_values:
            logging.warning(f"No SHAP values found for task group: {group_name}")
            continue

        # Aggregate SHAP values for this group
        aggregated_shap_values = np.concatenate(all_shap_values, axis=0)
        aggregated_test_features = pd.concat(test_features_list, ignore_index=True)

        if len(all_shap_values) > 0:
            save_raw_shap_values(
                all_shap_values, test_features_list, group_name, output_dir, args
            )

        # Create masks for valid values for each feature
        valid_data = {}
        valid_shap = {}

        for feature_idx, feature_name in enumerate(feature_cols):
            # Create mask for valid values (not -1 and not NA)
            feature_mask = (
                aggregated_test_features[feature_name] != -1
            ) & aggregated_test_features[feature_name].notna()

            if feature_mask.any():
                valid_data[feature_name] = aggregated_test_features.loc[
                    feature_mask, feature_name
                ].values
                valid_shap[feature_name] = aggregated_shap_values[
                    feature_mask, feature_idx
                ]

        # Convert back to format needed for summary plot
        valid_features = np.zeros_like(aggregated_test_features.values)
        valid_shaps = np.zeros_like(aggregated_shap_values)

        for idx, feature_name in enumerate(feature_cols):
            if feature_name in valid_data:
                mask = (
                    aggregated_test_features[feature_name] != -1
                ) & aggregated_test_features[feature_name].notna()
                valid_features[mask, idx] = valid_data[feature_name]
                valid_shaps[mask, idx] = valid_shap[feature_name]

        # Verify shapes match
        if aggregated_shap_values.shape[1] != len(feature_cols):
            logging.error(
                f"Shape mismatch for {group_name}: SHAP values shape {aggregated_shap_values.shape} vs features shape {feature_matrix.shape}"
            )
            continue

        # Create plot
        plt.rcParams.update({"font.size": 18})

        plt.figure(figsize=(12, 8))

        cmap = colors.red_blue

        shap.summary_plot(
            valid_shaps,
            valid_features,
            feature_names=feature_cols,
            cmap=cmap,
            show=False,  # because we want to customize the plot
        )

        ax = plt.gca()

        # Create legends for categorical features
        # Create legends for categorical features
    if encoders:
        legend_handles = []
        legend_labels = []

        for feat, encoder in encoders.items():
            # Only build a legend if this feature is actually being plotted
            if feat not in feature_cols:
                continue

            # Get the column of data that SHAP is plotting
            col_idx = feature_cols.index(feat)
            col_data = valid_features[:, col_idx]  # numeric-encoded column

            # Compute the per-feature color scale (5th-95th percentile)
            vmin = np.nanpercentile(col_data, 5)
            vmax = np.nanpercentile(col_data, 95)
            if vmin > vmax:
                vmin = vmax  # edge-case fallback

            # Add a "header" entry in the legend for this feature's name
            legend_handles.append(plt.Rectangle((0, 0), 0, 0, fill=False))
            legend_labels.append(f"\n{feat}:")

            # For each category string, map it to its numeric code, then to the colormap
            categories = encoder.categories_[0]
            for cat_str in categories:
                # Convert category → numeric encoding
                cat_value = encoder.transform([[cat_str]])[0, 0]
                # Clip and normalize
                clipped = min(max(cat_value, vmin), vmax)
                norm_val = (clipped - vmin) / (vmax - vmin + 1e-8)
                # Get the color from the same cmap shap uses
                color = cmap(norm_val)

                legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, ec="none"))
                legend_labels.append(cat_str)

        # Place legend on the right side of the plot
        plt.legend(
            legend_handles,
            legend_labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=12,
        )

        plt.title(f"SHAP Values for {group_name.upper()} Tasks")
        plt.tight_layout()

        has_freegen_feats = args.pseudo_feats_csv is not None
        # Save plot
        plt.savefig(
            figures_dir
            / f"shap_values_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.savefig(
            figures_dir
            / f"shap_values_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # Save SHAP values summary
        shap_summary = pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_shap": np.mean(aggregated_shap_values, axis=0),
            }
        ).sort_values("mean_shap", ascending=False)

        shap_summary.to_csv(
            output_dir
            / f"shap_summary_{group_name}_{args.regressor}_{args.predictor_type}_{args.metric}.csv",
            index=False,
        )

        logging.info(f"\nSHAP Summary for {group_name}:")
        logging.info(f"Features: {feature_cols}")
        # sort by mean abs shap
        for _, row in shap_summary.sort_values("mean_shap", ascending=False).iterrows():
            logging.info(f"{row['feature']}: {row['mean_shap']:.4f}")


def custom_shap_summary_plot(shap_values, features, feature_names, encoders=None):
    """
    Create a SHAP-style beeswarm plot but with proper categorical feature handling
    """
    plt.figure(figsize=(12, 8))

    # Calculate feature importance
    importance = np.abs(shap_values).mean(0)
    # Sort by importance
    feature_order = np.argsort(-importance)  # - for descending order

    # Get number of features
    n_features = len(feature_names)

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    maxval = np.max(np.abs(shap_values))

    # Plot each feature
    for pos, idx in enumerate(feature_order):
        feature_name = feature_names[idx]
        values = shap_values[:, idx]

        if feature_name in encoders:
            # Categorical feature
            categories = encoders[feature_name].categories_[0]
            feature_vals = features[:, idx]

            # Use same red-blue colormap as SHAP for consistency
            colors = plt.cm.RdBu_r(np.linspace(0, 1, len(categories)))

            for cat_idx, category in enumerate(categories):
                mask = feature_vals == cat_idx
                if mask.any():
                    points = values[mask]
                    # Use SHAP-style jittering based on density
                    ys = np.ones(len(points)) * (n_features - 1 - pos)
                    ys = ys + np.random.normal(0, 0.02, size=len(points))

                    plt.scatter(
                        points,
                        ys,
                        c=[colors[cat_idx]],
                        label=category if pos == 0 else "",
                        alpha=0.5,
                        s=16,
                    )
        else:
            # Numeric feature - use regular SHAP coloring
            ys = np.ones(len(values)) * (n_features - 1 - pos)
            ys = ys + np.random.normal(0, 0.02, size=len(values))

            # Normalize feature values for coloring
            feature_vals = features[:, idx]
            norm_vals = (feature_vals - feature_vals.min()) / (
                feature_vals.max() - feature_vals.min()
            )

            plt.scatter(values, ys, c=norm_vals, cmap=plt.cm.RdBu_r, alpha=0.5, s=16)

    # Add feature names
    plt.yticks(range(n_features - 1, -1, -1), [feature_names[i] for i in feature_order])

    # Add SHAP value axis
    plt.xlabel("SHAP value (impact on model output)")

    # Add legend for categorical features only
    if encoders:
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            title="Categories",
        )

    # Add color bar for numeric features
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r)
    sm.set_array([])
    plt.colorbar(sm, label="Feature value", ax=ax)

    plt.tight_layout()
    return plt.gcf(), ax


def compile_per_model_predictions(all_results: Dict) -> pd.DataFrame:
    """
    Compiles predictions and true scores into a per-model format
    Returns pd.DataFrame with columns [y_col, Model, True, Predicted]
    """
    records = []

    for task, results in all_results.items():
        benchmark, setting = task.rsplit("_", 1)
        task_name = f"{benchmark}_{setting}"

        # Get predictions, true values, and model IDs from results
        predictions = results["predictions"]
        true_values = results["true_values"]
        model_ids = results["model_ids"]

        for model_id, true_val, pred_val in zip(model_ids, true_values, predictions):
            record = {
                "y_col": task_name,
                "Model": model_id,
                "True": true_val,
                "Predicted": pred_val,
            }
            records.append(record)

    return pd.DataFrame(records)


def remove_extrapolation_data(features: pd.DataFrame, labels: pd.Series) -> Tuple:
    # First ensure indices match by aligning the data
    features, labels = features.align(labels, join="inner", axis=0)

    # Now find largest models
    largest_idx = features["total_params"].nlargest(5).index

    largest_model_features = features.loc[largest_idx]
    largest_model_labels = labels.loc[largest_idx]

    features = features.drop(largest_idx)
    labels = labels.drop(largest_idx)

    return features, labels, largest_model_features, largest_model_labels


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)

    # Load data from DuckDB
    df = load_data_from_db(
        args.db_path, args.predictor_type, args.metric, args.drop_instruction_tuned
    )

    df = standardize_task_names(df)
    if args.pseudo_feats_csv:
        pseudo_feats = pd.read_csv(args.pseudo_feats_csv)
        pseudo_feats.drop_duplicates(subset="id", inplace=True)

        # NOTE: trying this: /home/gneubig/sand/not_just_scaling/all_models_feature_stats_2_03_with_ratios.csv
        # CURR_FREEGEN_FEATS = ["id", "edu_classifier_mean"]
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
            "facebook/xglm-564M",
        ]
        # pseudo_feats = pseudo_feats[~pseudo_feats["id"].isin(models_to_drop)]
        cols_from_freegens = list(pseudo_feats.columns)
        # keep non instruct tuned
        non_instruct = df["id"].unique()
        # pseudo_feats = pseudo_feats[pseudo_feats["id"].isin(non_instruct)]
        # existing_ids = set(pseudo_feats["id"])
        # missing_ids = set(non_instruct) - set(existing_ids)
        # empty_rows = pd.DataFrame({
        #    "id": list(missing_ids),
        #    **{col: np.nan for col in df.columns if col != "id"}
        # })
        # pseudo_feats = pd.concat([pseudo_feats, empty_rows], ignore_index=True)

        # models_with_pseudo_feats = set(pseudo_feats["id"])

        df = df.merge(pseudo_feats, on="id", how="left")
        pseudo_feats = pseudo_feats[cols_from_freegens]

        # pseudo_feats.to_csv("/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/tagged_feats.csv", index=False)
        # TEMP: drop models without pseudo features
        # df = df.dropna(subset=[col for col in pseudo_feats.columns if col != "id"])

    # Preprocess data
    pseudo_feats_lst = (
        []
        if not args.pseudo_feats_csv
        else list(set(pseudo_feats.columns) - set(["id"]))
    )
    df, enc = preprocess_data(
        df,
        args.predictor_type,
        pseudo_feats=pseudo_feats_lst,
        use_freegens_only=args.pseudo_feats_only,
    )

    # TODO: this should be an option since we may want to see individual subtasks
    df = aggregate_multi_part_evals(df)

    # NOTE: unselect this after
    # SEL_FEATURES = ['total_params', 'pretraining_summary_total_tokens_billions', 'question_words_ratio', 'layer_norm_type', 'pct_english_mean', 'positional_embeddings', 'pretraining_summary_percentage_books', 'pretraining_summary_percentage_code', 'block_type', 'id']

    # SEL_FEATURES = ['pretraining_summary_total_tokens_billions', 'total_params','question_words_ratio', 'layer_norm_type', 'dimension', 'pretraining_summary_percentage_code', 'id']
    # df = feat_transform(df)
    # Get unique task/setting combinations
    task_settings = df.groupby(["benchmark", "setting"]).size().reset_index()

    # NOTE: testing this
    # if args.predictor_type == "all" and args.pseudo_feats_csv:
    #     TOP_FEATURES = [
    #         'pretraining_summary_total_tokens_billions',
    #         'total_params',
    #         'content_function_ratio_mean',
    #         'num_heads',
    #         'const_parse_const_tree_depth_max_std',
    #         'dimension',
    #         'edu_classifier_std',
    #         'dep_parse_dep_root_dist_mean_std',
    #         'sequence_length',
    #         'unique_tokens_std',
    #         "pretraining_summary_percentage_web",
    #         "pretraining_summary_percentage_books",
    #         "pretraining_summary_percentage_code",
    #         "pretraining_summary_percentage_reference",
    #         "pretraining_summary_percentage_english",
    #     ]

    #     df = df[TOP_FEATURES + ['id', 'benchmark', 'setting', 'value']]

    all_results = {}
    errs_per_model = defaultdict(dict)

    for _, row in task_settings.iterrows():
        task = row["benchmark"]
        setting = row["setting"]

        if args.sel_tasks is not None and task not in args.sel_tasks:
            continue

        # TODO: add this back
        root_task = None
        for task_base in BENCHMARK_DEFAULTS:
            if task.startswith(task_base):
                root_task = task_base
                break
        if not root_task:
            continue
        if not setting in BENCHMARK_DEFAULTS[root_task]:
            continue
        print(f"=== TASK {task},{setting} ===")
        # Prepare data for this task/setting combination
        # exclude models from the lsit for test

        features, labels = prepare_task_data(df, task, setting)

        if len(features) < MIN_SAMPLES:
            continue

        if args.sel_features:
            features = features[args.sel_features]

        # there are some NaN values in brier score?? seems lm-eval is returning it
        mask = labels.notnull()
        features = features[mask]
        labels = labels[mask]

        # NOTE: trying excluding models
        # features = features[~features["id"].isin(models_to_exclude)]
        # labels = labels.loc[features.index]
        missing_percentage = (features == -1).sum() / len(features) * 100

        # Print the percentage of missing values per column
        print(missing_percentage)

        # features = features.drop(columns=["pretraining_summary_percentage_code", "pretraining_summary_percentage_academic", "pretraining_summary_percentage_web", "pretraining_summary_percentage_reference", "pretraining_summary_percentage_academic"])

        assert features["id"].value_counts().max() == 1, "Duplicate models found"

        print(f"Num models overall: {len(features)}")

        if not args.predictor_type == "non_scaling_laws":
            feat_transform(features)

        # if extrapolation mode, remove the 5 largest models
        if args.measure_extrapolation:
            (
                features,
                labels,
                final_test_features,
                final_test_labels,
            ) = remove_extrapolation_data(features, labels)

        if len(features) < MIN_SAMPLES:
            continue

        # Train and evaluate
        if args.hyperparam_search:
            results = train_and_evaluate_w_search(
                features,
                labels,
                args,
                task=task,
                setting=setting,
                missing_val=args.missing_val,
                seed=args.seed,
            )
        else:
            results = train_and_evaluate(
                features,
                labels,
                args,
                n_estimators=args.n_estimators,
                lr=args.lr,
                max_depth=args.max_depth,
                missing_val=args.missing_val,
                seed=args.seed,
            )

        signed_errs = []
        for pred, true in zip(results["predictions"], results["true_values"]):
            signed_err = pred - true
            signed_errs.append(signed_err)

        for i, model_id in enumerate(results["model_ids"]):
            if task not in errs_per_model[model_id]:
                errs_per_model[model_id][task] = []
            errs_per_model[model_id][task].append(signed_errs[i])

        # Store results
        result_key = f"{task}_{setting}"
        all_results[result_key] = {
            "mae": np.mean(results["mae"]),
            "mean_signed_err": np.mean(signed_errs),
            "signed_errs_per_model": [],
            "std_mae": np.std(results["mae"]),
            "predictions": results["predictions"],
            "true_values": results["true_values"],
            "feature_importance": np.mean(results["feature_importance"], axis=0),
            "median_baseline": np.mean(results["all_mae_median_baseline"]),
            "loglinear_baseline": np.mean(results["all_mae_loglinear_baseline"]),
            "all_shap_values": results["all_shap_values"],
            "test_features": results["test_features"],
            "model_ids": results["model_ids"],
        }

        logging.info(f"Task: {task}, Setting: {setting}, Metric: {args.metric}")
        logging.info(
            f"MAE: {np.mean(results['mae']):.4f} ± {np.std(results['mae']):.4f}"
        )
        logging.info(
            f"Median Baseline MAE: {np.mean(results['all_mae_median_baseline']):.4f}"
        )
        logging.info(
            f"Loglinear Baseline MAE: {np.mean(results['all_mae_loglinear_baseline']):.4f}"
        )

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
    has_freegen_feats = args.pseudo_feats_csv is not None

    model_errs_dir = Path("./performance_prediction/model_prediction_errs")
    model_errs_df.to_csv(
        f"./performance_prediction/model_prediction_errs/{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}_model_errs.csv",
        index=False,
    )

    # Save results with metric in filename
    output_dir = Path("./performance_prediction/results_db")
    output_dir.mkdir(exist_ok=True)

    per_model_predictions = compile_per_model_predictions(all_results)
    per_model_predictions.to_csv(
        output_dir
        / f"per_model_predictions_{args.regressor}_{args.predictor_type}_{args.metric}.csv",
        index=False,
    )

    results_df = pd.DataFrame(
        [
            {
                "task": task.rsplit("_", 1)[0],
                "setting": task.split("_")[-1],
                "mae": results["mae"],
                "std_mae": results["std_mae"],
                "med_baseline_mae": results["median_baseline"],
                "loglinear_baseline_mae": results["loglinear_baseline"],
            }
            for task, results in all_results.items()
        ]
    )

    results_df.to_csv(
        output_dir
        / f"performance_prediction_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}_alltasks.csv",
        index=False,
    )
    results_df = postprocess_results(results_df, args)
    results_df.to_csv(
        output_dir
        / f"performance_prediction_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}.csv",
        index=False,
    )

    logging.info(f"=== Average MAE across tasks: {results_df['mae'].mean():.4f} ===")
    logging.info(
        f"=== Median baseline MAE: {results_df['med_baseline_mae'].mean():.4f} ==="
    )
    logging.info(
        f"=== Loglinear baseline MAE: {results_df['loglinear_baseline_mae'].mean():.4f} ==="
    )

    if args.regressor == "xgboost":
        # Print average across all tasks
        feature_columns = [col for col in features.columns if col != "id"]

        all_task_importances = []
        importance_data = {"task": [], "feature": [], "importance": []}
        for task, results in all_results.items():
            all_task_importances.append(results["feature_importance"])
            task_importance = results["feature_importance"]
            for feat_name, imp in zip(feature_columns, task_importance):
                importance_data["task"].append(task)
                importance_data["feature"].append(feat_name)
                importance_data["importance"].append(imp)

        print("\nOverall average feature importances:")
        avg_importances = np.mean(all_task_importances, axis=0)
        sorted_importances = sorted(
            zip(feature_columns, avg_importances), key=lambda x: x[1], reverse=True
        )
        for fname, imp in sorted_importances:
            print(f"{fname}: {imp:.4f}")

        importance_df = pd.DataFrame(importance_data)
        # sort by data importance
        tmp = importance_df.pivot(index="task", columns="feature", values="importance")
        if "total_params" in tmp.columns:
            sorted_importance = tmp.sort_values("total_params", ascending=False)
        else:
            sorted_importance = tmp
        sorted_importance.to_csv(
            output_dir
            / f"feature_importance_{args.regressor}_{args.predictor_type}_{args.metric}_freegens_{has_freegen_feats}.csv"
        )

    # aggregating shap values
    if args.regressor == "xgboost" or args.regressor == "lgbm":
        # NOTE: trying this
        # plot_task_group_shap_values(all_results, features, args, output_dir, enc)

        # features = features[SEL_FEATURES]
        plot_task_group_shap_values_simple(all_results, features, args, output_dir, enc)


if __name__ == "__main__":
    main()
