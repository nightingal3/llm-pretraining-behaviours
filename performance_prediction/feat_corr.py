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

from common_args import add_common_args, load_data
from metadata.duckdb.model_metadata_db import AnalysisStore
from performance_predict_from_db import (
    load_data_from_db,
    get_args,
    standardize_task_names,
    preprocess_data,
)

BENCHMARK_DEFAULTS = {
    "arc_challenge": ["25-shot"],
    "hellaswag": ["10-shot"],
    "mmlu": ["5-shot"],
    "truthfulqa": ["0-shot"],
    "winogrande": ["5-shot"],
    "lambada": ["0-shot"],
    "gsm8k": ["5-shot"],
    "arithmetic": ["5-shot"],
    "minerva": ["5-shot"],
    #'mathqa': ['5-shot'],
    #'xnli': ['0-shot'],
    #'anli': '0-shot',
    #'logiqa2': '0-shot',
    #'fld': '0-shot',
    #'asdiv': '5-shot'
}
categorical_variables = [
    "activation",
    "attention_variant",
    "biases",
    "block_type",
    "layer_norm_type",
    "positional_embeddings",
    "benchmark",
    "setting",
    "value",
]

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


def calculate_correlations(
    features_df: pd.DataFrame,
    eval_pivot: pd.DataFrame,
    cols_to_correlate: list = [],
    categorical_variables: list = [
        "activation",
        "attention_variant",
        "biases",
        "block_type",
        "layer_norm_type",
        "positional_embeddings",
        "benchmark",
        "setting",
        "value",
    ],
) -> pd.DataFrame:
    """Calculate Kendall's tau correlations between features and evaluation metrics."""
    # First deduplicate features
    features_df = features_df.drop_duplicates(subset=["id"], keep="first")

    # Create averaged columns for task groups
    def add_average_columns(eval_pivot):
        # Average MMLU tasks
        mmlu_cols = [col for col in eval_pivot.columns if col[0].startswith("mmlu_")]
        if mmlu_cols:
            eval_pivot[("mmlu_average", "5-shot")] = eval_pivot[mmlu_cols].mean(axis=1)

        # Average arithmetic tasks
        arithmetic_cols = [
            col for col in eval_pivot.columns if col[0].startswith("arithmetic_")
        ]
        if arithmetic_cols:
            eval_pivot[("arithmetic_average", "5-shot")] = eval_pivot[
                arithmetic_cols
            ].mean(axis=1)

        # Average minerva math tasks
        minerva_cols = [
            col for col in eval_pivot.columns if col[0].startswith("minerva_math_")
        ]
        if minerva_cols:
            eval_pivot[("minerva_math_average", "5-shot")] = eval_pivot[
                minerva_cols
            ].mean(axis=1)

        return eval_pivot

    eval_pivot = add_average_columns(eval_pivot)

    # Get feature columns with filtering
    feature_cols = [
        col
        for col in features_df.columns
        if not col.endswith("_classified")
        and col != "id"
        and col not in categorical_variables
        and (col in cols_to_correlate if cols_to_correlate else True)
    ]

    correlations = []

    for eval_col in eval_pivot.columns:
        benchmark, setting = eval_col

        # Skip individual subtasks if we have averages
        if any(
            [
                (benchmark.startswith("mmlu_") and benchmark != "mmlu_average"),
                (
                    benchmark.startswith("arithmetic_")
                    and benchmark != "arithmetic_average"
                ),
                (
                    benchmark.startswith("minerva_math_")
                    and benchmark != "minerva_math_average"
                ),
            ]
        ):
            continue

        # Check against BENCHMARK_DEFAULTS
        root_task = next(
            (t for t in BENCHMARK_DEFAULTS if benchmark.startswith(t)), None
        )
        if not root_task or setting not in BENCHMARK_DEFAULTS[root_task]:
            continue

        eval_values = eval_pivot[eval_col].dropna()

        for feature in feature_cols:
            # Align features and labels after deduplication
            feature_values = features_df.set_index("id")[feature]
            common_indices = feature_values.index.intersection(eval_values.index)

            if len(common_indices) > 5:
                # Get values as arrays after alignment
                x = feature_values.loc[common_indices].values
                y = eval_values.loc[common_indices].values

                # Create mask on numpy arrays
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]

                if len(x) > 5 and not (x == x[0]).all() and not (y == y[0]).all():
                    tau, p_value = stats.kendalltau(x, y)
                    correlations.append(
                        {
                            "feature": feature,
                            "benchmark": benchmark,
                            "setting": setting,
                            "correlation": tau,
                            "p_value": p_value,
                            "n_samples": len(x),
                        }
                    )

    # Convert to dataframe and sort
    correlations_df = pd.DataFrame(correlations)
    if not correlations_df.empty:
        correlations_df["abs_correlation"] = abs(correlations_df["correlation"])
        return correlations_df.sort_values("abs_correlation", ascending=False)
    return pd.DataFrame()


def main():
    args = get_args()

    # Load data from DuckDB using your existing structure
    df = load_data_from_db(
        args.db_path, args.predictor_type, args.metric, args.drop_instruction_tuned
    )
    df = standardize_task_names(df)
    df = df[~df["id"].isin(models_to_drop)]

    # Add pseudo features if specified
    if args.pseudo_feats_csv:
        pseudo_feats = pd.read_csv(args.pseudo_feats_csv)
        df = df.merge(pseudo_feats, on="id", how="left")

    # Preprocess data using your existing pipeline
    pseudo_feats_lst = (
        []
        if not args.pseudo_feats_csv
        else list(set(pseudo_feats.columns) - set(["id"]))
    )
    df, _ = preprocess_data(
        df,
        args.predictor_type,
        pseudo_feats=pseudo_feats_lst,
        use_freegens_only=args.pseudo_feats_only,
    )

    # Create evaluation pivot table
    eval_pivot = pd.pivot_table(
        df, values="value", index="id", columns=["benchmark", "setting"], aggfunc="mean"
    )
    freegen_cols = [col for col in pseudo_feats.columns if col != "id"]
    # Calculate correlations
    breakpoint()
    correlations_df = calculate_correlations(
        df, eval_pivot, cols_to_correlate=freegen_cols
    )

    # Post-process and save results
    significant_corrs = correlations_df[correlations_df["p_value"] < 0.05]

    significant_corrs.to_csv("benchmark_corrs.csv", index=False)


if __name__ == "__main__":
    main()
