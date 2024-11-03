import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import gaussian_process
from sklearn.preprocessing import OrdinalEncoder
import argparse
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    KFold,
    GridSearchCV,
)
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
from datasets import Dataset
import logging
import joblib


# args shared across perf prediction scripts
from common_args import add_common_args, load_data

# min samples to consider a task
MIN_SAMPLES = 10


def fit_regressor(reg, train_feats, train_labels):
    reg.fit(train_feats, train_labels)
    return reg


def median_baseline(train_labels, test_feats):
    median = train_labels.median()
    return np.full(len(test_feats), median)


def train_regressor(
    train_feats,
    train_labels,
    regressor="xgboost",
    quantile=0.95,
    verbose=False,
    lr=0.1,
    max_depth=10,
    n_estimators=100,
    missing_val=-1,
    seed=42,
    **kwargs,
):
    logging.debug(
        f"Training a {regressor} model with training data of shape {train_feats.shape}."
    )
    train_feats = preprocess_features(train_feats)

    kwargs = {
        "objective": "reg:squarederror",
        "learning_rate": lr,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "enable_categorical": True,
        "missing": missing_val,
        "random_state": seed,
    }
    reg = get_regressor(regressor, **kwargs)

    reg = fit_regressor(reg, train_feats, train_labels)

    # Calculate feature importances
    if regressor == "xgboost":
        importances = reg.feature_importances_
    else:
        perm_importance = permutation_importance(
            reg, train_feats, train_labels, n_repeats=10, random_state=seed
        )
        importances = perm_importance.importances_mean

    return reg, importances


def preprocess_features(df):
    # Convert object columns to numeric if possible
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert categorical columns to numeric codes
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].cat.codes

    # Handle missing values
    df = df.fillna(-1)

    return df


def get_regressor(regressor_name, **params):
    if regressor_name == "xgboost":
        return xgb.XGBRegressor(**params)
    elif regressor_name == "linear":
        return make_pipeline(SimpleImputer(strategy="mean"), LinearRegression())
    elif regressor_name == "svr":
        return make_pipeline(SimpleImputer(strategy="mean"), SVR())
    else:
        raise ValueError(f"Unsupported regressor: {regressor_name}")


def train_regressor_with_hyperparameter_search(
    train_feats,
    train_labels,
    y_col,
    cv_folds=5,
    seed=42,
    model_dir="./models",
    force_new_search=False,
    fold=0,
):
    # NOTE: currently this only supports XGBoost, other regressors should also be added.
    params_dir = os.path.join(model_dir, "best_params")
    os.makedirs(params_dir, exist_ok=True)
    params_file = os.path.join(params_dir, f"best_params_{y_col}_fold_{fold}.joblib")

    if not force_new_search and os.path.exists(params_file):
        logging.info(f"Loading previous best parameters for {y_col} (Fold {fold})")
        best_params = joblib.load(params_file)
        xgb_model = get_regressor("xgboost", **best_params)
        xgb_model.fit(train_feats, train_labels)
        return xgb_model, xgb_model.feature_importances_, best_params

    logging.info(f"Performing hyperparameter search for {y_col}")

    train_feats = preprocess_features(train_feats)

    kwargs = {
        "objective": "reg:squarederror",
        "enable_categorical": True,
        "missing": -1,
    }
    xgb_model = get_regressor("xgboost", **kwargs)

    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "n_estimators": [10, 20, 50],
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",  # or 'neg_mean_squared_error'
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        verbose=1,
        n_jobs=10,
    )

    # Run the grid search
    grid_search.fit(train_feats, train_labels)

    # Output the best parameters and best score
    logging.debug("Best parameters:", grid_search.best_params_)
    logging.debug("Best score:", grid_search.best_score_)

    joblib.dump(grid_search.best_params_, params_file)
    logging.info(f"Saved best parameters for {y_col} (Fold {fold}) to {params_file}")

    return (
        grid_search.best_estimator_,
        grid_search.best_estimator_.feature_importances_,
        grid_search.best_params_,
    )


def preprocess_data(data):
    columns_to_convert = [
        "activation",
        "attention_variant",
        "biases",
        "block_type",
        "layer_norm_type",
        "positional_embeddings",
        "weight_tying",
    ]
    columns_to_convert_in_data = [c for c in columns_to_convert if c in data.columns]

    data = pd.get_dummies(data, columns=columns_to_convert_in_data)
    data = data.drop(["model_name", "id"], axis=1)
    return data


def get_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
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

    args = parser.parse_args()

    assert args.n_estimators > 0, "Number of trees must be greater than 0"
    assert args.lr > 0, "Learning rate must be greater than 0"
    assert args.max_depth > 0, "Max depth must be greater than 0"
    if not (args.model_feats or args.data_feats):
        raise ValueError("Please provide either model_feats or data_feats")

    return args


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
                "pretraining_summary:total_tokens_billions",
                "model_name",
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


def feat_transform(dataset: pd.DataFrame):
    # transform total_params and pretraining_summary:total_tokens_billions to log scale
    (
        dataset["total_params"],
        dataset["pretraining_summary:total_tokens_billions"],
    ) = pd.to_numeric(dataset["total_params"], errors="coerce"), pd.to_numeric(
        dataset["pretraining_summary:total_tokens_billions"], errors="coerce"
    )

    dataset["total_params"] = np.log(dataset["total_params"])
    dataset["pretraining_summary:total_tokens_billions"] = np.log(
        dataset["pretraining_summary:total_tokens_billions"]
    )
    return dataset


def cross_validation(feats, labels, y_col, args, n_folds: int = 5) -> dict:
    k_folds = KFold(n_splits=n_folds, random_state=args.seed, shuffle=True)
    test_features_list = []
    all_shap_values = []
    all_mae = []
    all_mae_median_baseline = []
    all_predictions = []
    all_test_labels = []
    all_test_indices = []
    best_params_per_fold = []

    feat_importances = []

    for i, (train_index, test_index) in enumerate(k_folds.split(feats)):
        train_feats, test_feats = feats.iloc[train_index], feats.iloc[test_index]
        train_labels, test_labels = (
            labels.iloc[train_index],
            labels.iloc[test_index],
        )

        train_labels = labels.iloc[train_index]

        # Calculate median baseline predictions
        median_predictions = median_baseline(train_labels, test_feats)
        mae_median = mean_absolute_error(test_labels, median_predictions)
        all_mae_median_baseline.append(mae_median)

        if args.hyperparam_search:
            (
                model,
                importances,
                fold_best_params,
            ) = train_regressor_with_hyperparameter_search(
                train_feats,
                train_labels,
                y_col,
                seed=args.seed,
                force_new_search=args.force_new_search,
                model_dir=f"./models_{args.predictor_type}",
                fold=i,
            )
            best_params_per_fold.append(fold_best_params)
        else:
            model, importances = train_regressor(
                train_feats,
                train_labels,
                regressor=args.regressor,
                lr=args.lr,
                max_depth=args.max_depth,
                n_estimators=args.n_estimators,
                missing_val=args.missing_val,
                seed=args.seed,
            )

        test_feats = preprocess_features(test_feats)
        predictions = model.predict(test_feats)

        all_predictions.append(predictions)
        all_test_labels.append(list(test_labels))
        all_test_indices.append(list(test_index))

        mae = mean_absolute_error(test_labels, predictions)
        all_mae.append(mae)
        feat_importances.append(importances)

        if args.regressor == "xgboost":
            # get shap values
            explainer = shap.Explainer(model)
            shap_values = explainer(test_feats)
            all_shap_values.append(shap_values.values)
            test_features_list.append(test_feats)

    # NOTE: each item in each of these lists represents results from a fold
    return {
        "all_mae": all_mae,
        "all_mae_median_baseline": all_mae_median_baseline,
        "all_predictions": all_predictions,
        "all_test_labels": all_test_labels,
        "all_test_indices": all_test_indices,
        "feat_importances": feat_importances,
        "all_shap_values": all_shap_values,
        "test_features_list": test_features_list,
        "best_params_per_fold": best_params_per_fold,
    }


def fit_predictors_on_datasets(args: argparse.Namespace, dataset: pd.DataFrame):
    mae_per_task = []
    med_baseline_mae_per_task = []  # To store MAE for median baseline
    successful_tasks = []
    mmlu_mae = []
    all_feat_importances = []
    mmlu_shap_values = []
    mmlu_test_features = []
    all_predictions = []
    all_scores = []

    scaling_laws_features = [
        "total_params",
        "pretraining_summary:total_tokens_billions",
    ]

    for y_col in y_cols:
        if args.new_task_only and "new_task_groups" not in y_col:
            continue
        if args.predictor_type == "scaling_laws":
            dataset_copy = (
                dataset.copy()
                .dropna(subset=[y_col] + scaling_laws_features)
                .reset_index(drop=True)
            )
        else:
            if args.feat_subset:
                dataset_copy = (
                    dataset[args.feat_subset + [y_col, "model_name", "id"]]
                    .copy()
                    .dropna()
                    .reset_index(drop=True)
                )
            else:
                dataset_copy = (
                    dataset.copy()
                    .dropna(subset=[y_col] + scaling_laws_features)
                    .reset_index(drop=True)
                )

        dataset_copy["total_params"] = pd.to_numeric(
            dataset_copy["total_params"], errors="coerce"
        )
        dataset_copy["pretraining_summary:total_tokens_billions"] = pd.to_numeric(
            dataset_copy["pretraining_summary:total_tokens_billions"], errors="coerce"
        )

        if len(dataset_copy) <= MIN_SAMPLES:
            warnings.warn(
                f"Skipping {y_col} as there are not enough samples for training"
            )
            continue

        original_order = dataset_copy.index.copy()

        trainset = preprocess_data(dataset_copy)

        # Ensure consistent ordering after preprocessing
        trainset = trainset.reindex(original_order)

        feats = trainset.drop(columns=cols_from_results, errors="ignore")
        labels = trainset[y_col]

        # Reset index to ensure KFold gets data in same order
        feats = feats.reset_index(drop=True)
        labels = labels.reset_index(drop=True)

        model_names = dataset_copy["model_name"]
        trainset = preprocess_data(dataset_copy)

        feats = trainset.drop(columns=cols_from_results, errors="ignore")
        labels = trainset[y_col]

        cross_val_results = cross_validation(feats, labels, y_col, args)
        all_mae = cross_val_results["all_mae"]
        all_mae_median_baseline = cross_val_results["all_mae_median_baseline"]
        all_predictions_in_fold = cross_val_results["all_predictions"]
        all_test_labels = cross_val_results["all_test_labels"]
        all_test_indices = cross_val_results["all_test_indices"]
        feat_importances = cross_val_results["feat_importances"]
        all_shap_values = cross_val_results["all_shap_values"]
        test_features_list = cross_val_results["test_features_list"]

        # Reconstruct errors using test indices
        task_predictions = {}
        task_scores = {}
        task_absolute_errors = {}

        for test_idx, test_preds, test_labels in zip(
            all_test_indices, all_predictions_in_fold, all_test_labels
        ):
            task_predictions.update(
                {name: pred for (name, pred) in zip(model_names[test_idx], test_preds)}
            )

            task_scores.update(
                {
                    name: score
                    for (name, score) in zip(model_names[test_idx], test_labels)
                }
            )
            absolute_errors = {
                name: ae
                for (name, ae) in zip(
                    model_names[test_idx], abs(test_labels - test_preds)
                )
            }
            task_absolute_errors.update(absolute_errors)

        mean_importances = np.mean(feat_importances, axis=0)
        importances_series = pd.Series(mean_importances, index=feats.columns)
        logging.debug(f"Feature Importances for task {y_col}: ")
        logging.debug(importances_series)
        all_feat_importances.append(importances_series)

        all_predictions.append({y_col: task_predictions})
        all_scores.append({y_col: task_scores})
        os.makedirs("./logs", exist_ok=True)
        with open(f"./logs/perf_pred_{y_col}_{args.predictor_type}.txt", "w") as f:
            f.write(
                f"=== Absolute Error for each model: ===\n"
                + "".join(
                    [
                        f"{model}: {error}\n"
                        for model, error in task_absolute_errors.items()
                    ]
                )
            )
            f.write(
                f"=== Average Mean Absolute Error across folds for task : {np.mean(all_mae)} ===\n"
            )
            f.write(
                f"=== Median Baseline MAE across folds : {np.mean(all_mae_median_baseline)} ===\n"
            )
            f.write("=== Feature Importances: ===\n")
            f.write(importances_series.sort_values(ascending=False).to_string())
            f.write("\n")

        mae_per_task.append(np.mean(all_mae))
        med_baseline_mae_per_task.append(np.mean(all_mae_median_baseline))

        if any([y_col.startswith(t) for t in mmlu_tasks]):
            mmlu_mae.extend(all_mae)

        successful_tasks.append(y_col)
        logging.info(
            f"Average Mean Absolute Error across folds for {y_col}: {np.mean(all_mae)}"
        )
        logging.info(
            f"Median Baseline MAE across folds for {y_col}: {np.mean(all_mae_median_baseline)}"
        )

        if args.regressor == "xgboost":
            # Aggregating SHAP values
            aggregated_shap_values = np.concatenate(all_shap_values, axis=0)
            aggregated_test_features = pd.concat(test_features_list, ignore_index=True)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                aggregated_shap_values, aggregated_test_features, show=False
            )

            os.makedirs("./performance_prediction/figures", exist_ok=True)
            plt.savefig(
                f"./performance_prediction/figures/aggregate_shap_{y_col}_{args.predictor_type}.png"
            )
            plt.gcf().clear()

    return (
        successful_tasks,
        mae_per_task,
        med_baseline_mae_per_task,
        mmlu_mae,
        all_feat_importances,
        mmlu_shap_values,
        mmlu_test_features,
        all_predictions,
        all_scores,
    )


def process_predictions_and_scores(all_predictions, all_scores):
    task_names = [list(d.keys())[0] for d in all_predictions]

    def process_dict_list(dict_list, prefix):
        dicts = [list(d.values())[0] for d in dict_list]
        df = pd.DataFrame.from_records(dicts, index=task_names).transpose()
        df.columns = [f"{prefix}_{col}" for col in df.columns]
        return df

    df_preds = process_dict_list(all_predictions, "pred")
    df_scores = process_dict_list(all_scores, "true")

    return pd.merge(df_preds, df_scores, left_index=True, right_index=True), task_names


def calculate_errors(df, task_names):
    for task in task_names:
        df[f"SErr_{task}"] = df[f"pred_{task}"] - df[f"true_{task}"]
        df[f"AErr_{task}"] = abs(df[f"SErr_{task}"])
    return df.reindex(sorted(df.columns, key=lambda x: x[4:]), axis=1)


def calculate_mean_errors(df):
    signed_errors = df.filter(regex="^SErr_")
    absolute_errors = df.filter(regex="^AErr_")
    return (
        pd.DataFrame(
            {
                "mean_signed_error": signed_errors.mean(axis=1).sort_values(
                    ascending=True
                )
            }
        ),
        pd.DataFrame(
            {
                "mean_absolute_error": absolute_errors.mean(axis=1).sort_values(
                    ascending=True
                )
            }
        ),
    )


def plot_mmlu_shap_values(
    mmlu_shap_values, mmlu_test_features, y_cols_joined, predictor_type
):
    if not mmlu_shap_values:
        return

    aggregated_mmlu_shap_values = np.concatenate(mmlu_shap_values, axis=0)
    aggregated_mmlu_test_features = pd.concat(mmlu_test_features, ignore_index=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        aggregated_mmlu_shap_values, aggregated_mmlu_test_features, show=False
    )
    plt.title("Aggregated SHAP Values for MMLU Tasks")
    plt.savefig(
        f"./performance_prediction/figures/aggregate_shap_mmlu_{y_cols_joined}_{predictor_type}.png"
    )
    plt.close()


def save_dataframe(
    df, filename, directory="./performance_prediction/generated_data", keep_index=False
):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, filename), index=keep_index)


def compile_per_model_predictions(all_predictions, all_scores, successful_tasks):
    """
    Compiles predictions and true scores into a per-model format

    Args:
        all_predictions (list): List of dictionaries containing predictions for each task
        all_scores (list): List of dictionaries containing true scores for each task
        successful_tasks (list): List of task names that were successfully processed

    Returns:
        pd.DataFrame: DataFrame with columns [y_col, Model, True, Predicted]
    """
    records = []

    for task, pred_dict, score_dict in zip(
        successful_tasks, all_predictions, all_scores
    ):
        predictions = list(pred_dict.values())[0]
        true_scores = list(score_dict.values())[0]

        for model_name in predictions.keys():
            record = {
                "y_col": task,
                "Model": model_name,
                "True": true_scores[model_name],
                "Predicted": predictions[model_name],
            }
            records.append(record)

    return pd.DataFrame(records)


def save_compiled_predictions(df, args, predictor_type):
    """
    Save compiled predictions to CSV file
    """
    output_dir = "./logs"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"compiled_predictions_{args.metric}_{args.regressor}_{predictor_type}.csv",
    )
    df.to_csv(output_file, index=False)
    logging.info(f"Saved compiled predictions to {output_file}")


def postprocess_results(
    args,
    df_results,
    all_predictions,
    all_scores,
    mmlu_shap_values,
    mmlu_test_features,
    all_feat_importances,
):
    if len(mmlu_mae) > 0 and args.merge_mmlu:
        df_results = pd.concat(
            [df_results, pd.DataFrame({"task": ["mmlu"], "mae": [np.mean(mmlu_mae)]})]
        )
        # delete all the individual mmlu tasks
        df_results = df_results[~df_results["task"].str.startswith("hendrycksTest-")]
    if args.merge_arithmetic:
        df_results = pd.concat(
            [
                df_results,
                pd.DataFrame(
                    {
                        "task": ["arithmetic"],
                        "mae": [
                            np.mean(
                                [
                                    mae
                                    for task, mae in zip(successful_tasks, mae_per_task)
                                    if "arithmetic" in task
                                ]
                            )
                        ],
                    }
                ),
            ]
        )
        # delete all the individual arithmetic tasks
        df_results = df_results[~df_results["task"].str.startswith("arithmetic_")]
    print("Debug information:")
    print(f"Number of successful tasks: {len(successful_tasks)}")
    print(f"MAE per task: {mae_per_task}")
    print(f"Median baseline MAE per task: {med_baseline_mae_per_task}")
    print(
        f"Improvement over baseline: {list(np.array(mae_per_task) - np.array(med_baseline_mae_per_task))}"
    )

    for task, mae, baseline_mae in zip(
        successful_tasks, mae_per_task, med_baseline_mae_per_task
    ):
        print(f"Task: {task}")
        print(f"  MAE: {mae}")
        print(f"  Baseline MAE: {baseline_mae}")
        print(f"  Improvement: {baseline_mae - mae}")

    print("\nDataFrame contents:")
    print(df_results)
    sorted_tasks_by_mae = sorted(
        zip(successful_tasks, mae_per_task), key=lambda x: x[1]
    )

    # Print tasks sorted by MAE
    for task, mae in sorted_tasks_by_mae:
        logging.info(f"Task {task} MAE: {mae}")

    best_task_mae = successful_tasks[np.argmin(mae_per_task)]
    worst_task_mae = successful_tasks[np.argmax(mae_per_task)]
    logging.info(
        f"\nMost predictable task by MAE: {best_task_mae} with MAE: {min(mae_per_task)}"
    )
    logging.info(
        f"Least predictable task by MAE: {worst_task_mae} with MAE: {max(mae_per_task)}"
    )

    logging.info(f"Overall Median Baseline MAE: {np.mean(med_baseline_mae_per_task)}")

    # print average MAE
    logging.info(f"Average MAE: {df_results['mae'].mean()}")

    y_cols_joined = ",".join(args.y_cols)
    save_dataframe(
        df_results,
        f"summary_{y_cols_joined}_{args.predictor_type}_metric_{args.metric}_{args.regressor}.csv",
    )

    df_errors, task_names = process_predictions_and_scores(all_predictions, all_scores)
    df_errors = calculate_errors(df_errors, task_names)

    save_dataframe(
        df_errors,
        f"absolute_errors_{y_cols_joined}_{args.predictor_type}_metric_{args.metric}_{args.regressor}.csv",
        keep_index=True,
    )

    df_mean_signed_errors, df_mean_absolute_errors = calculate_mean_errors(df_errors)

    save_dataframe(
        df_mean_signed_errors,
        f"mean_signed_errors_{y_cols_joined}_{args.predictor_type}_metric_{args.metric}_{args.regressor}.csv",
        directory="./performance_prediction/mispredictions",
        keep_index=True,
    )

    save_dataframe(
        df_mean_absolute_errors,
        f"mean_absolute_errors_{y_cols_joined}_{args.predictor_type}_metric_{args.metric}_{args.regressor}.csv",
        directory="./performance_prediction/mispredictions",
        keep_index=True,
    )

    logging.info(
        f"Mean signed errors and mean absolute errors saved to {os.getcwd()}/performance_prediction/mispredictions/"
    )

    importance_df = pd.concat(all_feat_importances, axis=1)
    mean_importances = importance_df.mean(axis=1).sort_values(ascending=False)
    logging.info("Mean Feature Importances overall:")
    logging.info(mean_importances)

    save_dataframe(
        mean_importances.reset_index(),
        f"feature_importances_{y_cols_joined}_{args.predictor_type}.csv",
    )

    plot_mmlu_shap_values(
        mmlu_shap_values, mmlu_test_features, y_cols_joined, args.predictor_type
    )

    # log individual model performances vs predicted performances for further visualization
    per_model_df = compile_per_model_predictions(
        all_predictions, all_scores, successful_tasks
    )
    save_compiled_predictions(per_model_df, args, args.predictor_type)


def consolidate_total_params(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    if "total_params" in dataset.columns and "safetensors:total" in dataset.columns:
        # fill in total_params with values from safetensors:total as the canonical column
        dataset["total_params"] = np.where(
            dataset["total_params"].isna(),
            dataset["safetensors:total"],
            dataset["total_params"],
        )
        dataset = dataset.drop(columns=["safetensors:total"])
    return dataset


if __name__ == "__main__":
    args = get_args()

    # set the logger
    logging.basicConfig(level=args.log_level)

    # join the metadata and scores
    dataset = load_data(args)

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open("./eval_task_groups/mmlu_deprecated.yaml", "r") as f:
        mmlu_tasks = yaml.safe_load(f)["task"]

    training_scores = pd.read_csv(args.train_labels)

    cols_from_results = set(training_scores.columns) - {"model_name", "id"}

    if args.y_cols == ["all"]:
        y_cols = [t for t in list(cols_from_results) if t.endswith(f"_{args.metric}")]
    else:
        y_cols = args.y_cols

    dataset = consolidate_total_params(dataset)
    dataset = process_data(dataset, args)
    dataset = feat_transform(dataset)

    (
        successful_tasks,
        mae_per_task,
        med_baseline_mae_per_task,
        mmlu_mae,
        all_feat_importances,
        mmlu_shap_values,
        mmlu_test_features,
        all_predictions,
        all_scores,
    ) = fit_predictors_on_datasets(args, dataset)

    df_results = pd.DataFrame(
        {
            "task": successful_tasks,
            "mae": mae_per_task,
            "improvement_over_baseline": list(
                np.array(mae_per_task) - np.array(med_baseline_mae_per_task)
            ),
        }
    )

    logging.info(f"Overall mae: {df_results['mae'].mean()}")
    logging.info(f"Median baseline mae: {np.mean(med_baseline_mae_per_task)}")
    postprocess_results(
        args,
        df_results,
        all_predictions,
        all_scores,
        mmlu_shap_values,
        mmlu_test_features,
        all_feat_importances,
    )
