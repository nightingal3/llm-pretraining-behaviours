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


def fit_regressor(reg, train_feats, train_labels):
    reg.fit(train_feats, train_labels)
    return reg


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
    print(
        f"Training a {regressor} model with training data of shape {train_feats.shape}."
    )

    if regressor == "xgboost":
        reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=lr,
            max_depth=max_depth,
            n_estimators=n_estimators,
            enable_categorical=True,
            missing=missing_val,
            nrounds=10000,
            random_state=seed,
        )
    elif regressor == "linear":
        reg = make_pipeline(SimpleImputer(strategy="mean"), LinearRegression())
    elif regressor == "svr":
        reg = make_pipeline(SimpleImputer(strategy="mean"), SVR())
    else:
        raise ValueError(f"Unsupported regressor: {regressor}")

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


def train_regressor_with_hyperparameter_search(train_feats, train_labels, cv_folds=5):
    # Set up the model
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=10,  # Fixed as per your requirement
        enable_categorical=True,
        missing=-1,
    )

    # Define the hyperparameter grid
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],  # Example values
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 2, 4],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",  # or 'neg_mean_squared_error'
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        verbose=2,
    )

    # Run the grid search
    grid_search.fit(train_feats, train_labels)

    # Output the best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    return grid_search.best_estimator_


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
    parser.add_argument(
        "--regressor",
        type=str,
        default="xgboost",
        choices=["xgboost", "linear", "svr"],
        help="Type of regressor to use",
    )
    parser.add_argument(
        "--model_feats",
        type=str,
        help="The path to the CSV file containing the training features related to models.",
        default="./performance_prediction/gathered_data/training_model_final.csv"
    )
    parser.add_argument(
        "--data_feats",
        type=str,
        help="The path to the CSV file containing the training features related to datasets.",
        default="./performance_prediction/gathered_data/training_dataset_final_revised.csv"
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        help="The path to the CSV file containing the training labels",
        required=True,
        default="./performance_prediction/gathered_data/training_score_final.csv"
    )
    parser.add_argument(
        "--feat_subset",
        type=str,
        nargs="+",
        help="Subset of features to use for training",
    )
    parser.add_argument(
        "--y_cols",
        type=str,
        help="The name(s) of the column(s) containing the target variable(s) in the train_labels file",
        required=True,
        nargs="+",
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
        "--drop_instruction_tuned",
        action="store_true",
        help="Whether to drop models that are instruction tuned",
    )
    parser.add_argument(
        "--new_task_only", action="store_true", help="only keep new tasks"
    )
    parser.add_argument(
        "--metric", default="acc", choices=["acc", "brier_score", "perplexity"]
    )
    args = parser.parse_args()

    assert args.n_estimators > 0, "Number of trees must be greater than 0"
    assert args.lr > 0, "Learning rate must be greater than 0"
    assert args.max_depth > 0, "Max depth must be greater than 0"
    if not (args.model_feats or args.data_feats):
        raise ValueError("Please provide either model_feats or data_feats")

    return args


def map_importances_to_categories(
    feature_importances, feature_names, encoder, categorical_columns
):
    decoded_importances = pd.Series(0, index=feature_names)
    for idx, feature in enumerate(feature_names):
        if feature in categorical_columns:
            cat_index = categorical_columns.index(feature)
            # Map each category's importance
            for i, cat in enumerate(encoder.categories_[cat_index]):
                # This assumes that each category contributes to a separate element in feature_importances
                # which might need adjustment depending on how your model handles importances
                decoded_feature_name = f"{feature}_{cat}"
                decoded_importances[decoded_feature_name] = feature_importances[
                    idx * len(encoder.categories_[cat_index]) + i
                ]
        else:
            decoded_importances[feature] = feature_importances[idx]

    return decoded_importances.sort_values(ascending=False)


def load_data(args: argparse.Namespace):
    # Load the CSV files into pandas DataFrames
    training_scores = pd.read_csv(args.train_labels)
    
    if args.model_feats and args.data_feats:
        arch_metadata = pd.read_csv(args.model_feats)
        data_metadata = pd.read_csv(args.data_feats)
        metadata_feats = pd.merge(arch_metadata, data_metadata, on="id")
    elif args.model_feats:
        metadata_feats = pd.read_csv(args.model_feats)
    else:
        metadata_feats = pd.read_csv(args.data_feats)

    # Merge the DataFrames based on 'model_name' and 'id', dropping entries without matches
    dataset = pd.merge(
        training_scores,
        metadata_feats,
        how="inner",
        left_on="model_name",
        right_on="id",
    )

    return dataset


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
    dataset["total_params"] = np.log(dataset["total_params"])
    dataset["pretraining_summary:total_tokens_billions"] = np.log(
        dataset["pretraining_summary:total_tokens_billions"]
    )
    return dataset

def fit_predictors_on_datasets(args: argparse.Namespace, dataset: pd.DataFrame):
    mae_per_task = []
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
        # drop rows with missing score values
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

        if len(dataset_copy) <= max(5, args.n_estimators):
            warnings.warn(
                f"Skipping {y_col} as there are not enough samples for training"
            )
            continue

        model_names = dataset_copy["model_name"]
        trainset = preprocess_data(dataset_copy)

        feats = trainset.drop(columns=cols_from_results, errors="ignore")
        labels = trainset[y_col]

        # cross val
        k_folds = KFold(n_splits=5, random_state=args.seed, shuffle=True)
        test_features_list = []
        all_shap_values = []
        all_mae = []
        task_predictions = {}
        task_scores = {}
        task_absolute_errors = {}

        feat_importances = []

        for train_index, test_index in k_folds.split(feats):
            # train model, get MAEs
            train_feats, test_feats = feats.iloc[train_index], feats.iloc[test_index]
            train_labels, test_labels = (
                labels.iloc[train_index],
                labels.iloc[test_index],
            )
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

            predictions = model.predict(test_feats)

            task_predictions.update(
                {
                    name: pred
                    for (name, pred) in zip(model_names[test_index], predictions)
                }
            )

            task_scores.update(
                {
                    name: score
                    for (name, score) in zip(model_names[test_index], test_labels)
                }
            )
            absolute_errors = {
                name: ae
                for (name, ae) in zip(
                    model_names[test_index], abs(test_labels - predictions)
                )
            }
            task_absolute_errors.update(absolute_errors)

            mae = mean_absolute_error(test_labels, predictions)
            all_mae.append(mae)

            feat_importances.append(importances)

            if args.regressor == "xgboost":
                # get shap values
                explainer = shap.Explainer(model)
                shap_values = explainer(test_feats)
                all_shap_values.append(shap_values.values)
                test_features_list.append(test_feats)

                if any([y_col.startswith(t) for t in mmlu_tasks]):
                    mmlu_shap_values.append(shap_values.values)
                    mmlu_test_features.append(test_feats)

        mean_importances = np.mean(feat_importances, axis=0)
        importances_series = pd.Series(mean_importances, index=feats.columns)
        print("Feature Importances: ")
        print(importances_series)
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
                f"=== Average Mean Absolute Error across folds: {np.mean(all_mae)} ===\n"
            )
            f.write("=== Feature Importances: ===\n")
            f.write(importances_series.sort_values(ascending=False).to_string())
            f.write("\n")

        mae_per_task.append(np.mean(all_mae))
        if any([y_col.startswith(t) for t in mmlu_tasks]):
            mmlu_mae.extend(all_mae)

        successful_tasks.append(y_col)
        print(f"Average Mean Squared Error across folds: {np.mean(all_mae)}")

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

    return successful_tasks, mae_per_task, mmlu_mae, all_feat_importances, mmlu_shap_values, mmlu_test_features, all_predictions, all_scores

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
        pd.DataFrame({"mean_signed_error": signed_errors.mean(axis=1).sort_values(ascending=True)}),
        pd.DataFrame({"mean_absolute_error": absolute_errors.mean(axis=1).sort_values(ascending=True)})
    )

def plot_mmlu_shap_values(mmlu_shap_values, mmlu_test_features, y_cols_joined, predictor_type):
    if not mmlu_shap_values:
        return
    
    aggregated_mmlu_shap_values = np.concatenate(mmlu_shap_values, axis=0)
    aggregated_mmlu_test_features = pd.concat(mmlu_test_features, ignore_index=True)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(aggregated_mmlu_shap_values, aggregated_mmlu_test_features, show=False)
    plt.title("Aggregated SHAP Values for MMLU Tasks")
    plt.savefig(f"./performance_prediction/figures/aggregate_shap_mmlu_{y_cols_joined}_{predictor_type}.png")
    plt.close()

def save_dataframe(df, filename, directory="./performance_prediction/generated_data", keep_index=False):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, filename), index=keep_index)

def postprocess_results(args, df_results, all_predictions, all_scores, mmlu_shap_values, mmlu_test_features, all_feat_importances):
    if len(mmlu_mae) > 0:
        df_results = pd.concat([
            df_results,
            pd.DataFrame({"task": ["mmlu"], "mae": [np.mean(mmlu_mae)]})
        ])

    y_cols_joined = ",".join(args.y_cols)
    save_dataframe(df_results, f"summary_{y_cols_joined}_{args.predictor_type}_metric_{args.metric}_{args.regressor}.csv")

    df_errors, task_names = process_predictions_and_scores(all_predictions, all_scores)
    df_errors = calculate_errors(df_errors, task_names)
    
    save_dataframe(df_errors, f"absolute_errors_{y_cols_joined}_{args.predictor_type}_metric_{args.metric}_{args.regressor}.csv", keep_index=True)

    df_mean_signed_errors, df_mean_absolute_errors = calculate_mean_errors(df_errors)
    
    save_dataframe(df_mean_signed_errors, f"mean_signed_errors_{y_cols_joined}_{args.predictor_type}_metric_{args.metric}_{args.regressor}.csv", 
                   directory="./performance_prediction/mispredictions", keep_index=True)

    save_dataframe(df_mean_absolute_errors, f"mean_absolute_errors_{y_cols_joined}_{args.predictor_type}_metric_{args.metric}_{args.regressor}.csv", directory="./performance_prediction/mispredictions", keep_index=True)

    print(f"Mean signed errors and mean absolute errors saved to {os.getcwd()}/performance_prediction/mispredictions/")

    importance_df = pd.concat(all_feat_importances, axis=1)
    mean_importances = importance_df.mean(axis=1).sort_values(ascending=False)
    print("Mean Feature Importances:")
    print(mean_importances)
    save_dataframe(mean_importances.reset_index(), f"feature_importances_{y_cols_joined}_{args.predictor_type}.csv")

    plot_mmlu_shap_values(mmlu_shap_values, mmlu_test_features, y_cols_joined, args.predictor_type)

if __name__ == "__main__":
    args = get_args()
    
    # join the metadata and scores
    dataset = load_data(args)
    
    with open("./eval_task_groups/mmlu_deprecated.yaml", "r") as f:
        mmlu_tasks = yaml.safe_load(f)["task"]

    training_scores = pd.read_csv(args.train_labels)

    cols_from_results = set(training_scores.columns) - {"model_name", "id"}

    if args.y_cols == ["all"]:
        y_cols = [t for t in list(cols_from_results) if t.endswith(f"_{args.metric}")]
    else:
        y_cols = args.y_cols

    dataset = process_data(dataset, args)

    if "total_params" not in dataset.columns:
        dataset["total_params"] = np.where(
            dataset["safetensors:total"].isna(),
            dataset["total_params"],
            dataset["safetensors:total"],
        )
        dataset = dataset.drop(columns=["safetensors:total"])
    if "total_params" in dataset.columns and "safetensors:total" in dataset.columns:
        # drop safetensors
        dataset = dataset.drop(columns=["safetensors:total"])

    dataset = feat_transform(dataset)   

    successful_tasks, mae_per_task, mmlu_mae, all_feat_importances, mmlu_shap_values, mmlu_test_features, all_predictions, all_scores = fit_predictors_on_datasets(args, dataset)

    df_results = pd.DataFrame(
        {
            "task": successful_tasks,
            "mae": mae_per_task,
        }
    )

    print("Overall mae: ", df_results["mae"].mean())

    postprocess_results(args, df_results, all_predictions, all_scores, mmlu_shap_values, mmlu_test_features, all_feat_importances)