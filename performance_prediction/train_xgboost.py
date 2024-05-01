import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import gaussian_process
from sklearn.preprocessing import OrdinalEncoder
import argparse
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import warnings
import os

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
    **kwargs,
):
    print(
        f"Training a {regressor} model with training data of shape {train_feats.shape}."
    )

    reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=lr,
        max_depth=max_depth,
        n_estimators=n_estimators,
        enable_categorical=True,
        missing=missing_val,
    )

    fit_regressor(reg, train_feats, train_labels)
    return reg


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_feats",
        type=str,
        help="The path to the CSV file containing the training features related to models.",
    )
    parser.add_argument(
        "--data_feats",
        type=str,
        help="The path to the CSV file containing the training features related to datasets.",
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        help="The path to the CSV file containing the training labels",
        required=True,
    )
    parser.add_argument(
        "--y_col",
        type=str,
        help="The name of the column containing the target variable in the train_labels file",
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
        default=100,
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

    args = parser.parse_args()
    assert args.n_estimators > 0, "Number of trees must be greater than 0"
    assert args.lr > 0, "Learning rate must be greater than 0"
    assert args.max_depth > 0, "Max depth must be greater than 0"
    if not (args.model_feats or args.data_feats):
        raise ValueError("Please provide either model_feats or data_feats")

    categorical_variables = [
        "activation",
        "attention_variant",
        "batch_instances",
        "biases",
        "block_type",
        "layer_norm_type",
    ]

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

    cols_from_results = set(training_scores.columns) - {"model_name", "id"}
    # Merge the DataFrames based on 'model_name' and 'id', dropping entries without matches
    dataset = pd.merge(
        training_scores,
        metadata_feats,
        how="inner",
        left_on="model_name",
        right_on="id",
    )

    dataset = dataset.drop(columns=["assigned person", "notes"])

    if args.predictor_type == "scaling_laws":
        # drop all but total params and num tokens
        dataset = dataset[
            [
                "total_params",
                "total_summary:total_size_tokens_billions",
                "model_name",
                "id",
            ]
            + list(cols_from_results)
        ]
        categorical_variables = []

    if args.y_col == ["all"]:
        y_cols = [t for t in list(cols_from_results) if t.endswith("_acc")]
    else:
        y_cols = args.y_col

    mae_per_task = []
    successful_tasks = []

    for y_col in y_cols:
        # drop rows with missing score values
        dataset = dataset.dropna(subset=y_col)
        if len(dataset) <= args.n_estimators:
            warnings.warn(
                f"Skipping {y_col} as there are not enough samples for training"
            )
            continue

        # ordinal encode
        if args.predictor_type == "all":
            if "is_instruction_tuned" in dataset.columns:
                dataset["is_instruction_tuned"] = dataset["is_instruction_tuned"].map(
                    {True: 1, False: 0, np.nan: -1}
                )

            for var in categorical_variables:
                dataset[var] = dataset[var].astype("category")

            enc = OrdinalEncoder()
            dataset[categorical_variables] = enc.fit_transform(
                dataset[categorical_variables]
            )

        trainset = preprocess_data(dataset)

        feats = trainset.drop(columns=cols_from_results, errors="ignore")
        labels = trainset[y_col]

        # cross val
        k_folds = KFold(n_splits=5, random_state=42, shuffle=True)
        test_features_list = []
        all_shap_values = []
        all_mae = []
        feat_importances = []

        for train_index, test_index in k_folds.split(feats):
            train_feats, test_feats = feats.iloc[train_index], feats.iloc[test_index]
            train_labels, test_labels = (
                labels.iloc[train_index],
                labels.iloc[test_index],
            )
            model = train_regressor(
                train_feats,
                train_labels,
                lr=args.lr,
                max_depth=args.max_depth,
                n_estimators=args.n_estimators,
                missing_val=args.missing_val,
            )
            predictions = model.predict(test_feats)
            mae = mean_absolute_error(test_labels, predictions)
            all_mae.append(mae)

            importances = model.feature_importances_
            feat_importances.append(importances)

            explainer = shap.Explainer(model)
            shap_values = explainer(test_feats)
            all_shap_values.append(shap_values.values)
            test_features_list.append(test_feats)

        mean_importances = np.mean(feat_importances, axis=0)
        importances_series = pd.Series(mean_importances, index=feats.columns)
        print("Feature Importances: ")
        print(importances_series)

        os.makedirs("./logs", exist_ok=True)
        with open(f"./logs/perf_pred_{y_col}_{args.predictor_type}.txt", "w") as f:
            f.write(
                f"=== Average Mean Absolute Error across folds: {np.mean(all_mae)} ===\n"
            )
            f.write("=== Feature Importances: ===\n")
            f.write(importances_series.sort_values(ascending=False).to_string())
            f.write("\n")

        mae_per_task.append(np.mean(all_mae))
        successful_tasks.append(y_col)
        # Aggregating SHAP values
        aggregated_shap_values = np.concatenate(all_shap_values, axis=0)
        aggregated_test_features = pd.concat(test_features_list, ignore_index=True)

        print(f"Average Mean Squared Error across folds: {np.mean(all_mae)}")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(aggregated_shap_values, aggregated_test_features, show=False)

        os.makedirs("./figures", exist_ok=True)
        plt.savefig(f"./figures/aggregate_shap_{y_col}_{args.predictor_type}.png")
        plt.gcf().clear()

    df_results = pd.DataFrame(
        {
            "task": successful_tasks,
            "mae": mae_per_task,
        }
    )
    y_cols_joined = ",".join(args.y_col)
    df_results.to_csv(f"summary_{y_cols_joined}_{args.predictor_type}.csv", index=False)
