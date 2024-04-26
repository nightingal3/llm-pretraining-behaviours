import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import gaussian_process
from sklearn.preprocessing import OrdinalEncoder
import argparse
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, train_test_split
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
    data = pd.get_dummies(data, columns=columns_to_convert)
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
        "is_instruction_tuned",
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

    # drop rows with missing values
    dataset = dataset.dropna(subset=args.y_col)

    dataset = dataset.drop(columns=["assigned person", "notes"])

    # ordinal encode
    dataset["is_instruction_tuned"] = dataset["is_instruction_tuned"].map(
        {True: 1, False: 0, np.nan: -1}
    )

    for var in categorical_variables:
        dataset[var] = dataset[var].astype("category")

    enc = OrdinalEncoder()
    dataset[categorical_variables] = enc.fit_transform(dataset[categorical_variables])

    trainset = preprocess_data(dataset)

    feats = trainset.drop(columns=cols_from_results, errors="ignore")
    labels = trainset[args.y_col]

    train_feats, test_feats, train_labels, test_labels = train_test_split(
        feats, labels, test_size=0.2, random_state=42
    )

    if args.n_estimators > len(train_feats):
        warnings.warn(
            f"Number of trees ({args.n_estimators}) is greater than the number of training samples ({len(train_feats)}). You will likely overfit."
        )

    breakpoint()
    model = train_regressor(
        train_feats,
        train_labels,
        lr=args.lr,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        missing_val=args.missing_val,
    )

    test_predictions = model.predict(test_feats, output_margin=True)

    print(
        f"Model Hyperparameters:\n Learning Rate: {args.lr}\n Max Depth: {args.max_depth}\n Number of Estimators: {args.n_estimators}"
    )
    print(f"Mean Squared Error: {mean_squared_error(test_labels, test_predictions)}")
    feature_importances = model.feature_importances_
    feature_names = train_feats.columns

    # Create a pandas series to associate feature names with their importance scores
    importances = pd.Series(feature_importances, index=feature_names)
    print("Feature Importances: ")
    print(importances)

    # view feature importance/directionality
    if args.interpret_plot == "shap":
        explainer = shap.TreeExplainer(model)
        explanation = explainer(test_feats)
        shap_values = explanation.values
        np.abs(
            shap_values.sum(axis=1) + explanation.base_values - test_predictions
        ).max()
        shap.plots.beeswarm(explanation)
        plt.tight_layout()

        os.makedirs("./figures", exist_ok=True)
        plt.savefig(f"./figures/shap_{args.y_col}.png")
    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    test_predictions = pd.DataFrame(test_predictions, columns=["predicted_score"])
    test_predictions.to_csv(args.output_file, index=False)
    print(f"Test predictions saved to {args.output_file}")
