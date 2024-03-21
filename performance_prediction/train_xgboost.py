import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


def fit_regressor(reg, train_feats, train_labels):
    reg.fit(train_feats, train_labels)
    return reg


def train_regressor(
    train_feats,
    train_labels,
    regressor="xgboost",
    quantile=0.95,
    verbose=False,
    **kwargs,
):
    print(
        f"Training a {regressor} model with training data of shape {train_feats.shape}."
    )
    reg = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=0.1, max_depth=10, n_estimators=100
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


# Load the CSV files into pandas DataFrames
training_scores = pd.read_csv(
    "../metadata/training_data/training_score_final.csv"
)  # Replace with the actual path to your first CSV file
training = pd.read_csv(
    "../metadata/training_data/training_model_final.csv"
)  # Replace with the actual path to your second CSV file

# Merge the DataFrames based on 'model_name' and 'id', dropping entries without matches
dataset = pd.merge(
    training_scores, training, how="inner", left_on="model_name", right_on="id"
)

trainset = preprocess_data(dataset)

train_feats = trainset.drop(["hellaswag"], axis=1)[:-1]
train_labels = trainset["hellaswag"][:-1]

model = train_regressor(train_feats, train_labels)

test_feat = train_feats

test_predictions = model.predict(test_feat)

print(test_predictions)
