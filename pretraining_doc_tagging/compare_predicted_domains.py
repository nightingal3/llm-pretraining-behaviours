import pandas as pd
from scipy import stats
import os
import re
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


from generate_generation_tagging_jobs import get_latest_jsonl


def check_encoding_issues(filepath):
    """Check if a file contains escaped Unicode sequences."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            sample = f.read(1000)  # Check first 1000 characters
            if "\\u" in sample:
                return True
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return False


ground_truth_df = pd.read_csv(
    "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretraining_doc_tagging/ground_truth_dataset_info.csv"
)
predicted_df = pd.read_csv(
    "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/all_models_feature_stats_2_03.csv"
)

ground_truth_df = ground_truth_df[
    [
        "id",
        "pretraining_summary_percentage_web",
        "pretraining_summary_percentage_code",
        "pretraining_summary_percentage_books",
        "pretraining_summary_percentage_reference",
        "pretraining_summary_percentage_academic",
    ]
]

breakpoint()
predicted_df = predicted_df[
    [
        "id",
        "domain_academic_pct_mean",
        "domain_books_pct_mean",
        "domain_code_pct_mean",
        "domain_reference_pct_mean",
        "domain_specific_datasets_pct_mean",
        "domain_web_pct_mean",
    ]
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

combined_df = ground_truth_df.merge(predicted_df, on="id", how="inner")
combined_df = combined_df[~combined_df["id"].isin(models_to_drop)]

# get basic pearson correlation
correlations = []
for feature in ground_truth_df.columns[1:]:
    feature_name = feature.split("_")[-1]
    x = combined_df[feature]
    y = combined_df["domain_" + feature_name + "_pct_mean"]

    mask = ~(x.isna() | y.isna())
    x = x[mask]
    y = y[mask]

    if len(x) > 5 and not (x == x.iloc[0]).all() and not (y == y.iloc[0]).all():
        tau, p_value = stats.pearsonr(x, y)
        correlations.append(
            {
                "feature": feature_name,
                "correlation": tau,
                "p_value": p_value,
                "n_samples": len(x),
            }
        )
correlations_df = pd.DataFrame(correlations)
print(correlations_df)


# academic, code have highest corrs. try keeping them and normalizing the remaining percentage between the three remaining categories.

revised_pred_df = predicted_df.copy()
revised_pred_df = revised_pred_df.loc[
    revised_pred_df["domain_academic_pct_mean"].notna()
    & revised_pred_df["domain_code_pct_mean"].notna()
]
revised_pred_df["domain_books_pct_mean"] = 0
revised_pred_df["domain_reference_pct_mean"] = 0
revised_pred_df["domain_specific_datasets_pct_mean"] = 0

for i, row in revised_pred_df.iterrows():
    remaining_pct = 100 - row["domain_academic_pct_mean"] - row["domain_code_pct_mean"]
    original_preds = predicted_df.loc[predicted_df["id"] == row["id"]]
    orig_books_pct = original_preds["domain_books_pct_mean"].values[0]
    orig_reference_pct = original_preds["domain_reference_pct_mean"].values[0]
    orig_web_pct = original_preds["domain_web_pct_mean"].values[0]

    revised_pred_df.loc[i, "domain_books_pct_mean"] = orig_books_pct * (
        remaining_pct / 100
    )
    revised_pred_df.loc[i, "domain_reference_pct_mean"] = orig_reference_pct * (
        remaining_pct / 100
    )
    revised_pred_df.loc[i, "domain_web_pct_mean"] = orig_web_pct * (remaining_pct / 100)
revised_pred_df = revised_pred_df.rename(
    columns={c: f"revised_{c}" for c in predicted_df.columns if c != "id"}
)

# Merge with combined_df on 'id'
combined_df = combined_df.merge(
    revised_pred_df,
    on="id",
    how="left",  # Use "inner" if you only want overlapping IDs
)

# Now calculate correlations using the merged DataFrame
correlations = []
for feature in ground_truth_df.columns[1:]:
    feature_name = feature.split("_")[-1]
    x = combined_df[feature]
    y = combined_df[f"revised_domain_{feature_name}_pct_mean"]  # Use renamed column

    mask = ~(x.isna() | y.isna())
    x = x[mask]
    y = y[mask]

    if len(x) > 5 and not (x == x.iloc[0]).all() and not (y == y.iloc[0]).all():
        tau, p_value = stats.pearsonr(x, y)
        correlations.append(
            {
                "feature": feature_name,
                "correlation": tau,
                "p_value": p_value,
                "n_samples": len(x),
            }
        )
correlations_df = pd.DataFrame(correlations)
print("revised: ", correlations_df)

categories = ["web", "code", "books", "reference", "academic"]
for cat in categories:
    combined_df[f"error_{cat}"] = abs(
        combined_df[f"pretraining_summary_percentage_{cat}"]
        - combined_df[f"revised_domain_{cat}_pct_mean"]
    )

# Total error across all categories
combined_df["total_error"] = combined_df[[f"error_{cat}" for cat in categories]].sum(
    axis=1
)
# exclude nans from the total error
combined_df = combined_df.loc[combined_df["domain_web_pct_mean"].notna()]

# Get top 10 worst-performing models
worst_models = combined_df.sort_values("total_error", ascending=False).head(10)
print("Top 10 worst-predicted models:")
print(worst_models[["id", "total_error"]])

worst_models = combined_df.sort_values("total_error", ascending=False).head(10)[
    ["id", "total_error"]
]
best_models = combined_df.sort_values("total_error", ascending=True).head(10)[
    ["id", "total_error"]
]

base_dir = "/data/tir/projects/tir5/users/mengyan3/freegens_all_corrected"


### 3. Check Encoding in Their Files ###
def check_unicode_escapes(text):
    """Check if text contains unicode escape sequences like \\u3000"""
    return bool(re.search(r"\\u[0-9a-fA-F]{4}", text))


def check_file_encoding(model_id):
    """Check a model's files for encoding issues using your existing path logic"""
    org_dir, model_dir = model_id.split("/")
    model_path = os.path.join(base_dir, org_dir, model_dir)

    jsonl_file = get_latest_jsonl(model_path, org_dir, model_dir)
    if not jsonl_file:
        return False, "No JSONL file found"

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            # Check first 100 lines for efficiency
            for _ in range(100):
                line = f.readline()
                if not line:
                    break
                if check_unicode_escapes(line):
                    return True, f"Found escaped Unicode in {jsonl_file}"

        return False, "No escaped Unicode found in sample"
    except Exception as e:
        return True, f"Error reading file: {str(e)}"


### 4. Add Encoding Check Results to Worst Models ###
worst_models["encoding_issue"] = False
worst_models["encoding_details"] = ""

for idx, row in worst_models.iterrows():
    has_issue, details = check_file_encoding(row["id"])
    worst_models.at[idx, "encoding_issue"] = has_issue
    worst_models.at[idx, "encoding_details"] = details

### 5. Display Results ###
print("\nTop 10 Worst Performing Models with Encoding Checks:")
print(worst_models[["id", "total_error", "encoding_issue", "encoding_details"]])

### 6. Correlation Analysis with Encoding Flag ###
# Add encoding issue flag to main dataframe
combined_df = combined_df.merge(
    worst_models[["id", "encoding_issue"]], on="id", how="left"
)
combined_df["encoding_issue"] = combined_df["encoding_issue"].fillna(False)

# Compare errors for models with/without encoding issues
print("\nError Comparison:")
print(combined_df.groupby("encoding_issue")["total_error"].describe())


def constrained_composition_predictor(train_df, categories, use_total_params=False):
    """Train model on pre-split training data, return model artifacts"""
    # Prepare features and target from training data
    if use_total_params:
        X_train = train_df[
            [f"domain_{cat}_pct_mean" for cat in categories] + ["total_params"]
        ].copy()
    else:
        X_train = train_df[[f"domain_{cat}_pct_mean" for cat in categories]].copy()
    y_train = train_df[[f"pretraining_summary_percentage_{cat}" for cat in categories]]

    # Log-transform model size
    if use_total_params:
        X_train["total_params"] = np.log(X_train["total_params"] + 1e-8)

    # Fit scaler on training data
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    constraints = {"type": "eq", "fun": lambda p: np.sum(p) - len(categories)}
    bounds = [(0, 1) for _ in range(X_scaled.shape[1] * len(categories))]

    # Define and optimize weights
    def loss(weights):
        W = weights.reshape(len(scaler.feature_names_in_), len(categories))
        logits = X_scaled @ W
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        pred = exp / exp.sum(axis=1, keepdims=True) * 100
        return np.mean((pred - y_train.values) ** 2)

    result = minimize(
        loss,
        np.random.randn(X_scaled.shape[1] * len(categories)) * 0.01,
        method="L-BFGS-B",
    )

    weights = result.x.reshape(len(scaler.feature_names_in_), len(categories))

    return {"weights": weights, "scaler": scaler}


def constrained_composition_predictor_simple(
    combined_df, categories, use_total_params=False
):
    """Simplified version with explicit bounds and model size handling"""
    # Prepare features: 5 domain percentages + log(size)
    if use_total_params:
        X = combined_df[
            [f"domain_{cat}_pct_mean" for cat in categories] + ["total_params"]
        ].copy()
        X["total_params"] = np.log(
            X["total_params"] + 1e-8
        )  # Log transform with stability
    else:
        X = combined_df[[f"domain_{cat}_pct_mean" for cat in categories]].copy()

    y = combined_df[[f"pretraining_summary_percentage_{cat}" for cat in categories]]

    # Initialize weights matrix (6 features × 5 categories)
    n_features = X.shape[1]
    n_categories = len(categories)

    def loss(params):
        weights = params.reshape(n_features, n_categories)
        pred = X.values @ weights
        pred = np.clip(pred, 0, 100)  # Force reasonable range
        return np.mean((pred - y.values) ** 2)

    # Constraints and bounds
    bounds = [
        (0, 1) for _ in range(n_features * n_categories)
    ]  # Non-negative weights ≤1
    cons = {
        "type": "eq",
        "fun": lambda p: np.sum(p) - n_features,
    }  # Sum of weights = num features

    # Optimize with better initialization
    initial_weights = np.ones(n_features * n_categories) / n_features
    result = minimize(
        loss,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-6},
    )

    # Get optimal weights
    weights = result.x.reshape(n_features, n_categories)

    # Generate predictions
    pred = X.values @ weights
    pred = np.clip(pred, 0, 100)  # Clip before normalization
    pred = pred / pred.sum(axis=1, keepdims=True) * 100

    return pd.DataFrame(pred, columns=categories), weights


def predict_with_size(model, test_df, categories, use_total_params=False):
    """Make predictions using pre-split test data"""
    # Prepare test features
    if use_total_params:
        X_test = test_df[
            [f"domain_{cat}_pct_mean" for cat in categories] + ["total_params"]
        ].copy()
    else:
        X_test = test_df[[f"domain_{cat}_pct_mean" for cat in categories]].copy()

    if use_total_params:
        X_test["total_params"] = np.log(X_test["total_params"] + 1e-8)

    # Transform using training scaler
    X_scaled = model["scaler"].transform(X_test)

    # Calculate predictions
    logits = X_scaled @ model["weights"]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    pred = exp / exp.sum(axis=1, keepdims=True) * 100

    return pd.DataFrame(pred, index=test_df.index, columns=categories)


def analyze_performance_simple(true_df, pred_df, categories):
    """Calculate MAE and correlations with proper index alignment"""
    # Reset indices for alignment
    true_df = true_df.reset_index(drop=True)
    pred_df = pred_df.reset_index(drop=True)

    # Filter NaN predictions
    valid_mask = pred_df[categories].notna().all(axis=1)
    filtered_pred = pred_df.loc[valid_mask].copy()
    filtered_true = true_df.loc[valid_mask].copy()

    # Calculate metrics
    errors = {}
    for cat in categories:
        true_col = f"pretraining_summary_percentage_{cat}"
        pred_col = cat

        errors[cat] = {
            "mae": np.mean(np.abs(filtered_true[true_col] - filtered_pred[pred_col])),
            "corr": stats.pearsonr(filtered_true[true_col], filtered_pred[pred_col])[0]
            if len(filtered_true) > 1
            else np.nan,
        }

    # Calculate total MAE
    total_mae = (
        np.abs(
            filtered_true[
                [f"pretraining_summary_percentage_{cat}" for cat in categories]
            ].values
            - filtered_pred[categories].values
        )
        .sum(axis=1)
        .mean()
    )

    errors["total"] = {"mae": total_mae, "corr": np.nan}

    return pd.DataFrame(errors).T


def analyze_performance(true_values_df, predictions_df, categories):
    """Calculate MAE and correlations using DataFrame indices for alignment"""
    errors = {}

    for cat in categories:
        true_col = f"pretraining_summary_percentage_{cat}"
        pred_col = cat

        # Align using indices
        aligned_true = true_values_df[true_col].loc[predictions_df.index]
        aligned_pred = predictions_df[pred_col]

        errors[cat] = {
            "mae": np.mean(np.abs(aligned_true - aligned_pred)),
            "corr": stats.pearsonr(aligned_true, aligned_pred)[0],
        }

    # Calculate total error
    aligned_true_all = true_values_df[
        [f"pretraining_summary_percentage_{cat}" for cat in categories]
    ].loc[predictions_df.index]
    errors["total"] = np.mean(
        np.abs(aligned_true_all.values - predictions_df.values).sum(axis=1)
    )

    return pd.DataFrame(errors)


# Main workflow
categories = ["web", "code", "books", "reference", "academic"]
USE_TOTAL_PARAMS = False

model_feats_df = pd.read_csv(
    "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_02_02_imputed.duckdb/model_annotations.csv"
)
model_feats_df = model_feats_df[["id", "total_params"]]
# combined_df = combined_df.merge(model_feats_df, on="id", how="left")


combined_df = ground_truth_df.merge(predicted_df, on="id").merge(
    model_feats_df, on="id"
)

# Train/test split
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Train predictor
train_preds, model_weights = constrained_composition_predictor_simple(
    train_df, categories, use_total_params=USE_TOTAL_PARAMS
)

# Predict on test set
if USE_TOTAL_PARAMS:
    test_feats = test_df[
        [f"domain_{cat}_pct_mean" for cat in categories] + ["total_params"]
    ]
else:
    test_feats = test_df[[f"domain_{cat}_pct_mean" for cat in categories]]

# test_feats = test_df[[f"domain_{cat}_pct_mean" for cat in categories]]
test_preds = test_feats.values @ model_weights
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True) * 100
test_preds = pd.DataFrame(test_preds, columns=categories, index=test_df.index)

# Analyze performance
train_errors = analyze_performance_simple(train_df, train_preds, categories)
test_errors = analyze_performance_simple(test_df, test_preds, categories)

print("Training Errors:")
print(train_errors)
print("\nTest Errors:")
print(test_errors)

# # Split data
# train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# # Train model
# model = constrained_composition_predictor_simple(train_df, categories)

# # Make predictions
# train_preds = predict_with_size(model, train_df, categories)
# test_preds = predict_with_size(model, test_df, categories)

# # Analyze performance
# train_errors = analyze_performance(train_df, train_preds, categories)
# test_errors = analyze_performance(test_df, test_preds, categories)

# print("Training Errors:")
# print(train_errors)
# print("\nTest Errors:")
# print(test_errors)

### Preds for unknown models
# Load data
ground_truth_df = pd.read_csv(
    "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/pretraining_doc_tagging/ground_truth_dataset_info.csv"
)[
    [
        "id",
        "pretraining_summary_percentage_web",
        "pretraining_summary_percentage_code",
        "pretraining_summary_percentage_books",
        "pretraining_summary_percentage_reference",
        "pretraining_summary_percentage_academic",
    ]
]

predicted_df = pd.read_csv("all_models_feature_stats_2_03.csv")[
    [
        "id",
        "domain_academic_pct_mean",
        "domain_books_pct_mean",
        "domain_code_pct_mean",
        "domain_reference_pct_mean",
        "domain_web_pct_mean",
    ]
]

# Define models to exclude
models_to_exclude = [
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

# Merge and filter data
combined = ground_truth_df.merge(
    predicted_df[~predicted_df["id"].isin(models_to_exclude)], on="id", how="right"
)
combined = combined.merge(model_feats_df, on="id", how="left")

# Identify models with missing ground truth
missing_ground_truth_mask = (
    combined[
        [
            "pretraining_summary_percentage_web",
            "pretraining_summary_percentage_code",
            "pretraining_summary_percentage_books",
            "pretraining_summary_percentage_reference",
            "pretraining_summary_percentage_academic",
        ]
    ]
    .isna()
    .all(axis=1)
)

# Filter for models with complete feature data
unknown_models = combined[missing_ground_truth_mask].dropna(
    subset=[
        "domain_web_pct_mean",
        "domain_code_pct_mean",
        "domain_books_pct_mean",
        "domain_reference_pct_mean",
        "domain_academic_pct_mean",
    ]
)

# Prepare features for prediction
prediction_features = unknown_models[
    [
        "domain_web_pct_mean",
        "domain_code_pct_mean",
        "domain_books_pct_mean",
        "domain_reference_pct_mean",
        "domain_academic_pct_mean",
    ]
]

breakpoint()
predictions = prediction_features.values @ model_weights
predictions = predictions / predictions.sum(axis=1, keepdims=True) * 100
predictions = pd.DataFrame(predictions, columns=categories, index=unknown_models.index)
predictions["id"] = unknown_models["id"]
predictions = predictions.round(2)
print(predictions)
