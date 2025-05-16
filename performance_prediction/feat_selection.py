import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel


def load_and_prepare_data(
    tagged_feats_path: str, dataset_info_path: str, model_info_path: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and merge feature data."""
    # Load data
    tagged_feats = pd.read_csv(tagged_feats_path)
    dataset_info = pd.read_csv(dataset_info_path)
    model_info = pd.read_csv(model_info_path)
    # Merge datasets
    df = tagged_feats.merge(dataset_info, on="id", how="inner")
    df = df.merge(model_info, on="id", how="inner")

    # drop "id" and "id.1" cols
    df = df.drop(["id.1"], axis=1)

    # Define feature groups for analysis
    SCALING_FEATURES = ["total_params", "pretraining_summary_total_tokens_billions"]

    FREEGEN_FEATURES = [
        "content_function_ratio_mean",
        "edu_classifier_std",
        "const_parse_const_tree_depth_max_std",
        "dep_parse_dep_root_dist_mean_std",
        "entropy_mean",
        "unique_tokens_std",
        "ttr_std",
    ]

    # Return merged data and feature lists
    return df, SCALING_FEATURES + FREEGEN_FEATURES


def elasticnet_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_alphas: int = 100,
    cv: int = 5,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
) -> Tuple[List[str], Dict]:
    """
    Perform feature selection using ElasticNet with cross-validation.
    Returns selected features and importance metrics.
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit ElasticNetCV
    model = ElasticNetCV(
        l1_ratio=l1_ratio, n_alphas=n_alphas, cv=cv, max_iter=max_iter, random_state=42
    )
    model.fit(X_scaled, y)

    # Get feature importance
    importance = np.abs(model.coef_)

    # Create feature importance dict
    feature_importance = {name: imp for name, imp in zip(feature_names, importance)}

    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )

    # Select features with non-zero coefficients
    selected_features = [feat for feat, imp in sorted_features if imp > 0]

    # Additional metrics
    metrics = {
        "alpha": model.alpha_,
        "l1_ratio": l1_ratio,
        "n_selected_features": len(selected_features),
        "feature_importance": feature_importance,
    }

    return selected_features, metrics


def plot_feature_importance(
    feature_importance: Dict[str, float],
    output_path: str,
    title: str = "Feature Importance from ElasticNet",
):
    """Plot feature importance scores."""
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=False
    )

    # Create plot
    plt.figure(figsize=(10, max(8, len(feature_importance) * 0.3)))

    # Plot horizontal bars
    features, scores = zip(*sorted_features)
    y_pos = np.arange(len(features))
    plt.barh(y_pos, scores)

    # Customize plot
    plt.yticks(y_pos, features)
    plt.xlabel("Absolute Coefficient Value")
    plt.title(title)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_selected_features(
    X: pd.DataFrame, y: pd.Series, selected_features: List[str], test_size: float = 0.2
) -> Dict:
    """
    Evaluate the performance of selected features using a simple ElasticNet model.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=test_size, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    # Calculate metrics
    metrics = {
        "train_mae": mean_absolute_error(y_train, train_preds),
        "test_mae": mean_absolute_error(y_test, test_preds),
        "n_features": len(selected_features),
    }

    return metrics


def simple_feature_selection(X, y):
    """Simple feature selection using HGBR importance scores."""
    # Create and fit model
    model = HistGradientBoostingRegressor(random_state=42, max_iter=100)

    # Fit model
    model.fit(X, y)
    breakpoint()

    # Get importance scores
    importance_scores = dict(zip(X.columns, model.feature_importances_))

    # Sort features by importance
    sorted_features = sorted(
        importance_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Print top features
    print("\nTop 10 most important features:")
    for feat, score in sorted_features[:10]:
        print(f"{feat}: {score:.4f}")

    return sorted_features


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Define paths
    tagged_feats_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/tagged_feats.csv"
    dataset_info_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_01_29.duckdb/dataset_info.csv"
    model_info_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_01_29.duckdb/model_annotations.csv"
    evals_lst = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_01_29.duckdb/evaluation_results.csv"
    output_dir = Path("feature_selection_results")
    output_dir.mkdir(exist_ok=True)

    evals_lamb = pd.read_csv(evals_lst)
    evals_lamb = evals_lamb[evals_lamb["benchmark"] == "lambada_standard"]
    evals_lamb = evals_lamb[evals_lamb["metric"] == "accuracy"]
    evals_lamb = evals_lamb[evals_lamb["setting"] == "0-shot"]

    # Load data
    logging.info("Loading data...")
    df, feature_names = load_and_prepare_data(
        tagged_feats_path, dataset_info_path, model_info_path
    )

    # filter evals by id
    labels = evals_lamb[["id", "metric_value"]].copy()

    # Merge features and labels on id
    aligned_data = df.merge(labels, on="id", how="inner")
    categorical_columns = [
        "activation",
        "attention_variant",
        "biases",
        "block_type",
        "layer_norm_type",
        "positional_embeddings",
        "btach_tokens",
        "weight_tying",
    ]
    # Now we have features and labels aligned by id
    X = aligned_data.drop(
        ["id", "metric_value", "is_instruction_tuned", "is_preference_tuned"], axis=1
    )  # Drop non-feature columns
    y = aligned_data["metric_value"]

    X = X[[col for col in X.columns if col not in categorical_columns]]

    breakpoint()

    # Perform feature selection
    logging.info("Performing feature selection...")
    selected_features, metrics = simple_feature_selection(X, y)

    # Plot feature importance
    logging.info("Plotting feature importance...")
    plot_feature_importance(
        metrics["feature_importance"], output_dir / "feature_importance.png"
    )

    # Evaluate selected features
    logging.info("Evaluating selected features...")
    eval_metrics = evaluate_selected_features(X, y, selected_features)

    # Save results
    results = {
        "selected_features": selected_features,
        "feature_importance": metrics["feature_importance"],
        "elasticnet_params": {
            "alpha": metrics["alpha"],
            "l1_ratio": metrics["l1_ratio"],
        },
        "evaluation_metrics": eval_metrics,
    }

    # Save as CSV
    results_df = pd.DataFrame(
        [
            {"feature": feat, "importance": imp}
            for feat, imp in metrics["feature_importance"].items()
        ]
    )

    results_df.to_csv(output_dir / "feature_selection_results.csv", index=False)

    logging.info(f"Selected {len(selected_features)} features")
    logging.info(f"Test MAE: {eval_metrics['test_mae']:.4f}")


if __name__ == "__main__":
    main()
