import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def prepare_feature(df, feature):
    """Prepare a single feature for modeling"""
    x = df[feature].copy()
    feature_name = feature

    # Take log of specific features
    if feature in ["pretraining_summary_total_tokens_billions", "total_params"]:
        x = np.log(x)
        feature_name = f"log_{feature}"

    if x.dtype == "object":
        # For categorical features, create dummy variables
        x = pd.get_dummies(x, dummy_na=True)
        # If multiple columns were created, use all of them
        if isinstance(x, pd.DataFrame):
            x.columns = [str(col) for col in x.columns]
            return x, feature_name
    else:
        # Treat both NaN and -1 as missing values
        missing_mask = (x == -1) | pd.isna(x)
        valid_values = x[~missing_mask]

        if len(valid_values) == 0:  # If all values are missing
            x[missing_mask] = 0
        else:
            mean_val = valid_values.mean()
            x[missing_mask] = mean_val

    return pd.DataFrame({str(feature): x}), feature_name


def get_mae_scores(X, y):
    """Get cross-validated MAE scores for linear regression"""
    # Convert to numpy array
    X = X.values

    # Handle NaN values in target
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    if len(y) < 30:  # Skip if too few samples
        return np.inf, []

    # Fill any remaining NaN values with 0
    X = np.nan_to_num(X, 0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use 3-fold cross-validation
    cv = min(3, len(y))
    mae_scores = cross_val_score(
        LinearRegression(), X_scaled, y, scoring="neg_mean_absolute_error", cv=cv
    )
    return -mae_scores.mean(), mae_scores


def greedy_feature_selection(df, y, feature_names, threshold=0.001):
    """Greedily select features that reduce MAE the most"""
    results = []

    # Calculate baseline MAE (mean prediction)
    valid_y = y[~np.isnan(y)]
    if len(valid_y) < 10:
        print("Not enough valid samples for this benchmark")
        return results

    baseline_mae = np.abs(valid_y - np.mean(valid_y)).mean()
    results.append(("baseline", None, baseline_mae))
    print(f"0. mean baseline: {baseline_mae:.4f} MAE")

    selected_features = []
    selected_feature_names = []  # Track the display names
    best_mae = baseline_mae

    while True:
        best_improvement = -threshold
        best_feature = None
        best_feature_name = None
        best_X = None
        best_scores = None

        # Try adding each remaining feature
        for feature in feature_names:
            if feature in selected_features:
                continue

            try:
                # Prepare current feature
                X_new, feature_name = prepare_feature(df, feature)

                if len(selected_features) > 0:
                    # Combine with previously selected features
                    X_prev_list = []
                    for f in selected_features:
                        X_f, _ = prepare_feature(df, f)
                        X_prev_list.append(X_f)
                    X_prev = pd.concat(X_prev_list, axis=1)
                    X_current = pd.concat([X_prev, X_new], axis=1)
                else:
                    X_current = X_new

                # Calculate MAE with this feature added
                mae, scores = get_mae_scores(X_current, y)
                improvement = best_mae - mae

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_feature = feature
                    best_feature_name = feature_name
                    best_X = X_current
                    best_scores = scores
            except Exception as e:
                print(f"Warning: Error processing feature {feature}: {str(e)}")
                continue

        # If no feature improves MAE by more than threshold, stop
        if best_improvement < threshold:
            break

        # Add best feature to selected features
        selected_features.append(best_feature)
        selected_feature_names.append(best_feature_name)
        best_mae = best_mae - best_improvement

        # Record result
        results.append((best_feature_name, best_improvement, best_mae))

        # Print result with both MAE and -MAE scores
        print(f"{len(selected_features)}. add [{best_feature_name}]:")
        print(f"   MAE: {best_mae:.4f} (improved by {best_improvement:.4f})")
        print(f"   Cross-validation scores (-MAE):")
        for i, score in enumerate(best_scores, 1):
            print(f"     Fold {i}: {score:.4f}")
        print(f"   Mean -MAE: {np.mean(best_scores):.4f} ± {np.std(best_scores):.4f}")

    return results


def plot_benchmark_progression(results, benchmark_name):
    """Plot MAE progression for a benchmark"""
    features = [r[0] for r in results]
    maes = [r[2] for r in results]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(features)), maes)
    plt.title(f"MAE Progression for {benchmark_name}")
    plt.xlabel("Features Added")
    plt.ylabel("MAE")

    # Rotate feature names for better readability
    plt.xticks(range(len(features)), features, rotation=45, ha="right")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(f'mae_progression_{benchmark_name.replace("/", "_")}.png')
    plt.close()


def plot_feature_frequency(feature_counts):
    """Plot feature frequency across all benchmarks"""
    # Sort features by frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    features, counts = zip(*sorted_features)

    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(features)), counts)
    plt.title("Feature Frequency Across All Benchmarks")
    plt.xlabel("Features")
    plt.ylabel("Number of Times Selected")

    # Rotate feature names for better readability
    plt.xticks(range(len(features)), features, rotation=45, ha="right")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("feature_frequency.png")
    plt.close()


def plot_average_mae_reduction(all_results, feature_names):
    """Plot average MAE reduction by each feature"""
    # Initialize dictionary to store MAE reductions
    mae_reductions = {feature: [] for feature in feature_names}

    # Collect MAE reductions for each feature across all benchmarks
    for benchmark_results in all_results:
        for feature_name, improvement, _ in benchmark_results[1:]:  # Skip baseline
            mae_reductions[feature_name].append(improvement)

    # Calculate average MAE reduction
    avg_reductions = {}
    for feature, reductions in mae_reductions.items():
        if len(reductions) > 0:
            avg_reductions[feature] = np.mean(reductions)
        else:
            avg_reductions[feature] = 0

    # Sort features by average reduction
    sorted_features = sorted(avg_reductions.items(), key=lambda x: x[1], reverse=True)
    features, reductions = zip(*sorted_features)

    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(features)), reductions)
    plt.title("Average MAE Reduction by Feature")
    plt.xlabel("Features")
    plt.ylabel("Average MAE Reduction")

    # Rotate feature names for better readability
    plt.xticks(range(len(features)), features, rotation=45, ha="right")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("average_mae_reduction.png")
    plt.close()


def main():
    # Read all input files
    model_annotations = pd.read_csv(
        "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_02_09.duckdb/model_annotations.csv"
    )
    all_models = pd.read_csv("./all_models_feature_stats_2_03.csv")
    dataset_info = pd.read_csv(
        "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_02_09.duckdb/dataset_info.csv"
    )
    aggregated_evals = pd.read_csv("./aggregated_evals.csv")

    # Create pivot table for benchmarks
    output_features = aggregated_evals.pivot(
        index="id", columns=["benchmark", "setting"], values="value"
    )
    output_features.columns = [
        f"{benchmark}_{setting}" for benchmark, setting in output_features.columns
    ]

    # Merge all input feature dataframes
    input_features = model_annotations.set_index("id")
    input_features = input_features.join(all_models.set_index("id"))
    input_features = input_features.join(dataset_info.set_index("id"))

    # Create final dataframe with input and output features
    final_df = pd.concat([input_features, output_features], axis=1)
    mask = ~final_df["truthfulqa_0-shot"].isna() & ~final_df["total_params"].isna()
    plt.scatter(final_df[mask]["total_params"], final_df[mask]["truthfulqa_0-shot"])
    plt.xlabel("Total params (%)")
    plt.ylabel("TruthfulQA Score")
    plt.title("Total Params vs TruthfulQA Performance")
    plt.savefig("total_params_vs_truthfulqa.png")

    # Get input features (excluding metadata column which is text)
    input_feature_names = [col for col in input_features.columns if col != "metadata"]
    output_feature_names = output_features.columns.tolist()

    # Print dataset info
    print("Number of models:", len(final_df))

    # Track feature frequency and results
    feature_counts = {}
    all_results = []
    benchmark_maes = []  # New list to track benchmark MAEs

    # For each benchmark
    for benchmark in output_feature_names:
        print(f"\nExperiment for {benchmark}:")
        y = final_df[benchmark].values
        valid_samples = np.sum(~np.isnan(y))
        print(f"Number of models with scores: {valid_samples}")

        # Run greedy feature selection
        results = greedy_feature_selection(final_df, y, input_feature_names)
        all_results.append(results)

        # Store final MAE for this benchmark if we have results
        if len(results) > 0:
            final_mae = results[-1][2]  # Last result contains final MAE
            benchmark_maes.append(
                {
                    "benchmark": benchmark,
                    "mae": final_mae,
                    "n_samples": valid_samples,
                    "n_features": len(results) - 1,  # Subtract 1 to exclude baseline
                }
            )

            plot_benchmark_progression(results, benchmark)

            # Update feature counts
            for feature, _, _ in results[1:]:  # Skip baseline
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Print feature frequency
    print("\nFeature frequency across all benchmarks:")
    for feature, count in sorted(
        feature_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature}: {count}")

    # Plot feature frequency and average MAE reduction
    plot_feature_frequency(feature_counts)
    plot_average_mae_reduction(all_results, list(feature_counts.keys()))

    # Create and display summary table
    mae_df = pd.DataFrame(benchmark_maes)
    mae_df = mae_df.sort_values("mae")  # Sort by MAE

    # Calculate mean/std across benchmarks
    mean_mae = mae_df["mae"].mean()
    std_mae = mae_df["mae"].std()

    # Add summary row
    summary_row = pd.DataFrame(
        [
            {
                "benchmark": "AVERAGE",
                "mae": mean_mae,
                "n_samples": mae_df["n_samples"].mean(),
                "n_features": mae_df["n_features"].mean(),
            }
        ]
    )
    mae_df = pd.concat([mae_df, summary_row])

    # Format the table
    mae_df["mae"] = mae_df["mae"].round(4)
    mae_df["n_samples"] = mae_df["n_samples"].astype(int)
    mae_df["n_features"] = mae_df["n_features"].astype(int)

    # Save to CSV
    mae_df.to_csv("linear_regression_mae_summary.csv", index=False)

    # Print formatted table
    print("\nMAE Summary for All Benchmarks:")
    print("=" * 80)
    print(mae_df.to_string(index=False))
    print("=" * 80)
    print(f"\nOverall MAE: {mean_mae:.4f} ± {std_mae:.4f}")

    # Print feature frequency
    print("\nFeature frequency across all benchmarks:")
    for feature, count in sorted(
        feature_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature}: {count}")

    # Plot feature frequency and average MAE reduction
    plot_feature_frequency(feature_counts)
    plot_average_mae_reduction(all_results, list(feature_counts.keys()))


if __name__ == "__main__":
    main()
