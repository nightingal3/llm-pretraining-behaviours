import argparse
import pandas as pd
import altair as alt
from pathlib import Path
import statsmodels.api as sm
import numpy as np
from metadata.duckdb.model_metadata_db import AnalysisStore

from performance_predict_from_db import load_data_from_db


def calculate_r_squared(x, y):
    """Fit a regression model and calculate R squared"""
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    return results.rsquared


def plot_line_plot(
    sel_data: pd.DataFrame,
    data_feat: str,
    apply_log: bool = False,
    threshold: float = None,
) -> alt.Chart:
    if apply_log:
        sel_data[data_feat] = pd.to_numeric(sel_data[data_feat], errors="coerce")
        sel_data[data_feat] = np.log(sel_data[data_feat])
        data_feat_label = f"log({data_feat})"
        min_val, max_val = sel_data[data_feat].min(), sel_data[data_feat].max()
    else:
        data_feat_label = data_feat
        min_val, max_val = sel_data[data_feat].min(), sel_data[data_feat].max()

    # Calculate R^2 value
    try:
        r_squared = calculate_r_squared(sel_data[data_feat], sel_data["SignedError"])
    except ValueError:
        sel_data[data_feat] = sel_data[data_feat].astype(float)
        r_squared = calculate_r_squared(sel_data[data_feat], sel_data["SignedError"])
    r_squared_text = f"R^2 = {r_squared:.2f}"

    # Create the points chart
    points = (
        alt.Chart(sel_data)
        .mark_point()
        .encode(
            x=alt.X(
                data_feat,
                title=data_feat_label,
                scale=alt.Scale(domain=(min_val, max_val)),
            ),
            y=alt.Y("SignedError", title="Mean Error"),
        )
    )

    # Create the text labels only if a threshold is specified
    if threshold is not None:
        # Filter data for labels based on the threshold
        label_data = sel_data[sel_data["SignedError"].abs() > threshold]
        labels = (
            alt.Chart(label_data)
            .mark_text(align="left", dx=5, dy=-5, fontSize=8)
            .encode(
                x=alt.X(
                    data_feat,
                    title=data_feat_label,
                    scale=alt.Scale(domain=(min_val, max_val)),
                ),
                y=alt.Y("SignedError", title="Mean Error"),
                text="Model",
            )
        )
        final_chart = points + labels

    else:
        final_chart = points

    regression_line = (
        alt.Chart(sel_data)
        .transform_regression(data_feat, "SignedError", method="linear")
        .mark_line(color="red")
        .encode(
            x=alt.X(data_feat, title=data_feat),
            y=alt.Y("SignedError", title="Mean Error"),
        )
    )

    r_squared_label = (
        alt.Chart({"values": [{}]})
        .mark_text(align="left", dx=5, dy=-5)
        .encode(
            x=alt.value(5),
            y=alt.value(5),
            text=alt.value(r_squared_text),
        )
    )

    final_chart = final_chart + regression_line + r_squared_label
    return final_chart


def plot_box_plot(sel_data: pd.DataFrame, model_feat: str) -> alt.Chart:
    plot = (
        alt.Chart(sel_data)
        .mark_boxplot()
        .encode(
            x=alt.X(model_feat, title=model_feat),
            y=alt.Y("SignedError", title="Mean Error"),
            color=alt.Color(model_feat, legend=None),
        )
    )
    return plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_csv",
        type=str,
        help="Path to features CSV file (optional if using DB)",
        required=True,
    )
    parser.add_argument(
        "--data_feats_csv",
        type=str,
        help="Path to data features CSV file (optional if using DB)",
        default=None,
    )
    parser.add_argument(
        "--model_feats_csv",
        type=str,
        help="Path to model features CSV file (optional if using DB)",
        default=None,
    )
    parser.add_argument(
        "--db_path", type=str, help="Path to DuckDB database", default=None
    )
    args = parser.parse_args()

    # Check if we're using DB or CSV files
    if args.db_path:
        all_feats = load_data_from_db(args.db_path, "all", "accuracy")

        df_1 = all_feats.drop(
            columns=[
                "is_instruction_tuned",
                "is_preference_tuned",
                "metadata",
                "benchmark",
                "setting",
                "value",
                "value_stderr",
                "id_1",
            ]
        )

        # just keep one copy of each model since we just need the feats and not performance
        df_1 = df_1.drop_duplicates(subset=["id"])

        model_feats = [
            "dimension",
            "num_heads",
            "mlp_ratio",
            "layer_norm_type",
            "positional_embeddings",
            "attention_variant",
            "biases",
            "block_type",
            "activation",
            "sequence_length",
            "batch_instances",
            "batch_tokens",
            "weight_tying",
            "total_params",
        ]
        df_data_feats = df_1.drop(columns=model_feats)
        df_model_feats = df_1[model_feats + ["id"]]
    else:
        if not all([args.features_csv, args.data_feats_csv, args.model_feats_csv]):
            raise ValueError("Must provide either db_path or all CSV file paths")
        df_errors = pd.read_csv(args.features_csv)
        df_data_feats = pd.read_csv(args.data_feats_csv)
        df_model_feats = pd.read_csv(args.model_feats_csv)

    # Merge dataframes
    if not args.db_path:
        df_1 = pd.merge(df_data_feats, df_model_feats, on="id")
    if args.db_path:
        df_errors = pd.read_csv(args.features_csv)  # Still need errors CSV
    df_2 = pd.merge(df_1, df_errors, left_on="id", right_on="Model")

    # Define feature sets (excluding certain columns)
    data_feats = list(
        set(df_data_feats.columns) - set(["id", "sft_summary:total_tokens_billions"])
    )
    model_feats = list(
        set(df_model_feats.columns) - set(["id", "deep_key", "safetensors:total"])
    )

    # Plot data features
    for data_feat in data_feats:
        print("Processing data feat:", data_feat)
        sel_data = df_2.copy(deep=True)[[data_feat, "SignedError", "Model"]]
        sel_data = sel_data.dropna()

        if len(sel_data) <= 10:
            print(f"Skipping {data_feat} due to insufficient data")
            continue

        apply_log = (
            data_feat == "pretraining_summary_total_tokens_billions"
            or data_feat == "total_params"
        )
        if ":" in data_feat:
            sel_data = sel_data.rename(columns={data_feat: data_feat.replace(":", "_")})
            data_feat = data_feat.replace(":", "_")

        threshold = None if not apply_log else 0.05
        final_chart = plot_line_plot(
            sel_data, data_feat, apply_log=apply_log, threshold=threshold
        )

        # Save plot
        Path(
            "./performance_prediction/figures/mean_errs_revised_1_16_nonsl/data"
        ).mkdir(parents=True, exist_ok=True)
        final_chart.save(
            f"./performance_prediction/figures/mean_errs_revised_1_16_nonsl/data/mean_error_vs_{data_feat}.pdf"
        )

    # Plot model features
    numeric_features = [
        "batch_instances",
        "dimension",
        "mlp_ratio",
        "num_heads",
        "sequence_length",
        "total_params",
    ]

    for model_feat in model_feats:
        print("Processing model feat:", model_feat)
        sel_data = df_2[[model_feat, "SignedError", "Model"]].dropna()
        apply_log = model_feat == "total_params"

        if ":" in model_feat:
            sel_data = sel_data.rename(
                columns={model_feat: model_feat.replace(":", "_")}
            )
            model_feat = model_feat.replace(":", "_")

        if len(sel_data) <= 10:
            print(f"Skipping {model_feat} due to insufficient data")
            continue

        threshold = None if not apply_log else 0.05
        if model_feat in numeric_features:
            plot = plot_line_plot(
                sel_data, model_feat, apply_log=apply_log, threshold=threshold
            )
        else:
            plot = plot_box_plot(sel_data, model_feat)

        # Save plot
        Path(
            "./performance_prediction/figures/mean_errs_revised_1_16_nonsl/model"
        ).mkdir(parents=True, exist_ok=True)
        plot.save(
            f"./performance_prediction/figures/mean_errs_revised_1_16_nonsl/model/mean_error_vs_{model_feat}.pdf"
        )
