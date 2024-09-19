import argparse
import pandas as pd
import altair as alt
from pathlib import Path
import statsmodels.api as sm
import numpy as np

def calculate_r_squared(x, y):
    """Fit a regression model and calculate R squared"""
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    return results.rsquared
def plot_line_plot(sel_data: pd.DataFrame, data_feat: str, apply_log: bool = False, threshold: float = None) -> alt.Chart:
    if apply_log:
        sel_data[data_feat] = pd.to_numeric(sel_data[data_feat], errors='coerce')
        sel_data[data_feat] = np.log(sel_data[data_feat])
        data_feat_label = f"log({data_feat})"

        min_val, max_val = sel_data[data_feat].min(), sel_data[data_feat].max()
    else:
        data_feat_label = data_feat

        min_val, max_val = sel_data[data_feat].min(), sel_data[data_feat].max()

    # Calculate R^2 value
    try:
        r_squared = calculate_r_squared(sel_data[data_feat], sel_data["mean_signed_error"])
    except ValueError:
        sel_data[data_feat] = sel_data[data_feat].astype(float)
        r_squared = calculate_r_squared(sel_data[data_feat], sel_data["mean_signed_error"])
    r_squared_text = f"R^2 = {r_squared:.2f}"

    # Create the points chart
    points = (
        alt.Chart(sel_data)
        .mark_point()
        .encode(
            x=alt.X(data_feat, title=data_feat_label, scale=alt.Scale(domain=(min_val, max_val))),
            y=alt.Y("mean_signed_error", title="Mean Error"),
        )
    )

    # Create the text labels only if a threshold is specified
    if threshold is not None:
        # Filter data for labels based on the threshold
        label_data = sel_data[(sel_data["mean_signed_error"].abs() > threshold)]
        labels = (
            alt.Chart(label_data)
            .mark_text(align='left', dx=5, dy=-5, fontSize=8)  # Smaller font size
            .encode(
                x=alt.X(data_feat, title=data_feat_label, scale=alt.Scale(domain=(min_val, max_val))),
                y=alt.Y("mean_signed_error", title="Mean Error"),
                text='id'  # Use the 'id' column for the text
            )
        )
        final_chart = points + labels  # Add labels only if there are labels to add
    else:
        final_chart = points

    regression_line = (
        alt.Chart(sel_data)
        .transform_regression(data_feat, "mean_signed_error", method="linear")
        .mark_line(color="red")
        .encode(
            x=alt.X(data_feat, title=data_feat),
            y=alt.Y("mean_signed_error", title="Mean Error"),
        )
    )

    r_squared_label = (
        alt.Chart({"values": [{}]})
        .mark_text(align="left", dx=5, dy=-5)
        .encode(
            x=alt.value(5),  # pixel offset from left
            y=alt.value(5),  # pixel offset from top
            text=alt.value(r_squared_text),
        )
    )

    final_chart = final_chart + regression_line + r_squared_label

    return final_chart

def _plot_line_plot(sel_data: pd.DataFrame, data_feat: str, apply_log: bool = False) -> alt.Chart:
    if apply_log:
        sel_data[data_feat] = pd.to_numeric(sel_data[data_feat], errors='coerce')
        sel_data[data_feat] = np.log(sel_data[data_feat])
        data_feat_label = f"log({data_feat})"

        min_val, max_val = sel_data[data_feat].min(), sel_data[data_feat].max()
    else:
        data_feat_label = data_feat

        min_val, max_val = sel_data[data_feat].min(), sel_data[data_feat].max()

    # Calculate R^2 value
    try:
        r_squared = calculate_r_squared(sel_data[data_feat], sel_data["mean_signed_error"])
    except ValueError:
        sel_data[data_feat] = sel_data[data_feat].astype(float)
        r_squared = calculate_r_squared(sel_data[data_feat], sel_data["mean_signed_error"])
    r_squared_text = f"R^2 = {r_squared:.2f}"

    # Create the chart
    points = (
        alt.Chart(sel_data)
        .mark_point()
        .encode(
            x=alt.X(data_feat, title=data_feat_label, scale=alt.Scale(domain=(min_val, max_val))),
            y=alt.Y("mean_signed_error", title="Mean Error"),
            text="id"
        )
    )

    regression_line = (
        alt.Chart(sel_data)
        .transform_regression(data_feat, "mean_signed_error", method="linear")
        .mark_line(color="red")
        .encode(
            x=alt.X(data_feat, title=data_feat),
            y=alt.Y("mean_signed_error", title="Mean Error"),
        )
    )

    text = (
        alt.Chart({"values": [{}]})
        .mark_text(align="left", dx=5, dy=-5)
        .encode(
            x=alt.value(5),  # pixel offset from left
            y=alt.value(5),  # pixel offset from top
            text=alt.value(r_squared_text),
        )
    )

    final_chart = points + regression_line + text

    return final_chart


def plot_box_plot(sel_data: pd.DataFrame, model_feat: str) -> alt.Chart:
    # Create the boxplot
    plot = (
        alt.Chart(sel_data)
        .mark_boxplot()
        .encode(
            x=alt.X(model_feat, title=model_feat),
            y=alt.Y("mean_signed_error", title="Mean Error"),
            color=alt.Color(model_feat, legend=None),
        )
    )

    return plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_csv",
        type=str,
        default="./performance_prediction/mispredictions/mean_signed_errors_all_all_metric_acc_xgboost.csv",
    )
    parser.add_argument(
        "--data_feats_csv",
        type=str,
        default="./performance_prediction/gathered_data/training_dataset_final_revised.csv",
    )
    parser.add_argument(
        "--model_feats_csv",
        type=str,
        default="./performance_prediction/gathered_data/training_model_final.csv",
    )
    args = parser.parse_args()

    df_errors = pd.read_csv(args.features_csv)
    df_data_feats = pd.read_csv(args.data_feats_csv)
    df_model_feats = pd.read_csv(args.model_feats_csv)

    df_1 = pd.merge(df_data_feats, df_model_feats, on="id")
    df_2 = pd.merge(df_1, df_errors, left_on="id", right_on="model")

    data_feats = list(
        set(df_data_feats.columns) - set(["id", "sft_summary:total_tokens_billions"])
    )
    model_feats = list(
        set(df_model_feats.columns) - set(["id", "deep_key", "safetensors:total"])
    )

    for data_feat in data_feats:
        print(data_feat)

        sel_data = df_2.copy(deep=True)[[data_feat, "mean_signed_error", "id"]]
        sel_data = sel_data.dropna()

        if len(sel_data) <= 10:
            print(f"Skipping {data_feat} due to insufficient data")
            continue

        apply_log = (data_feat == "pretraining_summary:total_tokens_billions" or data_feat == "total_params")
        if ":" in data_feat:  # rename the col to avoid clash with altair
            sel_data = sel_data.rename(columns={data_feat: data_feat.replace(":", "_")})
            data_feat = data_feat.replace(":", "_")

        threshold = None if not apply_log else 0.05
        final_chart = plot_line_plot(sel_data, data_feat, apply_log =apply_log, threshold=threshold)

        # save
        Path("./performance_prediction/figures/mean_errs_revised/data").mkdir(
            parents=True, exist_ok=True
        )
        final_chart.save(
            f"./performance_prediction/figures/mean_errs_revised/data/mean_error_vs_{data_feat}.pdf"
        )

    numeric_features = [
        "batch_instances",
        "dimension",
        "mlp_ratio",
        "num_heads",
        "sequence_length",
        "total_params",
    ]
    for model_feat in model_feats:
        sel_data = df_2[[model_feat, "mean_signed_error", "id"]].dropna()
        apply_log = (model_feat == "total_params")
        if ":" in model_feat:  # rename the col to avoid clash with Altair
            sel_data = sel_data.rename(
                columns={model_feat: model_feat.replace(":", "_")}
            )
            model_feat = model_feat.replace(":", "_")

        if len(sel_data) <= 10:
            print(f"Skipping {model_feat} due to insufficient data")
            continue
        
        threshold = None if not apply_log else 0.05
        if model_feat in numeric_features:
            plot = plot_line_plot(sel_data, model_feat, apply_log=apply_log, threshold=threshold)

        else:
            # Create the boxplot
            plot = plot_box_plot(sel_data, model_feat)

        # Save the chart
        Path("./performance_prediction/figures/mean_errs_revised/model").mkdir(
            parents=True, exist_ok=True
        )
        plot.save(
            f"./performance_prediction/figures/mean_errs_revised/model/mean_error_vs_{model_feat}.pdf"
        )
