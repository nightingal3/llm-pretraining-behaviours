import argparse
import pandas as pd
import altair as alt
from pathlib import Path
import statsmodels.api as sm

def calculate_r_squared(x, y):
    """ Fit a regression model and calculate R squared """
    x_with_const = sm.add_constant(x)  # adding a constant
    model = sm.OLS(y, x_with_const)
    results = model.fit()
    return results.rsquared

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", type=str, default="/data/tir/projects/tir6/general/mengyan3/tower-llm-training/mispredictions/mean_errors.csv")
    parser.add_argument("--data_feats_csv", type=str, default="/data/tir/projects/tir6/general/mengyan3/tower-llm-training/performance_prediction/gathered_data/training_dataset_final_revised.csv")
    parser.add_argument("--model_feats_csv", type=str, default="/data/tir/projects/tir6/general/mengyan3/tower-llm-training/performance_prediction/gathered_data/training_model_final.csv")
    args = parser.parse_args()

    df_errors = pd.read_csv(args.features_csv)
    df_data_feats = pd.read_csv(args.data_feats_csv)
    df_model_feats = pd.read_csv(args.model_feats_csv)

    df_1 = pd.merge(df_data_feats, df_model_feats, on="id")
    df_2 = pd.merge(df_1, df_errors, left_on="id", right_on="model")
    
    data_feats = list(set(df_data_feats.columns) - set(["id", "sft_summary:total_tokens_billions"]))
    model_feats = list(set(df_model_feats.columns) - set(["id", "deep_key", "safetensors:total"]))

    for data_feat in data_feats:
        print(data_feat)

        sel_data = df_2.copy(deep=True)[[data_feat, "mean_error", "id"]]
        sel_data = sel_data.dropna()


        if len(sel_data) <= 10:
            print(f"Skipping {data_feat} due to insufficient data")
            continue
        if ":" in data_feat: # rename the col to avoid clash with altair
            sel_data = sel_data.rename(columns={data_feat: data_feat.replace(":", "_")})
            data_feat = data_feat.replace(":", "_")


        # Calculate R^2 value
        r_squared = calculate_r_squared(sel_data[data_feat], sel_data["mean_error"])
        r_squared_text = f"R^2 = {r_squared:.2f}"

        # Create the chart
        points = alt.Chart(sel_data).mark_point().encode(
            x=alt.X(data_feat, title=data_feat),
            y=alt.Y("mean_error", title="Mean Error"),
            tooltip="id"
        )
        
        regression_line = alt.Chart(sel_data).transform_regression(
            data_feat, "mean_error", method="linear"
        ).mark_line(color='red').encode(
            x=alt.X(data_feat, title=data_feat),
            y=alt.Y("mean_error", title="Mean Error")
        )

        text = alt.Chart({'values':[{}]}).mark_text(
            align='left', dx=5, dy=-5
        ).encode(
            x=alt.value(5),  # pixel offset from left
            y=alt.value(5),  # pixel offset from top
            text=alt.value(r_squared_text)
        )

        final_chart = points + regression_line + text

        # save
        Path("./performance_prediction/figures/mean_errs/data").mkdir(parents=True, exist_ok=True)
        final_chart.save(f"./performance_prediction/figures/mean_errs/data/mean_error_vs_{data_feat}.pdf")

    numeric_features = ["batch_instances", "dimension", "mlp_ratio", "num_heads", "sequence_length", "total_params"]
    for model_feat in model_feats:
        sel_data = df_2[[model_feat, "mean_error", "id"]].dropna()
        if ":" in model_feat:  # rename the col to avoid clash with Altair
            sel_data = sel_data.rename(columns={model_feat: model_feat.replace(":", "_")})
            model_feat = model_feat.replace(":", "_")

        if len(sel_data) <= 10:
            print(f"Skipping {model_feat} due to insufficient data")
            continue
        
        if model_feat in numeric_features:
            # Calculate R^2 value
            r_squared = calculate_r_squared(sel_data[model_feat], sel_data["mean_error"])
            r_squared_text = f"R^2 = {r_squared:.2f}"

            # Create the chart
            points = alt.Chart(sel_data).mark_point().encode(
                x=alt.X(model_feat, title=model_feat),
                y=alt.Y("mean_error", title="Mean Error"),
                tooltip="id"
            )
            
            regression_line = alt.Chart(sel_data).transform_regression(
                model_feat, "mean_error", method="linear"
            ).mark_line(color='red').encode(
                x=alt.X(model_feat, title=model_feat),
                y=alt.Y("mean_error", title="Mean Error")
            )

            text = alt.Chart({'values':[{}]}).mark_text(
                align='left', dx=5, dy=-5
            ).encode(
                x=alt.value(5),  # pixel offset from left
                y=alt.value(5),  # pixel offset from top
                text=alt.value(r_squared_text)
            )

            plot = points + regression_line + text

        else:
        # Create the boxplot
            plot = alt.Chart(sel_data).mark_boxplot().encode(
                x=alt.X(model_feat, title=model_feat),
                y=alt.Y('mean_error', title='Mean Error'),
                color=alt.Color(model_feat, legend=None)
            )

        # Save the chart
        Path("./performance_prediction/figures/mean_errs/model").mkdir(parents=True, exist_ok=True)
        plot.save(f"./performance_prediction/figures/mean_errs/model/mean_error_vs_{model_feat}.pdf")

