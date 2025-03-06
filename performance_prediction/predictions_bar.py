import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from performance_predict_from_db import load_data_from_db

all_preds_df = pd.read_csv("/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/results_db/per_model_predictions_xgboost_all_accuracy.csv")
sl_preds_df = pd.read_csv("/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/results_db/per_model_predictions_xgboost_scaling_laws_accuracy.csv")

db_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2024_12_05.duckdb"
model_info_df = load_data_from_db(db_path, "all", "accuracy")
model_info_df = model_info_df.drop_duplicates(subset=["id"])

df_merged = all_preds_df.merge(
    sl_preds_df, on=["y_col", "Model"], suffixes=("_all", "_sl")
)
df_merged = df_merged.merge(model_info_df, left_on="Model", right_on="id")
df_merged = df_merged.sort_values(by="True_all")

all_tasks = df_merged["y_col"].unique()
all_tasks = [
    task for task in all_tasks if "hendrycks" not in task and "arithmetic" not in task
]
all_tasks.extend(["mmlu", "arithmetic"])

bar_width = 0.25
for task in all_tasks:
    if task == "mmlu":  # Aggregate all the hendrycksTest-related tasks
        df_task = df_merged[df_merged["y_col"].str.contains("hendrycksTest")]
        df_task_grouped = (
            df_task.groupby("Model")
            .agg(
                {
                    "True_all": "mean",
                    "Predicted_all": "mean",
                    "Predicted_sl": "mean",
                    "total_params": "mean",
                }
            )
            .reset_index()
        )
        df_task_grouped.sort_values(by="total_params", inplace=True)

    elif task == "arithmetic":  # Aggregate all the arithmetic-related tasks
        df_task = df_merged[df_merged["y_col"].str.contains("arithmetic")]
        df_task_grouped = (
            df_task.groupby("Model")
            .agg(
                {
                    "True_all": "mean",
                    "Predicted_all": "mean",
                    "Predicted_sl": "mean",
                    "total_params": "mean",
                }
            )
            .reset_index()
        )
        df_task_grouped.sort_values(by="True_all", inplace=True)

    else:  # No aggregation for other tasks
        df_task_grouped = df_merged[df_merged["y_col"] == task]

    # Set the positions of the bars
    r1 = np.arange(len(df_task_grouped))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(
        r1,
        df_task_grouped["True_all"],
        color="b",
        width=bar_width,
        edgecolor="grey",
        label="True",
    )
    plt.bar(
        r2,
        df_task_grouped["Predicted_all"],
        color="r",
        width=bar_width,
        edgecolor="grey",
        label="Predicted (all feats)",
    )
    plt.bar(
        r3,
        df_task_grouped["Predicted_sl"],
        color="g",
        width=bar_width,
        edgecolor="grey",
        label="Predicted (scaling laws)",
    )

    # Add the model names as x-axis labels (sorted by size)
    plt.xlabel("Model", fontweight="bold")
    plt.xticks(
        [r + bar_width for r in range(len(df_task_grouped))],
        df_task_grouped["Model"],
        rotation=90,
    )

    # Add a title and labels
    plt.title(f"True vs Predicted for {task}")

    # Add a legend
    plt.legend()

    # Show or save the plot
    plt.tight_layout()
    # mkdir -p
    Path(f"./preds_and_true_perf_acc_from_db").mkdir(
        parents=True, exist_ok=True
    )
    plt.savefig(f"./preds_and_true_perf_acc_from_db/predictions_{task}.png")
    plt.close()
