import pandas as pd
import numpy as np
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cutoff_n",
        type=int,
        default=3,
        help="The top cutoff_n highest errors will be reported for each task ",
    )
    args = parser.parse_args()
    assert args.cutoff_n > 0, "cutoff_n must be positive"

    scores = pd.read_csv("absolute_errors_all_all.csv")
    scores.rename(columns={scores.columns[0]: "model name"}, inplace=True)
    scores.set_index("model name", inplace=True)

    # For each task separately, find the cutoff_n worst performing models
    worst_per_task = {}
    for task in scores.columns:
        cutoff = sorted(scores[task].dropna())[-args.cutoff_n]
        worst_per_task[task] = [
            (model, score)
            for model, score in scores[scores[task] >= cutoff][task].items()
        ]
    worst_per_task_json = json.dumps(worst_per_task, indent=2)
    with open("highest_errors_per_task.json", "w") as file:
        file.write(worst_per_task_json)

    # Find the 10 models with the highest MAE across all tasks
    scores["mae"] = scores.mean(axis=1)
    cutoff = sorted(scores["mae"])[-10]
    worst_overall = scores.loc[scores["mae"] >= cutoff, ["mae"]].sort_values(by="mae")
    worst_overall_json = worst_overall.to_json(indent=2)
    with open("highest_errors_overall.json", "w") as file:
        file.write(worst_overall_json)
