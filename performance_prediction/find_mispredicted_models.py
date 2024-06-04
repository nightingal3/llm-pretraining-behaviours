import pandas as pd
import numpy as np
import json
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per_task_cutoff_n",
        type=int,
        default=3,
        help="The top per_task_cutoff_n largest positive/negative errors will be reported for each task",
    )
    parser.add_argument(
        "--all_tasks_cutoff_n",
        type=int,
        default=5,
        help="The top all_tasks_cutoff_n largest positive/negative errors across all task/model pairs will be reported",
    )
    args = parser.parse_args()
    assert args.per_task_cutoff_n > 0, "per_task_cutoff_n must be positive"
    assert args.all_tasks_cutoff_n > 0, "all_tasks_cutoff_n must be positive"

    errors = pd.read_csv("errors_all_all.csv")
    errors.rename(columns={errors.columns[0]: "model name"}, inplace=True)
    errors.set_index("model name", inplace=True)
    signed_errors = errors.copy().filter(regex="^SErr")
    signed_errors.columns = [col[5:] for col in signed_errors.columns]
    unsigned_errors = errors.copy().filter(regex="^AErr")
    unsigned_errors.columns = [col[5:] for col in unsigned_errors.columns]

    output_dir = "mispredictions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each task separately, find the per_task_cutoff_n largest positive/negative errors
    worst_per_task_pos = {}  # overpredictions
    worst_per_task_neg = {}  # underpredictions
    for task in signed_errors.columns:
        pos_errors = sorted(
            [score for score in signed_errors[task].dropna() if score > 0]
        )
        neg_errors = sorted(
            [score for score in signed_errors[task].dropna() if score < 0]
        )
        pos_cutoff = (
            0
            if len(pos_errors) < args.per_task_cutoff_n
            else pos_errors[-args.per_task_cutoff_n]
        )
        neg_cutoff = (
            0
            if len(neg_errors) < args.per_task_cutoff_n
            else neg_errors[args.per_task_cutoff_n]
        )
        worst_per_task_pos[task] = [
            (model, score)
            for model, score in signed_errors[signed_errors[task] >= pos_cutoff][
                task
            ].items()
        ]
        worst_per_task_neg[task] = [
            (model, score)
            for model, score in signed_errors[signed_errors[task] < neg_cutoff][
                task
            ].items()
        ]
    worst_per_task_pos_json = json.dumps(worst_per_task_pos, indent=2)
    with open(
        os.path.join(output_dir, "largest_pos_errors_per_task.json"), "w"
    ) as file:
        file.write(worst_per_task_pos_json)
    worst_per_task_neg_json = json.dumps(worst_per_task_neg, indent=2)
    with open(
        os.path.join(output_dir, "largest_neg_errors_per_task.json"), "w"
    ) as file:
        file.write(worst_per_task_neg_json)

    # Across all task/model pairs, find the all_tasks_cutoff_n largest positive/negative errors
    all_signed_errors = []
    all_signed_errors = [
        (model, task, score)
        for task, col in signed_errors.items()
        for model, score in col.items()
        if not pd.isnull(score)
    ]
    all_signed_errors.sort(key=lambda x: x[2])
    worst_all_pairs_neg = all_signed_errors[: args.all_tasks_cutoff_n]
    worst_all_pairs_neg_json = json.dumps(
        [(model, task, score) for model, task, score in worst_all_pairs_neg], indent=2
    )
    with open(os.path.join(output_dir, "largest_single_neg_errors.json"), "w") as file:
        file.write(worst_all_pairs_neg_json)
    worst_all_pairs_pos = all_signed_errors[-args.all_tasks_cutoff_n :]
    worst_all_pairs_pos_json = json.dumps(
        [(model, task, score) for model, task, score in worst_all_pairs_pos], indent=2
    )
    with open(os.path.join(output_dir, "largest_single_pos_errors.json"), "w") as file:
        file.write(worst_all_pairs_pos_json)

    # Find the 10 models with the highest MAE across all tasks
    unsigned_errors["mae"] = unsigned_errors.mean(axis=1)
    cutoff = sorted(unsigned_errors["mae"])[-10]
    worst_overall = unsigned_errors.loc[
        unsigned_errors["mae"] >= cutoff, ["mae"]
    ].sort_values(by="mae")
    worst_overall_json = worst_overall.to_json(indent=2)
    with open(os.path.join(output_dir, "highest_abs_errors_overall.json"), "w") as file:
        file.write(worst_overall_json)

    # Find the 5 models with largest mean positive/negative errors across all tasks
    signed_errors["mean_error"] = signed_errors.mean(axis=1)
    pos_cutoff = sorted(signed_errors["mean_error"])[-5]
    neg_cutoff = sorted(signed_errors["mean_error"])[5]
    worst_overall_pos = signed_errors.loc[
        signed_errors["mean_error"] >= pos_cutoff, ["mean_error"]
    ].sort_values(by="mean_error")
    worst_overall_neg = signed_errors.loc[
        signed_errors["mean_error"] < neg_cutoff, ["mean_error"]
    ].sort_values(by="mean_error")
    worst_overall_pos_json = worst_overall_pos.to_json(indent=2)
    worst_overall_neg_json = worst_overall_neg.to_json(indent=2)
    with open(os.path.join(output_dir, "highest_errors_overall_pos.json"), "w") as file:
        file.write(worst_overall_pos_json)
    with open(os.path.join(output_dir, "highest_errors_overall_neg.json"), "w") as file:
        file.write(worst_overall_neg_json)
