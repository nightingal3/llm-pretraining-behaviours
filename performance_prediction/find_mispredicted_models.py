import pandas as pd
import numpy as np
import json
import copy

scores = pd.read_csv("absolute_errors_all_scaling_laws.csv")
scores.rename(columns={scores.columns[0]: "model name"}, inplace=True)
scores.set_index("model name", inplace=True)

# For each task separately, find the 3 worst performing models
worst_per_task = {}
for task in scores.columns:
    cutoff = sorted(scores[task].dropna())[-3]
    worst_per_task[task] = [
        (model, score) for model, score in scores[scores[task] >= cutoff][task].items()
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

"""
Seems like most of the prediction errors come from Qwen-7B, falcon-7b, theseed-v0.3, and Jallabi-34B. Let's find tasks where a different model is in the top 3:
"""

most_mispredicted = worst_overall.sort_values(by="mae")[-4:].index.tolist()
outliers = copy.deepcopy(worst_per_task)
for task in outliers:
    outliers[task] = [(m, e) for (m, e) in outliers[task] if m not in most_mispredicted]
outliers = {task: models for task, models in outliers.items() if models}
outliers_json = json.dumps(outliers, indent=2)
with open("highest_errors_non_top_4.json", "w") as file:
    file.write(outliers_json)
