import os
import pandas as pd
import glob
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# ===========================
# 📌 Step 1: Load SHAP Summary Files
# ===========================

shap_files_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/results_db/shap_summary_*.csv"

shap_data = {}
tasks_included = [
    "arc_challenge",
    "gsm8k",
    "hellaswag",
    "humaneval",
    "lambada",
    "truthfulqa",
    "winogrande",
]

# Automatically detect and add all MMLU tasks
for file in glob.glob(shap_files_path):
    task_name = (
        os.path.basename(file)
        .replace("shap_summary_", "")
        .replace("_xgboost_all_accuracy.csv", "")
    )

    if task_name.startswith("mmlu_") or "hendrycksTest-" in task_name:
        # just select a few mmlu bc there's too many
        if "0-shot" in task_name:
            continue
        selected_tasks = [
            "professional_medicine",
            "professional_psychology",
            "moral_disputes",
            "sociology",
            "international_law",
            "anatomy",
            "nutrition",
            "formal_logic",
            "abstract_algebra",
            "foreign_policy",
        ]
        if any(task in task_name for task in selected_tasks):
            tasks_included.append(task_name)

# Load SHAP values for selected tasks
for file in glob.glob(shap_files_path):
    task_name = (
        os.path.basename(file)
        .replace("shap_summary_", "")
        .replace("_xgboost_all_accuracy.csv", "")
    )

    if task_name not in tasks_included:
        continue

    df = pd.read_csv(file)

    # Store mean absolute SHAP values (indexed by feature)
    shap_data[task_name] = df.set_index("feature")["mean_abs_shap"]

# Convert dictionary to DataFrame (task-feature matrix)
shap_df = pd.DataFrame(shap_data).T.fillna(0)  # Fill missing features with 0

# Save structured matrix for reference
shap_matrix_path = "/data/tir/projects/tir5/users/mengyan3/task_shap_matrix.csv"
shap_df.to_csv(shap_matrix_path)
print(f"Saved SHAP matrix to {shap_matrix_path}")

# ===========================
# 📌 Step 2: Generate Task Dendrogram
# ===========================

# Perform hierarchical clustering using Ward's method
Z = linkage(shap_df, method="ward")

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=shap_df.index, leaf_rotation=90, leaf_font_size=14)

# Formatting
plt.title("Task Clustering Based on SHAP Feature Contributions", fontsize=18)
plt.ylabel("Dissimilarity", fontsize=16)
plt.xticks(fontsize=12)

# Save dendrogram plot
dendrogram_path = "./task_shap_dendrogram.pdf"
plt.savefig(dendrogram_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"Saved dendrogram to {dendrogram_path}")
