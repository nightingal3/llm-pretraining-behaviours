import pandas as pd
import duckdb

from metadata.duckdb.model_metadata_db import AnalysisStore

# Read the CSVs
arc_models = pd.read_csv(
    "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/done_models/evaled_models_arc_challenge_25-shot.csv"
)
initial_models = pd.read_csv("initial_models.csv")

# Convert to sets for comparison
arc_model_ids = set(arc_models["id"])
initial_model_ids = set(initial_models["id"])

# Find differences
extra_in_arc = arc_model_ids - initial_model_ids
missing_from_arc = initial_model_ids - arc_model_ids

print(f"Arc challenge models count: {len(arc_model_ids)}")
print(f"Initial models count: {len(initial_model_ids)}")

print(f"\nModels in arc_challenge but not in initial_models ({len(extra_in_arc)}):")
for model in sorted(extra_in_arc):
    print(f"- {model}")

print(f"\nModels in initial_models but not in arc_challenge ({len(missing_from_arc)}):")
for model in sorted(missing_from_arc):
    print(f"- {model}")


def compare_queries(db_path: str):
    """Compare results between the two approaches"""
    store = AnalysisStore.from_existing(db_path)

    # Approach from check-missing-evals
    query1 = """
    SELECT DISTINCT m.id, m.is_instruction_tuned
    FROM model_annotations m
    LEFT JOIN dataset_info d ON m.id = d.id
    WHERE m.total_params IS NOT NULL 
    AND d.pretraining_summary_total_tokens_billions IS NOT NULL
    """
    models1 = store.con.execute(query1).df()
    models1 = models1[models1["is_instruction_tuned"] != True]

    # Approach from main script
    query2 = """
    SELECT 
        m.id,
        m.total_params,
        m.is_instruction_tuned,
        d.pretraining_summary_total_tokens_billions,
        e.benchmark,
        e.setting,
        e.metric_value as value,
        e.metric_stderr as value_stderr
    FROM model_annotations m
    LEFT JOIN dataset_info d ON m.id = d.id
    LEFT JOIN evaluation_results e ON m.id = e.id
    WHERE m.total_params IS NOT NULL 
    AND d.pretraining_summary_total_tokens_billions IS NOT NULL
    AND m.is_instruction_tuned != True
    AND e.metric = 'accuracy'
    """
    models2 = store.con.execute(query2).df()

    # Compare
    models1_set = set(models1["id"])
    models2_set = set(models2["id"])

    print(f"Models from query 1: {len(models1)}")
    print(f"Models from query 2: {len(models2)}")

    # Check differences
    only_in_1 = models1_set - models2_set
    print("\nModels only in first query:")
    if only_in_1:
        print(sorted(only_in_1))

    only_in_2 = models2_set - models1_set
    print("\nModels only in second query:")
    if only_in_2:
        print(sorted(only_in_2))

    store.con.close()
    return models1, models2


if __name__ == "__main__":
    db_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_01_27.duckdb"
    models1, models2 = compare_queries(db_path)
