import duckdb
import pandas as pd
from metadata.duckdb.model_metadata_db import AnalysisStore

our_model_paths = {
    "llama2_460M_nl_code_shuf-hf": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts/llama2_460M_nl_code_shuf-hf",
    "llama2_220M_nl_only_shuf-hf": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts/llama2_220M_nl_only_shuf-hf/",
    "llama2_220M_nl_code_shuf-hf": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts/llama2_220M_nl_code_shuf-hf/",
    "llama2_220M_nl_40_code_60": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts_hf_final/llama2_220M_nl_40_code_60/",
    "llama2_220M_nl_20_code_80": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts_hf_final/llama2_220M_nl_20_code_80/",
    "llama2_220M_nl_0_code_100": "/data/tir/projects/tir5/users/mengyan3/dolma_checkpts_hf_final/llama2_220M_nl_0_code_100/",
}

# oops, actually invert our_model_paths
our_model_paths = {v: k for k, v in our_model_paths.items()}


def standardize_task_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize duplicate task names and remove duplicates from name variations"""
    df = df.copy()

    # Convert hendrycksTest to mmlu
    hendrycks_mask = df["benchmark"].str.startswith("hendrycksTest-")
    if hendrycks_mask.any():
        df.loc[hendrycks_mask, "benchmark"] = df.loc[
            hendrycks_mask, "benchmark"
        ].str.replace("hendrycksTest-", "mmlu_")

    # Fix ARC challenge naming
    df.loc[df["benchmark"] == "arc:challenge", "benchmark"] = "arc_challenge"

    # Remove duplicates, keeping first occurrence
    df = df.drop_duplicates(subset=["id", "benchmark", "setting"], keep="first")

    return df


def get_completed_evals(store: AnalysisStore) -> pd.DataFrame:
    """Get all completed model evaluations from the database"""
    query = """
        SELECT DISTINCT 
            m.id,
            m.is_instruction_tuned,
            e.benchmark,
            e.setting
        FROM model_annotations m
        JOIN evaluation_results e ON m.id = e.id
        JOIN dataset_info d ON m.id = d.id  -- Added join
        WHERE m.total_params IS NOT NULL
        AND d.pretraining_summary_total_tokens_billions IS NOT NULL  -- Added filter
    """

    df = store.con.execute(query).df()
    # Filter out instruction tuned models in pandas
    df = df[df["is_instruction_tuned"] != True]
    df = standardize_task_names(df)
    return df.drop(columns=["is_instruction_tuned"])


def find_missing_evals(db_path: str) -> pd.DataFrame:
    """Find missing evaluations for non-instruction tuned models"""
    store = AnalysisStore.from_existing(db_path)

    # Get all models with required fields
    base_query = """
        SELECT DISTINCT 
            m.id,
            m.is_instruction_tuned
        FROM model_annotations m
        LEFT JOIN dataset_info d ON m.id = d.id
        WHERE m.total_params IS NOT NULL 
        AND d.pretraining_summary_total_tokens_billions IS NOT NULL
    """
    initial_models = store.con.execute(base_query).df()
    # Filter out instruction tuned models
    initial_models = initial_models[initial_models["is_instruction_tuned"] != True]
    initial_models = initial_models.drop(columns=["is_instruction_tuned"])
    initial_models.to_csv("initial_models.csv", index=False)

    # exclude some models
    excludes = ["llama2_460M_nl_code_shuf-hf", "llama2_220M_nl_60_code_40"]
    initial_models = initial_models[~initial_models["id"].isin(excludes)]

    # Get completed evaluations
    completed_evals = get_completed_evals(store)

    # Define benchmark-setting combinations and their sub-benchmarks
    benchmark_patterns = {
        "arc_challenge": ["arc_challenge", "arc:challenge"],
        "hellaswag": ["hellaswag"],
        "mmlu": [
            "mmlu_",
            "hendrycksTest",
        ],  # Will match any benchmark starting with mmlu_
        "truthfulqa": ["truthfulqa", "truthfulqa:mc"],
        "winogrande": ["winogrande"],
        "lambada": ["lambada"],
        "gsm8k": ["gsm8k"],
        "humaneval": ["humaneval"],
        #'xnli': ['xnli'],
        #'anli': ['anli'],
        #'logiqa2': ['logiqa2'],
        #'mathqa': ['mathqa'],
        #'arithmetic': ['arithmetic_'],  # Will match arithmetic_* sub-benchmarks
        #'minerva_math': ['minerva_']  # Will match minerva_math_* sub-benchmarks
    }

    benchmark_settings = [
        ("arc_challenge", "25-shot"),
        ("hellaswag", "10-shot"),
        ("mmlu", "0-shot"),
        ("mmlu", "5-shot"),
        ("truthfulqa", "0-shot"),
        ("winogrande", "5-shot"),
        ("lambada", "0-shot"),
        ("gsm8k", "5-shot"),
        # ('arithmetic', '5-shot'),
        # ('minerva_math', '5-shot'),
        ("humaneval", "0-shot"),
        # ('xnli', '0-shot'),
        # ('anli', '0-shot'),
        # ('logiqa2', '0-shot'),
        # ('mathqa', '0-shot')
    ]

    # For each model and benchmark-setting combination, check if any matching sub-benchmarks exist
    missing_evals = []

    for model_id in initial_models["id"]:
        for benchmark, setting in benchmark_settings:
            # Get patterns to match for this benchmark
            patterns = benchmark_patterns[benchmark]

            # Check if any matching sub-benchmarks exist for this model
            has_eval = False
            for pattern in patterns:
                matching_evals = completed_evals[
                    (completed_evals["id"] == model_id)
                    & (completed_evals["benchmark"].str.startswith(pattern))
                    & (completed_evals["setting"] == setting)
                ]
                if len(matching_evals) > 0:
                    has_eval = True
                    break

            # If no matching sub-benchmarks found, this is a missing evaluation

            if model_id in our_model_paths:
                model_id = our_model_paths[model_id]

            if not has_eval:
                missing_evals.append(
                    {"id": model_id, "benchmark": benchmark, "setting": setting}
                )

    missing_evals_df = pd.DataFrame(missing_evals)
    # TODO: excluding minerva math for now, add back in when 5 shot clarified
    missing_evals_df = missing_evals_df[missing_evals_df["benchmark"] != "minerva_math"]

    # Print summary statistics
    print(f"\nTotal models to evaluate: {len(initial_models)}")
    initial_models.to_csv("initial_models.csv", index=False)
    print(f"Total evaluations (including sub-benchmarks): {len(completed_evals)}")

    # Print stats per benchmark
    print("\nCompleted evaluations per benchmark pattern:")
    for benchmark, patterns in benchmark_patterns.items():
        for setting in [s for b, s in benchmark_settings if b == benchmark]:
            completed_models = set()
            for pattern in patterns:
                matching = completed_evals[
                    (completed_evals["benchmark"].str.startswith(pattern))
                    & (completed_evals["setting"] == setting)
                ]
                completed_models.update(matching["id"])
            print(f"{benchmark} ({setting}): {len(completed_models)} models")

    print(f"\nTotal missing evaluations: {len(missing_evals)}")

    store.con.close()
    return missing_evals_df


if __name__ == "__main__":
    db_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_03_03.duckdb"
    output_tsv = "./missing_evals_303.tsv"

    missing_evals = find_missing_evals(db_path)

    missing_evals.to_csv(output_tsv, sep="\t", index=False)
