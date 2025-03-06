import os
import pandas as pd
import glob
import re

# Base directories
base_dir = "/data/tir/projects/tir5/users/mengyan3/freegens_all"
output_dir = "/data/tir/projects/tir5/users/mengyan3/pretraining_features/generations"

# Load CSV file
csv_file_path = "./all_models_feature_stats_1_29.csv"
df = pd.read_csv(csv_file_path)

# Models missing free generations
models_missing_free_gens = {
    "cerebras/Cerebras-GPT-6.7B",
    "EleutherAI/gpt-j-6b",
    "Qwen/Qwen-7B",
    "Qwen/Qwen2-72B",
    "aisingapore/sea-lion-7b",
    "Qwen/Qwen1.5-14B",
    "google/gemma-2-9b",
    "cerebras/btlm-3b-8k-base",
    "Qwen/Qwen2.5-72B",
    "allenai/OLMo-7B-hf",
    "EleutherAI/pythia-12b-deduped",
}

# Full list of features to check
features = [
    "char_len",
    "num_tokens",
    "unique_tokens",
    # "seq_ind_tok",
    "edu_classifier",
    "const_parse",
    "dep_parse",
    # "code_features",
    "ttr",
    "content_function_ratio",
    "entropy",
]


def get_latest_jsonl(model_path):
    """Find the newest JSONL file by timestamp in filename."""
    json_files = []
    for subdir in os.listdir(model_path):
        subdir_path = os.path.join(model_path, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Look for JSONL/JSON files in this subdirectory
        subdir_files = glob.glob(os.path.join(subdir_path, "*.[jJ][sS][oO][nN][lL]*"))
        # Filter out files with "results" in the name
        subdir_files = [f for f in subdir_files if "results" not in f]
        json_files.extend(subdir_files)

    if not json_files:
        return None

    # Extract timestamp from filename and use it to find the latest
    def get_timestamp(filepath):
        match = re.search(
            r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+", os.path.basename(filepath)
        )
        return match.group(0) if match else ""

    latest_file = max(json_files, key=get_timestamp)
    return latest_file


def generate_commands():
    commands = []
    id_counter = 1

    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        model_identifier = row[
            0
        ]  # Assume the first column contains the model identifiers

        # Skip models missing free generations
        if model_identifier in models_missing_free_gens:
            print(f"Skipping model missing free generations: {model_identifier}")
            continue

        # Build the model's directory path
        org_dir, model_dir = model_identifier.split("/", 1)
        model_path = os.path.join(base_dir, org_dir, model_dir)

        # Check if model path exists
        if not os.path.isdir(model_path):
            print(f"Warning: Directory not found for model {model_identifier}")
            continue

        # Get the newest JSONL file for this model
        input_file = get_latest_jsonl(model_path)
        if not input_file:
            print(f"Warning: No valid JSONL file found for {model_identifier}")
            continue

        # Create output directory structure matching input
        output_subdir = os.path.join(output_dir, org_dir, model_dir)

        # Generate commands only for features where feature_*_mean is NaN
        for feature in features:
            # Find all relevant columns for this feature (e.g., "feature_*_mean")
            feature_columns = [
                col
                for col in df.columns
                if col.startswith(f"{feature}_") and col.endswith("_mean")
            ]

            # If any of these columns are NaN, we need to run the tagger for this feature
            if any(pd.isnull(row[col]) for col in feature_columns):
                output_file = os.path.join(output_subdir, f"samples_{feature}.parquet")
                commands.append(
                    [
                        str(id_counter),
                        feature,
                        input_file,
                        output_file,
                        model_identifier,
                    ]
                )
                id_counter += 1

    return commands


# Generate and save commands to a TSV file
all_commands = generate_commands()
command_df = pd.DataFrame(
    all_commands, columns=["id", "feature", "input_file", "output_file", "model"]
)
command_df = command_df.drop_duplicates(subset=["id"])
output_tsv_path = "./pretraining_doc_tagging/tag_freegens_commands_remaining_129.tsv"
command_df.to_csv(output_tsv_path, sep="\t", index=False)

print(f"\nGenerated commands written to {output_tsv_path}")
print("\nFirst few lines of generated TSV:")
print(command_df.head())
