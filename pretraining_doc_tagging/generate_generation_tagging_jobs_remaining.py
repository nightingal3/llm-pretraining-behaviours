import pandas as pd
import os
import argparse
from pathlib import Path
import re
import glob

# Base directories to search for model files
BASE_DIRS = [
    "/data/tir/projects/tir5/users/mengyan3/freegens_all_corrected",
    "/data/tir/projects/tir5/users/mengyan3/freegens_all",
]
OUTPUT_DIR = "/data/tir/projects/tir5/users/mengyan3/pretraining_features/generations"


def get_latest_jsonl(model_path, model_org, model_name):
    """Find the newest JSONL file by timestamp in filename that starts with filtered_."""
    json_files = []

    # Debug: Show exactly what we're looking at
    print(f"Exploring model_path: {model_path}")
    print(f"Organization: {model_org}, Model name: {model_name}")

    # Handle consistent pattern seen in screenshots: org/name/org__name/[filtered/]files.jsonl
    nested_dir = os.path.join(model_path, f"{model_org}__{model_name}")
    if os.path.isdir(nested_dir):
        print(f"Found nested directory: {nested_dir}")

        # Check for direct JSONL file in nested dir (like Qwen__Qwen1.5-110B.jsonl)
        direct_file = os.path.join(nested_dir, f"{model_org}__{model_name}.jsonl")
        if os.path.isfile(direct_file):
            print(f"Found direct JSONL file: {direct_file}")
            return direct_file

        # Check for filtered directory inside nested dir
        filtered_dir = os.path.join(nested_dir, "filtered")
        if os.path.isdir(filtered_dir):
            print(f"Found filtered directory: {filtered_dir}")
            # Look for files in this filtered directory
            filtered_files = glob.glob(os.path.join(filtered_dir, "*.jsonl"))
            # Prefer files with "filtered_" prefix
            preferred_files = [
                f for f in filtered_files if "filtered_" in os.path.basename(f)
            ]
            # Prefer files with "sample_" prefix next
            if not preferred_files:
                preferred_files = [
                    f
                    for f in filtered_files
                    if "samples_" in os.path.basename(f)
                    or "sample_" in os.path.basename(f)
                ]
            # If still nothing, take any JSONL
            if not preferred_files and filtered_files:
                preferred_files = filtered_files

            if preferred_files:
                print(f"Found {len(preferred_files)} JSONL files in filtered directory")
                json_files.extend(preferred_files)

    # If we found files in the nested directory, select the latest one
    if json_files:
        try:
            # Try to find files with timestamps in the name
            timestamped_files = [
                f
                for f in json_files
                if re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(f))
            ]
            if timestamped_files:
                latest_file = max(
                    timestamped_files,
                    key=lambda f: re.search(
                        r"\d{4}-\d{2}-\d{2}", os.path.basename(f)
                    ).group(0),
                )
            else:
                # If no timestamps, just use the first file
                latest_file = json_files[0]

            print(f"Selected file: {latest_file}")
            return latest_file
        except Exception as e:
            print(f"Error selecting latest file: {e}, defaulting to first file")
            return json_files[0]

    # If we couldn't find anything in the nested structure, try a broader search
    print("No files found in expected nested structure, trying broader search...")

    # Walk through all subdirectories looking for JSONL files
    for root, _, files in os.walk(model_path):
        jsonl_files = [
            os.path.join(root, f)
            for f in files
            if f.endswith(".jsonl") and "results" not in f
        ]
        if jsonl_files:
            print(f"Found {len(jsonl_files)} JSONL files in {root}")
            json_files.extend(jsonl_files)

    if json_files:
        # Choose the first file (or we could implement more sophisticated selection)
        selected_file = json_files[0]
        print(f"Selected file from broader search: {selected_file}")
        return selected_file

    print(f"No JSONL files found for {model_org}/{model_name}")
    return None


def handle_all_feature(row, tasks_df):
    """Handle the 'all' feature tag by expanding it to individual feature tags"""
    model = row["model"]
    # Define all available features
    all_features = [
        "char_len",
        "num_tokens",
        "unique_tokens",
        "edu_classifier",
        "entropy",
        "ttr",
        "content_function_ratio",
        "const_parse",
        "dep_parse",
        "domain_report",
        "seq_ind_tok",
    ]

    # Create a new row for each feature
    expanded_rows = []
    for feature in all_features:
        expanded_rows.append(
            {"model": model, "task_type": "feature", "missing_component": feature}
        )

    # Return the expanded rows
    return expanded_rows


def generate_tagging_jobs(missing_tasks_file, output_file):
    """Generate tagging jobs based on missing_tasks.csv"""
    # Read the missing tasks CSV
    tasks_df = pd.read_csv(missing_tasks_file)
    print(f"Loaded {len(tasks_df)} tasks from {missing_tasks_file}")

    # Handle 'all' feature tag by expanding to individual features
    expanded_rows = []
    for _, row in tasks_df.iterrows():
        if row["missing_component"] == "all":
            expanded_rows.extend(handle_all_feature(row, tasks_df))
        else:
            expanded_rows.append(row.to_dict())

    # Create expanded DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    print(f"Expanded to {len(expanded_df)} tasks after handling 'all' tag")

    # Group by model and feature to avoid duplicates
    unique_model_features = expanded_df.drop_duplicates(
        subset=["model", "missing_component"]
    )
    print(f"Reduced to {len(unique_model_features)} unique model-feature combinations")

    # Generate commands
    commands = []
    id_counter = 1
    not_found_models = []

    # Process each model-feature combination
    for _, task in unique_model_features.iterrows():
        model_identifier = task["model"]
        feature = task["missing_component"]

        # Skip if model or feature is missing
        if not model_identifier or not feature:
            continue

        # Split identifier into org and name
        if "/" in model_identifier:
            model_org, model_name = model_identifier.split("/", 1)
        else:
            # Handle models without organization prefix
            model_org = ""
            model_name = model_identifier

        # Try to find the model in either of the base directories
        model_path = None
        input_file = None

        for base_dir in BASE_DIRS:
            # Construct the path
            if model_org:
                candidate_path = os.path.join(base_dir, model_org, model_name)
            else:
                # For models without organization, try to find them at the root
                candidate_path = os.path.join(base_dir, model_name)

            print(f"Looking for model at path: {candidate_path}")

            # Check if the model directory exists
            if os.path.isdir(candidate_path):
                model_path = candidate_path
                # Get the newest JSONL file for this model
                candidate_file = get_latest_jsonl(model_path, model_org, model_name)
                if candidate_file:
                    input_file = candidate_file
                    print(f"Found model in {base_dir}")
                    break

        if not model_path or not input_file:
            print(f"Warning: Could not find input file for {model_identifier}")
            not_found_models.append(model_identifier)
            continue

        # Create output directory structure matching input
        if model_org:
            output_subdir = os.path.join(OUTPUT_DIR, model_org, model_name)
        else:
            output_subdir = os.path.join(OUTPUT_DIR, model_name)

        # Set the output file path based on feature
        if feature == "domain_report":
            output_file_path = os.path.join(
                output_subdir, "pred_domains_filtered", "predicted_domains.csv"
            )
        elif feature == "entropy":
            output_file_path = os.path.join(
                output_subdir, "entropy", "generation_entropy.json"
            )
        else:
            output_file_path = os.path.join(output_subdir, f"samples_{feature}.parquet")

        # Add to commands
        commands.append(
            {
                "id": str(id_counter),
                "feature": feature,
                "input_file": input_file,
                "output_file": output_file_path,
                "model": model_identifier,
            }
        )
        id_counter += 1

    # Convert to DataFrame and save
    commands_df = pd.DataFrame(commands)
    commands_df.to_csv(output_file, sep="\t", index=False)

    print(f"\nGenerated {len(commands_df)} tagging jobs in {output_file}")

    if not_found_models:
        print(f"\nWarning: Could not find {len(set(not_found_models))} models:")
        for model in sorted(set(not_found_models)):
            print(f"  - {model}")

    # Print feature distribution summary
    feature_counts = commands_df["feature"].value_counts()
    print("\nFeature distribution in generated tasks:")
    for feature, count in feature_counts.items():
        print(f"  {feature}: {count} tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tagging jobs from missing_tasks.csv"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="missing_tasks.csv",
        help="Input CSV file with missing tasks",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./pretraining_doc_tagging/tag_missing_tasks.tsv",
        help="Output TSV file for tagging jobs",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Generate tagging jobs
    generate_tagging_jobs(args.input, args.output)


if __name__ == "__main__":
    main()
