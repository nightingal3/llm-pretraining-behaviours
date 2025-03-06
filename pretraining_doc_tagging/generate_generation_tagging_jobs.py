import glob
import os
import argparse
from pathlib import Path
import re
import pandas as pd

# Base directories
base_dirs = [
    "/data/tir/projects/tir5/users/mengyan3/freegens_all_corrected",
    "/data/tir/projects/tir5/users/mengyan3/freegens_all",
]
output_dir = "/data/tir/projects/tir5/users/mengyan3/pretraining_features/generations"

# List of models that we still need to tag (identified from our analysis)
untagged_models = [
    "cerebras/Cerebras-GPT-6.7B",
    "Qwen/Qwen2-72B",
    "EleutherAI/pythia-12b",
    "Qwen/Qwen1.5-110B",
    "llama2_220M_nl_code_shuf-hf",
    "aisingapore/sea-lion-7b",
    "cerebras/btlm-3b-8k-base",
    "llama2_220M_nl_only_shuf-hf",
    "llama2_220M_nl_40_code_60",
    "llama2_220M_nl_0_code_100",
    "Qwen/Qwen2.5-72B",
    "llama2_220M_nl_20_code_80",
    "allenai/OLMo-7B-hf",
    "EleutherAI/pythia-12b-deduped",
]

# All features
ALL_FEATURES = [
    # "char_len",
    # "num_tokens",
    # "unique_tokens",
    "edu_classifier",
    # "entropy",
    # "ttr",
    # "content_function_ratio",
    # "const_parse",
    # "dep_parse",
    # "domain_report"  # Include domain classification
]

# Natural language features that should be applied only to non-code documents
NL_FEATURES = [
    "edu_classifier",
    "ttr",
    "content_function_ratio",
    "const_parse",
    "dep_parse",
]


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


def generate_commands(mode="all"):
    """
    Generate tagging commands.

    Args:
        mode (str): Either "all" (process all documents) or "non_code" (process only confirmed non-code documents)
    """
    commands = []
    id_counter = 1
    processed_models = set()

    # Track models that we couldn't find
    not_found_models = []
    output_dir_suffix = "" if mode == "all" else "_non_code"

    print(f"Generating commands for mode: {mode}")

    for model_identifier in untagged_models:
        print(f"\nProcessing untagged model: {model_identifier}")

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

        for base_dir in base_dirs:
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

        if not model_path:
            print(
                f"Warning: Model directory not found for {model_identifier} in any base directory"
            )
            not_found_models.append(model_identifier)
            continue

        if not input_file:
            print(f"Warning: No valid JSONL file found for {model_identifier}")
            not_found_models.append(model_identifier)
            continue

        # Create output directory structure matching input
        if model_org:
            output_subdir = os.path.join(
                output_dir, model_org, model_name + output_dir_suffix
            )
        else:
            output_subdir = os.path.join(output_dir, model_name + output_dir_suffix)

        # Special handling for domain_report - always include it so we can filter by domain later
        domain_feature = "domain_report"
        domain_output_file = os.path.join(
            output_subdir, "pred_domains_filtered", "predicted_domains.csv"
        )
        commands.append(
            [
                str(id_counter),
                domain_feature,
                input_file,
                domain_output_file,
                model_identifier,
                "required=True",  # Mark this as required for filtering
            ]
        )
        id_counter += 1

        # Add commands for other features
        for feature in ALL_FEATURES:
            # Skip domain_report as we already added it
            if feature == "domain_report":
                continue

            # In non_code mode, add special filter tags for NL features
            filter_tag = ""
            if mode == "non_code" and feature in NL_FEATURES:
                filter_tag = "filter=non_code"

            if feature == "entropy":
                output_file = os.path.join(
                    output_subdir, "entropy", "generation_entropy.json"
                )
            else:
                output_file = os.path.join(
                    output_subdir, f"samples_{feature}.parquet"  # Simplified filename
                )

            command_row = [
                str(id_counter),
                feature,
                input_file,
                output_file,
                model_identifier,
            ]

            if filter_tag:
                command_row.append(filter_tag)

            commands.append(command_row)
            id_counter += 1

        processed_models.add(model_identifier)

    print(f"\nTotal untagged models processed: {len(processed_models)}")
    print("Processed models:")
    for model in sorted(processed_models):
        print(f"  - {model}")

    if not_found_models:
        print(f"\nWarning: Could not find {len(not_found_models)} models:")
        for model in not_found_models:
            print(f"  - {model}")

    return commands


def main():
    """Main function to handle command line arguments and generate command files."""
    parser = argparse.ArgumentParser(
        description="Generate tagging commands for untagged models"
    )
    parser.add_argument(
        "--modes",
        choices=["all", "non_code", "both"],
        default="both",
        help="Which mode(s) to generate commands for (default: both)",
    )

    args = parser.parse_args()

    # Generate commands based on requested modes
    if args.modes in ["all", "both"]:
        all_commands = generate_commands(mode="all")
        # Prepare columns (some rows might have extra columns)
        max_cols = max(len(row) for row in all_commands)
        columns = ["id", "feature", "input_file", "output_file", "model"]
        if max_cols > 5:
            columns.append("filter_tag")

        # Write to TSV
        command_df = pd.DataFrame(all_commands, columns=columns)
        command_df.to_csv(
            "./pretraining_doc_tagging/tag_untagged_models_edu.tsv",
            sep="\t",
            index=False,
        )
        print(f"\nGenerated {len(command_df)} commands in tag_untagged_models.tsv")

    if args.modes in ["non_code", "both"]:
        non_code_commands = generate_commands(mode="non_code")
        # Prepare columns
        max_cols = max(len(row) for row in non_code_commands)
        columns = ["id", "feature", "input_file", "output_file", "model"]
        if max_cols > 5:
            columns.append("filter_tag")

        # Write to TSV
        command_df = pd.DataFrame(non_code_commands, columns=columns)
        command_df.to_csv(
            "./pretraining_doc_tagging/tag_untagged_models_non_code.tsv",
            sep="\t",
            index=False,
        )
        print(
            f"\nGenerated {len(command_df)} commands in tag_untagged_models_non_code.tsv"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
