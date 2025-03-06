import pandas as pd
import json
import glob
import os
import re
import tqdm

# Read in the CSV file
df = pd.read_csv("./all_models_feature_stats_3_03.csv")

# Define keywords to look for
keywords = {
    "question_words": r"\b(How|What|Why|When|Where|Who|Which|Whose)\b",
    "imperative_verbs": r"\b(Do|Make|Consider|Take|Use|Ensure|Check|Build|Apply|Run|Create|Find|Go|Try|Turn|Start|Stop|Put|Keep|Leave|Get|Move)\b",
    "conjunctions": r"\b(and|but|or|so|because|although|however|therefore|yet)\b",
    "instructions_words": r"(Question:|Answer:|Instruction:|User:|Assistant:|Q:|A:)",
    "numbers": r"\b\d+\b|\b\d+\.\d+\b|\b\d+%\b",  # Matches whole numbers, decimals, and percentages
}

# Initialize results dictionary
results = {}

# Iterate through IDs
for id_val in tqdm.tqdm(df["id"]):
    # Construct paths with preference order
    corrected_path = f"/data/tir/projects/tir5/users/mengyan3/freegens_all_corrected/{id_val}/filtered/filtered_samples_generate_only_*.jsonl"
    original_path = f"/data/tir/projects/tir5/users/mengyan3/freegens_all/{id_val}/*/filtered/filtered_samples_generate_only_*.jsonl"
    unfiltered_path = (
        f"/data/tir/projects/tir5/users/mengyan3/freegens_all/{id_val}/*/*.jsonl"
    )

    # Try to use the corrected path first
    files = glob.glob(corrected_path)
    if not files:
        files = glob.glob(original_path)
        if not files:
            files = glob.glob(unfiltered_path)

    if not files:
        print(f"Warning: No files found for ID {id_val}")
        continue

    # Use only the first found file
    filepath = files[0]

    # Initialize counters
    keyword_counts = {k: 0 for k in keywords}
    total_chars = 0

    try:
        with open(filepath, "r") as f:
            for line in f:
                try:
                    # Parse JSON and extract response
                    data = json.loads(line)
                    response = data["resps"][0][0]

                    # Count characters
                    total_chars += len(response)

                    # Count keywords
                    for keyword, pattern in keywords.items():
                        keyword_counts[keyword] += len(
                            re.findall(pattern, response, re.IGNORECASE)
                        )

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error processing line in {filepath}: {e}")
                    continue

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        continue

    if total_chars == 0:
        print(f"Warning: Total characters is 0 for ID {id_val}")
        continue

    # Store results for this ID
    results[id_val] = {"total_chars": total_chars}
    for keyword, count in keyword_counts.items():
        results[id_val][keyword] = count
        results[id_val][f"{keyword}_ratio"] = 100000 * count / total_chars

# Merge the new ratio features into the dataframe
df_with_ratios = df.copy()
for keyword in keywords:
    df_with_ratios[f"{keyword}_ratio"] = df_with_ratios["id"].map(
        lambda x: results.get(x, {}).get(f"{keyword}_ratio", float("nan"))
    )

# Save the updated CSV
output_file = "./all_models_feature_stats_3_03_with_ratios.csv"
df_with_ratios.to_csv(output_file, index=False)
