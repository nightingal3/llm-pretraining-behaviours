import pandas as pd

# Load the CSV
df = pd.read_csv(
    "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_02_09.duckdb/evaluation_results.csv"
)

# Print info about duplicates
dupes = df.duplicated(subset=["id", "benchmark", "setting", "metric"], keep=False)
print(f"Found {sum(dupes)} duplicate entries")

# Show the duplicate rows
print("\nDuplicate rows:")
print(df[dupes].sort_values(["id", "benchmark", "setting", "metric"]))
breakpoint()
# Remove duplicates, keeping first occurrence
df_deduped = df.drop_duplicates(
    subset=["id", "benchmark", "setting", "metric"], keep="first"
)

# Verify no duplicates remain
assert not df_deduped.duplicated(
    subset=["id", "benchmark", "setting", "metric"]
).any(), "Still found duplicates!"

# Save cleaned data
df_deduped.to_csv("./metadata/duckdb/2025_02_09.duckdb/cleaned_file.csv", index=False)

print(f"\nReduced from {len(df)} to {len(df_deduped)} rows")
