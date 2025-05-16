import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = "/data/tir/projects/tir5/users/mengyan3/pretraining_features/new_subsets"

DATASETS = [
    "pile-uncopyrighted",
    "c4",
    "RedPajama-Data-1T",
    "starcoderdata",
    "fineweb"
]

FEATURES_TO_EXTRACT = [
    "char_len",
    "num_tokens",
    "unique_tokens",
    "edu_classifier",
    "keyword_ratios"
]

def find_feature_files():
    """Find all feature files for each dataset."""
    dataset_files = {}
    
    for dataset in DATASETS:
        dataset_path = os.path.join(BASE_DIR, dataset)
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path {dataset_path} does not exist")
            continue
            
        dataset_files[dataset] = {}
        
        # Find parquet files for each feature
        for feature in FEATURES_TO_EXTRACT:
            feature_file = os.path.join(dataset_path, f"{feature}.parquet")
            if os.path.exists(feature_file):
                dataset_files[dataset][feature] = feature_file
            else:
                print(f"Warning: Feature file {feature_file} not found")
                
    return dataset_files

def process_nested_feature(df, col_name):
    """Process features that are stored as nested dictionaries."""
    # Create a dictionary to store each ratio
    extracted_features = {}
    
    # Check if we have data to process
    if len(df) == 0 or not isinstance(df[col_name].iloc[0], dict):
        return {}
    
    # Extract keys from the first row (assuming all rows have same keys)
    keys = df[col_name].iloc[0].keys()
    
    # For each key, extract values and compute stats
    for key in keys:
        try:
            # Extract values for this key into a new column
            values = df[col_name].apply(lambda x: x.get(key, np.nan))
            
            # Only include valid numeric values
            valid_values = pd.to_numeric(values, errors='coerce').dropna()
            
            if len(valid_values) > 0:
                extracted_features[f"{col_name}_{key}"] = {
                    "mean": valid_values.mean(),
                    "std": valid_values.std() if len(valid_values) > 1 else 0,
                    "count": len(valid_values)
                }
        except Exception as e:
            print(f"Error processing {col_name}_{key}: {str(e)}")
    
    return extracted_features

def compute_stats(dataset_name, feature_files):
    """Compute statistics for a dataset's features."""
    results = []
    
    for feature, file_path in feature_files.items():
        try:
            # Read parquet file
            df = pq.read_table(file_path).to_pandas()
            
            # Get the column name for the feature
            col_name = next((col for col in df.columns if feature in col.lower()), None)
            if not col_name:
                print(f"Warning: No column found for feature {feature} in {file_path}")
                continue
                
            # Process based on feature type
            if feature in ["edu_classifier", "keyword_ratios"] and isinstance(df[col_name].iloc[0], dict):
                # Process nested features
                nested_stats = process_nested_feature(df, col_name)
                
                # Add each nested feature as a separate result
                for nested_feature, stats in nested_stats.items():
                    stats_with_meta = stats.copy()
                    stats_with_meta["dataset"] = dataset_name
                    stats_with_meta["feature"] = nested_feature
                    results.append(stats_with_meta)
                
                # For edu_classifier, also include the raw_score if it exists
                if feature == "edu_classifier" and "raw_score" in df[col_name].iloc[0]:
                    raw_scores = df[col_name].apply(lambda x: x.get("raw_score", np.nan))
                    valid_scores = pd.to_numeric(raw_scores, errors='coerce').dropna()
                    
                    if len(valid_scores) > 0:
                        results.append({
                            "dataset": dataset_name,
                            "feature": "edu_classifier",
                            "mean": valid_scores.mean(),
                            "std": valid_scores.std() if len(valid_scores) > 1 else 0,
                            "count": len(valid_scores)
                        })
            else:
                # Standard processing for other features
                stats = {
                    "dataset": dataset_name,
                    "feature": feature,
                    "mean": df[col_name].mean(),
                    "std": df[col_name].std(),
                    "count": len(df),
                }
                results.append(stats)
            
        except Exception as e:
            print(f"Error processing {dataset_name} - {feature}: {str(e)}")
            
    return results

def main():
    output_file = "dataset_feature_stats.csv"
    
    print("Finding feature files...")
    dataset_files = find_feature_files()
    
    if not dataset_files:
        print("No datasets or features found!")
        return
        
    all_results = []
    for dataset_name, feature_files in tqdm(dataset_files.items(), desc="Processing datasets"):
        results = compute_stats(dataset_name, feature_files)
        all_results.extend(results)
        
    if not all_results:
        print("No results generated!")
        return
        
    results_df = pd.DataFrame(all_results)
    
    # Reshape to wide format
    results_df.drop_duplicates(subset=["dataset", "feature"], inplace=True)
    mean_df = results_df.pivot(index="dataset", columns="feature", values="mean").add_suffix("_mean")
    std_df = results_df.pivot(index="dataset", columns="feature", values="std").add_suffix("_std")
    
    wide_df = pd.concat([mean_df, std_df], axis=1)
    wide_df = wide_df.reset_index().rename(columns={"dataset": "id"})
    
    # Save both detailed and wide format results
    results_df.to_csv("dataset_feature_stats_detailed.csv", index=False)
    wide_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file} and dataset_feature_stats_detailed.csv")
    
if __name__ == "__main__":
    main()