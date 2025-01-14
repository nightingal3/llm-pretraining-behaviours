import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path
from tqdm import tqdm

def extract_feature_type(path):
    """Extract feature type from path."""
    features = ['char_len', 'num_tokens', 'unique_tokens', 'edu_classifier']
    for part in path.parts:
        for feature in features:
            if feature in part:
                return feature
    return None

def find_parquet_files(base_path):
    """Recursively find all parquet files and organize by model and feature."""
    model_files = {}
    base = Path(base_path)
    
    # Walk through all directories
    for path in base.rglob('*.parquet'):
        # Extract feature type
        feature = extract_feature_type(path)
        if not feature:
            print(f"Warning: Could not determine feature type for {path}")
            continue
            
        # Get model name from parent directories
        model_parts = []
        current = path.parent
        while current != base:
            # Skip feature-containing directories for model name
            if not any(feat in current.name for feat in ['char_len', 'num_tokens', 'seq_ind_tok', 'unique_tokens', 'edu_classifier']):
                model_parts.append(current.name)
            current = current.parent
            
        model_name = '/'.join(reversed(model_parts))
        if not model_name:
            print(f"Warning: Could not determine model name for {path}")
            continue
        
        # Initialize dict for this model if needed
        if model_name not in model_files:
            model_files[model_name] = {}
            
        # Store path for this feature
        model_files[model_name][feature] = str(path)
    
    return model_files

def compute_stats(model_name, feature_files):
    """Compute statistics for a single model's features."""
    results = []
    
    for feature, file_path in feature_files.items():
        try:
            df = pq.read_table(file_path).to_pandas()
            col_name = next(col for col in df.columns if feature in col.lower())
            if feature == "edu_classifier":
                # needs a bit of proc
                df["edu_classifier_raw"] = df["edu_classifier"]
                df["edu_classifier"] = df["edu_classifier_raw"].apply(lambda x: x['raw_score'])
            
            stats = {
                'model': model_name,
                'feature': feature,
                'mean': df[col_name].mean(),
                'std': df[col_name].std(),
                'count': len(df)
            }
            results.append(stats)
        except Exception as e:
            print(f"Error processing {model_name} - {feature}: {str(e)}")
    
    return results

def main():
    base_path = '/data/tir/projects/tir5/users/mengyan3/pretraining_features/generations'
    output_file = 'all_models_feature_stats.csv'
    
    print("Finding parquet files...")
    model_files = find_parquet_files(base_path)
    
    if not model_files:
        print("No models found! Check the base path and file structure.")
        return
        
    print(f"\nFound {len(model_files)} models to process")
    
    all_results = []
    for model_name, feature_files in tqdm(model_files.items(), desc="Processing models"):
        results = compute_stats(model_name, feature_files)
        all_results.extend(results)
    
    if not all_results:
        print("No results generated! Check if parquet files contain expected data.")
        return
        
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\nResults saved to {output_file}")
    print("\nSummary by feature:")
    summary = results_df.groupby('feature').agg({
        'mean': ['mean', 'std'],
        'std': ['mean', 'std'],
        'count': 'mean'
    })
    print(summary)
    # Create wide format version
    wide_df = results_df.pivot(
        index='model',
        columns='feature',
        values=['mean', 'std']
    )
    
    # Flatten column names and make them more readable
    wide_df.columns = [f"{feat}_{stat}" for stat, feat in wide_df.columns]
    
    # Reset index to make 'model' a regular column
    wide_df = wide_df.reset_index()
    
    # Rename model column to id for consistency
    wide_df = wide_df.rename(columns={'model': 'id'})
    
    # Save wide format
    wide_output = output_file.replace('.csv', '_wide.csv')
    wide_df.to_csv(wide_output, index=False)
    
    print(f"\nResults saved to {output_file} (long format)")
    print(f"Wide format results saved to {wide_output}")
    
    print("\nProcessed models:")
    for model in sorted(wide_df['id'].unique()):
        print(f"- {model}")

if __name__ == "__main__":
    main()