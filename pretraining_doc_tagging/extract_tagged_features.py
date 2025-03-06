import pandas as pd
import pyarrow.parquet as pq
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pathlib import Path

from metadata.duckdb.model_metadata_db import AnalysisStore
from generate_generation_tagging_jobs import get_latest_jsonl

FEATURES_TO_EXTRACT = ['char_len', 'num_tokens', 'unique_tokens', 'edu_classifier', 'entropy', 'ttr', "content_function_ratio", "const_parse", "dep_parse"]

def get_filtered_ids(model_path, model_org, model_name):
    """Load document IDs from filtered JSONL file"""
    
    latest_jsonl = get_latest_jsonl(model_path, model_org, model_name)
    if not latest_jsonl:
        return set()
    
    filtered_ids = set()
    with open(latest_jsonl) as f:
        for line in f:
            doc = json.loads(line)
            filtered_ids.add(doc['doc_id'])
    return filtered_ids


def load_initial_models(db_path: str) -> pd.DataFrame:
    """Load initial set of models from database"""
    store = AnalysisStore.from_existing(db_path)
    query = """
        SELECT DISTINCT m.*
        FROM model_annotations m
        LEFT JOIN dataset_info d ON m.id = d.id
        WHERE m.total_params IS NOT NULL 
        AND d.pretraining_summary_total_tokens_billions IS NOT NULL
    """
    df = store.con.execute(query).df()
    store.con.close()
    df = df[df["is_instruction_tuned"] != True]
    return df[["id"]]

# TODO: content function ratio seems missing
def extract_feature_type(path):
    """Extract feature type from path."""
    for part in path.parts:
        for feature in FEATURES_TO_EXTRACT:
            if feature in part:
                return feature
    return None

def compute_domain_stats(domain_file):
    """Compute statistics for domain classifications."""
    try:
        df = pd.read_csv(domain_file)
        # Count each domain type

        # NOTE: exclude unknown/incoherent for now
        df = df[df['prediction'] != 'unknown']
        df = df[df['prediction'] != 'incoherent']

        # media should fall under reference
        df['prediction'] = df['prediction'].apply(lambda x: 'reference' if x == 'media' else x)

        domain_counts = df['prediction'].value_counts()
        total = len(df)
        
        # Calculate percentages
        stats = {}
        for domain in ['academic', 'books', 'code', 'reference', 'web', 'specific_datasets']:
            count = domain_counts.get(domain, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            stats[f'domain_{domain}_pct'] = percentage

        stats["pct_english"] = (df['language'] == 'en').mean() * 100
            
        return stats

    except Exception as e:
        print(f"Error processing domain file {domain_file}: {str(e)}")
        return {}

def find_feature_files(base_path):
    """Recursively find all feature files and organize by model and feature."""
    model_files = {}
    base = Path(base_path)
    
    # Walk through all directories
    for path in base.rglob('*'):
        # Extract feature type
        feature = extract_feature_type(path)
        if not feature:
            continue
        
        # Check if file type matches the expected format for the feature
        if feature == "entropy" and not path.name.endswith('.json'):
            continue
        elif feature != "entropy" and not path.name.endswith('.parquet'):
            continue

        # Get model name from parent directories
        model_parts = []
        current = path.parent
        while current != base:
            if not any(feat in current.name for feat in FEATURES_TO_EXTRACT):
                model_parts.append(current.name)
            current = current.parent

        model_name = '/'.join(reversed(model_parts))
        if not model_name:
            continue

        # Initialize dict for this model if needed
        if model_name not in model_files:
            model_files[model_name] = {}

        # Store path for this feature
        model_files[model_name][feature] = str(path)
    
    return model_files

def compute_parse_stats(parse_dict):
    """Compute statistics for a single document."""
    stats = {}
    
    if 'const_tree_depth' in parse_dict:
        tree_depth = parse_dict['const_tree_depth']
        word_depth = parse_dict['const_word_depth']
        words_per_sent = parse_dict['num_words_sentence']
        
        if len(tree_depth) > 0:
            stats.update({
                'const_tree_depth_max': np.max(tree_depth),
                'const_tree_depth_mean': np.mean(tree_depth),
                'const_word_depth_mean': np.mean(word_depth),
                'const_word_depth_std': np.std(word_depth) if len(word_depth) > 1 else 0,
                'words_per_sent_mean': np.mean(words_per_sent),
                'words_per_sent_std': np.std(words_per_sent) if len(words_per_sent) > 1 else 0,
                'num_words': parse_dict['num_words_input'][0],
                'num_sentences': parse_dict['num_sentences_input'][0]
            })
    
    if 'dist_to_head' in parse_dict:
        head_distances = np.array(parse_dict['dist_to_head'])
        root_distances = np.array(parse_dict['dist_to_root'])
        
        if len(head_distances) > 0:
            stats.update({
                'dep_head_dist_median': np.median(head_distances),
                'dep_head_dist_90p': np.percentile(head_distances, 90),
                'dep_head_dist_max': np.max(head_distances),
                'dep_root_dist_median': np.median(root_distances),
                'dep_root_dist_mean': np.mean(root_distances),
                'dep_root_dist_max': np.max(root_distances)
            })
    
    return stats

def compute_stats(model_name, feature_files):
    """Compute statistics for a single model's features."""
    results = []
    
    # Get organization and model name for domain file path
    model_parts = model_name.split('/')
    model_full_name = model_name
    
    if len(model_parts) == 1:
        # No organization prefix, treat the whole name as model_name
        model_org = ""  # Empty org
        model_name_short = model_name  # Use the full name as the model name
        
        filtered_ids = get_filtered_ids(
            f'/data/tir/projects/tir5/users/mengyan3/freegens_all/{model_name_short}', 
            model_org, 
            model_name_short
        )
        
        domain_file = f'/data/tir/projects/tir5/users/mengyan3/pretraining_features/generations/{model_name_short}/pred_domains_filtered/predicted_domains.csv'
    elif len(model_parts) == 2:
        model_org, model_name = model_parts
        filtered_ids = get_filtered_ids(f'/data/tir/projects/tir5/users/mengyan3/freegens_all/{model_full_name}', model_org, model_name)

        domain_file = f'/data/tir/projects/tir5/users/mengyan3/pretraining_features/generations/{model_org}/{model_name}/pred_domains_filtered/predicted_domains.csv'
        
        # Add domain statistics if file exists
        if os.path.exists(domain_file):
            domain_stats = compute_domain_stats(domain_file)
            for feature, value in domain_stats.items():
                results.append({
                    'model': model_full_name,
                    'feature': feature,
                    'mean': value,
                })
    
    for feature, file_path in feature_files.items():
        try:
            if feature == "entropy":
                # Process entropy JSON
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                stats = {
                    'model': model_full_name,
                    'feature': feature,
                    'mean': data.get('mean_corpus_entropy', None),
                    'median': data.get('median_corpus_entropy', None),
                    'count': data.get('total_contexts', None)
                }
            elif feature in ["const_parse", "dep_parse"]:
                df = pq.read_table(file_path).to_pandas()
                
                len_before_filter = len(df)
                df = df.loc[df['id'].isin(filtered_ids)]
                if len(df) < len_before_filter * 0.5:
                    print(f"Warning: {model_name} - {feature} has less than 50% of documents after filtering")
                    continue

                # Compute stats for each document
                doc_stats = []
                for parse_dict in df[feature]:
                    doc_stats.append(compute_parse_stats(parse_dict))
                
                # Average across documents
                if doc_stats:
                    agg_stats = {}
                    for key in doc_stats[0].keys():
                        values = [d[key] for d in doc_stats if key in d]
                        if values:
                            agg_stats[key] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'count': len(values)
                            }
                    
                    # Convert to results format
                    for key, stats in agg_stats.items():
                        results.append({
                            'model': model_full_name,
                            'feature': f"{feature}_{key}",
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'count': stats['count']
                        })
            else:
                # Process parquet files
                df = pq.read_table(file_path).to_pandas()
                col_name = next(col for col in df.columns if feature in col.lower())

                # filter by filtered_ids
                len_before_filter = len(df)
                df = df.loc[df['id'].isin(filtered_ids)]
                if len(df) < len_before_filter * 0.5:
                    print(f"Warning: {model_name} - {feature} has less than 50% of documents after filtering")
                    continue

                if feature == "edu_classifier":
                    df["edu_classifier_raw"] = df["edu_classifier"]
                    df["edu_classifier"] = df["edu_classifier_raw"].apply(lambda x: x['raw_score'])

                stats = {
                    'model': model_full_name,
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
    output_file = 'all_models_feature_stats_3_03.csv'
    
    print("Finding feature files...")
    model_files = find_feature_files(base_path)

    breakpoint()
    
    if not model_files:
        print("No models found!")
        return
    
    all_results = []
    for model_name, feature_files in tqdm(model_files.items(), desc="Processing models"):
        #if model_name == "Qwen/Qwen1.5-110B":
            #breakpoint()
        results = compute_stats(model_name, feature_files)
        all_results.extend(results)
    
    if not all_results:
        print("No results generated!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Reshape to wide format
    results_df.drop_duplicates(subset=['model', 'feature'], inplace=True)
    results_df.dropna(subset=["model"], inplace=True)
    mean_df = results_df.pivot(index='model', columns='feature', values='mean').add_suffix('_mean')
    std_df = results_df.pivot(index='model', columns='feature', values='std').add_suffix('_std')
    
    wide_df = pd.concat([mean_df, std_df], axis=1)
    wide_df = wide_df.reset_index().rename(columns={'model': 'id'})
    
    wide_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()