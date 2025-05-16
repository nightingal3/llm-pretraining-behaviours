import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.stats import spearmanr, pearsonr

# Paths to the CSV files
DATASET_FEATURES_CSV = "dataset_feature_stats.csv"
MODEL_FEATURES_CSV   = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/all_models_feature_stats_3_03_with_ratios.csv"

# Define mapping between pretrain datasets and models
DATASET_TO_MODEL_MAPPING = {
    "pile-uncopyrighted": [
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "EleutherAI/gpt-neox-20b",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-70m-deduped",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-160m-deduped",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-410m-deduped",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1b-deduped",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-1.4b-deduped",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9b",
        "EleutherAI/pythia-6.9b-deduped",
        "EleutherAI/pythia-12b",
        "EleutherAI/pythia-12b-deduped",
        "cerebras/Cerebras-GPT-1.3B",
        "cerebras/Cerebras-GPT-13B",
        "cerebras/Cerebras-GPT-2.7B",
        "cerebras/Cerebras-GPT-6.7B",
    ],
}

def load_data():
    dataset_df = pd.read_csv(DATASET_FEATURES_CSV)

    # some columns in the dataset CSV have "keyword_ratios" prefix
    dataset_df.columns = [
        col.replace("keyword_ratios_", "") if col.startswith("keyword_ratios_") else col for col in dataset_df.columns
    ]
    model_df   = pd.read_csv(MODEL_FEATURES_CSV)

    # similar matching 
    ratio_feats = [
        "question_words_ratio",
        "imperative_verbs_ratio",
        "conjunctions_ratio",
        "instructions_words_ratio",
        "numbers_ratio"
    ]
    for ratio in ratio_feats:
        if ratio in model_df.columns:
            model_df.rename(columns={ratio: f"{ratio}_mean"}, inplace=True)

    return dataset_df, model_df

def common_features(dataset_df, model_df):
    ds_feats = {
        col.rsplit('_', 1)[0]
        for col in dataset_df.columns
        if col.endswith('_mean')
    }
    mdl_feats = {
        col.rsplit('_', 1)[0]
        for col in model_df.columns
        if col.endswith('_mean')
    }
    common = sorted(ds_feats & mdl_feats)
    print(f"Detected {len(common)} common features:\n", common)
    return common

def gather_pairs(dataset_df, model_df, common_feats):
    rows = []
    for ds_id in dataset_df['id']:
        models = DATASET_TO_MODEL_MAPPING.get(ds_id, [])
        if not models:
            continue
        ds_row = dataset_df[dataset_df['id']==ds_id].iloc[0]
        for mdl_id in models:
            mdl_rows = model_df[model_df['id']==mdl_id]
            if mdl_rows.empty:
                continue
            mdl_row = mdl_rows.iloc[0]
            for feat in common_feats:
                ds_val = ds_row.get(f"{feat}_mean", np.nan)
                mdl_val= mdl_row.get(f"{feat}_mean", np.nan)
                if not pd.isna(ds_val) and not pd.isna(mdl_val):
                    rows.append({
                        "dataset_id": ds_id,
                        "model_id":   mdl_id,
                        "feature":    feat,
                        "ds_value":   ds_val,
                        "mdl_value":  mdl_val
                    })
    return pd.DataFrame(rows)

def compute_spearman(corr_df):
    results = []
    for feat, sub in corr_df.groupby('feature'):
        if len(sub) < 2:
            continue
        rho, p = spearmanr(sub['ds_value'], sub['mdl_value'])
        results.append({
            "feature":    feat,
            "spearman_r": rho,
            "p_value":    p,
            "n_pairs":    len(sub)
        })
    return pd.DataFrame(results).sort_values('spearman_r', ascending=False)

def main():
    ds_df, mdl_df = load_data()
    common = common_features(ds_df, mdl_df)
    pairs  = gather_pairs(ds_df, mdl_df, common)
    
    if pairs.empty:
        print("No matching dataset–model pairs found. Check your mapping.")
        return
    
    # Save the raw paired values if desired
    pairs.to_csv("dataset_model_pairs.csv", index=False)
    
    # Compute and display Spearman correlations
    spearman_df = compute_spearman(pairs)
    print("\nSpearman correlations per feature:")
    print(spearman_df.to_string(index=False))
    
    # per model
    rho_all, p_all = spearmanr(
        pairs["ds_value"],
        pairs["mdl_value"],
    )
    print(f"\nPooled Spearman ρ across all pairs: {rho_all:.3f} (p={p_all:.3g})")

    r_all, p_all = pearsonr(
        pairs["ds_value"],
        pairs["mdl_value"],
    )
    print(f"Pooled Pearson r across all pairs: {r_all:.3f} (p={p_all:.3g})")
    # Persist results
    spearman_df.to_csv("spearman_correlations.csv", index=False)
    print("\nSaved per-feature Spearman correlations to spearman_correlations.csv")

if __name__ == "__main__":
    main()
