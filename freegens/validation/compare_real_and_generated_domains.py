import pandas as pd
import glob
import os
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

# Directory containing domain classification CSVs
DOMAIN_DIR = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/domain_classification_results"
# Path to model generations/features CSV
MODEL_FEATURES_CSV = "all_models_feature_stats_3_03.csv"
# Output directory
OUTPUT_DIR = Path("domain_analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mapping of dataset IDs to their trained models (only Pile for now)
DATASET_TO_MODELS = {
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

def extract_dataset_domain_stats(domain_dir):
    """Reads each CSV, filters out 'unknown', and computes domain percentages per dataset."""
    records = []
    for path in glob.glob(os.path.join(domain_dir, "*_sample_predicted_domains.csv")):
        df = pd.read_csv(path)
        df = df[df["prediction"] != "unknown"]
        basename = os.path.basename(path)
        dataset_id = basename.split("_", 2)[1]
        total = len(df)
        counts = df["prediction"].value_counts()
        for domain, count in counts.items():
            records.append({
                "dataset_id": dataset_id,
                "domain": domain,
                "dataset_pct": count / total * 100
            })
    return pd.DataFrame(records)

def compare_to_model_features(domain_stats, model_df, mapping):
    """Joins dataset percentages with model mean percentages for corresponding models."""
    comp = []
    for ds_id, group in domain_stats.groupby("dataset_id"):
        models = mapping.get(ds_id, [])
        for _, row in group.iterrows():
            domain = row["domain"]
            ds_pct = row["dataset_pct"]
            for mdl in models:
                mrow = model_df[model_df["id"] == mdl]
                col = f"domain_{domain}_pct_mean"
                if not mrow.empty and col in mrow.columns:
                    comp.append({
                        "dataset_id": ds_id,
                        "model_id": mdl,
                        "domain": domain,
                        "dataset_pct": ds_pct,
                        "model_pct_mean": mrow.iloc[0][col]
                    })
    return pd.DataFrame(comp)

def compute_model_correlations(comp_df, output_dir):
    """Compute Spearman correlation of dataset vs. model pct across domains for each model."""
    results = []
    for mdl, group in comp_df.groupby("model_id"):
        if len(group) < 2:
            continue
        rho, pval = spearmanr(group["dataset_pct"], group["model_pct_mean"])
        results.append({
            "model_id": mdl,
            "spearman_r": rho,
            "p_value": pval,
            "n_domains": len(group)
        })
    corr_df = pd.DataFrame(results).sort_values("spearman_r", ascending=False)
    corr_df.to_csv(output_dir / "domain_spearman_correlations.csv", index=False)
    print(f"Saved Spearman correlations to {output_dir/'domain_spearman_correlations.csv'}")
    print(corr_df)
    return corr_df

def main():
    # Extract dataset domain percentages
    domain_stats = extract_dataset_domain_stats(DOMAIN_DIR)
    domain_stats.to_csv(OUTPUT_DIR / "dataset_domain_percentages.csv", index=False)
    print(f"Saved dataset domain percentages to {OUTPUT_DIR/'dataset_domain_percentages.csv'}")
    
    # Load model features
    model_df = pd.read_csv(MODEL_FEATURES_CSV)
    
    # Compare dataset to model
    comp_df = compare_to_model_features(domain_stats, model_df, DATASET_TO_MODELS)

    rho_all, p_all = spearmanr(
        comp_df["dataset_pct"],
        comp_df["model_pct_mean"],
    )
    print(f"Pooled Spearman ρ across all pairs: {rho_all:.3f} (p={p_all:.3g})")

    r_all, p_all = pearsonr(
        comp_df["dataset_pct"],
        comp_df["model_pct_mean"],
    )
    print(f"Pooled Pearson r across all pairs: {r_all:.3f} (p={p_all:.3g})")
    
    comp_df.to_csv(OUTPUT_DIR / "domain_pct_comparison.csv", index=False)
    print(f"Saved domain vs. model comparison to {OUTPUT_DIR/'domain_pct_comparison.csv'}")
    
    # Compute and print correlations
    compute_model_correlations(comp_df, OUTPUT_DIR)

if __name__ == "__main__":
    main()

