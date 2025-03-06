import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from model_metadata_db import load_table_from_db


task_groups = {
    "mmlu": ["hendrycksTest-", "mmlu_"],
    "arithmetic": ["arithmetic_"],
    "arc_challenge": ["arc:challenge", "arc_challenge"],
    "truthfulqa": ["truthfulqa_mc1", "truthfulqa_mc2"],
    "minerva_math": ["minerva_math_"],
    "lambada": ["lambada_standard", "lambada_openai"],
    "gsm8k": ["gsm8k", "gsm8k_cot"]
}

def standardize_task_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize duplicate task names while preserving domains"""
    df = df.copy()
    
    # Convert hendrycksTest to mmlu
    hendrycks_mask = df['benchmark'].str.startswith('hendrycksTest-')
    if hendrycks_mask.any():
        df.loc[hendrycks_mask, 'benchmark'] = df.loc[hendrycks_mask, 'benchmark'].str.replace('hendrycksTest-', 'mmlu_')
    
    # Fix ARC challenge naming
    df.loc[df['benchmark'] == 'arc:challenge', 'benchmark'] = 'arc_challenge'

    df.drop_duplicates(subset=["id", "benchmark", "setting"], keep="first", inplace=True)
    
    return df

def merge_similar_benchmarks(pivoted_df: pd.DataFrame, task_groups: dict) -> pd.DataFrame:
    """
    Merge task groups using your specific patterns and return DataFrame with aggregated columns.
    Handles NaN values by only aggregating available results.
    """
    df_merged = pivoted_df.copy()
    
    for group_name, patterns in task_groups.items():
        # Find columns matching any of the group's patterns
        matched_cols = [
            col for col in df_merged.columns
            if any(pattern.lower() in col.lower() for pattern in patterns)
        ]
        
        if not matched_cols:
            print(f"No columns matched for group: {group_name}")
            continue
            
        # Calculate mean while ignoring NaN values
        df_merged[group_name] = df_merged[matched_cols].mean(axis=1, skipna=True)
        
        # Remove original columns that were aggregated
        df_merged = df_merged.drop(columns=matched_cols)
        
        # Print debug info
        print(f"Group '{group_name}' aggregated columns: {matched_cols}")
        print(f"New column stats:\n{df_merged[group_name].describe()}\n")
    
    return df_merged

def visualize_correlation(corr_matrix):
    plt.figure(figsize=(12, 10))
    
    # Create mask for NaN values and diagonal
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask |= np.isnan(corr_matrix)
    
    # Custom color map
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Plot heatmap with annotations
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8}
    )
    
    # Improve readability
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title("Task Performance Correlation Matrix", pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig("/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/correlation_matrix.png")

def perform_pca_analysis(df: pd.DataFrame, n_components: int = 5):
    """Perform PCA analysis on the benchmark-model matrix"""
    # 1. Handle missing values (mean imputation)
    df_imputed = df.fillna(df.mean())
    
    # 2. Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_imputed)
    
    # 3. Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    return pca, pd.DataFrame(principal_components, index=df.index)

def plot_pca_results(pca, feature_names):
    """Visualize PCA results similar to paper's Figure 2"""
    plt.figure(figsize=(12, 6))
    
    # Plot explained variance
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('(a) PCA Explained Variance')
    plt.grid(True)
    
    # Plot component weights
    plt.subplot(1, 2, 2)
    loadings = pd.DataFrame(pca.components_[:5], columns=feature_names)
    sns.heatmap(loadings, cmap='coolwarm', center=0, 
                annot=True, fmt=".2f", cbar=False)
    plt.title('(b) Principal Component Weights')
    plt.xlabel('Benchmarks')
    plt.ylabel('PC')
    plt.yticks(ticks=np.arange(5)+0.5, labels=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5'])
    
    plt.tight_layout()
    plt.savefig("/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/pca_analysis.png")
    plt.close()

evals = load_table_from_db(db_path="/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_01_22.duckdb", table_to_load="evaluation", metric="accuracy")  # Replace 'accuracy' with your desired metric
evals["task_setting"] = evals["benchmark"] + "_" + evals["setting"]
evals = standardize_task_names(evals)

evals_pivot = evals.pivot(index="id", columns="task_setting", values="metric_value")
# drop fld_default_0-shot and fld_star_0-shot
evals_pivot = evals_pivot.drop(columns=["fld_default_0-shot", "fld_star_0-shot"])
evals_pivot = evals_pivot.dropna(axis=1, thresh=30)

evals_pivot = merge_similar_benchmarks(evals_pivot, task_groups)
evals_pivot = evals_pivot[['hellaswag_10-shot', 'winogrande_5-shot', 'mmlu','arithmetic', 'arc_challenge', 'truthfulqa', 'minerva_math', 'lambada','gsm8k']]
model_missingness = evals_pivot.isna().mean(axis=1)
evals_filtered = evals_pivot[model_missingness <= 0.5]
evals_imputed = evals_filtered.fillna(evals_filtered.mean())

# Perform PCA analysis
# center data per
pca, pca_results = perform_pca_analysis(evals_imputed, n_components=5)
plot_pca_results(pca, evals_imputed.columns)
assert False
correlation_matrix = evals_pivot.corr(method="pearson")
correlation_matrix.to_csv("/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/correlation_matrix.csv")
visualize_correlation(correlation_matrix)