from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("./all_models_feature_stats_2_03.csv")
# Select only free-gen features (excluding model names)
drop_cols = [
    "id",
    "entropy_std",
    "domain_academic_pct_std",
    "domain_books_pct_std",
    "domain_code_pct_std",
    "domain_reference_pct_std",
    "domain_specific_datasets_pct_std",
    "domain_web_pct_std",
]

feature_cols = [col for col in df.columns if col not in drop_cols]
# drop empty cols
df_features = df[feature_cols]

df_cleaned = df_features.dropna(axis=1, how="all")

# Drop rows with missing data
df_cleaned = df_cleaned.dropna(axis=0, how="any")

# Select only numeric columns (excluding model name or other non-feature columns)
numeric_cols = df_cleaned.select_dtypes(include=["number"]).columns

X = df_cleaned[numeric_cols]


# Standardize the features
X_scaled = StandardScaler().fit_transform(X)

# Reduce dimensions to 5 principal components
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print(
    f"Explained variance by each principal component: {pca.explained_variance_ratio_}"
)

# Create dataframe with PCA features
df_pca = pd.DataFrame(X_pca, columns=[f"pca_{i+1}" for i in range(5)])
df_pca["id"] = df["id"]
pca_loadings = pd.DataFrame(
    pca.components_, columns=numeric_cols, index=[f"PC{i+1}" for i in range(5)]
)

# Find the top contributing features for each principal component
top_features_per_pc = {}
for i in range(5):
    top_features = pca_loadings.iloc[i].abs().nlargest(5).index.tolist()
    top_features_per_pc[f"PC{i+1}"] = top_features

# Convert to DataFrame for better readability
df_top_features = pd.DataFrame.from_dict(top_features_per_pc, orient="index")
df_top_features.to_csv("pca_freegen_top_features.csv")

# Save reduced feature set
df_pca.to_csv("pca_freegen_features.csv", index=False)

print("PCA-reduced features saved!")
pc_pairs = [
    ("pca_1", "pca_2"),
    ("pca_1", "pca_3"),
    ("pca_1", "pca_4"),
    ("pca_1", "pca_5"),
]

# Create scatter plots
for pc_x, pc_y in pc_pairs:
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca[pc_x], df_pca[pc_y], alpha=0.7)

    # Add model names as labels
    for i, txt in enumerate(df_pca["id"]):
        plt.annotate(txt, (df_pca[pc_x][i], df_pca[pc_y][i]), fontsize=8, alpha=0.7)

    plt.xlabel(pc_x.replace("_", " ").title())
    plt.ylabel(pc_y.replace("_", " ").title())
    plt.title(f"{pc_x.replace('_', ' ').title()} vs {pc_y.replace('_', ' ').title()}")
    plt.grid(True)
    plt.savefig(
        f"/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/pca_{pc_x}_{pc_y}.png"
    )
    plt.close()
