import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/gathered_data/merged_model_stats_1_16.csv"
data = pd.read_csv(file_path)
# drop id col
data = data.drop(columns=["id"])

# Drop > 0.5 NaN columns, interpolate the rest
nan_threshold = 0.5
data = data.dropna(thresh=data.shape[0] * nan_threshold, axis=1)
data = data.interpolate()

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Variance explained by each principal component
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

print(f"Explained variance by each principal component: {explained_variance}")
print(f"Cumulative explained variance: {cumulative_variance}")
# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker="o",
    linestyle="--",
)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Explained Variance by Principal Components")
plt.grid()
plt.savefig(
    "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/pca_variance.png"
)

# Return a DataFrame with the principal components
pca_df = pd.DataFrame(
    pca_result, columns=[f"PC{i+1}" for i in range(pca_result.shape[1])]
)
pca_df.head()

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
    for i, txt in enumerate(df_pca["model_name"]):
        plt.annotate(txt, (df_pca[pc_x][i], df_pca[pc_y][i]), fontsize=8, alpha=0.7)

    plt.xlabel(pc_x.replace("_", " ").title())
    plt.ylabel(pc_y.replace("_", " ").title())
    plt.title(f"{pc_x.replace('_', ' ').title()} vs {pc_y.replace('_', ' ').title()}")
    plt.grid(True)
    plt.savefig(
        f"/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/pca_{pc_x}_{pc_y}.png"
    )
    plt.close()
