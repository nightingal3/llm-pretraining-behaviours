import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from matplotlib.patches import Patch

def plot_feature_importances(df, feature_groups, title="Overall Feature Importance by Category"):
    """
    Generates a horizontal bar chart of feature importances, grouped by category.
    
    Args:
        df (pd.DataFrame): DataFrame containing feature importance values.
        feature_groups (dict): Dictionary mapping feature names to their respective category.
        title (str): Title of the plot.
    
    Returns:
        None (Displays the plot).
    """
    # Compute mean feature importances
    mean_importances = df.mean()

    # Sum sequence_length and dimension into total_params
    scaling_law_features = ["sequence_length", "dimension", "total_params"]
    merged_value = sum(mean_importances.get(f, 0) for f in scaling_law_features)

    # Drop the individual features and add the merged one under a new name
    mean_importances = mean_importances.drop(index=[f for f in scaling_law_features if f in mean_importances])
    mean_importances["total_params (+ seq_length + dimension)"] = merged_value

    # Sort after merging
    mean_importances = mean_importances.sort_values(ascending=False)

    # Define Set2 color mapping for categories
    set2_colors = cm.get_cmap("Set2").colors
    category_colors = {
        "Scaling Laws": set2_colors[0],  # Soft Green
        "Model Architecture": set2_colors[1],  # Soft Orange
        "Data": set2_colors[2],  # Soft Blue
        "Other": "#bdbdbd"  # Gray for unclassified features
    }

    # Explicitly map features to Scaling Laws (since they got dropped in previous versions)
    feature_category_mapping = {
        "total_params (+ seq_length + dimension)": "Scaling Laws",
        "pretraining_summary_total_tokens_billions": "Scaling Laws",
    }

    # Assign colors based on feature categories, ensuring Scaling Laws stays green
    bar_colors = [
        category_colors.get(feature_category_mapping.get(feat, feature_groups.get(feat, "Other")), "#bdbdbd") 
        for feat in mean_importances.index
    ]

    # Create the plot
    plt.figure(figsize=(14, 10))
    bars = plt.barh(mean_importances.index, mean_importances.values, color=bar_colors)

    # Formatting
    plt.xlabel("Mean Feature Importance (Gain)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)

    # Create a legend using the Set2 color scheme, ensuring Scaling Laws is included
    legend_patches = [Patch(color=color, label=category) for category, color in category_colors.items() if category in feature_groups.values() or category == "Scaling Laws"]
    plt.legend(handles=legend_patches, title="Feature Categories", fontsize=16, title_fontsize=18)

    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.tight_layout()

    # Save outputs
    plt.savefig("feature_importance_set2_fixed_final.png", dpi=300)
    plt.savefig("feature_importance_set2_fixed_final.pdf", dpi=300)

    plt.show()

# Load DataFrame
df = pd.read_csv("/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/results_db/feature_importance_xgboost_all_accuracy_freegens_False.csv")

# Define feature categories (excluding sequence_length, dimension, and total_params since they are merged)
feature_groups = {
    "Scaling Laws": "Scaling Laws",
    
    "activation": "Model Architecture",
    "attention_variant": "Model Architecture",
    "biases": "Model Architecture",
    "mlp_ratio": "Model Architecture",
    "num_heads": "Model Architecture",
    "positional_embeddings": "Model Architecture",

    "pretraining_summary_percentage_academic": "Data",
    "pretraining_summary_percentage_books": "Data",
    "pretraining_summary_percentage_code": "Data",
    "pretraining_summary_percentage_english": "Data",
    "pretraining_summary_percentage_reference": "Data",
    "pretraining_summary_percentage_web": "Data"
}

# Call the function
plot_feature_importances(df.drop(columns=["task"], errors="ignore"), feature_groups)
