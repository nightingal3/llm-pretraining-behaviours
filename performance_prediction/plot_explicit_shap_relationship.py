import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import linregress
import statsmodels.api as sm

# Set global font sizes for better readability
plt.rcParams.update({
    "font.size": 20,  # Base font size
    "axes.titlesize": 25,
    "axes.labelsize": 20,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 18
})

def analyze_feature_shap_loess(df, feature="code"):
    """
    Plots a scatterplot of SHAP value impact vs. feature percentage with LOESS trendlines and confidence intervals.
    
    - Small and large models are plotted separately with different colors.
    - Uses locally weighted regression (LOESS) for trend estimation with a shaded uncertainty region.
    
    Parameters:
      df: DataFrame containing:
          - pretraining_summary_percentage_{feature}
          - shap_pretraining_summary_percentage_{feature}
          - total_params (if feature == "code")
      feature: string, e.g. "code" or "english"
    
    Returns:
      Matplotlib figure.
    """
    # Define column names
    col_percentage = f'pretraining_summary_percentage_{feature}'
    col_shap = f'shap_pretraining_summary_percentage_{feature}'

    # Remove invalid values
    df = df[df[col_percentage] != -1].copy()

    # Split models into small and large groups based on median parameter count
    median_params = df['total_params'].median()
    small_models = df[df['total_params'] < median_params].copy()
    large_models = df[df['total_params'] >= median_params].copy()

    # Convert median params to a readable format
    median_params_real_num_rounded = round((math.e ** median_params) / 1e9, 2)

    # Create scatterplot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter points
    sns.scatterplot(x=col_percentage, y=col_shap, data=small_models, color="blue", label="Small Models", alpha=0.6, ax=ax)
    sns.scatterplot(x=col_percentage, y=col_shap, data=large_models, color="red", label="Large Models", alpha=0.6, ax=ax)

    # Function to fit and plot LOESS with confidence intervals
    def plot_loess_trendline(data, color, label):
        if len(data) > 2:  # Need at least 3 points for LOESS
            lowess = sm.nonparametric.lowess(data[col_shap], data[col_percentage], frac=0.5)  # frac controls smoothness
            x_vals, y_vals = lowess[:, 0], lowess[:, 1]

            # Compute confidence intervals (approximate method using bootstrapping)
            boot_samples = 100
            y_boot = np.zeros((boot_samples, len(x_vals)))
            for i in range(boot_samples):
                sample = data.sample(frac=1, replace=True)  # Resampling
                boot_lowess = sm.nonparametric.lowess(sample[col_shap], sample[col_percentage], frac=0.5)
                y_boot[i, :] = np.interp(x_vals, boot_lowess[:, 0], boot_lowess[:, 1])

            y_lower = np.percentile(y_boot, 2.5, axis=0)
            y_upper = np.percentile(y_boot, 97.5, axis=0)

            # Plot trendline
            ax.plot(x_vals, y_vals, color=color, linestyle="solid", linewidth=2, label=f"{label} LOESS Trendline")
            
            # Plot confidence interval
            ax.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.2)

    # Fit and plot LOESS trendlines with confidence intervals
    plot_loess_trendline(small_models, "blue", "Small Models")
    plot_loess_trendline(large_models, "red", "Large Models")

    # Set labels and title
    ax.set_title(f'SHAP Impact by {feature.capitalize()} Percentage in Training Data\n(Small Models < {median_params_real_num_rounded}B params, Large Models ≥ {median_params_real_num_rounded}B params)')
    ax.set_xlabel(f'{feature.capitalize()} Percentage in Training Data')
    ax.set_ylabel('SHAP Value Impact')

    # Add legend
    ax.legend()

    plt.tight_layout()
    return plt

def analyze_feature_shap_quadratic(df, feature="code"):
    """
    Plots a scatterplot of SHAP value impact vs. feature percentage with quadratic trendlines for small and large models.
    
    - Small and large models are plotted separately with different colors.
    - Quadratic regression (2nd-degree polynomial) trendlines are added for each group.
    
    Parameters:
      df: DataFrame containing:
          - pretraining_summary_percentage_{feature}
          - shap_pretraining_summary_percentage_{feature}
          - total_params (if feature == "code")
      feature: string, e.g. "code" or "english"
    
    Returns:
      Matplotlib figure.
    """
    # Define column names
    col_percentage = f'pretraining_summary_percentage_{feature}'
    col_shap = f'shap_pretraining_summary_percentage_{feature}'

    # Remove invalid values
    df = df[df[col_percentage] != -1].copy()

    # Split models into small and large groups based on median parameter count
    median_params = df['total_params'].median()
    small_models = df[df['total_params'] < median_params].copy()
    large_models = df[df['total_params'] >= median_params].copy()

    # Convert median params to a readable format
    median_params_real_num_rounded = round((math.e ** median_params) / 1e9, 2)

    # Create scatterplot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter points
    sns.scatterplot(x=col_percentage, y=col_shap, data=small_models, color="blue", label="Small Models", alpha=0.6, ax=ax)
    sns.scatterplot(x=col_percentage, y=col_shap, data=large_models, color="red", label="Large Models", alpha=0.6, ax=ax)

    # Function to fit and plot quadratic regression
    def plot_quadratic_trendline(data, color, label):
        if len(data) > 2:  # Need at least 3 points for a quadratic fit
            coeffs = np.polyfit(data[col_percentage], data[col_shap], deg=2)  # Fit quadratic (2nd-degree polynomial)
            poly_eq = np.poly1d(coeffs)  # Convert to polynomial equation
            x_vals = np.linspace(data[col_percentage].min(), data[col_percentage].max(), 100)
            y_vals = poly_eq(x_vals)
            ax.plot(x_vals, y_vals, color=color, linestyle="dashed", linewidth=2, label=f"{label} Quadratic Trendline")

    # Fit and plot trendlines
    plot_quadratic_trendline(small_models, "blue", "Small Models")
    plot_quadratic_trendline(large_models, "red", "Large Models")

    # Set labels and title
    ax.set_title(f'SHAP Impact by {feature.capitalize()} Percentage in Training Data\n(Small Models < {median_params_real_num_rounded}B params, Large Models ≥ {median_params_real_num_rounded}B params)')
    ax.set_xlabel(f'{feature.capitalize()} Percentage in Training Data')
    ax.set_ylabel('SHAP Value Impact')

    # Add legend
    ax.legend()

    plt.tight_layout()
    return plt

def analyze_feature_shap_trend(df, feature="code"):
    """
    Plots a scatterplot of SHAP value impact vs. feature percentage with linear trendlines for small and large models.
    
    - Small and large models are plotted separately with different colors.
    - Linear regression trendlines are added for each group.
    - Marks and annotates the code percentage with the highest SHAP value for both groups.
    
    Parameters:
      df: DataFrame containing:
          - pretraining_summary_percentage_{feature}
          - shap_pretraining_summary_percentage_{feature}
          - total_params (if feature == "code")
      feature: string, e.g. "code" or "english"
    
    Returns:
      Matplotlib figure.
    """
    # Define column names
    col_percentage = f'pretraining_summary_percentage_{feature}'
    col_shap = f'shap_pretraining_summary_percentage_{feature}'

    # Remove invalid values
    df = df[df[col_percentage] != -1].copy()

    # Split models into small and large groups based on median parameter count
    median_params = df['total_params'].median()
    small_models = df[df['total_params'] < median_params].copy()
    large_models = df[df['total_params'] >= median_params].copy()

    # Find the point with the max SHAP value
    max_shap_small = small_models.loc[small_models[col_shap].idxmax()]
    max_shap_large = large_models.loc[large_models[col_shap].idxmax()]

    # Convert median params to a readable format
    median_params_real_num_rounded = round((math.e ** median_params) / 1e9, 2)

    # Create scatterplot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Scatter points
    sns.scatterplot(x=col_percentage, y=col_shap, data=small_models, color="blue", label="Small Models", alpha=0.6, ax=ax, s=80)
    sns.scatterplot(x=col_percentage, y=col_shap, data=large_models, color="red", label="Large Models", alpha=0.6, ax=ax, s=80)

    # Function to fit and plot linear regression trendlines
    def plot_linear_trendline(data, color, label):
        if len(data) > 1:  # Need at least 2 points for a linear fit
            slope, intercept, _, _, _ = linregress(data[col_percentage], data[col_shap])
            x_vals = np.linspace(data[col_percentage].min(), data[col_percentage].max(), 100)
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color=color, linestyle="dashed", linewidth=2, label=f"{label} Linear Trendline")

    # Fit and plot linear trendlines
    plot_linear_trendline(small_models, "blue", "Small Models")
    plot_linear_trendline(large_models, "red", "Large Models")

    # Mark max SHAP points with vertical dashed lines
    ax.axvline(max_shap_small[col_percentage], linestyle="dashed", color="blue", alpha=0.7, linewidth=2)
    ax.axvline(max_shap_large[col_percentage], linestyle="dashed", color="red", alpha=0.7, linewidth=2)

    ax.axhline(0, linestyle="dashed", color="black", linewidth=1, alpha=0.8, label="Zero SHAP Impact")


    # Annotate max SHAP points
    ax.annotate(f"{max_shap_small[col_percentage]:.1f}%", 
                xy=(max_shap_small[col_percentage], max_shap_small[col_shap]), 
                xytext=(5, 10), textcoords="offset points", fontsize=16, color="blue")

    ax.annotate(f"{max_shap_large[col_percentage]:.1f}%", 
                xy=(max_shap_large[col_percentage], max_shap_large[col_shap]), 
                xytext=(5, 10), textcoords="offset points", fontsize=16, color="red")

    # Set labels and title
    ax.set_title(f'Humaneval SHAP Impact by {feature.capitalize()} % in Training Data\n(Small < {median_params_real_num_rounded}B params, Large ≥ {median_params_real_num_rounded}B params)')
    ax.set_xlabel(f'{feature.capitalize()} Percentage in Training Data', fontsize=20)
    ax.set_ylabel('SHAP Value Impact', fontsize=20)

    # Add legend
    ax.legend()

    plt.tight_layout()
    return plt

def analyze_feature_shap_combined(df, feature="code", num_bins=5):
    """
    Plots the SHAP value impact vs. binned percentage for a given feature.
    
    - Small and large models are plotted on the same axis with different colors.
    - Each group is binned separately using quantile binning (`qcut`) to ensure equal models per bin.
    - Bin labels are formatted as percentage ranges.
    
    Parameters:
      df: DataFrame containing:
          - pretraining_summary_percentage_{feature}
          - shap_pretraining_summary_percentage_{feature}
          - total_params (if feature == "code")
      feature: string, e.g. "code" or "english"
      num_bins: int, number of bins for automatic binning
    
    Returns:
      Matplotlib figure.
    """
    # Define column names
    col_percentage = f'pretraining_summary_percentage_{feature}'
    col_shap = f'shap_pretraining_summary_percentage_{feature}'

    # Remove invalid values
    df = df[df[col_percentage] != -1].copy()

    # Split models into small and large groups based on median parameter count
    median_params = df['total_params'].median()
    small_models = df[df['total_params'] < median_params].copy()  # Fix warning
    large_models = df[df['total_params'] >= median_params].copy()  # Fix warning

    def bin_data(data):
        """Performs quantile binning and returns labeled bins with ranges."""
        bin_indices, bin_edges = pd.qcut(data[col_percentage], q=num_bins, retbins=True, labels=False, duplicates="drop")
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}%" for i in range(len(bin_edges) - 1)]
        data.loc[:, f"{feature}_bin"] = bin_indices.map(lambda x: bin_labels[x])  # Fix warning
        return data, bin_labels

    small_models, small_labels = bin_data(small_models)
    large_models, large_labels = bin_data(large_models)

    # Convert median params to a more readable format
    median_params_real_num_rounded = round((math.e ** median_params) / 1e9, 2)

    # Create a single-axis plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot small models in blue
    sns.boxplot(x=f"{feature}_bin", y=col_shap, data=small_models, notch=True, color="blue", ax=ax, width=0.4, dodge=True)
    sns.stripplot(x=f"{feature}_bin", y=col_shap, data=small_models, color='black', alpha=0.5, jitter=True, ax=ax)

    # Plot large models in red
    sns.boxplot(x=f"{feature}_bin", y=col_shap, data=large_models, notch=True, color="red", ax=ax, width=0.4, dodge=True)
    sns.stripplot(x=f"{feature}_bin", y=col_shap, data=large_models, color='black', alpha=0.5, jitter=True, ax=ax)

    # Annotate sample sizes above the boxplots
    for i, label in enumerate(small_labels):
        n_small = len(small_models[small_models[f"{feature}_bin"] == label])
        n_large = len(large_models[large_models[f"{feature}_bin"] == label])
        ax.annotate(f'n={n_small}', xy=(i-0.2, small_models[col_shap].max()), xytext=(0, 5), 
                    textcoords="offset points", ha='center', fontsize=20, color='blue')
        ax.annotate(f'n={n_large}', xy=(i+0.2, large_models[col_shap].max()), xytext=(0, 5), 
                    textcoords="offset points", ha='center', fontsize=20, color='red')

    # Add horizontal reference line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Set labels and title
    ax.set_title(f'Impact of {feature.capitalize()} on Predictions\n(Small Models < {median_params_real_num_rounded}B params, Large Models ≥ {median_params_real_num_rounded}B params)')
    ax.set_xlabel(f'{feature.capitalize()} Percentage in Training Data')
    ax.set_ylabel('SHAP Value Impact')

    # Set x-axis labels as percentage ranges
    ax.set_xticklabels(small_labels, rotation=30, ha="right")

    # Add legend
    handles = [
        plt.Line2D([0], [0], color="blue", lw=4, label="Small Models"),
        plt.Line2D([0], [0], color="red", lw=4, label="Large Models")
    ]
    ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    return plt


def analyze_feature_shap_relationship(df, feature="code", num_bins=5):
    """
    Plots the SHAP value impact vs. binned percentage for a given feature.
    
    - For 'code', models are split into small and large groups based on median total_params.
    - Each group is binned separately using quantile binning (`qcut`) so that bins have equal models.
    - Bin labels are formatted as percentage ranges rather than generic bin numbers.
    
    Parameters:
      df: DataFrame containing:
          - pretraining_summary_percentage_{feature}
          - shap_pretraining_summary_percentage_{feature}
          - total_params (if feature == "code")
      feature: string, e.g. "code" or "english"
      num_bins: int, number of bins for automatic binning
    
    Returns:
      Matplotlib figure.
    """
    # Define column names
    col_percentage = f'pretraining_summary_percentage_{feature}'
    col_shap = f'shap_pretraining_summary_percentage_{feature}'
    
    # Remove invalid values
    df = df[df[col_percentage] != -1].copy()
    
    if feature == "code":
        # Split into small/large models based on median parameter count
        median_params = df['total_params'].median()
        small_models = df[df['total_params'] < median_params]
        large_models = df[df['total_params'] >= median_params]

        def bin_data(data):
            """Performs quantile binning and returns labeled bins with ranges."""
            bin_indices, bin_edges = pd.qcut(data[col_percentage], q=num_bins, retbins=True, labels=False, duplicates="drop")
            bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}%" for i in range(len(bin_edges) - 1)]
            data[f"{feature}_bin"] = bin_indices.apply(lambda x: bin_labels[x])
            return data, bin_labels

        small_models, small_labels = bin_data(small_models)
        large_models, large_labels = bin_data(large_models)

        # Convert median params to a more readable format
        median_params_real_num_rounded = round((math.e ** median_params) / 1e9, 2)

        # Create two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        def make_boxplot(data, bin_labels, ax, title):
            sns.boxplot(x=f"{feature}_bin", y=col_shap, data=data, notch=True, palette="Blues", ax=ax)
            sns.stripplot(x=f"{feature}_bin", y=col_shap, data=data, color='black', alpha=0.3, jitter=True, ax=ax)

            # Annotate sample sizes above the boxplots
            for i, label in enumerate(bin_labels):
                n = len(data[data[f"{feature}_bin"] == label])
                ax.annotate(f'n={n}', xy=(i, data[col_shap].max()), xytext=(0, 5), 
                            textcoords="offset points", ha='center', fontsize=12, color='black')

            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel(f'{feature.capitalize()} Percentage in Training Data')
            ax.set_ylabel('SHAP Value Impact')
            ax.set_xticklabels(bin_labels, rotation=30, ha="right")  # Rotate labels for readability

        make_boxplot(small_models, small_labels, ax1, f'Models < {median_params_real_num_rounded}B params')
        make_boxplot(large_models, large_labels, ax2, f'Models ≥ {median_params_real_num_rounded}B params')

    else:  # For "english" or other features
        df, bin_labels = bin_data(df)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.boxplot(x=f"{feature}_bin", y=col_shap, data=df, notch=True, palette="Blues", ax=ax)
        sns.stripplot(x=f"{feature}_bin", y=col_shap, data=df, color='black', alpha=0.3, jitter=True, ax=ax)

        # Annotate sample sizes
        for i, label in enumerate(bin_labels):
            n = len(df[df[f"{feature}_bin"] == label])
            ax.annotate(f'n={n}', xy=(i, df[col_shap].max()), xytext=(0, 5), 
                        textcoords="offset points", ha='center', fontsize=12, color='black')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{feature.capitalize()} Percentage in Training Data vs. SHAP Impact')
        ax.set_xlabel(f'{feature.capitalize()} Percentage in Training Data')
        ax.set_ylabel('SHAP Value Impact')
        ax.set_xticklabels(bin_labels, rotation=30, ha="right")

    plt.tight_layout(pad=3.0)
    return plt

# Read the data
df = pd.read_csv('/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/performance_prediction/results_db/figures_02_13/raw_shap_humaneval_0-shot_xgboost_all_accuracy_freegens_True_303.csv')

fig_combined = analyze_feature_shap_trend(df, feature="code")
fig_combined.savefig("code_shap_combined_humaneval_303.png")
fig_combined.savefig("code_shap_combined_humaneval_303.pdf")
assert False
# Generate plots
fig_code = analyze_feature_shap_relationship(df, feature="code", num_bins=5)
fig_code.savefig("code_shap_relationship_lambada.png")
plt.close()

fig_english = analyze_feature_shap_relationship(df, feature="english", num_bins=5)
fig_english.savefig("english_shap_relationship_lambada.png")
plt.close()
