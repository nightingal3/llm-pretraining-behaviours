import argparse
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

from common_args import add_common_args
from common_args import load_data, process_data

def general_scaling_law(x, params_const, tokens_const, power_const_params, power_const_tokens):
    num_params, num_tokens = x
    return ((params_const/num_params) ** (power_const_params / power_const_tokens) + (tokens_const/num_tokens)) ** power_const_tokens 


def fit_and_plot_scaling_law(dataset, y_col):
    # Prepare the data
    X = dataset[['total_params', 'pretraining_summary:total_tokens_billions']].values.T
    y = dataset[y_col].values

    # Remove rows with NaN values
    mask = ~np.isnan(X).any(axis=0) & ~np.isnan(y)
    X = X[:, mask]
    y = y[mask]

    if len(y) == 0:
        print(f"No data available for {y_col}")
        return

    # reverse y to 1 - y since it's accuracy
    y = 1 - y

    # re-transform total tokens to billions
    X[1] = X[1] * 1e9

    # Fit the scaling law
    initial_guess = [1e5, 1e3, 0.1, 0.1]
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, 1, 1])
    
    try:
        popt, pcov = curve_fit(general_scaling_law, X, y, p0=initial_guess, bounds=bounds, maxfev=10000)
    except RuntimeError as e:
        print(f"Error fitting scaling law for {y_col}: {str(e)}")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for number of parameters
    ax1.scatter(X[0], 1 - y, alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel(f'{y_col}')
    ax1.set_title(f'Scaling Law: {y_col} vs Number of Parameters')

    # Generate points for the predicted curve (parameters)
    N_range = np.logspace(np.log10(X[0].min()), np.log10(X[0].max()), 100)
    y_pred = general_scaling_law((N_range, np.median(X[1]) * np.ones_like(N_range)), *popt)
    ax1.plot(N_range, 1 - y_pred, 'r-', label='Predicted')
    ax1.legend()

    # Plot for number of tokens
    ax2.scatter(X[1], 1 - y, alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Tokens')
    ax2.set_ylabel(f'1 - {y_col}')
    ax2.set_title(f'Scaling Law: {y_col} vs Number of Tokens')

    # Generate points for the predicted curve (tokens)
    D_range = np.logspace(np.log10(X[1].min()), np.log10(X[1].max()), 100)
    y_pred = general_scaling_law((np.median(X[0]) * np.ones_like(D_range), D_range), *popt)
    ax2.plot(D_range, 1 - y_pred, 'r-', label='Predicted')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"scaling_law_{y_col}.png")
    plt.close()

    # Print the fitted parameters
    print(f"\nFitted parameters for {y_col}:")
    print(f"params_const = {popt[0]:.4e}")
    print(f"tokens_const = {popt[1]:.4e}")
    print(f"power_const_params = {popt[2]:.4f}")
    print(f"power_const_tokens = {popt[3]:.4f}")

    # Calculate R-squared
    residuals = y - general_scaling_law(X, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r_squared:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    parser.add_argument(
        "--predictor_type",
        type=str,
        default="scaling_laws"
    )
    args = parser.parse_args()


    dataset = load_data(args)
    training_scores = pd.read_csv(args.train_labels)

    cols_from_results = set(training_scores.columns) - {"model_name", "id"}

    if args.y_cols == ["all"]:
        y_cols = [t for t in list(cols_from_results) if t.endswith(f"_{args.metric}")]
    else:
        y_cols = args.y_cols

    dataset = process_data(dataset, args, cols_from_results)
    cols_to_get = ["total_params", "pretraining_summary:total_tokens_billions"] + y_cols
    dataset = dataset[cols_to_get]
    

    for y_col in y_cols:
        fit_and_plot_scaling_law(dataset, y_col)