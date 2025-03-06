import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from performance_predict_from_db import load_data_from_db, aggregate_multi_part_evals, prepare_task_data

def _scaling_law(inputs, alpha_N, alpha_D, Nc, Dc):
    N, D = inputs
    return ((Nc / N)**(alpha_N / alpha_D) + (Dc / D))**alpha_D

def scaling_law(inputs, alpha_N, alpha_D, Nc, Dc):
    """
    Modified scaling law that prevents NaN generation during fitting and
    optionally inverts the output if 'increasing_good' is True.

    Parameters:
      inputs: tuple (N, D) where:
        N = total_params
        D = total_tokens
      alpha_N, alpha_D: exponents for parameters/data
      Nc, Dc: 'critical' scales for params/tokens
      increasing_good: 
        - False (default): output is smaller when N or D are larger (loss-like)
        - True: output is inverted, so larger = better (accuracy-like)

    Returns:
      A numpy array of the predicted metric values in [0,1].
    """
    N, D = inputs
    
    try:
        # Add small epsilon to prevent division by zero
        eps = 1e-10
        N = np.maximum(N, eps)
        D = np.maximum(D, eps)
        
        # Ensure parameters are positive
        alpha_N = abs(alpha_N) + eps
        alpha_D = abs(alpha_D) + eps
        Nc = abs(Nc) + eps
        Dc = abs(Dc) + eps
        
        # Handle the power terms carefully
        with np.errstate(all='ignore'):
            ratio_N = np.clip(Nc / N, eps, 1e6)
            ratio_D = np.clip(Dc / D, eps, 1e6)
            
            # First power term
            power1 = alpha_N / alpha_D
            term1 = np.power(ratio_N, power1, where=(ratio_N > 0))
            term1 = np.clip(term1, eps, 1e6)
            
            # Sum and final power
            sum_terms = term1 + ratio_D
            sum_terms = np.clip(sum_terms, eps, 1e6)
            
            result = np.power(sum_terms, alpha_D, where=(sum_terms > 0))
            
            # Ensure result is valid
            result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
            result = np.clip(result, 0, 1)
        
            
    except Exception as e:
        # Return a reasonable fallback value
        return np.full_like(N, 0.5, dtype=float)


def fit_with_scaling_law(N, D, y):
    """Helper function to fit scaling law with multiple attempts"""
    initial_guesses = [
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, np.median(N), np.median(D)],
        [0.1, 0.1, N.max(), D.max()]
    ]
    
    best_error = float('inf')
    best_popt = None
    
    for p0 in initial_guesses:
        try:
            popt, _ = curve_fit(
                scaling_law, 
                (N, D), 
                y,
                p0=p0,
                bounds=([eps, eps, eps, eps], [10.0, 10.0, 100.0, 100.0]),
                maxfev=10000,
                method='trf'
            )
            
            # Calculate error
            y_pred = scaling_law((N, D), *popt)
            error = np.mean((y - y_pred)**2)
            
            if error < best_error:
                best_error = error
                best_popt = popt
                
        except Exception as e:
            continue
            
    if best_popt is None:
        raise ValueError("Failed to fit with any initial guess")
        
    return best_popt

def plot_scaling_with_prediction(N, D, y, popt, benchmark, r2, mae, increasing_good=True):
    """Plots the observed data and overlay the predicted scaling law with robust error handling"""
    try:
        # Generate a grid for predictions
        grid_N, grid_D = np.meshgrid(
            np.linspace(min(N)-1, max(N)+1, 100),
            np.linspace(min(D)-1, max(D)+1, 100)
        )

        # Evaluate the scaling law on the grid with error handling
        try:
            grid_predictions = _scaling_law((grid_N, grid_D), *popt)
            grid_predictions = np.nan_to_num(grid_predictions, nan=np.mean(y))  # Replace NaNs with mean
            grid_predictions = np.clip(grid_predictions, y.min(), y.max())
        except:
            # If prediction fails, create a simple gradient
            print(f"Warning: Prediction failed for {benchmark}, using fallback visualization")
            xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
            grid_predictions = xx + yy  # Simple diagonal gradient
            grid_predictions = (grid_predictions - grid_predictions.min()) / (grid_predictions.max() - grid_predictions.min())
            grid_predictions = grid_predictions * (y.max() - y.min()) + y.min()

        # Plot contour map
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(grid_N, grid_D, grid_predictions, 
                              levels=np.linspace(y.min(), y.max(), 20), 
                              cmap='viridis', 
                              alpha=0.6)
        
        # Create colorbar with larger ticks
        cbar = plt.colorbar(contour, 
                           ticks=np.linspace(y.min(), y.max(), 6))
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label("Predicted Performance", fontsize=20)

        # Overlay observed data points
        scatter = plt.scatter(N, D, c=y, cmap='viridis', edgecolor='k', s=80,)

        # add fit r2/mae
        #plt.text(0.5, 0.9, f"R²: {r2:.2f}, MAE: {mae:.2f}", 
               #transform=plt.gca().transAxes, fontsize=20)

        # make ticks text larger
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # bound x/y to original scale
        plt.xlim(min(N)-1, max(N)+1)
        plt.ylim(min(D)-1, max(D)+1)
        plt.xlabel("Log Total Params (N)", fontsize=20)
        plt.ylabel("Log Total Tokens (D)", fontsize=20)
        plt.title(f"{benchmark} performance", fontsize=34)
        
        # Only add legend if we have labeled artists
        if len(plt.gca().get_legend_handles_labels()[0]) > 0:
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(f"./performance_prediction/scaling_deviation_303_new/scaling_law_fit_{benchmark}.png")
        plt.savefig(f"./performance_prediction/scaling_deviation_303_new/scaling_law_fit_{benchmark}.pdf")
        plt.close()
        
    except Exception as e:
        print(f"Warning: Plot failed for {benchmark} with error: {e}")
        plt.close()  # Make sure to close any open figures

def plot_log_linear_fit(N, D, y, model, benchmark, r2, mae):
    """
    Plots the observed data and overlays the predicted log-linear fit.
    
    Parameters:
      N: 1D array of log Total Params.
      D: 1D array of log Total Tokens.
      y: 1D array of observed metric values.
      model: A fitted LinearRegression (or similar) model.
      benchmark: String, benchmark name.
      r2: float, R² value of the fit.
      mae: float, MAE of the fit.
    """
    # Generate a grid for predictions over the range of N and D.
    grid_N, grid_D = np.meshgrid(
        np.linspace(min(N)-1, max(N)+1, 100),
        np.linspace(min(D)-1, max(D)+1, 100)
    )

    # For a log-linear model, predictions are computed as:
    #   y_pred = intercept + coef[0] * N + coef[1] * D.
    # Here, we prepare the grid points for prediction.
    grid_points = np.column_stack([grid_N.ravel(), grid_D.ravel()])
    grid_pred = model.predict(grid_points)
    grid_predictions = grid_pred.reshape(grid_N.shape)

    # Clip predictions to a reasonable range (based on observed y)
    grid_predictions = np.clip(grid_predictions, y.min(), y.max())

    # Create the contour plot
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(
        grid_N, grid_D, grid_predictions,
        levels=np.linspace(y.min(), y.max(), 20),
        cmap='viridis',
        alpha=0.6
    )
    cbar = plt.colorbar(contour, 
                       ticks=np.linspace(grid_predictions.min(), grid_predictions.max(), 6))
    cbar.ax.tick_params(labelsize=14)  # Make tick labels bigger
    cbar.set_label("Predicted Performance", fontsize=16)  # Make label bigger

    # Overlay the observed data points; colors represent performance.
    plt.scatter(N, D, c=y, cmap='viridis', edgecolor='k', label='Observed Data')

    # Display R² and MAE on the plot.
    plt.text(0.5, 0.9, f"R²: {r2:.2f}, MAE: {mae:.2f}", transform=plt.gca().transAxes)

    # Set the axis limits and labels.
    plt.xlim(min(N)-1, max(N)+1)
    plt.ylim(min(D)-1, max(D)+1)
    plt.xlabel("Log Total Params (N)")
    plt.ylabel("Log Total Tokens (D)")
    plt.title(f"Log-linear Fit for {benchmark}")
    plt.legend()
    plt.tight_layout()

    # Save the figure (adjust the path as needed).
    plt.savefig(f"./performance_prediction/scaling_deviation_303_new/log_linear_fit_{benchmark}.png")
    plt.savefig(f"./performance_prediction/scaling_deviation_303_new/log_linear_fit_{benchmark}.pdf")
    plt.close()


def summarize_results(results):
    results_list = []
    for benchmark, res in results.items():
        result_dict = {
            'benchmark': benchmark,
            'r2': res['r2'],
            'mae': res['mae'],
            'method': res['method']
        }
        
        # Add coefficient information based on method
        if res['method'] == 'log_linear':
            result_dict.update({
                'param_coef': res['params'][0],  # First coefficient is for params
                'token_coef': res['params'][1],  # Second coefficient is for tokens 
                'param_token_ratio': res['params'][0] / res['params'][1] if res['params'][1] != 0 else float('inf')
            })
        elif res['method'] == 'non_linear':
            result_dict.update({
                'alpha_N': res['params'][0],  # Parameter sensitivity
                'alpha_D': res['params'][1],  # Data sensitivity
                'Nc': res['params'][2],       # Critical compute
                'Dc': res['params'][3],       # Critical data
                'param_token_ratio': res['params'][0] / res['params'][1] if res['params'][1] != 0 else float('inf')
            })
            
        results_list.append(result_dict)
    
    results_df = pd.DataFrame(results_list)
    
    # Sort by R² and print detailed analysis
    sorted_results = results_df.sort_values(by='r2', ascending=False)
    
    print("\n=== Sorted Results by R² with Coefficient Analysis ===")
    if 'param_coef' in sorted_results.columns:  # Log-linear results
        print("\nLog-linear model coefficients:")
        for _, row in sorted_results.iterrows():
            print(f"\nBenchmark: {row['benchmark']}")
            print(f"R²: {row['r2']:.3f}, MAE: {row['mae']:.3f}")
            print(f"Parameter coefficient: {row['param_coef']:.3f}")
            print(f"Token coefficient: {row['token_coef']:.3f}")
            print(f"Param/Token ratio: {row['param_token_ratio']:.3f}")
            
    elif 'alpha_N' in sorted_results.columns:  # Non-linear results
        print("\nNon-linear scaling law parameters:")
        for _, row in sorted_results.iterrows():
            print(f"\nBenchmark: {row['benchmark']}")
            print(f"R²: {row['r2']:.3f}, MAE: {row['mae']:.3f}")
            print(f"α_N (param sensitivity): {row['alpha_N']:.3f}")
            print(f"α_D (token sensitivity): {row['alpha_D']:.3f}")
            print(f"Nc (critical params): {row['Nc']:.3e}")
            print(f"Dc (critical tokens): {row['Dc']:.3e}")
            print(f"Param/Token sensitivity ratio: {row['param_token_ratio']:.3f}")
    
    # Save detailed results
    sorted_results.to_csv("./performance_prediction/scaling_deviation_303_new/scaling_coefficients.csv", index=False)
    
    return sorted_results


def standardize_task_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize duplicate task names while preserving domains"""
    df = df.copy()
    
    # Convert hendrycksTest to mmlu
    hendrycks_mask = df['benchmark'].str.startswith('hendrycksTest-')
    if hendrycks_mask.any():
        df.loc[hendrycks_mask, 'benchmark'] = df.loc[hendrycks_mask, 'benchmark'].str.replace('hendrycksTest-', 'mmlu_')
    
    # Fix ARC challenge naming
    df.loc[df['benchmark'] == 'arc:challenge', 'benchmark'] = 'arc_challenge'
    
    return df

def validate_scaling_analysis(db_path: str, metric: str = "accuracy", use_non_linear: bool = False):
    """Main validation pipeline with MMLU averaging."""
    # Load data
    scaling_df = load_data_from_db(db_path, "scaling_laws", metric)

    # Filter and preprocess data
    scaling_df = scaling_df.dropna(subset=['total_params', 'pretraining_summary_total_tokens_billions', 'value'])
    scaling_df['total_params'] = np.log(scaling_df['total_params'])
    scaling_df['pretraining_summary_total_tokens_billions'] = np.log(scaling_df['pretraining_summary_total_tokens_billions'])
    scaling_df = standardize_task_names(scaling_df)
    scaling_df["overall_setting"] = scaling_df["benchmark"] + "_" + scaling_df["setting"]

    scaling_df = aggregate_multi_part_evals(scaling_df)

    task_settings = scaling_df.groupby(['benchmark', 'setting']).size().reset_index()

    results = {}
    # Iterate through benchmarks
    for _, row in task_settings.iterrows():
        benchmark = row["benchmark"]
        setting = row["setting"]
        print(f"\nAnalyzing {benchmark}:")

        if "fld" in benchmark:
            continue
        
        features, labels = prepare_task_data(scaling_df, benchmark, setting)

        if len(features) < 30:
            print(f"Skipping {benchmark} due to insufficient data points")
            continue

        
        # Extract features and target
        N = features['total_params'].values
        D = features['pretraining_summary_total_tokens_billions'].values
        y = labels

        # need to replace 0 with small value to avoid instability
        #y = np.where(y == 0, 1e-6, y)

        if use_non_linear:
            try:
                if benchmark in ["anli", "xnli", "logiqa2", "mathqa"]:
                    sl_fn = _scaling_law
                else:
                    sl_fn = _scaling_law

                popt, _ = curve_fit(
                    sl_fn, 
                    (N, D), 
                    y,
                    #p0=[1, 1, 1, 1],
                    p0=[0.5, 0.5, np.min(N), np.min(D)],
                    maxfev=100000,
                    method='trf'
                )

                y_pred = sl_fn((N, D), *popt)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                print(f"Non-linear fit R²: {r2:.2f}, MAE: {mae:.2f}")

                results[benchmark] = {
                    'method': 'non_linear',
                    'r2': r2,
                    'mae': mae,
                    'params': popt
                }
                
                print(f"Plotting {benchmark} with {len(N)} points")
                plot_scaling_with_prediction(N, D, y, popt, benchmark, r2, mae)
                
            except Exception as e:
                print(f"Non-linear fitting failed for {benchmark}: {e}")
        else:
            # Log-linear fitting remains the same...
            X = features[['total_params', 'pretraining_summary_total_tokens_billions']]
            y = labels

            # Fit linear model and cross-validate
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            print(f"Log-linear fit R²: {r2:.2f}, MAE: {mae:.2f}")

            results[benchmark] = {
                'method': 'log_linear',
                'r2': r2,
                'mae': mae,
                'params': model.coef_
            }

            plot_log_linear_fit(
                N=features['total_params'].values,
                D=features['pretraining_summary_total_tokens_billions'].values,
                y=labels,
                model=model,
                benchmark=benchmark,
                r2=r2,
                mae=mae
            )


    return results

# Visualization functions
def plot_results(scaling_df, benchmark, popt=None):
    """Plot the actual vs predicted values."""
    bench_df = scaling_df[scaling_df['benchmark'] == benchmark]
    N = bench_df['total_params'].values
    D = bench_df['pretraining_summary_total_tokens_billions'].values
    y = bench_df['value'].values

    if popt is not None:
        y_pred = scaling_law((N, D), *popt)
        plt.scatter(y, y_pred, label="Non-linear fit", alpha=0.7)
    else:
        plt.scatter(N, y, label="Observed", alpha=0.7)

    plt.xlabel("Observed", fontsize=16)
    plt.ylabel("Predicted", fontsize=16)
    plt.title(benchmark, fontsize=20)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    db_path = "/data/tir/projects/tir5/users/mengyan3/tower-llm-training/tower-llm-training/metadata/duckdb/2025_03_03.duckdb"
    metric = "accuracy"

    # Run log-linear validation
    #print("\n=== Log-linear Validation ===")
    #log_linear_results = validate_scaling_analysis(db_path, metric, use_non_linear=False)

    # Run non-linear validation
    print("\n=== Non-linear Validation ===")
    non_linear_results = validate_scaling_analysis(db_path, metric, use_non_linear=True)
    sorted_results = summarize_results(non_linear_results)
    sorted_results.to_csv("./performance_prediction/scaling_deviation_303_new/sorted_results.csv", index=False)

    # Analyze and visualize
    # print("\n=== Results ===")
    # for benchmark, result in non_linear_results.items():
    #     if result['method'] == 'non_linear':
    #         print(f"Benchmark: {benchmark}, R²: {result['r2']:.2f}, MAE: {result['mae']:.2f}, Params: {result['params']}")
