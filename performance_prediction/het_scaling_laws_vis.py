import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from scipy.interpolate import griddata

from metadata.duckdb.model_metadata_db import AnalysisStore
from performance_predict_from_db import standardize_task_names

def load_scaling_data(db_path: str, metric: str = "accuracy") -> pd.DataFrame:
    """Load relevant evaluation data from DuckDB for scaling analysis"""
    store = AnalysisStore.from_existing(db_path)
    
    query = """
        SELECT 
            m.id,
            m.total_params,
            d.pretraining_summary_total_tokens_billions,
            e.benchmark,
            e.setting,
            e.metric_value as value,
            e.metric_stderr as value_stderr
        FROM model_annotations m
        LEFT JOIN dataset_info d ON m.id = d.id
        LEFT JOIN evaluation_results e ON m.id = e.id
        WHERE m.total_params IS NOT NULL 
        AND d.pretraining_summary_total_tokens_billions IS NOT NULL
        AND e.metric = ?
        -- Filter out rows where either key scaling metric is NULL
        AND m.total_params > 0
        AND d.pretraining_summary_total_tokens_billions > 0
    """
    
    df = store.con.execute(query, [metric]).df()
    store.con.close()
    
    # Clean up benchmark names
    df = standardize_task_names(df)  # Using your existing function
    
    # Log scale conversion for plotting
    df['log_params'] = np.log(df['total_params'])
    df['log_tokens'] = np.log(df['pretraining_summary_total_tokens_billions'])
    
    # Print summary stats
    print(f"\nLoaded {len(df)} evaluations across {df['benchmark'].nunique()} tasks")
    print("\nTasks with most evaluations:")
    print(df['benchmark'].value_counts().head())
    print("\nParameter range:", 
          f"Min: {df['total_params'].min():.2e}",
          f"Max: {df['total_params'].max():.2e}")
    print("\nToken range (B):", 
          f"Min: {df['pretraining_summary_total_tokens_billions'].min():.2f}",
          f"Max: {df['pretraining_summary_total_tokens_billions'].max():.2f}")
    
    return df


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0,1] range for comparison across tasks"""
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
def create_dual_scaling_plots(df: pd.DataFrame, task_groups: Dict[str, List[str]], normalize: bool = False):
    """Create plots showing both parameter and token scaling for each group"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    colors = sns.color_palette("husl", 8)
    
    for col, (group_name, tasks) in enumerate(task_groups.items()):
        for task, color in zip(tasks, colors):
            task_data = df[df['benchmark'] == task].copy()
            if len(task_data) < 5:
                continue
            
            # Filter out zero scores for parameter-dependent tasks
            if group_name == "Parameter-Dependent":
                task_data = task_data[task_data['value'] > 0]
            
            # Skip if no data left after filtering
            if len(task_data) < 3:
                continue
            
            # Plot parameter scaling
            sns.scatterplot(
                data=task_data,
                x='log_params',
                y='value',
                label=task,
                color=color,
                ax=axes[0, col],
                alpha=0.7
            )
            
            # Plot token scaling
            sns.scatterplot(
                data=task_data,
                x='log_tokens',
                y='value',
                label=task,
                color=color,
                ax=axes[1, col],
                alpha=0.7
            )
            
            # Add trend lines
            # Parameters
            z = np.polyfit(task_data['log_params'], task_data['value'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(task_data['log_params'].min(), task_data['log_params'].max(), 100)
            axes[0, col].plot(x_range, p(x_range), '--', color=color, alpha=0.5)
            
            # Tokens
            z = np.polyfit(task_data['log_tokens'], task_data['value'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(task_data['log_tokens'].min(), task_data['log_tokens'].max(), 100)
            axes[1, col].plot(x_range, p(x_range), '--', color=color, alpha=0.5)
        
        # Set titles and labels
        axes[0, col].set_title(f"{group_name}\nParameter Scaling")
        axes[1, col].set_title(f"{group_name}\nToken Scaling")
        axes[0, col].set_xlabel('Log(Parameters)')
        axes[1, col].set_xlabel('Log(Tokens)')
        axes[0, col].set_ylabel('Score')
        axes[1, col].set_ylabel('Score')
        
        axes[0, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig

def create_individual_plots(df: pd.DataFrame, output_dir: Path, normalize: bool = True):
    """Create individual plots for each task with interpolation"""
    tasks = df['benchmark'].unique()
    
    for task in tasks:
        task_data = df[df['benchmark'] == task].copy()
        if len(task_data) < 5:
            continue
            
        if normalize:
            task_data['plot_score'] = normalize_scores(task_data['value'])
            score_label = 'Normalized Score'
        else:
            task_data['plot_score'] = task_data['value']
            score_label = 'Score'

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'Scaling Behavior: {task}', fontsize=14)
        
        # Add padding to the grid to ensure we cover all points
        param_range = task_data['log_params'].max() - task_data['log_params'].min()
        token_range = task_data['log_tokens'].max() - task_data['log_tokens'].min()
        
        xi = np.linspace(
            task_data['log_params'].min() - 0.1 * param_range,
            task_data['log_params'].max() + 0.1 * param_range,
            100
        )
        yi = np.linspace(
            task_data['log_tokens'].min() - 0.1 * token_range,
            task_data['log_tokens'].max() + 0.1 * token_range,
            100
        )
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate
        zi = griddata(
            (task_data['log_params'], task_data['log_tokens']),
            task_data['plot_score'],
            (xi, yi),
            method='cubic',
            fill_value=np.nan
        )
        
        # Fill NaN values with nearest neighbor interpolation
        mask = np.isnan(zi)
        if mask.any():
            zi[mask] = griddata(
                (task_data['log_params'], task_data['log_tokens']),
                task_data['plot_score'],
                (xi[mask], yi[mask]),
                method='nearest'
            )
        
        # Create contour plot
        contour = ax.contourf(xi, yi, zi, levels=15, cmap='viridis', alpha=0.5)
        
        # Add scatter points
        scatter = ax.scatter(
            task_data['log_params'],
            task_data['log_tokens'],
            c=task_data['plot_score'],
            cmap='viridis',
            alpha=0.7
        )
        
        plt.colorbar(contour, label=score_label)
        ax.set_xlabel('Log(Parameters)')
        ax.set_ylabel('Log(Tokens)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / f"{task}_{score_label.lower().replace(' ', '_')}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

def analyze_scaling_behavior(df: pd.DataFrame, task_groups: Dict[str, List[str]], normalize: bool = True):
    """Show scaling trends for groups of tasks"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = sns.color_palette("husl", len(task_groups))
    markers = ['o', 's', '^', 'D', 'v']  # different markers for different groups
    
    for (group_name, tasks), color, marker in zip(task_groups.items(), colors, markers):
        for task in tasks:
            task_data = df[df['benchmark'] == task].copy()
            if len(task_data) < 5:
                continue
                
            if normalize:
                task_data['plot_score'] = normalize_scores(task_data['value'])
            else:
                task_data['plot_score'] = task_data['value']
                
            ax.scatter(
                task_data['log_params'],
                task_data['plot_score'],
                label=f"{group_name}: {task}",
                color=color,
                marker=marker,
                alpha=0.7
            )
            
            # Add trend line
            z = np.polyfit(task_data['log_params'], task_data['plot_score'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(task_data['log_params'].min(), task_data['log_params'].max(), 100)
            ax.plot(x_range, p(x_range), '--', color=color, alpha=0.5)
    
    ax.set_xlabel('Log(Parameters)')
    ax.set_ylabel('Normalized Score' if normalize else 'Score')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


def analyze_scaling_behavior(df: pd.DataFrame, task_groups: Dict[str, List[str]], normalize: bool = True):
    """Analyze and visualize scaling behavior for grouped tasks"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    score_label = 'Normalized Score' if normalize else 'Score'
    fig.suptitle(f'Scaling Behavior Across Task Groups ({score_label})', fontsize=16, y=1.05)
    
    colors = sns.color_palette("husl", 8)
    
    for ax, (group_name, tasks) in zip(axes, task_groups.items()):
        ax.set_title(f"{group_name} Tasks")
        ax.set_xlabel("Log(Parameters)")
        ax.set_ylabel(score_label)
        
        for task, color in zip(tasks, colors):
            task_data = df[df['benchmark'] == task].copy()
            if len(task_data) < 5:
                continue
                
            task_data['log_params'] = np.log(task_data['total_params'])
            task_data['plot_score'] = normalize_scores(task_data['value']) if normalize else task_data['value']
            
            sns.scatterplot(
                data=task_data,
                x='log_params',
                y='plot_score',
                label=task,
                color=color,
                ax=ax,
                alpha=0.7
            )
            
            z = np.polyfit(task_data['log_params'], task_data['plot_score'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(task_data['log_params'].min(), task_data['log_params'].max(), 100)
            ax.plot(x_range, p(x_range), '--', color=color, alpha=0.5)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_scaling_laws(df: pd.DataFrame):
    """Main analysis function"""
    task_groups = {
        "Parameter-Dependent": [
            "arithmetic_2da",
            "arithmetic_3da",
            "arithmetic_4da",
            "arithmetic_5da",
            "gsm8k",
            "gsm8k_cot",
            "minerva_math_algebra",
            "minerva_math_geometry",
            "minerva_math_precalc",
            "minerva_math_intermediate_algebra"
        ],
        "Token-Dependent": [
            "mmlu_professional_law",
            "mmlu_professional_medicine",
            "mmlu_college_medicine",
            "mmlu_clinical_knowledge",
            "mmlu_world_religions",
            "mmlu_virology",
            "mmlu_high_school_us_history",
            "mmlu_high_school_world_history"
        ],
        "Balanced": [
            "mmlu_logical_fallacies",
            "mmlu_computer_security",  
            "mmlu_college_computer_science",
            "lambada",
            "mmlu_college_mathematics",
            "mmlu_high_school_mathematics",
            "mmlu_abstract_algebra"
        ]
    }

    
    # Create output directories
    base_dir = Path("./scaling_analysis")
    individual_dir = base_dir / "individual_tasks"
    grouped_dir = base_dir / "grouped_tasks"
    
    for dir_path in [base_dir, individual_dir, grouped_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Generate individual plots (both normalized and raw)
    create_individual_plots(df, individual_dir, normalize=True)
    create_individual_plots(df, individual_dir, normalize=False)
    
    # Generate grouped plots (both normalized and raw)
    grouped_norm = analyze_scaling_behavior(df, task_groups, normalize=True)
    grouped_norm.savefig(grouped_dir / "scaling_behavior_normalized.png", bbox_inches='tight', dpi=300)
    
    grouped_raw = analyze_scaling_behavior(df, task_groups, normalize=False)
    grouped_raw.savefig(grouped_dir / "scaling_behavior_raw.png", bbox_inches='tight', dpi=300)

    breakpoint()
    dual_scaling = create_dual_scaling_plots(df, task_groups)
    dual_scaling.savefig(grouped_dir / "dual_scaling_behavior.png", bbox_inches='tight', dpi=300)
    
    
    plt.close('all')
    
    # Calculate and save scaling coefficients
    results = []
    for group_name, tasks in task_groups.items():
        for task in tasks:
            task_data = df[df['benchmark'] == task].copy()
            if len(task_data) < 5:
                continue
                
            task_data['log_params'] = np.log(task_data['total_params'])
            task_data['log_tokens'] = np.log(task_data['pretraining_summary_total_tokens_billions'])
            
            param_coef = np.corrcoef(task_data['log_params'], task_data['value'])[0,1]
            token_coef = np.corrcoef(task_data['log_tokens'], task_data['value'])[0,1]
            
            results.append({
                'task': task,
                'group': group_name,
                'param_scaling': param_coef,
                'token_scaling': token_coef,
                'min_score': task_data['value'].min(),
                'max_score': task_data['value'].max(),
                'score_range': task_data['value'].max() - task_data['value'].min()
            })
    
    # Save scaling coefficients to CSV
    pd.DataFrame(results).to_csv(base_dir / "scaling_coefficients.csv", index=False)

def main():
    db_path = "/data/tir/projects/tir6/general/mengyan3/tower-llm-training/metadata/duckdb/2024_12_05.duckdb"
    
    # Load data specifically for scaling analysis
    
    df = load_scaling_data(db_path, "accuracy")
    analyze_scaling_laws(df)
    

if __name__ == "__main__":
    main()