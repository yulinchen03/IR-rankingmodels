import matplotlib.pyplot as plt
import numpy as np
import os
import re

# --- Plotting Configuration ---
METRICS_TO_PLOT_ORDER = ['nDCG@100', 'RR', 'R@100']

PLOT_COLORS = {
    'Baseline': '#1f77b4',  # Muted Blue
    'Avg Ens': '#ff7f0e',   # Safety Orange
    'Cond Ens': '#2ca02c',  # Cooked Asparagus Green
    'Wgt Ens': '#d62728',   # Brick Red
    'Wgt L Ens': '#FFFF00' # yellow
}


def create_comparison_plot(plot_data, metrics, model_name, dataset_name, save_dir):
    """
    Creates a bar chart comparing evaluation methods for a model/dataset.
    (Function body remains the same as the previous version)

    Args:
        plot_data (dict): Dict mapping method names (e.g., 'Baseline') to dicts
                          of their scores ({'nDCG@100': 0.5, ...}).
        metrics (list): List of metric names defining the order on the x-axis.
        model_name (str): Name of the model for the plot title.
        dataset_name (str): Name of the dataset for the plot title.
        save_dir (str): Directory where the plot image will be saved.
    """

    method_names = list(plot_data.keys())
    num_methods = len(method_names)
    num_metrics = len(metrics)

    if num_methods == 0:
        print(f"No data provided to plot for {model_name} on {dataset_name}.")
        return

    x = np.arange(num_metrics)
    total_width = 0.8
    bar_width = total_width / max(1, num_methods)
    start_offset = - (total_width - bar_width) / 2

    fig, ax = plt.subplots(figsize=(max(12, num_metrics * 1.2), 7)) # Dynamic width
    max_score_found = 0.0

    for i, method_name in enumerate(method_names):
        method_scores = plot_data.get(method_name, {})
        if not method_scores:
             print(f"Warning: No scores found for method '{method_name}' during plotting.")
             continue

        scores = [method_scores.get(metric, np.nan) for metric in metrics]
        current_max = np.nanmax(scores) if not all(np.isnan(s) for s in scores) else 0
        max_score_found = max(max_score_found, current_max)
        position = x + start_offset + i * bar_width
        rects = ax.bar(position, scores, bar_width,
                       label=method_name,
                       color=PLOT_COLORS.get(method_name, None))

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Evaluation Metrics Comparison: {model_name} on {dataset_name}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, max(1.0, max_score_found * 1.1))
    ax.legend(title="Evaluation Method", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout(rect=[0, 0, 0.9, 1])

    safe_model_name = re.sub(r'[\\/:"*?<>|]+', '_', model_name)
    plot_filename = f"comparison_plot_{safe_model_name}_{dataset_name}.png"
    plot_save_path = os.path.join(save_dir, plot_filename)
    try:
        plt.savefig(plot_save_path)
        print(f"Plot saved to {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot {plot_save_path}: {e}")
    finally:
        plt.close(fig)