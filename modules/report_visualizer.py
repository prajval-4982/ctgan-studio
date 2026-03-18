"""
Report Visualizer Module
=========================
Generates evaluation bar charts and helper functions
for the Flask dashboard.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_comparison_chart(
    ml_results: dict,
    output_dir: str = os.path.join("static", "plots"),
) -> str:
    """
    Create a grouped bar chart comparing ML model metrics
    when trained on real vs. synthetic data.

    Args:
        ml_results: Dict with 'real_results' and 'synthetic_results' lists.
        output_dir: Directory to save the chart.

    Returns:
        Path to the saved chart image.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    real_res = ml_results["real_results"]
    synth_res = ml_results["synthetic_results"]
    model_names = [r["Model"] for r in real_res]

    fig, axes = plt.subplots(1, len(model_names), figsize=(7 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]

    bar_width = 0.30
    x = np.arange(len(metrics))

    colors_real = "#4C72B0"
    colors_synth = "#DD8452"

    for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
        real_vals = [real_res[idx][m] for m in metrics]
        synth_vals = [synth_res[idx][m] for m in metrics]

        bars1 = ax.bar(x - bar_width / 2, real_vals, bar_width,
                       label="Real", color=colors_real, edgecolor="white")
        bars2 = ax.bar(x + bar_width / 2, synth_vals, bar_width,
                       label="Synthetic", color=colors_synth, edgecolor="white")

        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "ml_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    return path


def format_ml_results_table(ml_results: dict) -> tuple[list[dict], list[dict]]:
    """
    Format ML evaluation results for display in an HTML table.

    Args:
        ml_results: Dict with 'real_results' and 'synthetic_results'.

    Returns:
        (real_rows, synth_rows) — each a list of dicts suitable for
        rendering in Jinja2 templates.
    """
    return ml_results["real_results"], ml_results["synthetic_results"]


def format_stats_table(stats: dict) -> tuple[str, str]:
    """
    Convert statistical comparison DataFrames to HTML table strings.

    Args:
        stats: Dict from statistical_evaluator.compare_statistics().

    Returns:
        (mean_html, std_html) — HTML table strings.
    """
    mean_html = stats["mean_comparison"].to_html(
        classes="table table-dark table-striped", border=0, float_format="%.4f"
    )
    std_html = stats["std_comparison"].to_html(
        classes="table table-dark table-striped", border=0, float_format="%.4f"
    )
    return mean_html, std_html


def plot_feature_importance(
    feature_importance: dict,
    output_dir: str = os.path.join("static", "plots"),
) -> str:
    """
    Generate a horizontal bar chart comparing feature importances
    from Random Forest models trained on Real vs. Synthetic data.

    Args:
        feature_importance: Dict with 'features', 'real', 'synthetic' lists.
        output_dir:         Directory to save the chart.

    Returns:
        Path to the saved chart image.
    """
    os.makedirs(output_dir, exist_ok=True)

    features = feature_importance["features"]
    real_imp = np.array(feature_importance["real"])
    synth_imp = np.array(feature_importance["synthetic"])

    # Sort by real importance (descending)
    sorted_idx = np.argsort(real_imp)
    features = [features[i] for i in sorted_idx]
    real_imp = real_imp[sorted_idx]
    synth_imp = synth_imp[sorted_idx]

    y = np.arange(len(features))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.45)))
    ax.barh(y - bar_height / 2, real_imp, bar_height,
            label="Real", color="#111827", edgecolor="white")
    ax.barh(y + bar_height / 2, synth_imp, bar_height,
            label="Synthetic", color="#9ca3af", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importance Alignment (Random Forest)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    return path

