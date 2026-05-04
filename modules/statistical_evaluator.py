"""
Statistical Evaluation Module
==============================
Compares real vs. synthetic datasets using basic statistical
measures, distribution / correlation plots, and scientific
quality metrics (KS Test, JS Distance, Correlation Similarity).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


def compare_statistics(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """
    Compare mean, standard deviation, and correlation matrices
    between real and synthetic datasets.

    Only numeric columns present in both DataFrames are compared.

    Args:
        real:      The original (real) DataFrame.
        synthetic: The generated (synthetic) DataFrame.

    Returns:
        A dictionary with keys:
            'mean_comparison'  → DataFrame with real vs synthetic means
            'std_comparison'   → DataFrame with real vs synthetic std devs
            'corr_real'        → Correlation matrix of real data
            'corr_synthetic'   → Correlation matrix of synthetic data
    """
    # Use only shared numeric columns
    common_numeric = [
        c for c in real.select_dtypes(include="number").columns
        if c in synthetic.columns
    ]

    real_num = real[common_numeric]
    synth_num = synthetic[common_numeric]

    mean_cmp = pd.DataFrame({
        "Real Mean": real_num.mean(),
        "Synthetic Mean": synth_num.mean(),
        "Difference": (real_num.mean() - synth_num.mean()).abs(),
    })

    std_cmp = pd.DataFrame({
        "Real Std": real_num.std(),
        "Synthetic Std": synth_num.std(),
        "Difference": (real_num.std() - synth_num.std()).abs(),
    })

    return {
        "mean_comparison": mean_cmp,
        "std_comparison": std_cmp,
        "corr_real": real_num.corr(),
        "corr_synthetic": synth_num.corr(),
    }


def calculate_scientific_scores(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """
    Compute advanced distribution quality metrics:

    - Kolmogorov-Smirnov (KS) Test per column
    - Jensen-Shannon (JS) Distance per column
    - Correlation Matrix Similarity (Frobenius norm)

    Args:
        real:      The original (real) DataFrame.
        synthetic: The generated (synthetic) DataFrame.

    Returns:
        Dict with:
          - 'ks_results':     list of {column, statistic, p_value, pass}
          - 'ks_mean_score':  overall KS similarity (1 - mean_statistic)
          - 'js_results':     list of {column, distance}
          - 'js_mean_score':  overall JS similarity (1 - mean_distance)
          - 'corr_similarity': 0-100% similarity of correlation matrices
    """
    common_numeric = [
        c for c in real.select_dtypes(include="number").columns
        if c in synthetic.columns
    ]

    real_num = real[common_numeric]
    synth_num = synthetic[common_numeric]

    # --- Kolmogorov-Smirnov Test ---
    ks_results = []
    for col in common_numeric:
        stat, p_val = ks_2samp(real_num[col].dropna(), synth_num[col].dropna())
        ks_results.append({
            "column": col,
            "statistic": round(stat, 4),
            "p_value": round(p_val, 4),
            "pass": p_val > 0.05,  # True = distributions are similar
        })

    ks_stats = [r["statistic"] for r in ks_results]
    ks_mean_score = round(1.0 - np.mean(ks_stats), 4) if ks_stats else 0

    # --- Jensen-Shannon Distance ---
    js_results = []
    for col in common_numeric:
        r_vals = real_num[col].dropna().values
        s_vals = synth_num[col].dropna().values

        # Create shared bins for histogram-based probability distributions
        combined = np.concatenate([r_vals, s_vals])
        bins = np.histogram_bin_edges(combined, bins=50)

        r_hist, _ = np.histogram(r_vals, bins=bins, density=True)
        s_hist, _ = np.histogram(s_vals, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        r_hist = r_hist + 1e-10
        s_hist = s_hist + 1e-10

        # Normalize to proper probability distributions
        r_hist = r_hist / r_hist.sum()
        s_hist = s_hist / s_hist.sum()

        js_dist = jensenshannon(r_hist, s_hist)
        js_results.append({
            "column": col,
            "distance": round(float(js_dist), 4),
        })

    js_dists = [r["distance"] for r in js_results]
    js_mean_score = round(1.0 - np.mean(js_dists), 4) if js_dists else 0

    # --- Correlation Matrix Similarity ---
    corr_real = real_num.corr().fillna(0).values
    corr_synth = synth_num.corr().fillna(0).values

    # Frobenius norm of difference, normalized to 0-100%
    max_diff = np.sqrt(2 * corr_real.shape[0] * corr_real.shape[1])  # worst case
    actual_diff = np.linalg.norm(corr_real - corr_synth, "fro")
    corr_similarity = round(max(0, min(100, (1 - actual_diff / max_diff) * 100)), 2)

    return {
        "ks_results": ks_results,
        "ks_mean_score": ks_mean_score,
        "js_results": js_results,
        "js_mean_score": js_mean_score,
        "corr_similarity": corr_similarity,
    }


def get_histogram_data(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """
    Compute exactly 30 bins for each shared numeric column and return
    the raw frequencies and bin edges (labels) for frontend charting.

    Returns:
        dict: {
            "col_name": {
                "labels": ["0-10", "10-20", ...],
                "real_counts": [15, 42, ...],
                "synth_counts": [18, 39, ...]
            }
        }
    """
    common_numeric = [
        c for c in real.select_dtypes(include="number").columns
        if c in synthetic.columns
    ]

    hist_data = {}

    for col in common_numeric:
        real_vals = real[col].dropna()
        synth_vals = synthetic[col].dropna()

        # Define 30 bins spanning the combined min/max
        combined = np.concatenate([real_vals.values, synth_vals.values])
        if len(combined) == 0:
            continue

        bins = np.histogram_bin_edges(combined, bins=30)

        r_counts, _ = np.histogram(real_vals, bins=bins)
        s_counts, _ = np.histogram(synth_vals, bins=bins)

        # Format labels as ranges like "2.5 - 3.1"
        labels = []
        for i in range(len(bins) - 1):
            labels.append(f"{bins[i]:.2f} - {bins[i+1]:.2f}")

        hist_data[col] = {
            "labels": labels,
            "real_counts": r_counts.tolist(),
            "synth_counts": s_counts.tolist(),
        }

    return hist_data


def plot_distributions(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    output_dir: str = os.path.join("static", "plots"),
) -> list[str]:
    """
    Generate overlaid distribution histograms for each shared
    numeric column and save them as PNG images.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []

    common_numeric = [
        c for c in real.select_dtypes(include="number").columns
        if c in synthetic.columns
    ]

    import re
    for col in common_numeric:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(real[col].dropna(), bins=30, alpha=0.55, label="Real", color="#4C72B0")
        ax.hist(synthetic[col].dropna(), bins=30, alpha=0.55, label="Synthetic", color="#DD8452")
        ax.set_title(f"Distribution — {col}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()

        safe_col = re.sub(r'[\\/*?:"<>|]', "", col)
        path = os.path.join(output_dir, f"dist_{safe_col}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_correlation_heatmaps(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    output_dir: str = os.path.join("static", "plots"),
) -> str:
    """
    Generate side-by-side correlation heatmaps for real and
    synthetic data and save as a single image.
    """
    os.makedirs(output_dir, exist_ok=True)

    common_numeric = [
        c for c in real.select_dtypes(include="number").columns
        if c in synthetic.columns
    ]

    real_corr = real[common_numeric].corr()
    synth_corr = synthetic[common_numeric].corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(real_corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[0], vmin=-1, vmax=1, square=True)
    axes[0].set_title("Real Data Correlation", fontsize=13, fontweight="bold")

    sns.heatmap(synth_corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1], vmin=-1, vmax=1, square=True)
    axes[1].set_title("Synthetic Data Correlation", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmaps.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    return path
