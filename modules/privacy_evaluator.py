"""
Privacy Evaluation Module
==========================
Evaluates the privacy risk of synthetic data by computing
the Distance to Closest Record (DCR) between synthetic and
real datasets. A low DCR means potential data memorisation /
identity disclosure.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def calculate_privacy_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    threshold: float = 0.05,
) -> dict:
    """
    Calculate privacy risk metrics using Distance to Closest Record (DCR).

    For every synthetic row, we find its nearest neighbour in the real data
    (Euclidean distance after standard-scaling).  If the distance is below
    *threshold* the row is considered a potential privacy leak.

    Args:
        real:       Numeric / encoded real DataFrame.
        synthetic:  Numeric / encoded synthetic DataFrame.
        threshold:  Distance below which a record is "too close" (default 0.05).

    Returns:
        Dict with privacy metrics:
          - mean_dcr:          Average distance to closest real record
          - median_dcr:        Median distance
          - min_dcr:           Minimum distance (worst-case privacy leak)
          - max_dcr:           Maximum distance
          - pct_at_risk:       % of synthetic rows within *threshold*
          - privacy_score:     0-100 score (higher = safer)
          - risk_level:        'LOW' / 'MEDIUM' / 'HIGH'
          - distances:         Array of all DCR values (for plotting)
    """
    # Use only common numeric columns
    common_cols = [c for c in real.columns if c in synthetic.columns]
    real_num = real[common_cols].select_dtypes(include=[np.number])
    synth_num = synthetic[common_cols].select_dtypes(include=[np.number])

    # Ensure precise matching after dtype selection
    final_common = [c for c in real_num.columns if c in synth_num.columns]
    real_num = real_num[final_common]
    synth_num = synth_num[final_common]

    if real_num.empty or synth_num.empty:
        return _empty_result()

    # Fill NaN and scale
    real_arr = real_num.fillna(0).values
    synth_arr = synth_num.fillna(0).values

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_arr)
    synth_scaled = scaler.transform(synth_arr)

    # k-NN with k=1
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(real_scaled)
    distances, _ = nn.kneighbors(synth_scaled)
    distances = distances.flatten()

    # Metrics
    mean_dcr = float(np.mean(distances))
    median_dcr = float(np.median(distances))
    min_dcr = float(np.min(distances))
    max_dcr = float(np.max(distances))
    at_risk = int(np.sum(distances < threshold))
    pct_at_risk = round(at_risk / len(distances) * 100, 2)

    # Privacy score: 100 = perfectly private, 0 = all records leaked
    privacy_score = round(max(0, min(100, 100 - pct_at_risk)), 1)

    # Risk level
    if pct_at_risk <= 5:
        risk_level = "LOW"
    elif pct_at_risk <= 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "mean_dcr": round(mean_dcr, 4),
        "median_dcr": round(median_dcr, 4),
        "min_dcr": round(min_dcr, 4),
        "max_dcr": round(max_dcr, 4),
        "at_risk_count": at_risk,
        "total_synthetic": len(distances),
        "pct_at_risk": pct_at_risk,
        "privacy_score": privacy_score,
        "risk_level": risk_level,
        "distances": distances,
    }


def _empty_result() -> dict:
    """Return a safe default when there are no numeric columns."""
    return {
        "mean_dcr": 0,
        "median_dcr": 0,
        "min_dcr": 0,
        "max_dcr": 0,
        "at_risk_count": 0,
        "total_synthetic": 0,
        "pct_at_risk": 0,
        "privacy_score": 100,
        "risk_level": "LOW",
        "distances": np.array([]),
    }
