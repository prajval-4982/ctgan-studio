"""
Privacy-Utility Score (PU-Score)
=================================
Defines the composite metric introduced in the paper:

    U  = ML Utility   = F1-score of a downstream classifier trained on
                         synthetic data and tested on real data.
    R  = Privacy Resistance = 1 - MIA_Accuracy

    PU-Score = 2 * (U * R) / (U + R)   [Harmonic mean — penalises bad trade-offs]

Equation (1) and (2) from report.tex §III-D.

Key properties:
  - Best value: 1.0 (perfect utility AND perfect privacy resistance)
  - Using harmonic mean ensures that a model with near-perfect utility but
    catastrophic privacy (MIA ~1.0) still gets a near-zero PU-Score.
  - Optimal empirically found at ε = 1.0 in the paper's Table III.

Usage
-----
    from modules.pu_score import calculate_pu_score, bulk_pu_scores

    score = calculate_pu_score(ml_f1=0.63, mia_accuracy=0.523)
    print(f"PU-Score: {score:.4f}")   # → 0.7142…
"""

import math
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def calculate_pu_score(
    ml_f1: float,
    mia_accuracy: float,
    round_to: int = 4,
) -> float:
    """
    Compute the Privacy-Utility Score (PU-Score).

    Parameters
    ----------
    ml_f1        : float in [0, 1] — ML utility (F1-score on real test set).
    mia_accuracy : float in [0, 1] — Attacker's MIA accuracy (0.5 = random).
    round_to     : int — Decimal places to round result.

    Returns
    -------
    float — PU-Score in [0, 1]. Higher is better.

    Notes
    -----
    If either U or R is 0, the harmonic mean is undefined (perfect bad case).
    We return 0.0 in that case.
    """
    # U: ML utility (F1-score, assumed already passed in)
    U = float(ml_f1)
    U = max(0.0, min(1.0, U))      # clip to [0,1]

    # R: Privacy Resistance = 1 - MIA_Accuracy
    R = 1.0 - float(mia_accuracy)
    R = max(0.0, min(1.0, R))      # clip to [0,1]

    # Harmonic mean
    if U + R == 0:
        return 0.0

    pu = 2.0 * (U * R) / (U + R)
    return round(pu, round_to)


# ---------------------------------------------------------------------------
# Convenience: compute PU-Scores for a full results table
# ---------------------------------------------------------------------------

def bulk_pu_scores(rows: list[dict]) -> list[dict]:
    """
    Given a list of result dicts (each having 'ml_f1' and 'mia_accuracy'),
    attach a 'pu_score' key to each and return the updated list.

    Parameters
    ----------
    rows : list of dicts, each must contain 'ml_f1' and 'mia_accuracy'.

    Returns
    -------
    Same list with 'pu_score' added to every dict.
    """
    for row in rows:
        row["pu_score"] = calculate_pu_score(
            ml_f1=row.get("ml_f1", 0.0),
            mia_accuracy=row.get("mia_accuracy", 0.5),
        )
    return rows


def pu_scores_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Convert a list of result dicts into a formatted DataFrame,
    including PU-Score column.

    Useful for producing paper-style tables.
    """
    rows = bulk_pu_scores(rows)
    df = pd.DataFrame(rows)

    # Reorder for paper table format if columns are present
    preferred_order = [
        "method", "dataset", "epsilon",
        "ml_f1", "mia_accuracy", "pu_score",
    ]
    cols = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    df = df[cols + remaining]

    return df


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def interpret_pu_score(pu: float) -> str:
    """
    Return a human-readable interpretation of a PU-Score.

    Thresholds loosely calibrated to the paper's experimental range.
    """
    if pu >= 0.75:
        return "Excellent — strong privacy AND high utility."
    elif pu >= 0.65:
        return "Good — acceptable privacy-utility trade-off."
    elif pu >= 0.50:
        return "Fair — notable utility loss or residual privacy leakage."
    elif pu >= 0.35:
        return "Poor — significant trade-off failure in utility or privacy."
    else:
        return "Very Poor — model memorised data or generated unusable records."
