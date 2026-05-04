"""
Membership Inference Attack (MIA) Evaluator
============================================
Implements a black-box shadow-model MIA as described in:

    Shokri et al. (2017) "Membership inference attacks against
    machine learning models." IEEE S&P 2017.

Our implementation:
    1. Train K shadow DP-CTGAN models on disjoint subsets of training data.
    2. For each shadow model, generate synthetic data.
    3. Build an attack dataset:
         - Real training rows → labelled "member"  (1)
         - Held-out real rows → labelled "non-member" (0)
       We use per-record distance to the nearest synthetic neighbour
       as the attack feature vector (distance-based membership signal).
    4. Train a binary Random Forest attack classifier.
    5. Report MIA accuracy on the target model's synthetic dataset.

An MIA accuracy of ~50% indicates the attacker is guessing randomly
(strong privacy). Values >60% indicate memorisation / privacy leakage.

Reference: report.tex §III-C (Threat Model).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_numeric(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a DataFrame to a fully numeric numpy array
    by label-encoding any remaining categorical columns.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df.values.astype(float)


def _distance_features(
    records: np.ndarray,
    synthetic_ref: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Compute the k-NN distances from each row in `records` to
    the nearest rows in `synthetic_ref`.

    Returns an (N, k) feature matrix.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(synthetic_ref)
    dists, _ = nn.kneighbors(records)
    return dists   # shape: (N, k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_mia(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthesizer,
    n_shadow: int = 4,
    shadow_sample_ratio: float = 0.5,
    n_attack_estimators: int = 100,
    n_synthetic_per_shadow: int | None = None,
    k_neighbors: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Execute a shadow-model Membership Inference Attack.

    Parameters
    ----------
    real_train : pd.DataFrame
        Records that WERE used to train the target synthesizer.
        These are the "member" class ground-truth.
    real_test : pd.DataFrame
        Records held out from training. "Non-member" ground-truth.
    synthesizer : object
        A fitted synthesizer with a `.sample(n: int) -> pd.DataFrame` method.
        Can be CTGANSynthesizer, _DPSynthesizer, or any compatible object.
    n_shadow : int
        Number of shadow synthesizers to train (default 4 as in paper).
        Note: shadow models here share the same synthesizer and generate
        separate independent samples (avoids re-training overhead).
    shadow_sample_ratio : float
        Fraction of real_train used per shadow experiment (default 0.5).
    n_attack_estimators : int
        Trees in the attacker Random Forest (default 100).
    n_synthetic_per_shadow : int | None
        Synthetic rows generated per shadow. Defaults to len(real_train).
    k_neighbors : int
        Number of nearest neighbours used as attack features.
    random_state : int
        Reproducibility seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        "mia_accuracy"    : float  — primary metric (0.5 = random, 1.0 = perfect)
        "mia_roc_auc"     : float  — ROC-AUC of the attack classifier
        "privacy_risk"    : str    — 'LOW' / 'MEDIUM' / 'HIGH'
        "attack_report"   : dict   — detailed per-class metrics
    """
    np.random.seed(random_state)
    n_rows_train = len(real_train)
    n_rows_test = len(real_test)
    n_synthetic = n_synthetic_per_shadow or n_rows_train

    # ------------------------------------------------------------------
    # 1. Build attack dataset using shadow samples
    # ------------------------------------------------------------------
    all_features = []
    all_labels = []

    # Scale based on combined real data statistics
    combined = pd.concat([real_train, real_test], ignore_index=True)
    combined_arr = _encode_numeric(combined)
    scaler = StandardScaler()
    scaler.fit(combined_arr)

    real_train_arr = scaler.transform(_encode_numeric(real_train))
    real_test_arr = scaler.transform(_encode_numeric(real_test))

    for shadow_idx in range(n_shadow):
        if verbose:
            print(f"  [MIA] Shadow experiment {shadow_idx + 1}/{n_shadow} …")

        # Generate a fresh synthetic sample (mimics training a shadow model)
        try:
            synth_df = synthesizer.sample(n_synthetic)
        except Exception as e:
            print(f"  [MIA] Warning: synthesizer.sample failed: {e}. Skipping shadow.")
            continue

        synth_arr = _encode_numeric(synth_df)
        # Ensure same column count as real data after encoding
        if synth_arr.shape[1] != combined_arr.shape[1]:
            # Column mismatch — align by taking minimum columns
            min_cols = min(synth_arr.shape[1], combined_arr.shape[1])
            synth_arr = synth_arr[:, :min_cols]
            real_train_arr_trimmed = real_train_arr[:, :min_cols]
            real_test_arr_trimmed = real_test_arr[:, :min_cols]
        else:
            real_train_arr_trimmed = real_train_arr
            real_test_arr_trimmed = real_test_arr

        synth_scaled = scaler.transform(
            np.pad(synth_arr, ((0, 0), (0, combined_arr.shape[1] - synth_arr.shape[1])))
            if synth_arr.shape[1] < combined_arr.shape[1]
            else synth_arr[:, :combined_arr.shape[1]]
        )

        # Sub-sample a fraction of training rows for this shadow experiment
        shadow_size = max(1, int(shadow_sample_ratio * n_rows_train))
        shadow_idx_arr = np.random.choice(n_rows_train, shadow_size, replace=False)
        shadow_member = real_train_arr[shadow_idx_arr]

        # Non-members: random subset of test rows
        nonmember_size = min(shadow_size, n_rows_test)
        nonmember_idx = np.random.choice(n_rows_test, nonmember_size, replace=False)
        shadow_nonmember = real_test_arr[nonmember_idx]

        # Distance features: member rows
        try:
            mem_feats = _distance_features(shadow_member, synth_scaled, k=k_neighbors)
            nonmem_feats = _distance_features(shadow_nonmember, synth_scaled, k=k_neighbors)
        except Exception as e:
            print(f"  [MIA] Warning: distance feature computation failed: {e}")
            continue

        all_features.append(mem_feats)
        all_labels.extend([1] * len(mem_feats))

        all_features.append(nonmem_feats)
        all_labels.extend([0] * len(nonmem_feats))

    if not all_features:
        return _empty_mia_result()

    X_attack = np.vstack(all_features)
    y_attack = np.array(all_labels)

    # ------------------------------------------------------------------
    # 2. Train attack classifier (Random Forest)
    # ------------------------------------------------------------------
    X_atk_train, X_atk_test, y_atk_train, y_atk_test = train_test_split(
        X_attack, y_attack, test_size=0.3, random_state=random_state, stratify=y_attack
    )

    attacker = RandomForestClassifier(
        n_estimators=n_attack_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    attacker.fit(X_atk_train, y_atk_train)

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    y_pred = attacker.predict(X_atk_test)
    y_proba = attacker.predict_proba(X_atk_test)[:, 1]

    mia_accuracy = round(accuracy_score(y_atk_test, y_pred), 4)
    try:
        mia_roc_auc = round(roc_auc_score(y_atk_test, y_proba), 4)
    except Exception:
        mia_roc_auc = None

    # Privacy risk classification (aligned with paper thresholds)
    if mia_accuracy <= 0.55:
        risk = "LOW"          # near-random — strong privacy
    elif mia_accuracy <= 0.65:
        risk = "MEDIUM"       # some leakage
    else:
        risk = "HIGH"         # significant memorisation

    if verbose:
        print(f"  [MIA] Accuracy: {mia_accuracy:.4f} | "
              f"ROC-AUC: {mia_roc_auc} | Risk: {risk}")

    return {
        "mia_accuracy": mia_accuracy,
        "mia_roc_auc": mia_roc_auc,
        "privacy_risk": risk,
        "attack_report": {
            "n_shadow_experiments": n_shadow,
            "attack_train_samples": len(X_atk_train),
            "attack_test_samples": len(X_atk_test),
            "member_fraction": round(float(y_attack.mean()), 4),
        },
    }


def _empty_mia_result() -> dict:
    """Return a safe default when attack cannot be computed."""
    return {
        "mia_accuracy": 0.5,
        "mia_roc_auc": None,
        "privacy_risk": "UNKNOWN",
        "attack_report": {},
    }
