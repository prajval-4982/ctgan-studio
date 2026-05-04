"""
Automated Benchmark Experiment Runner
=======================================
Reproduces Tables I, II, III from the research paper:

    "Privacy-Preserving Synthetic Tabular Data Generation
     Using DP-CTGAN: A Unified Privacy-Utility Framework
     with Membership Inference Attack Evaluation"

Experiment matrix:
    Datasets  : ILPD (Liver), Pima Diabetes, Adult Income
    Methods   : Real (ceiling), Vanilla CTGAN, SMOTE, DP-CTGAN × 4 epsilons
    Epsilons  : 0.5, 1.0, 5.0, 10.0

For each (dataset, method, epsilon) combination the runner:
    1. Preprocesses data.
    2. Trains the synthesizer (or runs SMOTE).
    3. Generates synthetic data.
    4. Computes ML Utility F1-score (Random Forest on real test set).
    5. Runs shadow-model MIA.
    6. Computes PU-Score.

Outputs:
    - results/benchmark_results.csv  — full results table
    - results/benchmark_results.xlsx — Excel version
    - results/benchmark_summary.txt  — human-readable summary

Usage
-----
    python src/run_experiments.py [--epochs 100] [--n-synthetic 500]

    # Quick sanity check (25 epochs, 200 rows):
    python src/run_experiments.py --epochs 25 --n-synthetic 200 --quick

Note
----
Full run (300 epochs × 3 datasets × 5 configurations) takes ~20-90 min
depending on CPU. Use --quick for a fast test.
"""

import sys
import os
import argparse
import time
import warnings

# Ensure project root is on the path even when called directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ==========================================================================
# Dataset loaders
# ==========================================================================

def _load_liver(path: str) -> tuple[pd.DataFrame, str]:
    """Indian Liver Patient Dataset — 583 rows, 11 columns."""
    cols = [
        "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Protiens", "Albumin",
        "AG_Ratio", "Dataset",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    # Gender: Male=1, Female=0
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["Dataset"] = (df["Dataset"] == 1).astype(int)  # liver disease = 1
    df = df.dropna()
    return df, "Dataset"


def _load_pima(path: str) -> tuple[pd.DataFrame, str]:
    """Pima Indians Diabetes Dataset — 768 rows."""
    df = pd.read_csv(path)
    # Standard column names
    if "Outcome" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "Outcome"})
    target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
    df = df.dropna()
    return df, target


def _load_adult(path: str) -> tuple[pd.DataFrame, str]:
    """UCI Adult Income Dataset — ~48k rows."""
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income",
    ]
    df = pd.read_csv(path, header=None, names=cols, na_values=" ?")
    df = df.dropna()
    # Encode all categoricals
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.strip())
    return df, "income"


DATASET_LOADERS = {
    "liver": _load_liver,
    "pima":  _load_pima,
    "adult": _load_adult,
}

DATASET_PATHS = {
    "liver": os.path.join(ROOT, "datasets", "raw", "liver.csv"),
    "pima":  os.path.join(ROOT, "datasets", "raw", "pima.csv"),
    "adult": os.path.join(ROOT, "datasets", "raw", "adult.csv"),
}


# ==========================================================================
# Evaluation helpers
# ==========================================================================

def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode remaining non-numeric columns."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df.fillna(0)


def evaluate_ml_utility(
    synth_train: pd.DataFrame,
    real_test: pd.DataFrame,
    target_col: str,
) -> float:
    """
    Train a Random Forest on synthetic data, test on real data.
    Returns weighted F1-score (same metric as Table I in paper).
    """
    try:
        synth_enc = _encode_df(synth_train)
        real_enc  = _encode_df(real_test)

        # Align columns
        common_cols = [c for c in synth_enc.columns if c in real_enc.columns]
        if target_col not in common_cols:
            return 0.0

        X_train = synth_enc[common_cols].drop(columns=[target_col])
        y_train = synth_enc[target_col]
        X_test  = real_enc[common_cols].drop(columns=[target_col])
        y_test  = real_enc[target_col]

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)

    except Exception as e:
        print(f"    [ML Utility] Error: {e}")
        return 0.0


def run_mia_quick(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthesizer,
    n_shadow: int = 4,
) -> float:
    """Run MIA and return accuracy. Wrapper around modules.mia_evaluator."""
    try:
        from modules.mia_evaluator import run_mia
        result = run_mia(
            real_train=real_train,
            real_test=real_test,
            synthesizer=synthesizer,
            n_shadow=n_shadow,
            verbose=False,
        )
        return result["mia_accuracy"]
    except Exception as e:
        print(f"    [MIA] Error: {e}. Returning default 0.5.")
        return 0.5


# ==========================================================================
# Synthesizer builders
# ==========================================================================

def _build_vanilla_ctgan(train_df: pd.DataFrame, epochs: int):
    """Train standard CTGAN via SDV."""
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata
    metadata = Metadata.detect_from_dataframe(data=train_df)
    synth = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    synth.fit(train_df)
    return synth


def _build_dp_ctgan(train_df: pd.DataFrame, epsilon: float, epochs: int):
    """Train DP-CTGAN with given epsilon."""
    from modules.dp_ctgan_trainer import train_dp_ctgan
    result = train_dp_ctgan(
        data=train_df,
        epsilon=epsilon,
        epochs=epochs,
        verbose=False,
    )
    return result["synthesizer"]


def _build_smote(train_df: pd.DataFrame, target_col: str, n_samples: int):
    """
    Run SMOTE to oversample the minority class.
    Returns a wrapper with .sample(n) method to match GAN interface.
    """
    try:
        from imblearn.over_sampling import SMOTE
        enc_df = _encode_df(train_df)
        X = enc_df.drop(columns=[target_col])
        y = enc_df[target_col]
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        df_res = pd.DataFrame(X_res, columns=X.columns)
        df_res[target_col] = y_res.values

        class _SMOTESynthesizer:
            def __init__(self, pool: pd.DataFrame):
                self._pool = pool.reset_index(drop=True)

            def sample(self, n: int) -> pd.DataFrame:
                idx = np.random.choice(len(self._pool), n, replace=True)
                return self._pool.iloc[idx].reset_index(drop=True)

        return _SMOTESynthesizer(df_res)

    except ImportError:
        print("    [SMOTE] imbalanced-learn not installed. Skipping.")
        return None


# ==========================================================================
# Main experiment loop
# ==========================================================================

def run_experiments(
    epochs: int = 300,
    n_synthetic: int | None = None,
    n_shadow: int = 4,
    epsilons: list[float] | None = None,
    quick: bool = False,
) -> pd.DataFrame:
    """
    Run the full experiment matrix and return a results DataFrame.
    """
    from modules.pu_score import calculate_pu_score

    if quick:
        epochs = 25
        n_synthetic = 200
        n_shadow = 2
        print("[Runner] QUICK mode: epochs=25, n_synthetic=200, n_shadow=2")

    if epsilons is None:
        epsilons = [0.5, 1.0, 5.0, 10.0]

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_rows = []

    # ------------------------------------------------------------------
    # Check datasets — download if needed
    # ------------------------------------------------------------------
    dataset_available = {}
    for name, path in DATASET_PATHS.items():
        if os.path.exists(path):
            dataset_available[name] = True
        else:
            print(f"[Runner] Dataset '{name}' not found at {path}.")
            print(f"         Attempting download …")
            try:
                from modules.dp_ctgan_trainer import download_benchmark_datasets
                download_benchmark_datasets()
                dataset_available[name] = os.path.exists(path)
            except Exception as e:
                print(f"         Download failed: {e}. Skipping '{name}'.")
                dataset_available[name] = False

    # ------------------------------------------------------------------
    # Iterate over datasets
    # ------------------------------------------------------------------
    for dataset_name, loader_fn in DATASET_LOADERS.items():
        if not dataset_available.get(dataset_name, False):
            print(f"\n[Runner] Skipping '{dataset_name}' (not available).")
            continue

        print(f"\n{'='*60}")
        print(f" DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")

        # Load
        try:
            df_full, target_col = loader_fn(DATASET_PATHS[dataset_name])
        except Exception as e:
            print(f"[Runner] Failed to load '{dataset_name}': {e}")
            continue

        # Encode for uniform processing
        df_enc = _encode_df(df_full)
        n_synth = n_synthetic or len(df_enc)

        # 80/20 split (paper §III-E)
        train_df, test_df = train_test_split(
            df_enc, test_size=0.2, random_state=42
        )
        print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows | "
              f"Target: '{target_col}'")

        # ------------------------------------------------------------------
        # 1. Real Data Ceiling
        # ------------------------------------------------------------------
        print("\n  [1/5] Real Data (ceiling) …")
        real_f1 = evaluate_ml_utility(train_df, test_df, target_col)
        all_rows.append({
            "dataset": dataset_name,
            "method": "Real Data (Ceiling)",
            "epsilon": None,
            "ml_f1": real_f1,
            "mia_accuracy": None,
            "pu_score": None,
        })
        print(f"         F1 = {real_f1:.4f}")

        # ------------------------------------------------------------------
        # 2. Vanilla CTGAN
        # ------------------------------------------------------------------
        print("\n  [2/5] Vanilla CTGAN …")
        try:
            t0 = time.time()
            vanilla_synth = _build_vanilla_ctgan(train_df, epochs)
            synth_data = vanilla_synth.sample(n_synth)
            vanilla_f1 = evaluate_ml_utility(synth_data, test_df, target_col)
            vanilla_mia = run_mia_quick(train_df, test_df, vanilla_synth, n_shadow)
            vanilla_pu = calculate_pu_score(vanilla_f1, vanilla_mia)
            all_rows.append({
                "dataset": dataset_name,
                "method": "Vanilla CTGAN",
                "epsilon": None,
                "ml_f1": vanilla_f1,
                "mia_accuracy": vanilla_mia,
                "pu_score": vanilla_pu,
            })
            print(f"         F1={vanilla_f1:.4f} | MIA={vanilla_mia:.4f} | "
                  f"PU={vanilla_pu:.4f}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"         ERROR: {e}")

        # ------------------------------------------------------------------
        # 3. SMOTE Baseline
        # ------------------------------------------------------------------
        print("\n  [3/5] SMOTE …")
        try:
            smote_synth = _build_smote(train_df, target_col, n_synth)
            if smote_synth is not None:
                smote_data = smote_synth.sample(n_synth)
                smote_f1   = evaluate_ml_utility(smote_data, test_df, target_col)
                smote_mia  = run_mia_quick(train_df, test_df, smote_synth, n_shadow)
                smote_pu   = calculate_pu_score(smote_f1, smote_mia)
                all_rows.append({
                    "dataset": dataset_name,
                    "method": "SMOTE",
                    "epsilon": None,
                    "ml_f1": smote_f1,
                    "mia_accuracy": smote_mia,
                    "pu_score": smote_pu,
                })
                print(f"         F1={smote_f1:.4f} | MIA={smote_mia:.4f} | "
                      f"PU={smote_pu:.4f}")
        except Exception as e:
            print(f"         ERROR: {e}")

        # ------------------------------------------------------------------
        # 4. DP-CTGAN × 4 epsilon budgets
        # ------------------------------------------------------------------
        for i, eps in enumerate(epsilons, start=1):
            print(f"\n  [4+{i-1}/5] DP-CTGAN ε={eps} …")
            t0 = time.time()
            try:
                dp_synth = _build_dp_ctgan(train_df, eps, epochs)
                dp_data  = dp_synth.sample(n_synth)
                dp_f1    = evaluate_ml_utility(dp_data, test_df, target_col)
                dp_mia   = run_mia_quick(train_df, test_df, dp_synth, n_shadow)
                dp_pu    = calculate_pu_score(dp_f1, dp_mia)
                all_rows.append({
                    "dataset": dataset_name,
                    "method": f"DP-CTGAN",
                    "epsilon": eps,
                    "ml_f1": dp_f1,
                    "mia_accuracy": dp_mia,
                    "pu_score": dp_pu,
                })
                print(f"         F1={dp_f1:.4f} | MIA={dp_mia:.4f} | "
                      f"PU={dp_pu:.4f}  ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"         ERROR: {e}")

    # ------------------------------------------------------------------
    # Export results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_rows)

    if results_df.empty:
        print("\n[Runner] No results generated. Check that datasets are available.")
        return results_df

    csv_path   = os.path.join(results_dir, "benchmark_results.csv")
    xlsx_path  = os.path.join(results_dir, "benchmark_results.xlsx")
    txt_path   = os.path.join(results_dir, "benchmark_summary.txt")

    results_df.to_csv(csv_path, index=False)
    print(f"\n[Runner] Results saved → {csv_path}")

    try:
        results_df.to_excel(xlsx_path, index=False)
        print(f"[Runner] Results saved → {xlsx_path}")
    except Exception:
        pass

    # Human-readable summary
    with open(txt_path, "w") as f:
        f.write("BENCHMARK RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        for ds in results_df["dataset"].unique():
            f.write(f"\nDataset: {ds.upper()}\n")
            f.write("-" * 40 + "\n")
            sub = results_df[results_df["dataset"] == ds]
            for _, row in sub.iterrows():
                eps_str = f"ε={row['epsilon']}" if row['epsilon'] else "  —  "
                f1_str  = f"{row['ml_f1']:.4f}" if row['ml_f1'] is not None else "  —  "
                mia_str = f"{row['mia_accuracy']:.4f}" if row['mia_accuracy'] is not None else "  —  "
                pu_str  = f"{row['pu_score']:.4f}" if row['pu_score'] is not None else "  —  "
                f.write(f"  {row['method']:30s} {eps_str:8s} "
                        f"F1={f1_str}  MIA={mia_str}  PU={pu_str}\n")
    print(f"[Runner] Summary saved → {txt_path}")

    return results_df


# ==========================================================================
# CLI entry point
# ==========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DP-CTGAN benchmark experiments (reproduces paper tables)."
    )
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs per model (default: 300)")
    parser.add_argument("--n-synthetic", type=int, default=None,
                        help="Synthetic rows to generate (default: same as training set)")
    parser.add_argument("--n-shadow", type=int, default=4,
                        help="Shadow experiments for MIA (default: 4)")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.5, 1.0, 5.0, 10.0],
                        help="Privacy budgets to test (default: 0.5 1.0 5.0 10.0)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check (25 epochs, 200 rows, 2 shadows)")

    args = parser.parse_args()

    df_results = run_experiments(
        epochs=args.epochs,
        n_synthetic=args.n_synthetic,
        n_shadow=args.n_shadow,
        epsilons=args.epsilons,
        quick=args.quick,
    )

    print("\n\nFINAL RESULTS TABLE")
    print("=" * 80)
    print(df_results.to_string(index=False))
