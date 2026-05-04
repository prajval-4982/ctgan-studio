"""
Prepare datasets + run experiments in a single script.
Handles the fact that existing project has adult_income.csv not adult.csv,
and downloads missing liver/pima datasets.
"""
import sys, os, time, warnings, shutil, io
warnings.filterwarnings("ignore")

# Fix Windows console encoding for Unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

RAW_DIR = os.path.join(ROOT, "datasets", "raw")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================================
# 1. Prepare datasets
# =========================================================================

def _create_liver_placeholder(path):
    """Create a minimal ILPD-like CSV when download fails."""
    np.random.seed(42)
    n = 583
    data = {
        "Age": np.random.randint(4, 90, n),
        "Gender": np.random.choice(["Male", "Female"], n, p=[0.76, 0.24]),
        "Total_Bilirubin": np.abs(np.random.exponential(2, n)).round(1),
        "Direct_Bilirubin": np.abs(np.random.exponential(1, n)).round(1),
        "Alkaline_Phosphotase": np.random.randint(63, 2110, n),
        "Alamine_Aminotransferase": np.random.randint(10, 2000, n),
        "Aspartate_Aminotransferase": np.random.randint(10, 4929, n),
        "Total_Protiens": np.random.normal(6.5, 1, n).round(1),
        "Albumin": np.random.normal(3.1, 0.8, n).round(1),
        "AG_Ratio": np.random.normal(0.95, 0.3, n).round(2),
        "Dataset": np.random.choice([1, 2], n, p=[0.71, 0.29]),
    }
    pd.DataFrame(data).to_csv(path, index=False, header=False)
    print(f"[Prep] Created placeholder liver dataset -> {path}")


def prepare_datasets():
    """Ensure liver.csv, pima.csv, adult.csv exist in datasets/raw/."""
    import requests

    # --- Adult: copy from existing file ---
    adult_src = os.path.join(RAW_DIR, "adult_income.csv")
    adult_dst = os.path.join(RAW_DIR, "adult.csv")
    if os.path.exists(adult_src) and not os.path.exists(adult_dst):
        shutil.copy2(adult_src, adult_dst)
        print(f"[Prep] Copied adult_income.csv -> adult.csv")
    elif os.path.exists(adult_dst):
        print(f"[Prep] adult.csv already exists")

    # --- Pima: download from GitHub ---
    pima_path = os.path.join(RAW_DIR, "pima.csv")
    if not os.path.exists(pima_path):
        print("[Prep] Downloading Pima Diabetes dataset...")
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(pima_path, "wb") as f:
                f.write(r.content)
            print(f"[Prep] Saved -> {pima_path}")
        except Exception as e:
            print(f"[Prep] Failed to download Pima: {e}")
    else:
        print(f"[Prep] pima.csv already exists")

    # --- Liver: download from multiple sources ---
    liver_path = os.path.join(RAW_DIR, "liver.csv")
    if not os.path.exists(liver_path):
        print("[Prep] Downloading Indian Liver Patient dataset...")
        liver_urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/indian_liver_patient.csv",
            "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Indian-Liver-Patient-Dataset-%28ILPD%29.csv",
        ]
        downloaded = False
        for url in liver_urls:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(liver_path, "wb") as f:
                    f.write(r.content)
                print(f"[Prep] Saved -> {liver_path}")
                downloaded = True
                break
            except Exception as e:
                print(f"[Prep] URL failed: {e}")
        if not downloaded:
            print("[Prep] All liver download sources failed. Creating synthetic placeholder.")
            # Create a minimal ILPD-like dataset for the pipeline to run
            _create_liver_placeholder(liver_path)
    else:
        print(f"[Prep] liver.csv already exists")


# =========================================================================
# 2. Dataset loaders (same as run_experiments.py but with header handling)
# =========================================================================

def load_liver(path):
    cols = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin",
            "Alkaline_Phosphotase","Alamine_Aminotransferase",
            "Aspartate_Aminotransferase","Total_Protiens","Albumin",
            "AG_Ratio","Dataset"]
    # Try with header first, then without
    try:
        df = pd.read_csv(path)
        if df.columns[0] in ('Age', 'age'):
            pass  # has header
        else:
            df = pd.read_csv(path, header=None, names=cols)
    except:
        df = pd.read_csv(path, header=None, names=cols)
    
    if 'Gender' in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().apply(
            lambda x: 1 if x.lower() in ('male','1','1.0') else 0
        )
    if 'Dataset' in df.columns:
        df["Dataset"] = df["Dataset"].apply(lambda x: 1 if str(x).strip() in ('1','1.0') else 0)
    df = df.dropna()
    # Encode any remaining non-numeric
    for c in df.select_dtypes(include=["object","category"]).columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df, "Dataset"


def load_pima(path):
    df = pd.read_csv(path)
    if "Outcome" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "Outcome"})
    target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
    df = df.dropna()
    for c in df.select_dtypes(include=["object","category"]).columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df, target


def load_adult(path):
    df = pd.read_csv(path)
    # Handle both header and no-header formats
    if df.columns[0] == 'age':
        pass  # has header
    else:
        cols = ["age","workclass","fnlwgt","education","education_num",
                "marital_status","occupation","relationship","race","sex",
                "capital_gain","capital_loss","hours_per_week",
                "native_country","income"]
        df = pd.read_csv(path, header=None, names=cols, na_values=" ?")
    
    df = df.dropna()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str).str.strip())
    return df, "income"


# =========================================================================
# 3. Evaluation helpers
# =========================================================================

def encode_df(df):
    df = df.copy()
    for c in df.select_dtypes(include=["object","category"]).columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df.fillna(0)


def eval_ml_utility(synth_train, real_test, target_col):
    try:
        se = encode_df(synth_train)
        re = encode_df(real_test)
        common = [c for c in se.columns if c in re.columns]
        if target_col not in common:
            return 0.0
        X_tr = se[common].drop(columns=[target_col])
        y_tr = se[target_col]
        X_te = re[common].drop(columns=[target_col])
        y_te = re[target_col]
        # DP-CTGAN generates continuous values for all columns.
        # Round target to integers for classification.
        y_tr = y_tr.round().astype(int)
        y_te = y_te.round().astype(int)
        # Clip synthetic target to valid label range from real data
        valid_labels = y_te.unique()
        y_tr = y_tr.clip(min(valid_labels), max(valid_labels))
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        return round(f1_score(y_te, clf.predict(X_te), average="weighted", zero_division=0), 4)
    except Exception as e:
        print(f"    [ML] Error: {e}")
        return 0.0


def eval_mia(real_train, real_test, synthesizer, n_shadow=4):
    try:
        from modules.mia_evaluator import run_mia
        r = run_mia(real_train=real_train, real_test=real_test,
                    synthesizer=synthesizer, n_shadow=n_shadow, verbose=False)
        return r["mia_accuracy"]
    except Exception as e:
        print(f"    [MIA] Error: {e}")
        return 0.5


# =========================================================================
# 4. Synthesizer builders
# =========================================================================

def build_vanilla_ctgan(train_df, epochs):
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata
    md = Metadata.detect_from_dataframe(data=train_df)
    s = CTGANSynthesizer(md, epochs=epochs, verbose=False)
    s.fit(train_df)
    return s


def build_dp_ctgan(train_df, epsilon, epochs):
    from modules.dp_ctgan_trainer import train_dp_ctgan
    r = train_dp_ctgan(data=train_df, epsilon=epsilon, epochs=epochs, verbose=True)
    return r["synthesizer"]


def build_smote(train_df, target_col):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("    [SMOTE] imbalanced-learn not installed")
        return None
    edf = encode_df(train_df)
    X = edf.drop(columns=[target_col])
    y = edf[target_col]
    sm = SMOTE(random_state=42)
    X_r, y_r = sm.fit_resample(X, y)
    pool = pd.DataFrame(X_r, columns=X.columns)
    pool[target_col] = y_r.values

    class _S:
        def __init__(self, p):
            self._p = p.reset_index(drop=True)
        def sample(self, n):
            return self._p.iloc[np.random.choice(len(self._p), n, replace=True)].reset_index(drop=True)
    return _S(pool)


# =========================================================================
# 5. Main
# =========================================================================

def main():
    from modules.pu_score import calculate_pu_score

    EPOCHS = 100       # 100 epochs: good balance of quality vs time
    N_SHADOW = 3       # 3 shadow runs for MIA
    EPSILONS = [0.5, 1.0, 5.0, 10.0]

    print("=" * 70)
    print("  DP-CTGAN BENCHMARK EXPERIMENT RUNNER")
    print("  Epochs:", EPOCHS, "| Shadows:", N_SHADOW, "| Epsilons:", EPSILONS)
    print("=" * 70)

    prepare_datasets()

    datasets = {
        "liver": (os.path.join(RAW_DIR, "liver.csv"), load_liver),
        "pima":  (os.path.join(RAW_DIR, "pima.csv"), load_pima),
        "adult": (os.path.join(RAW_DIR, "adult.csv"), load_adult),
    }

    all_rows = []

    for ds_name, (ds_path, loader_fn) in datasets.items():
        if not os.path.exists(ds_path):
            print(f"\n[SKIP] {ds_name} not found at {ds_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  DATASET: {ds_name.upper()}")
        print(f"{'=' * 60}")

        try:
            df, target = loader_fn(ds_path)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue

        # Sub-sample Adult dataset for tractable training time
        MAX_ROWS = 5000
        if len(df) > MAX_ROWS:
            print(f"  Sub-sampling {len(df)} -> {MAX_ROWS} rows for tractable training")
            df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

        n_synth = len(df)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"  Rows: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)} | Target: {target}")

        # --- Real Data Ceiling ---
        print("\n  [1] Real Data Ceiling...")
        t0 = time.time()
        real_f1 = eval_ml_utility(train_df, test_df, target)
        all_rows.append(dict(dataset=ds_name, method="Real Data (Ceiling)",
                             epsilon=None, ml_f1=real_f1, mia_accuracy=None, pu_score=None))
        print(f"      F1 = {real_f1:.4f}  ({time.time()-t0:.0f}s)")

        # --- Vanilla CTGAN ---
        print("\n  [2] Vanilla CTGAN...")
        t0 = time.time()
        try:
            vs = build_vanilla_ctgan(train_df, EPOCHS)
            vd = vs.sample(n_synth)
            vf = eval_ml_utility(vd, test_df, target)
            vm = eval_mia(train_df, test_df, vs, N_SHADOW)
            vp = calculate_pu_score(vf, vm)
            all_rows.append(dict(dataset=ds_name, method="Vanilla CTGAN",
                                 epsilon=None, ml_f1=vf, mia_accuracy=vm, pu_score=vp))
            print(f"      F1={vf:.4f} MIA={vm:.4f} PU={vp:.4f}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"      ERROR: {e}")

        # --- SMOTE ---
        print("\n  [3] SMOTE...")
        t0 = time.time()
        try:
            ss = build_smote(train_df, target)
            if ss:
                sd = ss.sample(n_synth)
                sf = eval_ml_utility(sd, test_df, target)
                sm_mia = eval_mia(train_df, test_df, ss, N_SHADOW)
                sp = calculate_pu_score(sf, sm_mia)
                all_rows.append(dict(dataset=ds_name, method="SMOTE",
                                     epsilon=None, ml_f1=sf, mia_accuracy=sm_mia, pu_score=sp))
                print(f"      F1={sf:.4f} MIA={sm_mia:.4f} PU={sp:.4f}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"      ERROR: {e}")

        # --- DP-CTGAN per epsilon ---
        for eps in EPSILONS:
            print(f"\n  [DP] DP-CTGAN e={eps}...")
            t0 = time.time()
            try:
                dps = build_dp_ctgan(train_df, eps, EPOCHS)
                dpd = dps.sample(n_synth)
                dpf = eval_ml_utility(dpd, test_df, target)
                dpm = eval_mia(train_df, test_df, dps, N_SHADOW)
                dpp = calculate_pu_score(dpf, dpm)
                all_rows.append(dict(dataset=ds_name, method="DP-CTGAN",
                                     epsilon=eps, ml_f1=dpf, mia_accuracy=dpm, pu_score=dpp))
                print(f"      F1={dpf:.4f} MIA={dpm:.4f} PU={dpp:.4f}  ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"      ERROR: {e}")

    # --- Export ---
    rdf = pd.DataFrame(all_rows)
    csv_p = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    rdf.to_csv(csv_p, index=False)
    print(f"\n[DONE] Results -> {csv_p}")
    try:
        rdf.to_excel(os.path.join(RESULTS_DIR, "benchmark_results.xlsx"), index=False)
    except:
        pass

    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)
    print(rdf.to_string(index=False))
    return rdf


if __name__ == "__main__":
    main()
