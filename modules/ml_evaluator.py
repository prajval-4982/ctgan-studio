"""
Machine Learning Evaluation Module
====================================
Trains Logistic Regression, Decision Tree, and Random Forest
classifiers on both real and synthetic data, then compares
performance metrics, ROC-AUC, confusion matrices, and
feature importance alignment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import label_binarize


def _safe_roc_auc(y_true, y_pred_proba, classes):
    """
    Compute ROC-AUC safely for binary and multiclass problems.
    Returns None if the calculation is not possible.
    """
    try:
        if len(classes) == 2:
            # Binary classification
            return round(roc_auc_score(y_true, y_pred_proba[:, 1]), 4)
        else:
            # Multiclass — one-vs-rest
            y_bin = label_binarize(y_true, classes=classes)
            if y_bin.shape[1] == 1:
                return None  # Can't compute for single-class
            return round(roc_auc_score(y_bin, y_pred_proba, average="weighted", multi_class="ovr"), 4)
    except Exception:
        return None


def _train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name: str, classes) -> dict:
    """
    Train a classifier and return evaluation metrics including
    ROC-AUC and Confusion Matrix.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ROC-AUC: needs probability estimates
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = _safe_roc_auc(y_test, y_proba, classes)
        except Exception:
            roc_auc = None

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm.tolist(),
    }


def evaluate_ml(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str | None = None,
) -> dict:
    """
    Compare ML model performance when trained on real vs. synthetic
    data and tested on a held-out portion of the real data.

    If *target_col* is not provided, the last column is used as the
    target variable.

    The function trains:
        - Logistic Regression
        - Decision Tree
        - Random Forest

    Each model is trained twice:
        1. On real data → tested on real test set.
        2. On synthetic data → tested on real test set.

    Also extracts feature importance from Random Forest for alignment
    comparison.

    Returns:
        A dictionary with keys:
          - 'real_results':      list of metric dicts (inc. ROC-AUC, CM)
          - 'synthetic_results': list of metric dicts (inc. ROC-AUC, CM)
          - 'feature_importance': dict with 'real' and 'synthetic' importances
          - 'class_labels':      list of unique target classes
    """
    # Determine target column
    if target_col is None:
        target_col = real.columns[-1]

    if target_col not in real.columns or target_col not in synthetic.columns:
        raise ValueError(f"Target column '{target_col}' not found in both datasets.")

    # ---------- Real data split ----------
    X_real = real.drop(columns=[target_col])
    y_real = real[target_col]

    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42
    )

    # ---------- Synthetic features ----------
    X_synth = synthetic.drop(columns=[target_col])
    y_synth = synthetic[target_col]

    # Determine all classes from the real data
    classes = sorted(y_real.unique().tolist())

    # ---------- Models ----------
    models = [
        (LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
    ]

    real_results = []
    synth_results = []

    rf_importance_real = None
    rf_importance_synth = None

    for clf, name in models:
        # Train on real, test on real
        real_results.append(
            _train_and_evaluate(X_train_real, X_test_real, y_train_real, y_test_real, clf, name, classes)
        )

        # Capture feature importance from Random Forest (real)
        if name == "Random Forest":
            rf_importance_real = dict(zip(X_real.columns, clf.feature_importances_))

        # Train on synthetic, test on real (full real test set)
        synth_clf = clf.__class__(**clf.get_params())  # fresh clone
        synth_results.append(
            _train_and_evaluate(X_synth, X_test_real, y_synth, y_test_real, synth_clf, name, classes)
        )

        # Capture feature importance from Random Forest (synthetic)
        if name == "Random Forest":
            rf_importance_synth = dict(zip(X_synth.columns, synth_clf.feature_importances_))

    # Build feature importance comparison
    feature_importance = None
    if rf_importance_real and rf_importance_synth:
        feature_importance = {
            "features": list(rf_importance_real.keys()),
            "real": [round(v, 4) for v in rf_importance_real.values()],
            "synthetic": [round(v, 4) for v in rf_importance_synth.values()],
        }

    return {
        "real_results": real_results,
        "synthetic_results": synth_results,
        "feature_importance": feature_importance,
        "class_labels": [str(c) for c in classes],
    }
