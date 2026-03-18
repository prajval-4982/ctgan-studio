"""
Preprocessing Module
====================
Cleans and prepares raw datasets for CTGAN training.

Pipeline:
    1. Handle missing values (median for numeric, mode for categorical).
    2. Encode categorical columns via LabelEncoder.
    3. Return cleaned DataFrame + metadata dict.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.

    - Numeric columns → filled with the column median.
    - Categorical (object / category) columns → filled with the column mode.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with no missing values.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if df[col].dtype in ("object", "category"):
            # Use the most frequent value (mode)
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)
        else:
            # Use the median for numeric columns
            df[col].fillna(df[col].median(), inplace=True)

    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all object / category columns.

    Args:
        df: A DataFrame (should already be cleaned of NaN values).

    Returns:
        A tuple of:
        - Encoded DataFrame
        - Dictionary mapping column name → LabelEncoder instance
    """
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline.

    Steps:
        1. Handle missing values.
        2. Encode categorical columns.

    Args:
        df: Raw DataFrame.

    Returns:
        A tuple of:
        - Cleaned & encoded DataFrame
        - Metadata dict with keys:
            'encoders'         → column → LabelEncoder mapping
            'original_columns' → list of original column names
            'categorical_cols' → list of originally categorical column names
            'numeric_cols'     → list of numeric column names
    """
    # --- Step 1: missing values ---
    df_clean = handle_missing_values(df)

    # Identify column types before encoding
    categorical_cols = list(df_clean.select_dtypes(include=["object", "category"]).columns)
    numeric_cols = list(df_clean.select_dtypes(include=["number"]).columns)

    # --- Step 2: encode categoricals ---
    df_encoded, encoders = encode_categorical(df_clean)

    metadata = {
        "encoders": encoders,
        "original_columns": list(df.columns),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }

    return df_encoded, metadata
