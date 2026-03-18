"""
Dataset Loader Module
=====================
Handles loading, previewing, and validating CSV datasets.
"""

import os
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the given file path.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    if not file_path.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")

    return df


def preview_dataset(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return the first *n* rows of the DataFrame for preview purposes.

    Args:
        df: The DataFrame to preview.
        n:  Number of rows to return (default 10).

    Returns:
        A DataFrame containing the first n rows.
    """
    return df.head(n)


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate basic properties of the dataset.

    Checks performed:
        - Dataset is not empty.
        - Has at least 2 columns.
        - Has at least 10 rows.

    Args:
        df: The DataFrame to validate.

    Returns:
        A dict with keys 'valid' (bool) and 'message' (str).
    """
    if df.empty:
        return {"valid": False, "message": "Dataset is empty."}

    if df.shape[1] < 2:
        return {"valid": False, "message": "Dataset must have at least 2 columns."}

    if df.shape[0] < 10:
        return {"valid": False, "message": "Dataset must have at least 10 rows."}

    return {
        "valid": True,
        "message": (
            f"Dataset is valid — {df.shape[0]} rows × {df.shape[1]} columns."
        ),
    }
