"""
Synthesizer Training Module
============================
Trains CTGAN or TVAE synthesizers using the SDV library
and provides helpers to save / load the trained model.
"""

import os
import warnings
import pandas as pd
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata

# Silence specific library warnings for a cleaner console
warnings.filterwarnings("ignore", category=FutureWarning, module="ctgan")
warnings.filterwarnings("ignore", category=UserWarning, module="sdv")

# Default path for persisted models
DEFAULT_MODEL_PATH = os.path.join("models", "ctgan_model.pkl")

# Supported model types
SUPPORTED_MODELS = {"CTGAN", "TVAE"}


def train_model(
    data: pd.DataFrame,
    model_type: str = "CTGAN",
    epochs: int = 300,
    verbose: bool = True,
):
    """
    Train a synthesizer on the supplied data.

    Args:
        data:       Cleaned / encoded DataFrame.
        model_type: 'CTGAN' or 'TVAE'.
        epochs:     Number of training epochs (default 300).
        verbose:    If True, print training progress.

    Returns:
        A fitted synthesizer instance (CTGANSynthesizer or TVAESynthesizer).
    """
    model_type = model_type.upper().strip()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from: {SUPPORTED_MODELS}")

    # Auto-detect metadata from the DataFrame
    metadata = Metadata.detect_from_dataframe(data=data)

    # Save metadata to JSON for replicability as recommended by SDV
    metadata_path = os.path.join("models", "metadata.json")
    os.makedirs("models", exist_ok=True)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    metadata.save_to_json(metadata_path)
    print(f"[{model_type}] Metadata saved → {metadata_path}")

    # Build the synthesizer
    if model_type == "TVAE":
        synthesizer = TVAESynthesizer(
            metadata,
            epochs=epochs,
            verbose=verbose,
        )
    else:
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=epochs,
            verbose=verbose,
        )

    print(f"[{model_type}] Training started …")
    synthesizer.fit(data)
    print(f"[{model_type}] Training complete.")

    return synthesizer


# Backward-compatible alias
def train_ctgan(data, epochs=300, verbose=True):
    """Legacy wrapper — calls train_model with model_type='CTGAN'."""
    return train_model(data, model_type="CTGAN", epochs=epochs, verbose=verbose)


def save_model(model, path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Persist a trained model to disk.

    Args:
        model: Trained synthesizer (CTGAN or TVAE).
        path:  File path to save to.

    Returns:
        The absolute path where the model was saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[Model] Saved → {path}")
    return os.path.abspath(path)


def load_model(path: str = DEFAULT_MODEL_PATH):
    """
    Load a previously saved model.

    Args:
        path: Path to the saved model file.

    Returns:
        A synthesizer instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at: {path}")

    # Try CTGAN first, fallback to TVAE
    try:
        model = CTGANSynthesizer.load(path)
    except Exception:
        model = TVAESynthesizer.load(path)

    print(f"[Model] Loaded ← {path}")
    return model
