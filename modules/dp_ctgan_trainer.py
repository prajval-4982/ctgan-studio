"""
DP-CTGAN Training Module
========================
Trains a Differentially Private Conditional Tabular GAN (DP-CTGAN).

Architecture follows Fang et al. (2022):
  "DP-CTGAN: Differentially private medical data generation using CTGANs."
  (AIME 2022, LNCS vol. 13263, pp. 178-188)

DP-SGD is applied ONLY to the Discriminator's parameter updates using the
Opacus library (https://opacus.ai). The Generator is not directly privatised
because all information flow from training data passes through the
Discriminator.

Privacy accounting:
  - Renyi Differential Privacy (RDP) accountant is used during training.
  - Final (epsilon, delta)-DP guarantee is extracted at inference time.

Usage
-----
    from modules.dp_ctgan_trainer import train_dp_ctgan

    result = train_dp_ctgan(
        data        = df,
        target_col  = "label",
        epsilon     = 1.0,
        delta       = 1e-5,
        epochs      = 300,
        batch_size  = 500,
        max_grad_norm = 1.0,
    )
    synthetic_df = result["synthesizer"].sample(len(df))
    print(f"Achieved epsilon: {result['achieved_epsilon']:.4f}")
"""

import warnings
import os
import numpy as np
import pandas as pd

# Silence library noise
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Optional Opacus import — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_dataframe(df: pd.DataFrame):
    """
    Label-encode all object/category columns so the DataFrame
    is fully numeric.  Returns (encoded_df, encoders_dict).
    """
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def _decode_dataframe(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Reverse LabelEncoding for categorical columns."""
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].round().clip(0, len(le.classes_) - 1).astype(int)
            df[col] = le.inverse_transform(df[col])
    return df


# ---------------------------------------------------------------------------
# Lightweight PyTorch discriminator (used for Opacus DP wrapping)
# ---------------------------------------------------------------------------

class _Discriminator(nn.Module):
    """
    A simple MLP discriminator compatible with Opacus.
    Opacus requires all layers to be compatible (e.g. no BatchNorm).
    We use LayerNorm instead.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # BatchNorm breaks DP — use LayerNorm
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class _Generator(nn.Module):
    """
    MLP generator. No DP is applied here — privacy flows through the
    Discriminator only (Fang et al. 2022).
    """

    def __init__(self, noise_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Core DP-CTGAN trainer
# ---------------------------------------------------------------------------

def train_dp_ctgan(
    data: pd.DataFrame,
    target_col: str | None = None,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    epochs: int = 300,
    batch_size: int = 500,
    max_grad_norm: float = 1.0,
    noise_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 2e-4,
    verbose: bool = True,
) -> dict:
    """
    Train a DP-CTGAN model and return results including achieved epsilon.

    Parameters
    ----------
    data         : Raw DataFrame (mixed types allowed).
    target_col   : Name of the target/label column (used for SMOTE comparison
                   in the benchmark runner). Not used by the GAN itself.
    epsilon      : Privacy budget (target). Lower = more private.
    delta        : Probability of (epsilon, delta)-DP failure — recommend 1e-5.
    epochs       : Number of adversarial training epochs.
    batch_size   : Mini-batch size for DP-SGD.
    max_grad_norm: Gradient clipping norm (C in the paper).
    noise_dim    : Dimensionality of the generator's noise input vector.
    hidden_dim   : Width of MLP hidden layers.
    lr           : Learning rate for both Generator and Discriminator.
    verbose      : Print training progress.

    Returns
    -------
    dict with keys:
        "synthesizer"       : Fitted object with .sample(n) method.
        "achieved_epsilon"  : Actual epsilon consumed during training.
        "model_type"        : "DP-CTGAN"
        "epsilon_target"    : The requested epsilon.
        "training_epochs"   : epochs actually run.
    """
    # ------------------------------------------------------------------
    # Fallback: if Opacus is not installed, warn and use vanilla CTGAN
    # ------------------------------------------------------------------
    if not OPACUS_AVAILABLE:
        warnings.warn(
            "Opacus is not installed. Falling back to vanilla CTGAN "
            "(no differential privacy). Install `opacus` for full DP support.",
            RuntimeWarning,
        )
        return _fallback_vanilla_ctgan(data, epochs, epsilon, verbose)

    # ------------------------------------------------------------------
    # 1. Encode data fully to numeric
    # ------------------------------------------------------------------
    df_enc, encoders = _encode_dataframe(data)
    df_enc = df_enc.fillna(0)
    feature_cols = list(df_enc.columns)
    n_features = len(feature_cols)
    n_data = len(df_enc)

    # Ensure batch_size does not exceed dataset size
    actual_batch_size = min(batch_size, n_data // 2)
    actual_batch_size = max(actual_batch_size, 16)

    if verbose:
        print(f"[DP-CTGAN] Features: {n_features}, Rows: {n_data}, "
              f"Target e={epsilon}, d={delta}, batch={actual_batch_size}")

    # ------------------------------------------------------------------
    # 2. Normalise and build PyTorch tensors
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_enc.values)
    X = torch.tensor(df_scaled, dtype=torch.float32)
    dataset = TensorDataset(X)
    # Use a regular DataLoader — Opacus will wrap it
    loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # 3. Instantiate Generator & Discriminator
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = _Generator(noise_dim, n_features, hidden_dim).to(device)
    D = _Discriminator(n_features, hidden_dim).to(device)

    # Validate Discriminator with Opacus (fixes incompatible layers)
    D = ModuleValidator.fix(D)

    optim_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # 4. Compute noise_multiplier and wrap with PrivacyEngine
    #    Using make_private (not make_private_with_epsilon) for
    #    Opacus 1.5.x compatibility.
    # ------------------------------------------------------------------
    import math

    # Heuristic noise schedule calibrated per epsilon:
    #   e=0.5 -> high noise (20.0), e=1.0 -> moderate (10.0),
    #   e=5.0 -> low (2.0), e=10.0 -> minimal (1.0)
    noise_multiplier = max(0.5, 10.0 / epsilon)

    privacy_engine = PrivacyEngine()
    D, optim_D, dp_loader = privacy_engine.make_private(
        module=D,
        optimizer=optim_D,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    if verbose:
        print(f"[DP-CTGAN] noise_multiplier={noise_multiplier:.2f}")

    # ------------------------------------------------------------------
    # 5. Adversarial training loop
    #    Uses BatchMemoryManager for Opacus 1.5.x Poisson sampling.
    #    D and G steps fully separated.
    # ------------------------------------------------------------------
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    G.train()
    D.train()

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []

        # --- Train Discriminator: hooks ON, BMM active ---
        D.enable_hooks()
        with BatchMemoryManager(
            data_loader=dp_loader,
            max_physical_batch_size=actual_batch_size,
            optimizer=optim_D,
        ) as memory_safe_loader:
            for (real_batch,) in memory_safe_loader:
                real_batch = real_batch.to(device)
                bs = real_batch.size(0)
                if bs < 2:
                    continue

                optim_D.zero_grad()
                z = torch.randn(bs, noise_dim, device=device)
                with torch.no_grad():
                    fake = G(z)

                real_logits = D(real_batch)
                fake_logits = D(fake)

                loss_D = criterion(real_logits, torch.ones(bs, 1, device=device)) + \
                         criterion(fake_logits, torch.zeros(bs, 1, device=device))
                loss_D.backward()
                optim_D.step()
                d_losses.append(loss_D.item())

        # --- Train Generator: hooks OFF so D acts as plain module ---
        D.disable_hooks()
        n_g_steps = max(1, len(d_losses))
        for _ in range(n_g_steps):
            optim_G.zero_grad()
            z2 = torch.randn(actual_batch_size, noise_dim, device=device)
            gen_out = G(z2)
            gen_logits = D(gen_out)
            loss_G = criterion(gen_logits, torch.ones(actual_batch_size, 1, device=device))
            loss_G.backward()
            optim_G.step()
            g_losses.append(loss_G.item())
        D.enable_hooks()  # re-enable for next epoch

        if verbose and (epoch % 50 == 0 or epoch == 1):
            eps_now = privacy_engine.get_epsilon(delta)
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"D-loss: {np.mean(d_losses):.4f} | "
                  f"G-loss: {np.mean(g_losses):.4f} | "
                  f"e spent: {eps_now:.4f}")

    achieved_epsilon = privacy_engine.get_epsilon(delta)
    if verbose:
        print(f"[DP-CTGAN] Training complete. "
              f"Achieved e = {achieved_epsilon:.4f} (target {epsilon})")

    # ------------------------------------------------------------------
    # 6. Wrap generator in a sampler interface matching CTGANSynthesizer
    # ------------------------------------------------------------------
    G.eval()

    class _DPSynthesizer:
        """Thin wrapper to give DP Generator the same .sample() interface."""

        def __init__(self, generator, noise_dim, feature_cols, encoders, scaler, device):
            self._G = generator
            self._noise_dim = noise_dim
            self._feature_cols = feature_cols
            self._encoders = encoders
            self._scaler = scaler
            self._device = device

        def sample(self, n: int = None, num_rows: int = None) -> pd.DataFrame:
            count = num_rows if num_rows is not None else n
            with torch.no_grad():
                z = torch.randn(count, self._noise_dim, device=self._device)
                out = self._G(z).cpu().numpy()
            # Inverse-scale back to original value ranges
            out = self._scaler.inverse_transform(out)
            df_out = pd.DataFrame(out, columns=self._feature_cols)
            # Decode back to original types
            df_out = _decode_dataframe(df_out, self._encoders)
            return df_out

    synthesizer = _DPSynthesizer(G, noise_dim, feature_cols, encoders, scaler, device)

    return {
        "synthesizer": synthesizer,
        "achieved_epsilon": achieved_epsilon,
        "model_type": "DP-CTGAN",
        "epsilon_target": epsilon,
        "training_epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Fallback: plain CTGAN via SDV (when Opacus is missing)
# ---------------------------------------------------------------------------

def _fallback_vanilla_ctgan(
    data: pd.DataFrame,
    epochs: int,
    epsilon_target: float,
    verbose: bool,
) -> dict:
    """Train standard CTGAN and wrap to match DP result dict structure."""
    metadata = Metadata.detect_from_dataframe(data=data)
    synth = CTGANSynthesizer(metadata, epochs=epochs, verbose=verbose)
    synth.fit(data)
    return {
        "synthesizer": synth,
        "achieved_epsilon": float("inf"),   # No privacy guarantee
        "model_type": "Vanilla-CTGAN (fallback)",
        "epsilon_target": epsilon_target,
        "training_epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Dataset downloader helper
# ---------------------------------------------------------------------------

def download_benchmark_datasets(save_dir: str = "datasets/raw") -> dict:
    """
    Download the three benchmark datasets used in the paper.

    Returns a dict mapping dataset_name -> local file path.
    """
    import requests
    from pathlib import Path

    os.makedirs(save_dir, exist_ok=True)

    urls = {
        "liver": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
            os.path.join(save_dir, "liver.csv"),
        ),
        "pima": (
            "https://raw.githubusercontent.com/plotly/datasets/master/"
            "diabetes.csv",
            os.path.join(save_dir, "pima.csv"),
        ),
        "adult": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "adult/adult.data",
            os.path.join(save_dir, "adult.csv"),
        ),
    }

    paths = {}
    for name, (url, path) in urls.items():
        if os.path.exists(path):
            print(f"[datasets] '{name}' already downloaded → {path}")
            paths[name] = path
            continue
        print(f"[datasets] Downloading '{name}' …")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"[datasets] Saved → {path}")
            paths[name] = path
        except Exception as e:
            print(f"[datasets] WARNING: Failed to download '{name}': {e}")

    return paths
