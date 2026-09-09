"""
Detection pipeline — FactoryShield-OT
======================================
Loads the trained LSTM Autoencoder checkpoint, reconstructs sliding windows
from a scaled test CSV, computes per-sensor residuals E_{t,f}, applies
Exponential Moving Average (EMA) smoothing on the global MAE, and raises
binary alerts against either the checkpoint's saved p99 threshold or a
user-supplied value.

Usage (from project root):
    python src/detection/inference.py --data data/processed/clean_100/test_set_final.csv
    python src/detection/inference.py --data <path> --threshold 0.05 --output out/scores.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import lfilter
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.lstm_autoencoder import LSTMAutoencoder, SlidingWindowDataset

# FIX BUG-2: point to an existing checkpoint file.
# The pre-trained files live under models_saved/<experiment>/lstm_autoencoder.pth.
# Default to the clean_100 experiment; pass --ckpt explicitly to select another.
_DEFAULT_CKPT  = _ROOT / "models_saved" / "clean_100" / "lstm_autoencoder.pth"
_DROP_COLS     = {"time", "timestamp", "attack", "label"}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_features(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Read a scaled CSV, drop metadata columns, return (array, feature_names)."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            "Ensure preprocessing has been run and the path is correct."
        )
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.lower() in _DROP_COLS],
                 errors="ignore")
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise ValueError(f"No numeric feature columns found in {csv_path}.")
    return df.to_numpy(dtype=np.float32), df.columns.tolist()


def load_model(ckpt_path: Path, device: str) -> Tuple[LSTMAutoencoder, dict]:
    """Load checkpoint, reconstruct model, return (model, ckpt_dict).

    Raises FileNotFoundError with a helpful message if the checkpoint is absent.
    Supports both old checkpoints (hidden_size / latent_dim keys) and new ones
    (hidden1 / latent_dim keys) so the pipeline is backwards-compatible.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt_path}\n"
            "Train the model first:\n"
            "  python src/models/train.py --train-csv <path>"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # backwards-compatibility: old checkpoints stored 'hidden_size' not 'hidden1'
    hidden1    = ckpt.get("hidden1",    ckpt.get("hidden_size",  128))
    latent_dim = ckpt.get("latent_dim", 64)
    n_features = ckpt["n_features"]
    window     = ckpt["window"]

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden1=hidden1,
        latent_dim=latent_dim,
        seq_len=window,
        dropout=ckpt.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Core reconstruction
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def reconstruct(
    model: LSTMAutoencoder,
    data: np.ndarray,
    window: int,
    stride: int,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the full forward pass and return per-sensor residuals and global MAE.

    For each window the residual is taken at the **last timestep** of the window
    (stride=1 semantics → one score per incoming sample in real-time mode).

    Returns:
        E   — (N_windows, F) absolute per-sensor reconstruction errors.
        mae — (N_windows,)   mean over features → global anomaly score.
    """
    loader = DataLoader(
        SlidingWindowDataset(data, window=window, stride=stride),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        recon = model(batch)
        # |x - x̂| at the last timestep of each window  →  (B, F)
        chunks.append((batch - recon).abs()[:, -1, :].cpu().numpy())

    E   = np.concatenate(chunks, axis=0)   # (N_windows, F)
    mae = E.mean(axis=1)                    # (N_windows,)
    return E, mae


# ─────────────────────────────────────────────────────────────────────────────
# EMA smoothing  (vectorised IIR — no Python loop)
# ─────────────────────────────────────────────────────────────────────────────
def ema_smooth(scores: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Apply causal EMA:  y_t = α·x_t + (1−α)·y_{t−1}.

    Implemented as a first-order IIR filter for numerical stability and speed.
    alpha=0.10 means ~10 % weight on the newest sample — matches the project
    specification and the Streamlit slider default.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
    return lfilter([alpha], [1.0, -(1.0 - alpha)], scores).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(
    data_path: Path,
    ckpt_path: Path,
    threshold: float | None,
    ema_alpha: float,
    stride: int,
    batch_size: int,
) -> pd.DataFrame:
    """Full pipeline: load → reconstruct → smooth → alert.

    If threshold is None the p99 value saved in the checkpoint is used.
    Raises FileNotFoundError (with actionable messages) for missing files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(ckpt_path, device)
    data, feature_names = load_features(data_path)

    # feature-count sanity check
    expected_f = ckpt["n_features"]
    if data.shape[1] != expected_f:
        raise ValueError(
            f"Feature mismatch: checkpoint expects {expected_f} features, "
            f"but {data_path.name} has {data.shape[1]}. "
            "Ensure the same preprocessing pipeline was used."
        )

    E, mae = reconstruct(model, data, ckpt["window"], stride, batch_size, device)
    smoothed = ema_smooth(mae, alpha=ema_alpha)

    # threshold selection: explicit arg > checkpoint p99 > fallback heuristic
    if threshold is None:
        if "threshold_p99" in ckpt:
            threshold = float(ckpt["threshold_p99"])
            print(f"[inference] Using checkpoint p99 threshold = {threshold:.6f}")
        else:
            threshold = float(np.percentile(smoothed, 99))
            print(f"[inference] No threshold in checkpoint — using p99 of "
                  f"current data = {threshold:.6f}  (re-train to persist this value)")

    alert = (smoothed > threshold).astype(np.int8)

    out = pd.DataFrame(E, columns=feature_names)
    out["mae_raw"]      = mae
    out["mae_smoothed"] = smoothed
    out["alert"]        = alert
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FactoryShield-OT detection pipeline")
    p.add_argument("--data",       type=Path, required=True,
                   help="Scaled test CSV (output of preprocessing)")
    p.add_argument("--ckpt",       type=Path, default=_DEFAULT_CKPT,
                   help="Model checkpoint .pth (default: models_saved/clean_100/lstm_autoencoder.pth)")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Alert threshold; omit to use the p99 value saved in the checkpoint")
    p.add_argument("--ema-alpha",  type=float, default=0.10,
                   help="EMA smoothing factor α ∈ (0, 1]  (default: 0.10)")
    p.add_argument("--stride",     type=int,   default=1)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--output",     type=Path,  default=None,
                   help="Optional CSV path to save scores + alerts")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        out = run_inference(
            args.data, args.ckpt, args.threshold,
            args.ema_alpha, args.stride, args.batch_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    n_alerts = int(out["alert"].sum())
    n_total  = len(out)
    print(f"windows={n_total}  alerts={n_alerts}  rate={n_alerts/n_total:.4f}  "
          f"peak_mae={out['mae_smoothed'].max():.5f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
