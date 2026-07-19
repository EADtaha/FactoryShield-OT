"""
Detection pipeline — FactoryShield-OT
Reconstructs sliding windows, computes per-sensor residuals E_{t,f}, smooths the
global MAE with an EMA, and raises a binary alert against --threshold.
Run: python src/detection/inference.py --data data/processed/test_clean_scaled.csv --threshold 0.05
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.signal import lfilter

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.lstm_autoencoder import LSTMAutoencoder, SlidingWindowDataset

DEFAULT_CKPT = Path("models_saved/lstm_autoencoder_best.pth")
DROP_COLS = {"time", "timestamp", "attack"}


def load_features(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.lower() in DROP_COLS], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    return df.to_numpy(dtype=np.float32), df.columns.tolist()


def load_model(ckpt_path: Path, device: str) -> Tuple[LSTMAutoencoder, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = LSTMAutoencoder(n_features=ckpt["n_features"], hidden_size=ckpt["hidden_size"],
                             latent_dim=ckpt["latent_dim"], seq_len=ckpt["window"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def reconstruct(model: LSTMAutoencoder, data: np.ndarray, window: int, stride: int,
                 batch_size: int, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """One residual vector per window, taken at the window's last timestep (real-time
    simulation semantics: stride=1 -> one score per incoming sample)."""
    loader = DataLoader(SlidingWindowDataset(data, window=window, stride=stride),
                         batch_size=batch_size, shuffle=False)
    chunks = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        recon = model(batch)
        chunks.append((batch - recon).abs()[:, -1, :].cpu().numpy())  # (B, F)

    E = np.concatenate(chunks, axis=0)  # (N_windows, F)
    mae = E.mean(axis=1)                # (N_windows,)
    return E, mae


def ema_smooth(scores: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """y_t = alpha * x_t + (1 - alpha) * y_{t-1}, computed via IIR filter (no Python loop)."""
    return lfilter([alpha], [1.0, -(1.0 - alpha)], scores)


def run_inference(data_path: Path, ckpt_path: Path, threshold: float, ema_alpha: float,
                   stride: int, batch_size: int) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(ckpt_path, device)
    data, feature_names = load_features(data_path)

    E, mae = reconstruct(model, data, ckpt["window"], stride, batch_size, device)
    smoothed = ema_smooth(mae, ema_alpha)
    alert = (smoothed > threshold).astype(int)

    out = pd.DataFrame(E, columns=feature_names)
    out["mae_raw"] = mae
    out["mae_smoothed"] = smoothed
    out["alert"] = alert
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FactoryShield-OT detection pipeline")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = run_inference(args.data, args.ckpt, args.threshold, args.ema_alpha,
                         args.stride, args.batch_size)
    n_alerts = int(out["alert"].sum())
    print(f"windows={len(out)} alerts={n_alerts} rate={n_alerts / len(out):.4f}")
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
