"""
Training entry point — FactoryShield-OT
Trains the LSTM Autoencoder exclusively on healthy HAIEnd 23.05 cycles.
Run from project root: python src/models/train.py [--train-csv ...] [--val-csv ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.lstm_autoencoder import LSTMAutoencoder, SlidingWindowDataset

DEFAULT_TRAIN_CSV = Path("data/processed/train_clean_scaled.csv")
DEFAULT_CKPT_DIR = Path("models_saved")


def load_features(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if c.lower() in ("time", "timestamp", "attack")],
                 errors="ignore")
    return df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)


def build_loader(csv_path: Path, window: int, stride: int, batch_size: int,
                  shuffle: bool, num_workers: int) -> DataLoader:
    data = load_features(csv_path)
    dataset = SlidingWindowDataset(data, window=window, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                       drop_last=shuffle)


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


def train(args: argparse.Namespace) -> LSTMAutoencoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    train_loader = build_loader(args.train_csv, args.window, args.stride,
                                 args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_loader(args.val_csv, args.window, args.stride, args.batch_size,
                               shuffle=False, num_workers=args.num_workers) if args.val_csv else None

    n_features = train_loader.dataset.data.shape[1]
    model = LSTMAutoencoder(n_features=n_features, hidden_size=args.hidden_size,
                             latent_dim=args.latent_dim, seq_len=args.window).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()  # MAE — identique à la métrique d'inférence/xAI

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / "lstm_autoencoder_best.pth"
    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mae = run_epoch(model, train_loader, criterion, device, optimizer)
        monitor_mae, monitor_name = train_mae, "train"
        msg = f"[{epoch:03d}/{args.epochs}] train_mae={train_mae:.5f}"

        if val_loader is not None:
            val_mae = run_epoch(model, val_loader, criterion, device)
            monitor_mae, monitor_name = val_mae, "val"
            msg += f" val_mae={val_mae:.5f}"
        print(msg)

        if monitor_mae < best_mae:
            best_mae = monitor_mae
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "mae": best_mae,
                "monitor": monitor_name,
                "n_features": n_features,
                "hidden_size": args.hidden_size,
                "latent_dim": args.latent_dim,
                "window": args.window,
            }, ckpt_path)
            print(f"  -> new best {monitor_name}_mae={best_mae:.5f}, saved to {ckpt_path}")

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FactoryShield-OT LSTM Autoencoder")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
