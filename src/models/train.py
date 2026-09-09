"""
Training entry point — FactoryShield-OT
Trains the LSTM Autoencoder exclusively on healthy HAIEnd 23.05 cycles.

Architecture: Encoder 225→128→64, Decoder 64→128→225
Loss:         L1 (MAE) — identical to the inference error metric
Threshold:    99th-percentile of train reconstruction errors, saved alongside weights

Usage (from project root):
    python src/models/train.py
    python src/models/train.py --train-csv data/processed/clean_100/train_clean_scaled.csv
    python src/models/train.py --config configs/hyperparameters.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.lstm_autoencoder import LSTMAutoencoder, SlidingWindowDataset

# ── optional YAML config ──────────────────────────────────────────────────────
try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_TRAIN = _ROOT / "data/processed/clean_100/train_clean_scaled.csv"
_DEFAULT_CKPT  = _ROOT / "models_saved"
_DROP_COLS     = {"time", "timestamp", "attack", "label"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_yaml_config(path: Path) -> dict:
    """Load hyperparameters.yaml and return the relevant sub-dicts."""
    if not _YAML_OK:
        print("WARNING: PyYAML not installed — ignoring --config flag.")
        return {}
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    merged: dict = {}
    for key in ("lstm_autoencoder", "data"):
        merged.update(raw.get(key, {}))
    return merged


def _load_features(csv_path: Path) -> np.ndarray:
    """Read a scaled CSV and return the numeric feature matrix (float32)."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {csv_path}\n"
            "Run preprocessing first or pass --train-csv with the correct path."
        )
    df = pd.read_csv(csv_path)
    drop = [c for c in df.columns if c.lower() in _DROP_COLS]
    df   = df.drop(columns=drop, errors="ignore")
    return df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)


def _build_loader(
    data: np.ndarray, window: int, stride: int,
    batch_size: int, shuffle: bool, num_workers: int,
) -> DataLoader:
    ds = SlidingWindowDataset(data, window=window, stride=stride)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=shuffle, persistent_workers=(num_workers > 0),
    )


def _run_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * batch.size(0)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def _compute_train_threshold(
    model: LSTMAutoencoder,
    data: np.ndarray,
    window: int, stride: int,
    batch_size: int, device: str,
    percentile: float = 99.0,
) -> float:
    """Run a full forward pass on training data and return the p-th percentile MAE."""
    loader = DataLoader(
        SlidingWindowDataset(data, window, stride),
        batch_size=batch_size, shuffle=False,
    )
    model.eval()
    maes = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        recon = model(batch)
        mae   = (batch - recon).abs().mean(dim=(1, 2))   # (B,)
        maes.append(mae.cpu().numpy())
    all_mae = np.concatenate(maes)
    return float(np.percentile(all_mae, percentile))


# ─────────────────────────────────────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> LSTMAutoencoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    # ── load data ─────────────────────────────────────────────────────────────
    train_data = _load_features(args.train_csv)
    val_data   = _load_features(args.val_csv) if args.val_csv else None
    n_features = train_data.shape[1]
    print(f"[train] train samples={len(train_data):,}  features={n_features}")
    if val_data is not None:
        print(f"[train] val   samples={len(val_data):,}")

    # ── build loaders ─────────────────────────────────────────────────────────
    train_loader = _build_loader(train_data, args.window, args.stride,
                                  args.batch_size, True,  args.num_workers)
    val_loader   = (_build_loader(val_data, args.window, args.stride,
                                   args.batch_size, False, args.num_workers)
                    if val_data is not None else None)

    # ── model ─────────────────────────────────────────────────────────────────
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden1=args.hidden1,
        latent_dim=args.latent_dim,
        seq_len=args.window,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model parameters={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()   # MAE — matches inference + threshold metric

    # ── output dir ────────────────────────────────────────────────────────────
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / "lstm_autoencoder_best.pth"

    # ── training loop with early stopping ────────────────────────────────────
    best_mae   = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_mae = _run_epoch(model, train_loader, criterion, device, optimizer)
        monitor, monitor_name = train_mae, "train"
        msg = f"[{epoch:03d}/{args.epochs}] train_mae={train_mae:.5f}"

        if val_loader is not None:
            val_mae = _run_epoch(model, val_loader, criterion, device)
            monitor, monitor_name = val_mae, "val"
            msg += f"  val_mae={val_mae:.5f}"

        print(msg)

        # ── save best checkpoint ──────────────────────────────────────────────
        if monitor < best_mae - 1e-7:
            best_mae   = monitor
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "mae":         best_mae,
                "monitor":     monitor_name,
                "n_features":  n_features,
                "hidden1":     args.hidden1,
                "latent_dim":  args.latent_dim,
                "window":      args.window,
                "dropout":     args.dropout,
            }, ckpt_path)
            print(f"  -> best {monitor_name}_mae={best_mae:.5f}  saved → {ckpt_path}")
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"[train] Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    # ── compute and save 99th-percentile threshold on training data ───────────
    print("[train] Computing 99th-percentile threshold on training data…")
    # reload best weights before threshold computation
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    threshold = _compute_train_threshold(
        model, train_data, args.window, args.stride,
        args.batch_size, device, percentile=99.0,
    )
    print(f"[train] Threshold (p99 train MAE) = {threshold:.6f}")

    # update checkpoint with threshold
    ckpt["threshold_p99"] = threshold
    torch.save(ckpt, ckpt_path)
    print(f"[train] Checkpoint updated with threshold → {ckpt_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FactoryShield-OT LSTM Autoencoder")
    p.add_argument("--config",      type=Path, default=None,
                   help="Path to configs/hyperparameters.yaml (optional override)")
    p.add_argument("--train-csv",   type=Path, default=_DEFAULT_TRAIN)
    p.add_argument("--val-csv",     type=Path, default=None)
    p.add_argument("--ckpt-dir",    type=Path, default=_DEFAULT_CKPT)
    p.add_argument("--window",      type=int,   default=60)
    p.add_argument("--stride",      type=int,   default=1)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--patience",    type=int,   default=7,
                   help="Early-stopping patience (0 = disabled)")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden1",     type=int,   default=128)
    p.add_argument("--latent-dim",  type=int,   default=64)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--num-workers", type=int,   default=0)
    args = p.parse_args()

    # YAML overrides (CLI flags still win if explicitly set by user)
    if args.config is not None:
        cfg = _load_yaml_config(args.config)
        defaults = {
            "window":     cfg.get("window_size",   args.window),
            "batch_size": cfg.get("batch_size",    args.batch_size),
            "epochs":     cfg.get("epochs",        args.epochs),
            "lr":         cfg.get("learning_rate", args.lr),
            "hidden1":    cfg.get("hidden_size",   args.hidden1),
            "latent_dim": cfg.get("latent_dim",    args.latent_dim),
            "dropout":    cfg.get("dropout",       args.dropout),
        }
        # only apply YAML value where the user did NOT explicitly pass a CLI flag
        cli_flags = {a.dest for a in p._actions if a.option_strings}
        for attr, val in defaults.items():
            flag = f"--{attr.replace('_', '-')}"
            if flag not in sys.argv:
                setattr(args, attr, val)

    return args


if __name__ == "__main__":
    train(_parse_args())
