"""
LSTM Autoencoder — FactoryShield-OT
Architecture: Encoder 225→128→64 (latent), Decoder 64→128→225
Trained exclusively on clean normal data (semi-supervised normality learning).
Window size T=60, stride=1, loss=L1 (MAE) to match the inference error metric.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset — zero-copy sliding window view
# ─────────────────────────────────────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    """Memory-efficient sliding window dataset using a pre-scaled (N, F) array.

    No duplication of the underlying array in RAM — each call to __getitem__
    returns a view slice, not a copy.
    """

    def __init__(self, data: np.ndarray, window: int = 60, stride: int = 1) -> None:
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.window = window
        self.stride = stride
        self.n_windows = max(0, (len(data) - window) // stride + 1)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        return self.data[start : start + self.window]  # (T, F)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder  225 → 128 → 64
# ─────────────────────────────────────────────────────────────────────────────
class LSTMEncoder(nn.Module):
    """Two-layer LSTM encoder that compresses (B, T, 225) to a latent (B, 64)."""

    def __init__(
        self,
        n_features: int = 225,
        hidden1: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1,   latent_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F=225)
        out1, _ = self.lstm1(x)            # (B, T, 128)
        out1    = self.drop(out1)
        _, (h_n, _) = self.lstm2(out1)     # h_n: (1, B, 64)
        return h_n.squeeze(0)              # (B, 64)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder  64 → 128 → 225
# ─────────────────────────────────────────────────────────────────────────────
class LSTMDecoder(nn.Module):
    """Two-layer LSTM decoder that reconstructs (B, 64) → (B, T, 225)."""

    def __init__(
        self,
        n_features: int = 225,
        hidden1: int = 128,
        latent_dim: int = 64,
        seq_len: int = 60,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len    = seq_len
        self.expand     = nn.Linear(latent_dim, hidden1)
        self.lstm1      = nn.LSTM(hidden1,   hidden1,   batch_first=True)
        self.drop       = nn.Dropout(dropout)
        self.lstm2      = nn.LSTM(hidden1,   n_features, batch_first=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, 64)
        seed = self.expand(z).unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, T, 128)
        out1, _ = self.lstm1(seed)          # (B, T, 128)
        out1    = self.drop(out1)
        out2, _ = self.lstm2(out1)          # (B, T, 225)
        return out2


# ─────────────────────────────────────────────────────────────────────────────
# Full LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """Encoder-decoder LSTM autoencoder for temporal anomaly detection.

    Default dimensions match HAIEnd 23.05:
        n_features = 225, hidden1 = 128, latent_dim = 64, seq_len = 60.
    """

    def __init__(
        self,
        n_features: int = 225,
        hidden1: int = 128,
        latent_dim: int = 64,
        seq_len: int = 60,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden1, latent_dim, dropout)
        self.decoder = LSTMDecoder(n_features, hidden1, latent_dim, seq_len, dropout)
        # store for checkpoint serialisation
        self.n_features = n_features
        self.hidden1    = hidden1
        self.latent_dim = latent_dim
        self.seq_len    = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))  # (B, T, F)


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch helper
# ─────────────────────────────────────────────────────────────────────────────
def _run_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
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
            total_loss += loss.item() * batch.size(0)
    return total_loss / max(len(loader.dataset), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience train() function (used by notebooks / quick experiments)
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model: LSTMAutoencoder,
    train_data: np.ndarray,
    val_data: np.ndarray | None = None,
    *,
    window: int = 60,
    stride: int = 1,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
    num_workers: int = 0,
    device: str | None = None,
) -> dict:
    """Train the autoencoder with optional early stopping.

    Args:
        model:       LSTMAutoencoder instance.
        train_data:  Scaled (N, F) float32 array — **normal samples only**.
        val_data:    Optional validation array (also normal-only or held-out).
        window:      Sliding window size T (default 60).
        stride:      Stride between windows (default 1).
        batch_size:  Mini-batch size (default 64).
        epochs:      Maximum number of epochs (default 50).
        lr:          Adam learning rate (default 1e-3).
        patience:    Early-stopping patience in epochs (default 7).
                     Set to 0 or None to disable early stopping.
        num_workers: DataLoader worker processes (0 = main thread).
        device:      'cuda' / 'cpu' / None (auto-detect).

    Returns:
        History dict with keys 'train_mae' and optionally 'val_mae'.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _loader_kw = dict(shuffle=True,  num_workers=num_workers,
                      pin_memory=(device == "cuda"), drop_last=True)
    train_loader = DataLoader(SlidingWindowDataset(train_data, window, stride),
                               batch_size=batch_size, **_loader_kw)
    val_loader = (
        DataLoader(SlidingWindowDataset(val_data, window, stride),
                   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=(device == "cuda"))
        if val_data is not None else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE — identical to the inference error metric

    history: dict[str, list[float]] = {"train_mae": [], "val_mae": []}
    best_val  = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_mae = _run_epoch(model, train_loader, criterion, device, optimizer)
        history["train_mae"].append(train_mae)
        msg = f"[{epoch:03d}/{epochs}] train_mae={train_mae:.5f}"

        monitor = train_mae
        if val_loader is not None:
            val_mae = _run_epoch(model, val_loader, criterion, device)
            history["val_mae"].append(val_mae)
            msg += f"  val_mae={val_mae:.5f}"
            monitor = val_mae

        print(msg)

        # early stopping
        if patience:
            if monitor < best_val - 1e-6:
                best_val   = monitor
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs).")
                    break

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy = np.random.rand(2000, 225).astype(np.float32)
    m = LSTMAutoencoder(n_features=225, hidden1=128, latent_dim=64, seq_len=60)
    total_params = sum(p.numel() for p in m.parameters())
    print(f"Model parameters: {total_params:,}")
    h = train(m, dummy, epochs=3, batch_size=64, patience=2)
    print("History:", {k: [f"{v:.5f}" for v in vs] for k, vs in h.items()})
