"""
LSTM Autoencoder — FactoryShield-OT
Learns absolute normality on HAIEnd 23.05 (F=225) via reconstruction of
T=60s sliding windows. Trained exclusively on healthy data (semi-supervised).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    """Zero-copy view of a scaled (N, F) array as overlapping (T, F) windows."""

    def __init__(self, data: np.ndarray, window: int = 60, stride: int = 1):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.window = window
        self.stride = stride
        self.n_windows = (len(data) - window) // stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        return self.data[start:start + self.window]


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, latent_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (h_n, _) = self.lstm(x)
        return self.to_latent(h_n[-1])  # (B, latent_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, latent_dim: int, seq_len: int, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.from_latent = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.to_features = nn.Linear(hidden_size, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim) -> replicated across T as decoder input
        seed = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(seed)
        return self.to_features(out)  # (B, T, F)


class LSTMAutoencoder(nn.Module):
    """Encoder-decoder pair reconstructing (B, T=60, F=225) windows."""

    def __init__(self, n_features: int = 225, hidden_size: int = 128, latent_dim: int = 32,
                 seq_len: int = 60, num_layers: int = 1):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_size, latent_dim, num_layers)
        self.decoder = LSTMDecoder(n_features, hidden_size, latent_dim, seq_len, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
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


def train(model: LSTMAutoencoder, train_data: np.ndarray, val_data: np.ndarray = None,
          window: int = 60, stride: int = 1, batch_size: int = 256, epochs: int = 50,
          lr: float = 1e-3, num_workers: int = 4, device: str = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        SlidingWindowDataset(train_data, window, stride),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(
            SlidingWindowDataset(val_data, window, stride),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE — matches the reconstruction-error metric used at inference

    history = {"train_mae": [], "val_mae": []}
    for epoch in range(1, epochs + 1):
        train_mae = _run_epoch(model, train_loader, criterion, device, optimizer)
        history["train_mae"].append(train_mae)
        msg = f"[{epoch:03d}/{epochs}] train_mae={train_mae:.5f}"
        if val_loader is not None:
            val_mae = _run_epoch(model, val_loader, criterion, device)
            history["val_mae"].append(val_mae)
            msg += f" val_mae={val_mae:.5f}"
        print(msg)

    return history


if __name__ == "__main__":
    # Sanity check with random data — replace with arrays from src/data/preprocess.py
    dummy_train = np.random.rand(5000, 225).astype(np.float32)
    model = LSTMAutoencoder(n_features=225, hidden_size=128, latent_dim=32, seq_len=60)
    train(model, dummy_train, epochs=2, batch_size=128)
