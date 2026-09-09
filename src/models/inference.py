"""
Script d'Inférence et d'Évaluation — FactoryShield-OT
Charge le LSTM Autoencoder entraîné, fusionne les labels de test depuis data/raw/label-test1.csv,
calcule les erreurs de reconstruction (MAE) et affiche le rapport complet.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Anchor all relative paths to the project root regardless of CWD.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.lstm_autoencoder import LSTMAutoencoder


def load_test_data_with_labels(test_csv_path: Path):
    """Charge le test CSV et fusionne rigoureusement les labels depuis data/raw/label-test1.csv.

    FIX BUG-6: label path is now anchored to the project root (_ROOT) so the
    script behaves identically regardless of the current working directory.
    """
    print(f"      Chargement des données de test : {test_csv_path}")
    df = pd.read_csv(test_csv_path)

    y_true = None

    # Always resolve relative to the project root — never relative to CWD.
    label_path = _ROOT / "data" / "raw" / "label-test1.csv"

    if label_path.exists():
        print(f"      [Label] Ingestion de : {label_path}")
        df_label = pd.read_csv(label_path)
        df_label.columns = df_label.columns.str.strip()

        # Recherche d'une colonne contenant 'attack' ou 'label'
        attack_cols = [
            c for c in df_label.columns
            if "attack" in c.lower() or "label" in c.lower()
        ]
        if not attack_cols:
            num_cols = df_label.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                attack_cols = [num_cols[-1]]

        if attack_cols:
            global_attack = (df_label[attack_cols].values > 0).any(axis=1).astype(int)

            # Align lengths: truncate or zero-pad as needed.
            if len(global_attack) >= len(df):
                y_true = global_attack[: len(df)]
            else:
                y_true = np.zeros(len(df), dtype=int)
                y_true[: len(global_attack)] = global_attack

            print(
                f"      ✅ Labels d'attaques fusionnés avec succès "
                f"({int(y_true.sum())} échantillons d'attaques)."
            )
    else:
        print(f"      ❌ Erreur critique : Fichier de labels introuvable à {label_path}")

    if y_true is None:
        print("      ⚠️ Attention : Aucun label trouvé, initialisation à 0.")
        y_true = np.zeros(len(df), dtype=int)

    # Nettoyage des features (garder uniquement les colonnes numériques de capteurs)
    df_features = df.drop(
        columns=[
            c for c in df.columns
            if c.lower() in ("time", "timestamp", "attack", "label")
        ],
        errors="ignore",
    )
    features = df_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)

    return features, y_true


def create_sliding_windows(data: np.ndarray, window: int = 60) -> np.ndarray:
    """Zero-copy sliding windows using numpy stride tricks.

    FIX PERF-2: the original implementation called np.array() on a Python list
    of slices, which materialised the full (N-T+1, T, F) tensor as a new
    allocation — up to ~48 GB for the full HAIEnd dataset.

    sliding_window_view returns a *view* (no copy).  np.swapaxes also returns a
    view.  Only the downstream consumer (the DataLoader / torch.tensor call)
    triggers an actual memory copy, and only one batch at a time.

    Returns shape: (n_windows, window, n_features)
    """
    # sliding_window_view output: (n_windows, n_features, window)
    raw = sliding_window_view(data, window_shape=window, axis=0)
    # Swap to (n_windows, window, n_features) — standard (B, T, F) convention
    return np.swapaxes(raw, 1, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Inférence LSTM Autoencoder - FactoryShield-OT"
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=_ROOT / "data" / "processed" / "clean_100" / "test_set_final.csv",
    )
    # FIX BUG-2 (also applied here): default points to an existing checkpoint.
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=_ROOT / "models_saved" / "clean_100" / "lstm_autoencoder.pth",
    )
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=98.0,
        help="Percentile pour fixer le seuil d'anomalie sur l'erreur",
    )
    args = parser.parse_args()

    print("=" * 75)
    print(" 🔍 FACTORYSHIELD-OT : MODULE D'INFÉRENCE & ÉVALUATION LSTM")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device utilisé : {device}")

    # ── 1. Chargement du modèle sauvegardé ───────────────────────────────────
    if not args.ckpt.exists():
        print(
            f"❌ Erreur : Aucun poids trouvés dans {args.ckpt}. "
            "Lance l'entraînement d'abord !"
        )
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    n_features = ckpt["n_features"]

    # FIX BUG-1: use 'hidden1' kwarg (the model's actual parameter name).
    # Backwards-compatible: also reads 'hidden_size' from older checkpoints.
    hidden1 = ckpt.get("hidden1", ckpt.get("hidden_size", 128))
    latent_dim = ckpt.get("latent_dim", 64)
    seq_len = ckpt.get("window", ckpt.get("seq_len", args.window))

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden1=hidden1,
        latent_dim=latent_dim,
        seq_len=seq_len,
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"      ✅ Modèle chargé depuis {args.ckpt}")
    print(f"         n_features={n_features}  hidden1={hidden1}  latent_dim={latent_dim}  seq_len={seq_len}")

    # ── 2. Chargement sécurisé des données de test et des labels ─────────────
    features, y_true_full = load_test_data_with_labels(args.test_csv)

    # Zero-copy windows — no full-tensor RAM spike
    X_windows = create_sliding_windows(features, window=args.window)

    # Align y_true with windows (label of the last timestep in each window)
    y_true_windows = y_true_full[args.window - 1 :]

    # ── 3. Calcul des erreurs de reconstruction (inférence par batch) ─────────
    print("      Calcul des erreurs de reconstruction (MAE par fenêtre)...")
    errors = []
    criterion = torch.nn.L1Loss(reduction="none")

    with torch.no_grad():
        for i in range(0, len(X_windows), args.batch_size):
            # torch.as_tensor avoids a copy when the source array is contiguous float32
            batch = torch.as_tensor(
                X_windows[i : i + args.batch_size], dtype=torch.float32
            ).to(device, non_blocking=True)
            recon = model(batch)
            loss = criterion(recon, batch).mean(dim=(1, 2))
            errors.extend(loss.cpu().numpy())

    errors = np.array(errors)

    # ── 4. Détermination du seuil et prédictions ──────────────────────────────
    threshold = np.percentile(errors, args.threshold_percentile)
    y_pred_windows = (errors > threshold).astype(int)

    # ── 5. Calcul des métriques de sécurité OT ────────────────────────────────
    accuracy  = accuracy_score(y_true_windows, y_pred_windows)
    precision = precision_score(y_true_windows, y_pred_windows, zero_division=0)
    recall    = recall_score(y_true_windows, y_pred_windows, zero_division=0)
    f1        = f1_score(y_true_windows, y_pred_windows, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true_windows, y_pred_windows).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    total_attacks = int(y_true_windows.sum())

    print("\n" + "=" * 75)
    print(" 📊 RÉSULTATS DE L'ÉVALUATION DU LSTM AUTOENCODER")
    print("=" * 75)
    print(f" Accuracy   : {accuracy:.4f}")
    print(f" Precision  : {precision:.4f}")
    print(f" Recall     : {recall:.4f}")
    print(f" F1-Score   : {f1:.4f}")
    print(f" FPR        : {fpr:.4f}")
    print(f" Détections : TP = {tp} / Total Attaques = {total_attacks}")
    print(f" Matrice    : TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print("=" * 75)


if __name__ == "__main__":
    main()
