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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.lstm_autoencoder import LSTMAutoencoder


def load_test_data_with_labels(test_csv_path: Path):
    """Charge le test CSV et fusionne rigoureusement les labels depuis data/raw/label-test1.csv."""
    print(f"      Chargement des données de test : {test_csv_path}")
    df = pd.read_csv(test_csv_path)

    y_true = None

    # 1. Recherche du fichier de labels dans data/raw/label-test1.csv
    label_path = Path("data/raw/label-test1.csv")
    if not label_path.exists():
        # Fallback si exécuté depuis un autre sous-dossier
        label_path = test_csv_path.parent.parent / "raw" / "label-test1.csv"

    if label_path.exists():
        print(f"      [Label] Ingestion de : {label_path}")
        df_label = pd.read_csv(label_path)
        df_label.columns = df_label.columns.str.strip()

        # Recherche d'une colonne contenant 'attack' ou 'label'
        attack_cols = [c for c in df_label.columns if 'attack' in c.lower() or 'label' in c.lower()]
        if not attack_cols:
            num_cols = df_label.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                attack_cols = [num_cols[-1]]

        if attack_cols:
            global_attack = (df_label[attack_cols].values > 0).any(axis=1).astype(int)

            # Gestion de la correspondance de taille (si test_mini.csv est plus petit)
            if len(global_attack) >= len(df):
                y_true = global_attack[:len(df)]
            else:
                # Si le fichier test_mini est un extrait, on complète ou ajuste
                y_true = np.zeros(len(df), dtype=int)
                y_true[:len(global_attack)] = global_attack

            print(f"      ✅ Labels d'attaques fusionnés avec succès ({int(y_true.sum())} échantillons d'attaques).")
    else:
        print(f"      ❌ Erreur critique : Fichier de labels introuvable à {label_path}")

    if y_true is None:
        print("      ⚠️ Attention : Aucun label trouvé, initialisation à 0.")
        y_true = np.zeros(len(df), dtype=int)

    # Nettoyage des features (garder uniquement les colonnes numériques de capteurs)
    df_features = df.drop(columns=[c for c in df.columns if c.lower() in ("time", "timestamp", "attack", "label")], errors="ignore")
    features = df_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)

    return features, y_true


def create_sliding_windows(data: np.ndarray, window: int = 60):
    """Crée des fenêtres glissantes pour le jeu de test."""
    windows = []
    for i in range(len(data) - window + 1):
        windows.append(data[i : i + window])
    return np.array(windows)


def main():
    parser = argparse.ArgumentParser(description="Inférence LSTM Autoencoder - FactoryShield-OT")
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/test_mini.csv"))
    parser.add_argument("--ckpt", type=Path, default=Path("models_saved/lstm_autoencoder_best.pth"))
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold-percentile", type=float, default=98.0,
                        help="Percentile pour fixer le seuil d'anomalie sur l'erreur")
    args = parser.parse_args()

    print("=" * 75)
    print(" 🔍 FACTORYSHIELD-OT : MODULE D'INFÉRENCE & ÉVALUATION LSTM")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device utilisé : {device}")

    # 1. Chargement du modèle sauvegardé
    if not args.ckpt.exists():
        print(f"❌ Erreur : Aucun poids trouvé dans {args.ckpt}. Lance l'entraînement d'abord !")
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location=device)
    n_features = ckpt["n_features"]

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=ckpt["hidden_size"],
        latent_dim=ckpt["latent_dim"]
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"      ✅ Modèle chargé depuis {args.ckpt}")

    # 2. Chargement sécurisé des données de test et des labels
    features, y_true_full = load_test_data_with_labels(args.test_csv)

    # Création des fenêtres glissantes
    X_windows = create_sliding_windows(features, window=args.window)

    # Aligner y_true avec les fenêtres (on prend le label de la fin de chaque fenêtre)
    y_true_windows = y_true_full[args.window - 1:]

    # 3. Calcul des erreurs de reconstruction (Inférence par Batch)
    print("      Calcul des erreurs de reconstruction (MAE par fenêtre)...")
    errors = []
    criterion = torch.nn.L1Loss(reduction='none')

    with torch.no_grad():
        for i in range(0, len(X_windows), args.batch_size):
            batch = torch.tensor(X_windows[i : i + args.batch_size], dtype=torch.float32).to(device)
            recon = model(batch)
            loss = criterion(recon, batch).mean(dim=(1, 2))
            errors.extend(loss.cpu().numpy())

    errors = np.array(errors)

    # 4. Détermination du seuil et prédictions
    threshold = np.percentile(errors, args.threshold_percentile)
    y_pred_windows = (errors > threshold).astype(int)

    # 5. Calcul des métriques de sécurité OT
    accuracy = accuracy_score(y_true_windows, y_pred_windows)
    precision = precision_score(y_true_windows, y_pred_windows, zero_division=0)
    recall = recall_score(y_true_windows, y_pred_windows, zero_division=0)
    f1 = f1_score(y_true_windows, y_pred_windows, zero_division=0)

    # Matrice de confusion (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_true_windows, y_pred_windows).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    total_attacks = int(y_true_windows.sum())

    # Affichage du rapport final
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
