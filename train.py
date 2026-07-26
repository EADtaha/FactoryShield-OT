import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import SGDOneClassSVM
from xgboost import XGBClassifier
from tqdm import tqdm

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXPERIMENTS = ['clean_100', 'mixed_70_30']
BASE_DATA_DIR = 'data/processed'
BASE_MODEL_DIR = 'models_saved'

# --- MODEL DEEP LEARNING ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

def create_sequences(data, seq_length=10):
    xs = [data[i:(i + seq_length)] for i in range(len(data) - seq_length)]
    return np.array(xs)

def train_models():
    print(f"🚀 Démarrage de l'entraînement sur : {DEVICE}")

    for exp in EXPERIMENTS:
        print(f"\n==========================================")
        print(f"🔥 EXPÉRIENCE : {exp.upper()}")
        print(f"==========================================")

        data_dir = os.path.join(BASE_DATA_DIR, exp)
        model_save_dir = os.path.join(BASE_MODEL_DIR, exp)
        os.makedirs(model_save_dir, exist_ok=True)

        train_file = "train_clean_scaled.csv" if exp == "clean_100" else "train_mixed_70_30.csv"
        df_train = pd.read_csv(os.path.join(data_dir, train_file))

        X_train = df_train.drop(columns=['label'], errors='ignore').values
        y_train = df_train['label'].values if 'label' in df_train.columns else np.zeros(len(df_train))

        # 1. Isolation Forest
        print("\n⏳ [1/4] Entraînement : Isolation Forest...")
        iso_contam = 0.01 if exp == "clean_100" else 0.30
        with tqdm(total=1, desc="Isolation Forest", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            iso = IsolationForest(contamination=iso_contam, random_state=42, n_jobs=-1)
            iso.fit(X_train)
            joblib.dump(iso, os.path.join(model_save_dir, "IsolationForest.joblib"))
            pbar.update(1)

        # 2. One-Class SVM
        print("⏳ [2/4] Entraînement : One-Class SVM...")
        with tqdm(total=1, desc="One-Class SVM", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            oc_svm = SGDOneClassSVM(random_state=42)
            oc_svm.fit(X_train)
            joblib.dump(oc_svm, os.path.join(model_save_dir, "OneClassSVM.joblib"))
            pbar.update(1)

        # Modèles Supervisés (seulement si données mixtes)
        if len(np.unique(y_train)) > 1:
            print("⏳ [3/4] Entraînement : Random Forest & XGBoost...")
            with tqdm(total=2, desc="Supervised ML", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                joblib.dump(rf, os.path.join(model_save_dir, "RandomForest.joblib"))
                pbar.update(1)

                xgb = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss', tree_method='hist', device='cuda' if DEVICE.type == 'cuda' else 'cpu')
                xgb.fit(X_train, y_train)
                joblib.dump(xgb, os.path.join(model_save_dir, "XGBoost.joblib"))
                pbar.update(1)
        else:
            print("⏭️ [3/4] Modèles supervisés ignorés (100% Clean).")

        # 3. LSTM Autoencoder
        print("⏳ [4/4] Entraînement : LSTM Autoencoder (PyTorch GPU)...")
        seq_len = 10
        X_train_seq = create_sequences(X_train, seq_len)
        train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)

        # Optimisation DataLoader : pin_memory accélère le transfert RAM -> VRAM
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=512, shuffle=True, pin_memory=True)

        model = LSTMAutoencoder(input_dim=X_train.shape[1], hidden_dim=64).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epochs = 10
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            # Barre de progression par batch
            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
            for batch in loop:
                # non_blocking=True pour des transferts asynchrones
                inputs = batch[0].to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), os.path.join(model_save_dir, "lstm_autoencoder.pth"))
        print(f"✅ Tous les modèles pour '{exp}' ont été sauvegardés.\n")

if __name__ == '__main__':
    train_models()
