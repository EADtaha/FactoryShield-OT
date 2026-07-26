import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def append_metrics_to_report(model_name, y_true, y_pred, y_scores, plt_obj):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    auc = roc_auc_score(y_true, y_scores) if y_scores is not None else None

    if auc is not None:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
        plt_obj.plot(fpr_curve, tpr_curve, label=f'{model_name} (AUC = {auc:.4f})')

    res = f"--------------------------------------------------\n"
    res += f"📌 MODÈLE : {model_name}\n"
    res += f"--------------------------------------------------\n"
    res += f"  • Accuracy            : {acc:.4f}\n"
    res += f"  • Precision           : {prec:.4f}\n"
    res += f"  • Recall (TPR)        : {rec:.4f}\n"
    res += f"  • F1-Score            : {f1:.4f}\n"
    res += f"  • FPR (Faux Positifs) : {fpr:.4f}\n"
    res += f"  • FNR (Faux Négatifs) : {fnr:.4f}\n"
    res += f"  • ROC-AUC             : {f'{auc:.4f}' if auc is not None else 'N/A'}\n"
    res += f"  • Matrice de Confusion:\n"
    res += f"      [TN: {tn:<6} | FP: {fp}]\n"
    res += f"      [FN: {fn:<6} | TP: {tp}]\n\n"
    return res

def evaluate_models():
    EXPERIMENTS = ['clean_100', 'mixed_70_30']
    BASE_DATA_DIR = 'data/processed'
    BASE_MODEL_DIR = 'models_saved'
    OUT_DIR = 'out'

    os.makedirs(OUT_DIR, exist_ok=True)

    for exp in EXPERIMENTS:
        print(f"\n==========================================")
        print(f"📊 ÉVALUATION : {exp.upper()}")
        print(f"==========================================")

        data_dir = os.path.join(BASE_DATA_DIR, exp)
        model_save_dir = os.path.join(BASE_MODEL_DIR, exp)
        test_path = os.path.join(data_dir, "test_set_final.csv")

        df_test = pd.read_csv(test_path)
        X_test = df_test.drop(columns=['label'], errors='ignore').values
        y_test = df_test['label'].values

        report_txt = f"==================================================\n"
        report_txt += f"      RAPPORT D'ÉVALUATION - {exp.upper()}\n"
        report_txt += f"==================================================\n\n"

        plt.figure(figsize=(10, 8))

        models_to_eval = [
            ("IsolationForest.joblib", "Isolation Forest"),
            ("OneClassSVM.joblib", "One-Class SVM"),
            ("RandomForest.joblib", "Random Forest"),
            ("XGBoost.joblib", "XGBoost"),
            ("lstm_autoencoder.pth", "LSTM Autoencoder")
        ]

        # Boucle d'évaluation avec barre de progression
        for file_name, model_name in tqdm(models_to_eval, desc="Évaluation des modèles", unit="modèle"):
            model_path = os.path.join(model_save_dir, file_name)
            if not os.path.exists(model_path):
                continue

            if file_name.endswith('.joblib'):
                model = joblib.load(model_path)
                if model_name in ["Isolation Forest", "One-Class SVM"]:
                    preds = np.where(model.predict(X_test) == -1, 1, 0)
                    scores = -model.score_samples(X_test)
                else:
                    preds = model.predict(X_test)
                    scores = model.predict_proba(X_test)[:, 1]

                report_txt += append_metrics_to_report(model_name, y_test, preds, scores, plt)

            elif file_name.endswith('.pth'):
                seq_len = 10
                X_test_seq = create_sequences(X_test, seq_len)
                y_test_seq = y_test[seq_len:]

                test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)
                lstm_model = LSTMAutoencoder(input_dim=X_test.shape[1], hidden_dim=64).to(DEVICE)
                lstm_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                lstm_model.eval()

                with torch.no_grad():
                    reconstructions = lstm_model(test_tensor)
                    rec_error = torch.mean((reconstructions - test_tensor) ** 2, dim=[1, 2]).cpu().numpy()

                threshold = np.percentile(rec_error, 95)
                preds_lstm = (rec_error > threshold).astype(int)
                report_txt += append_metrics_to_report(model_name, y_test_seq, preds_lstm, rec_error, plt)

        # Sauvegarde TXT
        txt_path = os.path.join(OUT_DIR, f"evaluation_report_{exp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_txt)

        # Sauvegarde PNG
        plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
        plt.xlabel('Taux de Faux Positifs (FPR)')
        plt.ylabel('Taux de Vrais Positifs (TPR)')
        plt.title(f'Courbes ROC-AUC - {exp.upper()}')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        roc_path = os.path.join(OUT_DIR, f"roc_curve_{exp}.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📄 Rapport : {txt_path}")
        print(f"🖼️ Graphique : {roc_path}\n")

if __name__ == '__main__':
    evaluate_models()
