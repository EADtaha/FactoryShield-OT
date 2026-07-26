"""Évaluation des modèles de détection d'anomalies pour FactoryShield-OT (CDC Section 7.4).

En sécurité OT/ICS, un Faux Négatif (attaque non détectée) peut avoir des
conséquences physiques bien plus graves qu'un Faux Positif (fausse alerte) —
c'est pourquoi le Rappel (Recall) et le Taux de Faux Négatifs (FNR) sont mis en
avant comme métriques prioritaires dans ce module, avant l'Accuracy globale qui
peut être trompeuse sur un dataset déséquilibré (peu d'attaques vs beaucoup de
normal).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class EvaluationError(Exception):
    """Erreur levée pour tout problème de calcul de métriques."""


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Modèle",
) -> Dict[str, Any]:
    """Calcule le jeu complet de métriques de détection pour un modèle.

    Args:
        y_true: Labels réels (0=normal, 1=attaque), tableau 1D.
        y_pred: Labels prédits (0=normal, 1=attaque), tableau 1D de même longueur.
        model_name: Nom du modèle, utilisé pour l'identification dans les
            résultats et les messages d'erreur.

    Returns:
        Dictionnaire contenant :
            - model_name: nom du modèle.
            - accuracy, precision, recall, f1: métriques standard (float, [0, 1]).
            - confusion_matrix: tableau NumPy 2x2 [[TN, FP], [FN, TP]].
            - fpr: Taux de Faux Positifs = FP / (FP + TN).
            - fnr: Taux de Faux Négatifs = FN / (FN + TP) — métrique la plus
              critique en contexte OT (une attaque manquée = un FN).
            - tn, fp, fn, tp: décompte brut des 4 cases de la matrice de confusion.

    Raises:
        EvaluationError: Si `y_true`/`y_pred` ont des longueurs différentes,
            sont vides, ou contiennent des valeurs hors {0, 1}.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if y_true_arr.shape[0] == 0 or y_pred_arr.shape[0] == 0:
        raise EvaluationError(f"[{model_name}] y_true/y_pred ne peuvent pas être vides.")
    if len(y_true_arr) != len(y_pred_arr):
        raise EvaluationError(
            f"[{model_name}] Longueurs incohérentes : y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}"
        )
    invalid_values = set(np.unique(y_true_arr)) - {0, 1} | set(np.unique(y_pred_arr)) - {0, 1}
    if invalid_values:
        raise EvaluationError(f"[{model_name}] Valeurs hors {{0, 1}} détectées : {invalid_values}")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "confusion_matrix": cm,
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compare_models_summary(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Construit un tableau comparatif de plusieurs modèles évalués.

    Args:
        results_dict: Dictionnaire {nom_modèle: résultat de `evaluate_model`}.
            Le nom de clé prime sur le champ `model_name` interne au résultat
            (permet de renommer un modèle sans ré-évaluer).

    Returns:
        DataFrame indexé par nom de modèle, colonnes : Accuracy, Precision,
        Recall, F1, FPR, FNR, TN, FP, FN, TP. Trié par Recall décroissant
        (métrique prioritaire en sécurité OT : un modèle qui rate moins
        d'attaques est classé en tête, même à Accuracy égale ou inférieure).

    Raises:
        EvaluationError: Si `results_dict` est vide.
    """
    if not results_dict:
        raise EvaluationError("results_dict est vide : aucun modèle à comparer.")

    rows = []
    for name, result in results_dict.items():
        rows.append(
            {
                "Modèle": name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "FPR": result["fpr"],
                "FNR": result["fnr"],
                "TN": result["tn"],
                "FP": result["fp"],
                "FN": result["fn"],
                "TP": result["tp"],
            }
        )

    summary = pd.DataFrame(rows).set_index("Modèle")
    summary = summary.sort_values(by="Recall", ascending=False)

    numeric_cols = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]
    summary[numeric_cols] = summary[numeric_cols].round(4)

    return summary


# ---------------------------------------------------------------------- #
# Test local : évaluation et comparaison sur des prédictions factices
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    n = 2000
    n_attacks = 100

    y_true = np.zeros(n, dtype=int)
    attack_idx = rng.choice(n, size=n_attacks, replace=False)
    y_true[attack_idx] = 1

    def _simulate_predictions(recall_target: float, fpr_target: float) -> np.ndarray:
        """Génère un y_pred factice avec un recall et un FPR approximatifs donnés."""
        y_pred = np.zeros(n, dtype=int)
        n_true_positive = int(recall_target * n_attacks)
        detected = rng.choice(attack_idx, size=n_true_positive, replace=False)
        y_pred[detected] = 1

        normal_idx = np.setdiff1d(np.arange(n), attack_idx)
        n_false_positive = int(fpr_target * len(normal_idx))
        false_alarms = rng.choice(normal_idx, size=n_false_positive, replace=False)
        y_pred[false_alarms] = 1
        return y_pred

    fake_results = {
        "IsolationForest": evaluate_model(y_true, _simulate_predictions(0.70, 0.08), "IsolationForest"),
        "SGDOneClassSVM": evaluate_model(y_true, _simulate_predictions(0.55, 0.15), "SGDOneClassSVM"),
        "RandomForest": evaluate_model(y_true, _simulate_predictions(0.92, 0.02), "RandomForest"),
        "XGBoost": evaluate_model(y_true, _simulate_predictions(0.95, 0.015), "XGBoost"),
    }

    print("--- evaluate_model() : détail RandomForest ---")
    for key, value in fake_results["RandomForest"].items():
        print(f"  {key}: {value}")

    print("\n--- compare_models_summary() : tableau comparatif (trié par Recall) ---")
    summary_df = compare_models_summary(fake_results)
    print(summary_df)

    print("\n--- Gestion d'erreurs (longueurs incohérentes) ---")
    try:
        evaluate_model(y_true, y_pred=np.zeros(10, dtype=int), model_name="Test")
    except EvaluationError as exc:
        print(f"  Exception correctement levée : {exc}")

    print("\nOK — module evaluation.py validé de bout en bout.")
