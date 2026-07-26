"""Pipeline de modèles ML classiques pour FactoryShield-OT, optimisée pour machines
à RAM/CPU restreints.

Ce module fournit `ClassicalModelsPipeline`, qui encapsule l'entraînement et
l'inférence de quatre modèles sur les données tabulaires 2D produites par
`HAIPreprocessor` (voir preprocessing.py) :

Non supervisés (fit sur y=0 uniquement — normalité thermodynamique) :
    - IsolationForest (max_samples borné, n_estimators réduit)
    - SGDOneClassSVM  (approximation linéaire par SGD, O(n) en mémoire — évite
      l'explosion O(n^2) de OneClassSVM à noyau RBF standard)

Supervisés (fit sur y=0/1) :
    - RandomForestClassifier (profondeur bornée)
    - XGBoost (tree_method='hist', histogrammes — rapide sur CPU, faible RAM)

Convention de score : pour tous les modèles, un score élevé = anomalie plus
probable. Cette convention unifiée permet de tracer des courbes ROC/PR
comparables entre modèles non supervisés et supervisés sans logique
conditionnelle côté appelant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import SGDOneClassSVM
from xgboost import XGBClassifier

# Modèles considérés comme non supervisés : leur .predict natif renvoie
# {-1, 1} (outlier/inlier) et non {0, 1}. Utilisé pour dispatcher la logique
# de conversion dans predict_all().
_UNSUPERVISED_MODELS = ("IsolationForest", "SGDOneClassSVM")
_SUPERVISED_MODELS = ("RandomForest", "XGBoost")


class ClassicalModelsPipelineError(Exception):
    """Erreur levée pour tout problème d'entraînement, de prédiction ou de persistance."""


class ClassicalModelsPipeline:
    """Entraîne, évalue et persiste les 4 modèles classiques de FactoryShield-OT.

    Attributes:
        random_state: Graine pour la reproductibilité des modèles stochastiques.
        models: Dictionnaire {nom_modèle: instance entraînée}.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initialise le pipeline, sans entraîner aucun modèle.

        Args:
            random_state: Graine de reproductibilité, propagée à tous les
                modèles qui l'acceptent.
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #
    def train_unsupervised(self, X_train_normal: np.ndarray) -> None:
        """Entraîne IsolationForest et SGDOneClassSVM sur des données strictement normales.

        Args:
            X_train_normal: Matrice 2D (n_samples, n_features) contenant
                exclusivement des échantillons sains (y=0). Convertie en
                float32 pour limiter l'empreinte mémoire.

        Raises:
            ClassicalModelsPipelineError: Si `X_train_normal` n'est pas 2D
                ou est vide.
        """
        X = self._validate_2d(X_train_normal, "X_train_normal")

        # max_samples borné : chaque arbre de la forêt n'est construit que sur
        # un sous-échantillon d'au plus 10 000 lignes, indépendamment de la
        # taille totale du dataset -> temps et mémoire d'entraînement bornés.
        iso_forest = IsolationForest(
            n_estimators=50,
            max_samples=min(10_000, X.shape[0]),
            n_jobs=-1,
            random_state=self.random_state,
        )
        iso_forest.fit(X)
        self.models["IsolationForest"] = iso_forest

        # SGDOneClassSVM : approximation linéaire par descente de gradient
        # stochastique, complexité O(n) en mémoire et en temps par époque,
        # contrairement à OneClassSVM (noyau RBF) qui est O(n^2) en mémoire
        # (matrice de Gram complète) — inadapté à une machine contrainte.
        sgd_ocsvm = SGDOneClassSVM(random_state=self.random_state)
        sgd_ocsvm.fit(X)
        self.models["SGDOneClassSVM"] = sgd_ocsvm

    def train_supervised(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entraîne RandomForest et XGBoost sur des données labellisées (0/1).

        Args:
            X_train: Matrice 2D (n_samples, n_features).
            y_train: Vecteur 1D de labels binaires (0=normal, 1=attaque).

        Raises:
            ClassicalModelsPipelineError: Dimensions invalides ou incohérentes
                entre `X_train` et `y_train`.
        """
        X = self._validate_2d(X_train, "X_train")
        y = np.asarray(y_train)
        if len(y) != X.shape[0]:
            raise ClassicalModelsPipelineError(
                f"y_train ({len(y)} échantillons) incohérent avec X_train ({X.shape[0]} échantillons)."
            )

        random_forest = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=self.random_state,
        )
        random_forest.fit(X, y)
        self.models["RandomForest"] = random_forest

        # tree_method='hist' : construction des arbres par histogrammes de
        # features plutôt que par tri exact des splits -> nettement plus
        # rapide et moins gourmand en RAM sur CPU, au prix d'une précision de
        # split marginalement réduite (négligeable ici avec max_depth=6).
        xgb_classifier = XGBClassifier(
            n_estimators=50,
            max_depth=6,
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        xgb_classifier.fit(X, y)
        self.models["XGBoost"] = xgb_classifier

    # ------------------------------------------------------------------ #
    # Prédiction
    # ------------------------------------------------------------------ #
    def predict_all(self, X_test: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Applique tous les modèles entraînés et renvoie prédictions + scores.

        Args:
            X_test: Matrice 2D (n_samples, n_features) à évaluer.

        Returns:
            Dictionnaire {nom_modèle: {"y_pred": ..., "scores": ...}} où :
                - y_pred: tableau 1D binaire (0=normal, 1=anomalie/attaque).
                - scores: tableau 1D de score brut, où une valeur plus élevée
                  indique une anomalie plus probable (convention unifiée entre
                  modèles non supervisés et supervisés).

        Raises:
            ClassicalModelsPipelineError: Si aucun modèle n'a été entraîné.
        """
        if not self.models:
            raise ClassicalModelsPipelineError(
                "Aucun modèle entraîné. Appelez train_unsupervised()/train_supervised() avant predict_all()."
            )

        X = self._validate_2d(X_test, "X_test")
        results: Dict[str, Dict[str, np.ndarray]] = {}

        for name, model in self.models.items():
            if name in _UNSUPERVISED_MODELS:
                raw_pred = model.predict(X)  # {-1: anomalie, 1: normal}
                y_pred = (raw_pred == -1).astype(int)
                # decision_function : positif = normal, négatif = anomalie.
                # On inverse le signe pour respecter la convention "score élevé = anomalie".
                scores = -model.decision_function(X)
            elif name in _SUPERVISED_MODELS:
                y_pred = model.predict(X).astype(int)
                scores = model.predict_proba(X)[:, 1]
            else:  # pragma: no cover - garde-fou si un modèle non répertorié est ajouté
                raise ClassicalModelsPipelineError(f"Modèle non reconnu pour la prédiction : '{name}'")

            results[name] = {"y_pred": y_pred, "scores": scores}

        return results

    # ------------------------------------------------------------------ #
    # Persistance
    # ------------------------------------------------------------------ #
    def save_models(self, output_dir: Union[str, Path] = "models/") -> None:
        """Sauvegarde tous les modèles entraînés (un fichier joblib par modèle).

        Args:
            output_dir: Répertoire de destination (créé si absent).

        Raises:
            ClassicalModelsPipelineError: Si aucun modèle n'a été entraîné.
        """
        if not self.models:
            raise ClassicalModelsPipelineError("Aucun modèle à sauvegarder.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, output_dir / f"{name}.joblib")

    def load_models(self, input_dir: Union[str, Path] = "models/") -> None:
        """Charge tous les modèles reconnus présents dans `input_dir`.

        Args:
            input_dir: Répertoire contenant les fichiers `<NomModèle>.joblib`.

        Raises:
            ClassicalModelsPipelineError: Répertoire introuvable ou aucun
                modèle reconnu trouvé.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise ClassicalModelsPipelineError(f"Répertoire introuvable : {input_dir}")

        known_names = _UNSUPERVISED_MODELS + _SUPERVISED_MODELS
        loaded_any = False
        for name in known_names:
            model_path = input_dir / f"{name}.joblib"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                loaded_any = True

        if not loaded_any:
            raise ClassicalModelsPipelineError(
                f"Aucun modèle reconnu trouvé dans {input_dir} (attendu parmi {known_names})."
            )

    # ------------------------------------------------------------------ #
    # Utilitaires internes
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_2d(X: np.ndarray, arg_name: str) -> np.ndarray:
        """Valide et caste une matrice en float32 (empreinte mémoire réduite de moitié vs float64).

        Args:
            X: Tableau à valider.
            arg_name: Nom de l'argument, pour les messages d'erreur.

        Returns:
            Le tableau converti en float32.

        Raises:
            ClassicalModelsPipelineError: Si `X` n'est pas 2D ou est vide.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ClassicalModelsPipelineError(f"{arg_name} doit être 2D, reçu un tableau de dimension {X.ndim}.")
        if X.shape[0] == 0:
            raise ClassicalModelsPipelineError(f"{arg_name} est vide (0 échantillon).")
        return X.astype(np.float32)


# ---------------------------------------------------------------------- #
# Test local : simulation complète sur données factices
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    rng = np.random.default_rng(seed=42)
    n_samples = 5000
    n_features = 10

    # Données saines : distribution gaussienne stable par capteur.
    X_normal_pool = rng.normal(loc=50.0, scale=5.0, size=(n_samples, n_features))

    # Jeu d'entraînement : 100% sain (méthodologie semi-supervisée).
    n_train = 3500
    X_train_normal = X_normal_pool[:n_train]

    # Jeu de test : le reste des données saines + 5% d'attaques injectées
    # (décalage de distribution simulant un comportement anormal du processus).
    X_test_normal = X_normal_pool[n_train:]
    n_test_normal = X_test_normal.shape[0]
    n_attacks = int(0.05 * n_test_normal / 0.95)  # pour obtenir ~5% d'attaques dans le test final
    X_test_attacks = rng.normal(loc=75.0, scale=8.0, size=(n_attacks, n_features))

    X_test = np.vstack([X_test_normal, X_test_attacks])
    y_test = np.concatenate([np.zeros(n_test_normal, dtype=int), np.ones(n_attacks, dtype=int)])

    # Mélange pour ne pas laisser toutes les attaques regroupées en fin de jeu de test.
    shuffle_idx = rng.permutation(len(X_test))
    X_test, y_test = X_test[shuffle_idx], y_test[shuffle_idx]

    print(f"Train (normal only): {X_train_normal.shape} | Test: {X_test.shape} "
          f"({y_test.sum()} attaques / {len(y_test)}, {100 * y_test.mean():.2f}%)\n")

    pipeline = ClassicalModelsPipeline(random_state=42)

    print("--- Entraînement non supervisé (IsolationForest, SGDOneClassSVM) ---")
    t0 = time.perf_counter()
    pipeline.train_unsupervised(X_train_normal)
    print(f"  Terminé en {time.perf_counter() - t0:.3f}s")

    # Pour les modèles supervisés, on a besoin d'un jeu labellisé : on réutilise
    # une petite fraction de train (normal) + quelques attaques factices supplémentaires.
    n_train_attacks = 100
    X_train_attacks = rng.normal(loc=75.0, scale=8.0, size=(n_train_attacks, n_features))
    X_train_sup = np.vstack([X_train_normal, X_train_attacks])
    y_train_sup = np.concatenate([np.zeros(n_train, dtype=int), np.ones(n_train_attacks, dtype=int)])

    print("\n--- Entraînement supervisé (RandomForest, XGBoost) ---")
    t0 = time.perf_counter()
    pipeline.train_supervised(X_train_sup, y_train_sup)
    print(f"  Terminé en {time.perf_counter() - t0:.3f}s")

    print("\n--- Prédiction sur le jeu de test (4 modèles) ---")
    t0 = time.perf_counter()
    results = pipeline.predict_all(X_test)
    print(f"  Terminé en {time.perf_counter() - t0:.3f}s")
    for name, out in results.items():
        print(f"  {name:16s} -> y_pred: {out['y_pred'].shape}, scores: {out['scores'].shape}, "
              f"anomalies détectées: {int(out['y_pred'].sum())}")

    print("\n--- Persistance (save_models / load_models) ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_models(tmp_dir)
        reloaded = ClassicalModelsPipeline()
        reloaded.load_models(tmp_dir)
        print(f"  Modèles rechargés : {list(reloaded.models.keys())}")
        reloaded_results = reloaded.predict_all(X_test)
        assert np.array_equal(
            reloaded_results["RandomForest"]["y_pred"], results["RandomForest"]["y_pred"]
        ), "Les prédictions doivent être identiques après rechargement."
        print("  OK — prédictions identiques avant/après rechargement.")

    print("\nOK — pipeline models.py validé de bout en bout.")
