"""Prétraitement du dataset HAI pour ML classique (tabulaire) et Deep Learning (séquentiel).

Ce module fournit `HAIPreprocessor`, qui prend en entrée le DataFrame nettoyé par
`HAIDataLoader` (voir data_loader.py) et produit :
  1. Des matrices tabulaires 2D normalisées, pour Isolation Forest / Random Forest / XGBoost.
  2. Des tenseurs 3D par fenêtrage glissant, pour LSTM Autoencoder / GRU / 1D-CNN.

Conformément à la méthodologie FactoryShield-OT (apprentissage semi-supervisé sur
données saines uniquement), `fit_transform_tabular` permet de restreindre le fit
du scaler aux échantillons normaux (fit_on_normal_only=True) tout en transformant
l'ensemble des données fournies.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ScalerType = Literal["standard", "minmax"]


class HAIPreprocessorError(Exception):
    """Erreur levée pour tout problème de prétraitement ou de dimensionnement."""


class HAIPreprocessor:
    """Prétraite les données HAI pour les pipelines ML classique et Deep Learning.

    Attributes:
        scaler_type: Type de normalisation utilisé ("standard" ou "minmax").
        scaler: Instance scikit-learn du scaler, ajustée après le premier
            appel à `fit_transform_tabular` avec `is_train=True`.
        feature_cols_: Liste des colonnes de features sur lesquelles le scaler
            a été ajusté (mémorisée pour valider la cohérence à l'inférence).
    """

    def __init__(self, scaler_type: ScalerType = "standard") -> None:
        """Initialise le préprocesseur.

        Args:
            scaler_type: "standard" pour StandardScaler (moyenne 0, écart-type 1),
                "minmax" pour MinMaxScaler (bornage [0, 1]).

        Raises:
            HAIPreprocessorError: Si `scaler_type` n'est ni "standard" ni "minmax".
        """
        if scaler_type not in ("standard", "minmax"):
            raise HAIPreprocessorError(
                f"scaler_type invalide : '{scaler_type}'. Attendu 'standard' ou 'minmax'."
            )
        self.scaler_type: ScalerType = scaler_type
        self.scaler: TransformerMixin = (
            StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        )
        self.feature_cols_: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Prétraitement tabulaire (ML classique)
    # ------------------------------------------------------------------ #
    def fit_transform_tabular(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
        is_train: bool = True,
        fit_on_normal_only: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Impute, normalise et retourne les matrices tabulaires X et y.

        Args:
            df: DataFrame source (typiquement `HAIDataLoader.df`), trié
                chronologiquement.
            feature_cols: Colonnes de capteurs à utiliser comme features. Si
                None, toutes les colonnes numériques hors `label_col` sont
                utilisées.
            label_col: Nom de la colonne de label binaire (0/1). Optionnelle
                — si absente, `y` retourné vaut None (cas inférence pure).
            is_train: Si True, ajuste (fit) le scaler sur ces données puis
                transforme. Si False, réutilise un scaler déjà ajusté
                (transform seul) — utile en inférence temps réel.
            fit_on_normal_only: Si True (et is_train=True et label_col fourni),
                le scaler est ajusté uniquement sur les lignes normales
                (label == 0), pertinent pour les modèles non supervisés
                entraînés sur la "normalité thermodynamique". La transformation
                finale s'applique néanmoins à tout `df`.

        Returns:
            Tuple (X_scaled, y) :
                - X_scaled: matrice 2D (n_samples, n_features), dtype float64.
                - y: vecteur 1D de labels (dtype int) si `label_col` est fourni,
                  sinon None.

        Raises:
            HAIPreprocessorError: Colonnes manquantes, is_train=False avant
                tout fit préalable, ou incompatibilité de colonnes entre
                fit et transform.
        """
        if feature_cols is None:
            candidate_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if label_col in candidate_cols:
                candidate_cols.remove(label_col)
            feature_cols = candidate_cols

        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise HAIPreprocessorError(f"Colonnes de features absentes du DataFrame : {missing_cols}")
        if label_col is not None and label_col not in df.columns:
            raise HAIPreprocessorError(f"Colonne de label absente du DataFrame : '{label_col}'")

        # Imputation temporelle : forward-fill puis backward-fill (adaptée aux
        # séries temporelles industrielles, évite d'introduire des valeurs
        # physiquement incohérentes comme une moyenne globale le ferait).
        features_df = df[feature_cols].ffill().bfill()

        if features_df.isna().any().any():
            still_nan = features_df.columns[features_df.isna().any()].tolist()
            raise HAIPreprocessorError(
                f"Colonnes entièrement vides (ffill/bfill inefficaces) : {still_nan}"
            )

        y: Optional[np.ndarray] = None
        if label_col is not None:
            y = df[label_col].fillna(0).to_numpy().astype(int)

        if is_train:
            if fit_on_normal_only:
                if y is None:
                    raise HAIPreprocessorError(
                        "fit_on_normal_only=True requiert un label_col pour identifier les échantillons normaux."
                    )
                normal_mask = y == 0
                if not normal_mask.any():
                    raise HAIPreprocessorError("Aucun échantillon normal (label == 0) trouvé pour le fit.")
                self.scaler.fit(features_df.loc[normal_mask])
            else:
                self.scaler.fit(features_df)
            self.feature_cols_ = list(feature_cols)
        else:
            if self.feature_cols_ is None:
                raise HAIPreprocessorError(
                    "is_train=False mais aucun scaler n'a été ajusté au préalable "
                    "(appelez fit_transform_tabular avec is_train=True, ou load_scaler())."
                )
            if list(feature_cols) != self.feature_cols_:
                raise HAIPreprocessorError(
                    "Les colonnes de features fournies ne correspondent pas à celles du fit : "
                    f"attendu {self.feature_cols_}, reçu {list(feature_cols)}"
                )

        X_scaled = self.scaler.transform(features_df).astype(np.float64)
        return X_scaled, y

    # ------------------------------------------------------------------ #
    # Fenêtrage temporel 3D (Deep Learning)
    # ------------------------------------------------------------------ #
    def create_sliding_windows(
        self,
        X_scaled: np.ndarray,
        y: Optional[np.ndarray] = None,
        window_size: int = 30,
        step: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transforme une matrice 2D en tenseur 3D de fenêtres glissantes.

        Utilise `numpy.lib.stride_tricks.sliding_window_view` pour générer les
        fenêtres sans dupliquer les données en mémoire (vue par strides) ; seule
        la sortie finale (de taille égale au tenseur produit) est matérialisée,
        ce qui évite tout pic mémoire intermédiaire lors du découpage de gros
        fichiers CSV.

        Args:
            X_scaled: Matrice 2D (n_samples, n_features), typiquement la sortie
                de `fit_transform_tabular`.
            y: Vecteur 1D de labels par échantillon (0/1), optionnel. Une
                fenêtre est étiquetée anormale (1) si au moins une mesure
                qu'elle contient est anormale.
            window_size: Longueur de chaque fenêtre temporelle.
            step: Pas de glissement entre deux fenêtres consécutives
                (step=1 : chevauchement maximal).

        Returns:
            Tuple (windows, y_windows) :
                - windows: tenseur 3D (num_windows, window_size, num_features).
                - y_windows: vecteur 1D (num_windows,) de labels de séquence,
                  ou None si `y` n'est pas fourni.

        Raises:
            HAIPreprocessorError: Dimensions invalides (X_scaled non-2D,
                window_size/step non positifs, window_size > n_samples,
                ou longueur de `y` incohérente avec `X_scaled`).
        """
        if X_scaled.ndim != 2:
            raise HAIPreprocessorError(f"X_scaled doit être 2D, reçu un tableau de dimension {X_scaled.ndim}.")
        if window_size <= 0 or step <= 0:
            raise HAIPreprocessorError("window_size et step doivent être des entiers strictement positifs.")

        n_samples, _n_features = X_scaled.shape
        if window_size > n_samples:
            raise HAIPreprocessorError(
                f"window_size ({window_size}) > nombre d'échantillons disponibles ({n_samples})."
            )
        if y is not None and len(y) != n_samples:
            raise HAIPreprocessorError(
                f"Longueur de y ({len(y)}) incohérente avec X_scaled ({n_samples} échantillons)."
            )

        # sliding_window_view(X, window_size, axis=0) -> shape (n_samples - window_size + 1, n_features, window_size)
        raw_windows = sliding_window_view(X_scaled, window_shape=window_size, axis=0)
        # Réordonnancement en (num_windows, window_size, num_features), format attendu par les frameworks DL.
        windows = np.swapaxes(raw_windows, 1, 2)[::step]
        windows = np.ascontiguousarray(windows)

        y_windows: Optional[np.ndarray] = None
        if y is not None:
            y_arr = np.asarray(y)
            raw_label_windows = sliding_window_view(y_arr, window_shape=window_size, axis=0)
            label_windows = raw_label_windows[::step]
            y_windows = (label_windows.max(axis=1) > 0).astype(int)

        return windows, y_windows

    # ------------------------------------------------------------------ #
    # Persistance du scaler
    # ------------------------------------------------------------------ #
    def save_scaler(self, filepath: Union[str, Path]) -> None:
        """Sauvegarde le scaler ajusté (et les métadonnées de colonnes) via joblib.

        Args:
            filepath: Chemin de destination du fichier (ex: 'artifacts/scaler.joblib').

        Raises:
            HAIPreprocessorError: Si aucun scaler n'a encore été ajusté.
        """
        if self.feature_cols_ is None:
            raise HAIPreprocessorError("Impossible de sauvegarder : le scaler n'a pas encore été ajusté.")

        payload = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "feature_cols": self.feature_cols_,
        }
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, filepath)

    def load_scaler(self, filepath: Union[str, Path]) -> None:
        """Charge un scaler préalablement sauvegardé via `save_scaler`.

        Args:
            filepath: Chemin du fichier joblib à charger.

        Raises:
            HAIPreprocessorError: Fichier introuvable ou contenu invalide.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise HAIPreprocessorError(f"Fichier de scaler introuvable : {filepath}")

        try:
            payload = joblib.load(filepath)
        except Exception as exc:  # noqa: BLE001 - message explicite pour l'utilisateur
            raise HAIPreprocessorError(f"Échec du chargement du scaler '{filepath}': {exc}") from exc

        required_keys = {"scaler", "scaler_type", "feature_cols"}
        if not required_keys.issubset(payload):
            raise HAIPreprocessorError(
                f"Fichier de scaler invalide, clés manquantes : {required_keys - set(payload)}"
            )

        self.scaler = payload["scaler"]
        self.scaler_type = payload["scaler_type"]
        self.feature_cols_ = payload["feature_cols"]


# ---------------------------------------------------------------------- #
# Test local : pipeline complet sur données factices au format HAI
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(seed=7)
    n_rows = 1500
    n_sensors = 8

    sensor_cols = [f"P1_SENSOR{i:02d}" for i in range(n_sensors)]
    data = {
        col: np.sin(np.linspace(0, 30, n_rows) + i) * 5 + 20 + rng.normal(0, 0.3, n_rows)
        for i, col in enumerate(sensor_cols)
    }
    label = np.zeros(n_rows, dtype=int)
    label[400:450] = 1
    label[1000:1030] = 1
    data["Attack"] = label
    df = pd.DataFrame(data)

    # Injection de valeurs manquantes pour tester l'imputation ffill/bfill.
    df.loc[0:2, sensor_cols[0]] = np.nan
    df.loc[10, sensor_cols[1]] = np.nan

    print("--- 1. fit_transform_tabular (entraînement, fit sur normal uniquement) ---")
    preprocessor = HAIPreprocessor(scaler_type="standard")
    X_scaled, y = preprocessor.fit_transform_tabular(
        df,
        feature_cols=sensor_cols,
        label_col="Attack",
        is_train=True,
        fit_on_normal_only=True,
    )
    print(f"  X_scaled shape: {X_scaled.shape}, dtype: {X_scaled.dtype}")
    print(f"  y shape: {y.shape}, classes: {np.unique(y, return_counts=True)}")
    print(f"  Aucune valeur NaN résiduelle : {not np.isnan(X_scaled).any()}")

    print("\n--- 2. create_sliding_windows (fenêtrage pour LSTM Autoencoder) ---")
    windows, y_windows = preprocessor.create_sliding_windows(X_scaled, y=y, window_size=30, step=1)
    print(f"  windows shape (num_windows, window_size, num_features): {windows.shape}")
    print(f"  y_windows shape: {y_windows.shape}, fenêtres anormales: {int(y_windows.sum())}")

    expected_num_windows = (n_rows - 30) // 1 + 1
    assert windows.shape == (expected_num_windows, 30, n_sensors), "Dimensionnement du tenseur incorrect."
    assert y_windows is not None and y_windows.shape[0] == expected_num_windows

    print("\n--- 3. save_scaler / load_scaler (persistance joblib) ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_path = Path(tmp_dir) / "scaler.joblib"
        preprocessor.save_scaler(scaler_path)
        print(f"  Scaler sauvegardé : {scaler_path} (existe: {scaler_path.exists()})")

        reloaded = HAIPreprocessor(scaler_type="standard")
        reloaded.load_scaler(scaler_path)

        # Simulation d'inférence temps réel : transform seul (is_train=False), pas de label.
        X_infer, y_infer = reloaded.fit_transform_tabular(
            df.iloc[:50],
            feature_cols=sensor_cols,
            label_col=None,
            is_train=False,
        )
        print(f"  Inférence (is_train=False) — X_infer shape: {X_infer.shape}, y_infer: {y_infer}")
        assert np.allclose(X_infer, X_scaled[:50]), "Le scaler rechargé doit reproduire la même transformation."

    print("\n--- 4. Gestion d'erreurs (window_size trop grand) ---")
    try:
        preprocessor.create_sliding_windows(X_scaled, window_size=n_rows + 10)
    except HAIPreprocessorError as exc:
        print(f"  Exception correctement levée : {exc}")

    print("\nOK — pipeline de prétraitement validé sur données factices.")
