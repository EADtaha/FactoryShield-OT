"""Chargement et inspection du dataset HAI (HIL-based Augmented ICS Security) v23.05.

Ce module fournit `HAIDataLoader`, responsable exclusivement de l'ingestion brute
des fichiers CSV du dataset HAI (turbines à vapeur, hydroélectricité, etc.) et de
leur inspection initiale. Il ne fait aucune transformation numérique (imputation,
normalisation, fenêtrage) : cette responsabilité appartient à `HAIPreprocessor`
(voir preprocessing.py), conformément à la séparation Chargement / Prétraitement
du CDC (Sections 7.1 et 7.2).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Toute colonne dont le nom contient ce motif (insensible à la casse) est
# considérée comme une colonne de label d'attaque (ex: Attack, Attack_P1,
# Attack_P2, Attack_P3, attack_label...).
_LABEL_PATTERN = re.compile(r"attack", re.IGNORECASE)
_TIMESTAMP_CANDIDATES = ("timestamp", "time")


class HAIDataLoaderError(Exception):
    """Erreur levée pour tout problème de chargement ou de structure du dataset HAI."""


class HAIDataLoader:
    """Charge un fichier CSV du dataset HAI et expose des méthodes d'inspection.

    Attributes:
        filepath: Chemin du fichier CSV source.
        df: DataFrame chargé, nettoyé et trié chronologiquement.
        timestamp_col: Nom réel de la colonne d'horodatage détectée.
        label_cols: Liste des colonnes de labels d'attaque détectées.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        """Initialise le loader et charge immédiatement le fichier.

        Args:
            filepath: Chemin vers le fichier CSV du dataset HAI.

        Raises:
            HAIDataLoaderError: Si le fichier n'existe pas, est vide, ou si
                aucune colonne d'horodatage n'est détectée.
        """
        self.filepath = Path(filepath)
        self.timestamp_col: str = ""
        self.label_cols: List[str] = []
        self.df: pd.DataFrame = self._load()

    # ------------------------------------------------------------------ #
    # Chargement
    # ------------------------------------------------------------------ #
    def _load(self) -> pd.DataFrame:
        """Charge, nettoie les noms de colonnes, convertit le timestamp et trie.

        Returns:
            Le DataFrame nettoyé et trié chronologiquement.

        Raises:
            HAIDataLoaderError: Fichier introuvable, vide, ou timestamp absent.
        """
        if not self.filepath.exists():
            raise HAIDataLoaderError(f"Fichier introuvable : {self.filepath}")

        try:
            df = pd.read_csv(self.filepath)
        except pd.errors.EmptyDataError as exc:
            raise HAIDataLoaderError(f"Fichier CSV vide : {self.filepath}") from exc
        except Exception as exc:  # noqa: BLE001 - on veut un message explicite
            raise HAIDataLoaderError(
                f"Échec de lecture du CSV '{self.filepath}': {exc}"
            ) from exc

        if df.empty:
            raise HAIDataLoaderError(f"Le fichier '{self.filepath}' ne contient aucune ligne.")

        # Nettoyage des noms de colonnes (espaces superflus en début/fin).
        df.columns = [str(c).strip() for c in df.columns]

        # Détection de la colonne timestamp.
        timestamp_col = self._detect_timestamp_column(df.columns)
        if timestamp_col is None:
            raise HAIDataLoaderError(
                "Aucune colonne 'timestamp' détectée dans le fichier. "
                f"Colonnes disponibles : {list(df.columns)}"
            )
        self.timestamp_col = timestamp_col

        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")
        n_unparsed = df[self.timestamp_col].isna().sum()
        if n_unparsed > 0:
            raise HAIDataLoaderError(
                f"{n_unparsed} valeurs de la colonne '{self.timestamp_col}' n'ont "
                "pas pu être converties en datetime64[ns]."
            )

        df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)

        # Détection des colonnes de label d'attaque.
        self.label_cols = [c for c in df.columns if _LABEL_PATTERN.search(c)]

        return df

    @staticmethod
    def _detect_timestamp_column(columns: List[str]) -> Optional[str]:
        """Trouve la colonne d'horodatage parmi les candidats connus (insensible à la casse)."""
        lower_map = {c.lower(): c for c in columns}
        for candidate in _TIMESTAMP_CANDIDATES:
            if candidate in lower_map:
                return lower_map[candidate]
        return None

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #
    def get_summary(self) -> Dict[str, Any]:
        """Résume le dataset chargé.

        Returns:
            Dictionnaire contenant :
                - n_samples: nombre total d'échantillons.
                - n_columns: nombre total de colonnes.
                - sensor_columns: liste des capteurs/variables industrielles.
                - label_columns: colonnes de labels d'attaque détectées.
                - class_distribution: pourcentages Normal (0) / Anormal (1),
                  calculés sur le label global (OR logique de toutes les
                  colonnes d'attaque). None si aucune colonne de label.
                - time_range: (début, fin) de la plage temporelle couverte.
        """
        summary: Dict[str, Any] = {
            "n_samples": int(len(self.df)),
            "n_columns": int(self.df.shape[1]),
            "sensor_columns": self.get_sensor_columns(),
            "label_columns": list(self.label_cols),
            "class_distribution": None,
            "time_range": (
                self.df[self.timestamp_col].min(),
                self.df[self.timestamp_col].max(),
            ),
        }

        if self.label_cols:
            global_label = self._global_attack_label()
            n_anomalous = int(global_label.sum())
            n_total = len(global_label)
            n_normal = n_total - n_anomalous
            summary["class_distribution"] = {
                "normal_pct": round(100.0 * n_normal / n_total, 4) if n_total else 0.0,
                "anomalous_pct": round(100.0 * n_anomalous / n_total, 4) if n_total else 0.0,
                "n_normal": n_normal,
                "n_anomalous": n_anomalous,
            }

        return summary

    def get_sensor_columns(self) -> List[str]:
        """Renvoie les colonnes prédictives (capteurs/actionneurs), hors timestamp et labels.

        Returns:
            Liste ordonnée des noms de colonnes de capteurs.
        """
        excluded = {self.timestamp_col, *self.label_cols}
        return [c for c in self.df.columns if c not in excluded]

    def check_missing_values(self) -> pd.DataFrame:
        """Résume les valeurs manquantes ou aberrantes (inf/-inf) par colonne.

        Returns:
            DataFrame indexé par nom de colonne avec les colonnes :
                - n_missing: nombre de NaN.
                - pct_missing: pourcentage de NaN.
                - n_infinite: nombre de valeurs infinies (colonnes numériques
                  uniquement, 0 sinon).
        """
        n_total = len(self.df)
        rows = []
        for col in self.df.columns:
            series = self.df[col]
            n_missing = int(series.isna().sum())
            n_infinite = (
                int(np.isinf(series.to_numpy(dtype=float, na_value=0.0)).sum())
                if pd.api.types.is_numeric_dtype(series)
                else 0
            )
            rows.append(
                {
                    "column": col,
                    "n_missing": n_missing,
                    "pct_missing": round(100.0 * n_missing / n_total, 4) if n_total else 0.0,
                    "n_infinite": n_infinite,
                }
            )

        result = pd.DataFrame(rows).set_index("column")
        return result

    def _global_attack_label(self) -> pd.Series:
        """Combine toutes les colonnes de labels en un label binaire global (OR logique).

        Returns:
            Series binaire (0/1) : 1 si au moins une colonne de label indique une attaque.

        Raises:
            HAIDataLoaderError: Si aucune colonne de label n'est disponible.
        """
        if not self.label_cols:
            raise HAIDataLoaderError("Aucune colonne de label d'attaque disponible dans ce fichier.")
        combined = (self.df[self.label_cols].fillna(0) != 0).any(axis=1).astype(int)
        return combined

    def get_global_labels(self) -> np.ndarray:
        """Expose le label binaire global (0=normal, 1=anormal) sous forme de tableau NumPy.

        Returns:
            Tableau 1D de labels binaires, aligné sur l'ordre chronologique du DataFrame.
        """
        return self._global_attack_label().to_numpy()


# ---------------------------------------------------------------------- #
# Test local avec données factices au format HAI
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(seed=42)
    n_rows = 2000
    n_sensors = 12

    dummy_timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="1s")
    dummy_data = {
        " timestamp ": dummy_timestamps,  # espaces volontaires pour tester le nettoyage
    }
    for i in range(n_sensors):
        # Signaux quasi-sinusoïdaux + bruit, pour simuler des grandeurs physiques (pression, débit...).
        dummy_data[f"P1_{['FCV', 'PIT', 'LIT', 'FT'][i % 4]}{i:02d}"] = (
            np.sin(np.linspace(0, 40, n_rows) + i) * 10 + 50 + rng.normal(0, 0.5, n_rows)
        )

    # Fenêtres d'attaque simulées : quelques segments contigus étiquetés à 1.
    attack_p1 = np.zeros(n_rows, dtype=int)
    attack_p1[300:340] = 1
    attack_p1[1200:1260] = 1
    attack_p2 = np.zeros(n_rows, dtype=int)
    attack_p2[900:930] = 1
    dummy_data["Attack_P1"] = attack_p1
    dummy_data["Attack_P2"] = attack_p2
    dummy_data["Attack"] = ((attack_p1 + attack_p2) > 0).astype(int)

    # Injection volontaire de quelques valeurs manquantes pour tester check_missing_values().
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.loc[5:8, dummy_df.columns[1]] = np.nan

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        dummy_df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    print(f"--- Fichier factice généré : {tmp_path} ---\n")

    loader = HAIDataLoader(tmp_path)

    print(">>> get_summary()")
    for key, value in loader.get_summary().items():
        print(f"  {key}: {value}")

    print("\n>>> get_sensor_columns() (5 premières)")
    print(" ", loader.get_sensor_columns()[:5])

    print("\n>>> check_missing_values() (colonnes avec valeurs manquantes)")
    missing = loader.check_missing_values()
    print(missing[missing["n_missing"] > 0])

    print("\n>>> get_global_labels() (shape)")
    print(" ", loader.get_global_labels().shape)

    Path(tmp_path).unlink(missing_ok=True)
    print("\nOK — pipeline de chargement validé sur données factices.")
