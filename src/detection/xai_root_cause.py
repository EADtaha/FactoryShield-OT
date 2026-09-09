"""
Root-cause localisation — FactoryShield-OT
===========================================
Ranks per-sensor reconstruction error at every alert onset to isolate the
attack injection point.  No external explainability library required.

Algorithm
---------
1. Find rows where alert transitions 0 → 1 (onset detection).
2. For each onset, extract the per-sensor error vector E_{t,f}.
3. Sort descending → top-K sensors are the most anomalous.
4. Rank-1 sensor = designated attack entry point.
5. Add pct_contribution (% share of total error) and subsystem label.

Input:  CSV produced by src/detection/inference.py
        Columns: <feature_1> … <feature_F>  mae_raw  mae_smoothed  alert

Output: DataFrame with columns
        timestep | rank | sensor | error | pct_contribution | subsystem | root_cause

Usage:
    python src/detection/xai_root_cause.py --errors-csv out/scores.csv --top-k 5
    python src/detection/xai_root_cause.py --errors-csv out/scores.csv --all-alerts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Subsystem map  (tag prefix → physical process)
# Covers Emerson Ovation DCS P1 Boiler control loops + P2 Turbine + P3 Water
# ─────────────────────────────────────────────────────────────────────────────
SUBSYSTEM_MAP: dict[str, str] = {
    # P1 — Pressure Control Loop
    "PIT01":  "P1 – Pressure Control",
    "PCV01D": "P1 – Pressure Control",
    "PCV02D": "P1 – Pressure Control",
    "B2016":  "P1 – Pressure Control",
    # P1 — Level Control Loop
    "LIT01":  "P1 – Level Control",
    "LCV01D": "P1 – Level Control",
    "B3004":  "P1 – Level Control",
    # P1 — Flow Control Loop
    "FT03":   "P1 – Flow Control",
    "FCV03D": "P1 – Flow Control",
    "B3005":  "P1 – Flow Control",
    # P1 — Temperature Control Loop
    "TIT01":  "P1 – Temperature Control",
    "FCV01D": "P1 – Temperature Control",
    "FCV02D": "P1 – Temperature Control",
    "B4022":  "P1 – Temperature Control",
    # P1 — Cooling Control
    "TIT02":  "P1 – Cooling Control",
    "TIT03":  "P1 – Cooling Control",
    "PP04":   "P1 – Cooling Control",
    # P2 — GE Mark VIe Steam Turbine
    "GT01":   "P2 – Turbine",
    "GT02":   "P2 – Turbine",
    "GT03":   "P2 – Turbine",
    "SPD01":  "P2 – Turbine",
    "SPD02":  "P2 – Turbine",
    "VLV_T":  "P2 – Turbine",
    # P3 — Siemens S7-300 Water Treatment
    "FT01":   "P3 – Water Treatment",
    "FT02":   "P3 – Water Treatment",
    "PT01":   "P3 – Water Treatment",
    "PT02":   "P3 – Water Treatment",
    "LT01":   "P3 – Water Treatment",
    "PMP01":  "P3 – Water Treatment",
    "PMP02":  "P3 – Water Treatment",
    # P4 — dSPACE SCALEXIO HIL Simulator
    "HIL01":  "P4 – HIL Simulator",
    "HIL02":  "P4 – HIL Simulator",
}

_META_COLS = {"mae_raw", "mae_smoothed", "alert"}


def _tag_subsystem(sensor: str) -> str:
    """Return the physical subsystem for a sensor tag, or 'Unknown'."""
    # exact match first
    if sensor in SUBSYSTEM_MAP:
        return SUBSYSTEM_MAP[sensor]
    # prefix match (handles numbered variants like PIT01A, FCV03D_OUT …)
    for prefix, subsystem in SUBSYSTEM_MAP.items():
        if sensor.startswith(prefix):
            return subsystem
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Alert onset detection
# ─────────────────────────────────────────────────────────────────────────────
def alert_onsets(df: pd.DataFrame) -> np.ndarray:
    """Row indices where alert transitions 0 → 1 (new event start)."""
    alert = df["alert"].to_numpy()
    prev  = np.concatenate(([0], alert[:-1]))
    return np.where((alert == 1) & (prev == 0))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main report builder
# ─────────────────────────────────────────────────────────────────────────────
def root_cause_report(
    df: pd.DataFrame,
    k: int = 5,
    onsets_only: bool = True,
) -> pd.DataFrame:
    """Build the top-K root-cause report for all alert rows or onset rows only.

    Args:
        df:          Output DataFrame from inference.py (feature cols + meta cols).
        k:           Number of top sensors to return per timestep.
        onsets_only: If True analyse only onset rows (0→1 transitions).
                     If False analyse every alerted row (can be large).

    Returns:
        DataFrame with columns:
            timestep | rank | sensor | error | pct_contribution | subsystem | root_cause
    """
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    if not feature_cols:
        raise ValueError(
            "No feature columns found in the input DataFrame. "
            "Expected columns produced by inference.py."
        )

    errors = df[feature_cols].to_numpy(dtype=np.float32)

    rows: np.ndarray = alert_onsets(df) if onsets_only else np.where(df["alert"].to_numpy() == 1)[0]

    if len(rows) == 0:
        return pd.DataFrame(
            columns=["timestep", "rank", "sensor", "error",
                     "pct_contribution", "subsystem", "root_cause"]
        )

    k = min(k, len(feature_cols))
    sub       = errors[rows]                               # (N_alerts, F)
    top_idx   = np.argsort(-sub, axis=1)[:, :k]            # (N_alerts, k)
    top_names = np.array(feature_cols)[top_idx]            # (N_alerts, k)
    top_errs  = np.take_along_axis(sub, top_idx, axis=1)   # (N_alerts, k)

    # percentage contribution of each top-k sensor relative to its row total
    row_totals = sub.sum(axis=1, keepdims=True)             # (N_alerts, 1)
    row_totals = np.where(row_totals == 0, 1.0, row_totals) # avoid /0
    top_pcts   = (top_errs / row_totals) * 100.0            # (N_alerts, k)

    report = pd.DataFrame({
        "timestep":         np.repeat(rows, k),
        "rank":             np.tile(np.arange(1, k + 1), len(rows)),
        "sensor":           top_names.ravel(),
        "error":            top_errs.ravel().round(6),
        "pct_contribution": top_pcts.ravel().round(2),
    })

    report["subsystem"]  = report["sensor"].map(_tag_subsystem)
    report["root_cause"] = report["rank"] == 1

    return report.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FactoryShield-OT root-cause localisation")
    p.add_argument("--errors-csv", type=Path, required=True,
                   help="Output CSV from inference.py (feature cols + mae/alert)")
    p.add_argument("--top-k",      type=int,  default=5,
                   help="Number of top sensors to report per onset (default: 5)")
    p.add_argument("--all-alerts", action="store_true",
                   help="Analyse every alerted row instead of onset rows only")
    p.add_argument("--output",     type=Path, default=None,
                   help="Optional CSV path to save the report")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.errors_csv.exists():
        print(f"ERROR: scores CSV not found: {args.errors_csv}")
        print("Run inference.py first:  python src/detection/inference.py --data <path>")
        raise SystemExit(1)

    df     = pd.read_csv(args.errors_csv)
    report = root_cause_report(df, k=args.top_k, onsets_only=not args.all_alerts)

    if report.empty:
        print("No alerts found in the input — nothing to report.")
        return

    n_onsets = report["timestep"].nunique()
    print(f"Alert onsets: {n_onsets}  |  Rows in report: {len(report)}\n")
    print(report.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
