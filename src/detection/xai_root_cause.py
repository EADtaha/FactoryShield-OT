"""
Root-cause localization — FactoryShield-OT
Ranks per-sensor reconstruction error at each alert to isolate the injection point.
Input: the CSV produced by src/detection/inference.py (feature columns + mae_raw/
mae_smoothed/alert). Top-1 sensor = designated attack injection point.
Run: python src/detection/xai_root_cause.py --errors-csv out/scores.csv --top-k 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = {"mae_raw", "mae_smoothed", "alert"}


def alert_onsets(df: pd.DataFrame) -> np.ndarray:
    """Row positions where alert transitions 0 -> 1 (new event, not a repeated alarm)."""
    alert = df["alert"].to_numpy()
    prev = np.concatenate(([0], alert[:-1]))
    return np.where((alert == 1) & (prev == 0))[0]


def root_cause_report(df: pd.DataFrame, k: int = 5, onsets_only: bool = True) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in META_COLS]
    errors = df[feature_cols].to_numpy()

    rows = alert_onsets(df) if onsets_only else np.where(df["alert"].to_numpy() == 1)[0]
    if len(rows) == 0:
        return pd.DataFrame(columns=["timestep", "rank", "sensor", "error", "root_cause"])

    sub = errors[rows]                                   # (N_alerts, F)
    top_idx = np.argsort(-sub, axis=1)[:, :k]             # vectorized per-row top-k
    top_sensors = np.array(feature_cols)[top_idx]         # (N_alerts, k)
    top_errors = np.take_along_axis(sub, top_idx, axis=1)  # (N_alerts, k)

    report = pd.DataFrame({
        "timestep": np.repeat(rows, k),
        "rank": np.tile(np.arange(1, k + 1), len(rows)),
        "sensor": top_sensors.ravel(),
        "error": top_errors.ravel(),
    })
    report["root_cause"] = report["rank"] == 1
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FactoryShield-OT root-cause localization")
    p.add_argument("--errors-csv", type=Path, required=True, help="Output of inference.py")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--all-alerts", action="store_true",
                    help="Report every alerted row instead of onsets only")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.errors_csv)
    report = root_cause_report(df, k=args.top_k, onsets_only=not args.all_alerts)

    if args.output:
        report.to_csv(args.output, index=False)
        print(f"saved -> {args.output}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
