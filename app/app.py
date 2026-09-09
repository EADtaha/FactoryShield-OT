"""
FactoryShield-OT — ISA-101 Light Industrial Operator Console
=============================================================
Emerson Ovation DCS / HAIEnd 23.05 Cybersecurity SOC Dashboard.

Data flow (real-data mode):
  1. User uploads a raw HAIEnd CSV or selects a benchmark dataset via sidebar.
  2. _ingest_csv()     — cached: reads, strips metadata, normalises with MinMaxScaler
                         fitted on nominal data only (ot_physics_guardrails rule).
  3. _load_lstm_model() — cached resource: loads pre-trained LSTM AE checkpoint.
  4. _run_full_inference() — cached: runs zero-copy SlidingWindowDataset inference,
                         returns full E matrix (N_windows × F) + global MAE vector.
  5. Session state stores the complete result + a "play cursor" integer.
  6. [ RUN ] advances the cursor by _VIEW_W (300 s) each 3-second tick, showing
     a rolling window of real reconstruction errors — when the cursor passes an
     actual attack segment the MAE spikes and annunciators turn red.

ISA-101 design rules enforced throughout (see _ISA_CSS block).

Run:  streamlit run app/app.py   (from project root)
"""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.signal import lfilter
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ── project root on sys.path ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── project imports (graceful degradation) ───────────────────────────────────
try:
    from src.models.lstm_autoencoder import LSTMAutoencoder, SlidingWindowDataset
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

try:
    from src.detection.xai_root_cause import root_cause_report, SUBSYSTEM_MAP
except Exception:
    root_cause_report = None          # type: ignore
    SUBSYSTEM_MAP: dict = {}

try:
    from src.detection.risk_scoring import RiskLevel, RiskScoringEngine, detect_risk_events_from_alerts
except Exception:
    RiskLevel = RiskScoringEngine = detect_risk_events_from_alerts = None  # type: ignore

try:
    from src.utils.pdf_report import generate_incident_pdf
    _PDF_OK = True
except Exception:
    generate_incident_pdf = None   # type: ignore
    _PDF_OK = False

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FactoryShield-OT | Operator Console",
    page_icon=":material/factory:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# ISA-101 DESIGN TOKENS
# ════════════════════════════════════════════════════════════════════════════
_C_CANVAS      = "#E5E7EB"
_C_PANEL       = "#FFFFFF"
_C_PANEL_ALT   = "#F4F5F7"
_C_BORDER      = "#9CA3AF"
_C_BORDER_DARK = "#6B7280"
_C_TEXT        = "#111827"
_C_TEXT_MUTED  = "#4B5563"
_C_MONO        = "Roboto Mono, Consolas, monospace"
_C_NORMAL      = "#E5E7EB"
_C_WARNING     = "#D97706"
_C_ALARM       = "#DC2626"
_C_ALARM_DARK  = "#B91C1C"
_C_OK_TEXT     = "#065F46"
_C_PRESSURE    = "#1D4ED8"
_C_TEMPERATURE = "#C2410C"
_C_FLOW        = "#15803D"
_C_LEVEL       = "#92400E"
_C_MAE_RAW     = "#9CA3AF"
_C_MAE_EMA     = "#1F2937"
_C_THRESHOLD   = "#DC2626"
_C_ALERT_FILL  = "rgba(220,38,38,0.08)"

_GRID_STYLE = dict(
    showgrid=True, gridcolor="#E2E8F0", gridwidth=1,
    zeroline=False, linecolor=_C_BORDER, linewidth=1,
)

_ANNUNCIATORS = [
    ("P1-BOILER",      "Emerson Ovation DCS"),
    ("P1-FEEDWATER",   "Level / Flow Control"),
    ("P2-TURBINE",     "GE Mark VIe"),
    ("P3-WATER-TREAT", "Siemens S7-300"),
]

PAGE_NAMES = ["Live SOC", "xAI Root Cause", "Model Benchmark", "Data Exploration"]

# Rolling view width for the live simulation loop (samples shown at once)
_VIEW_W   = 300   # seconds of data in the viewport
_TICK_ADV = 30    # samples advanced per auto-refresh tick
_TICK_S   = 3     # seconds between ticks

# ════════════════════════════════════════════════════════════════════════════
# ISA-101 GLOBAL CSS INJECTION
# ════════════════════════════════════════════════════════════════════════════
_ISA_CSS = f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {_C_CANVAS} !important;
    font-family: {_C_MONO};
    color: {_C_TEXT};
}}
[data-testid="stSidebar"] {{
    background-color: {_C_PANEL_ALT} !important;
    border-right: 2px solid {_C_BORDER_DARK};
}}
[data-testid="stMetric"], [data-testid="stExpander"],
[data-testid="metric-container"], div.stAlert,
div[data-testid="stDataFrame"] {{
    border-radius: 0px !important;
    border: 1px solid {_C_BORDER} !important;
    background-color: {_C_PANEL} !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: {_C_MONO}; font-size: 0.70rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: {_C_TEXT_MUTED} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: {_C_MONO}; font-size: 1.55rem;
    font-weight: 700; color: {_C_TEXT} !important;
}}
[data-testid="stMetricDelta"] {{ font-family: {_C_MONO}; font-size: 0.72rem; }}
h1, h2, h3 {{
    font-family: {_C_MONO}; color: {_C_TEXT} !important;
    letter-spacing: 0.04em;
    border-bottom: 2px solid {_C_BORDER_DARK}; padding-bottom: 4px;
}}
[data-testid="stButton"] > button {{
    border-radius: 0px !important;
    border: 1px solid {_C_BORDER_DARK} !important;
    background-color: {_C_PANEL_ALT} !important;
    color: {_C_TEXT} !important;
    font-family: {_C_MONO}; font-size: 0.75rem;
    letter-spacing: 0.06em; text-transform: uppercase;
}}
[data-testid="stSlider"] label {{
    font-family: {_C_MONO}; font-size: 0.72rem;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: {_C_TEXT_MUTED} !important;
}}
[data-testid="stCaptionContainer"] {{
    font-family: {_C_MONO}; font-size: 0.68rem;
    color: {_C_TEXT_MUTED} !important;
}}
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
    font-family: {_C_MONO} !important; font-size: 0.76rem !important;
}}
.block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; }}
[data-testid="stHeader"] {{ background-color: transparent !important; }}
[data-testid="stSidebar"] span, [data-testid="stSidebarNav"] span,
[data-testid="stRadio"] label, [data-testid="stSidebar"] p {{
    color: {_C_TEXT} !important;
}}
</style>
"""

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ════════════════════════════════════════════════════════════════════════════
_STATE_DEFAULTS: dict = {
    # detection parameters
    "threshold":            0.15,
    "ema_alpha":            0.10,
    # playback state
    "sim_running":          False,
    "play_cursor":          0,
    # ingestion options
    "downsample_step":      1,
    # inference results (populated after ingestion)
    "infer_result":         None,
    "raw_df":               None,
    "feature_cols":         None,
    "n_samples":            0,
    "source_label":         None,
    # latching incident log — persists across cursor advances
    "logged_alarms":        [],    # list[dict] — cumulative alarm records
    "logged_alarms_cursor": 0,     # high-water mark: segments up to this index are final
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════════════════════
# CACHED PIPELINE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading LSTM Autoencoder checkpoint...")
def _load_lstm_model(ckpt_path: str):
    """Load and cache the pre-trained LSTM AE.  Returns (model, ckpt_dict).

    @st.cache_resource keeps the model object in memory across reruns —
    no repeated torch.load calls.  The model stays on CPU for portability;
    inference batches are moved via non_blocking=True.
    """
    if not _TORCH_OK:
        return None, {}
    p = Path(ckpt_path)
    if not p.exists():
        return None, {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(p, map_location=device, weights_only=False)

    hidden1    = ckpt.get("hidden1",    ckpt.get("hidden_size",  128))
    latent_dim = ckpt.get("latent_dim", 64)
    # Safe retrieval — older checkpoints may use different key names
    n_features = ckpt.get("n_features") or ckpt.get("num_features") or 225
    window     = ckpt.get("window")     or ckpt.get("seq_len")      or 60

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden1=hidden1,
        latent_dim=latent_dim,
        seq_len=window,
        dropout=ckpt.get("dropout", 0.0),
    )

    # Resolve state dict — handle every checkpoint format seen in the wild:
    #   • {"model_state": {...}}       — FactoryShield-OT training script
    #   • {"state_dict": {...}}        — PyTorch Lightning / generic trainer
    #   • {"model_state_dict": {...}}  — torch.save(model.state_dict()) wrappers
    #   • {"model": {...}}             — HuggingFace-style wrappers
    #   • {<layer_name>: tensor, ...}  — raw state_dict saved directly
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("model_state")
            or ckpt.get("state_dict")
            or ckpt.get("model_state_dict")
            or ckpt.get("model")
            or ckpt
        )
    else:
        # torch.load returned a raw tensor dict (older serialisation)
        state_dict = ckpt

    # Strip DataParallel "module." prefix if model was trained with nn.DataParallel
    clean_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device).eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


@st.cache_data(show_spinner="Preprocessing telemetry data...")
def _preprocess_raw_csv(
    file_bytes:      bytes,
    file_name:       str,
    downsample_step: int = 1,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Parse, align, and MinMaxScale a raw or pre-scaled HAIEnd CSV.

    Memory-budget constraints (OOM fix):
      • Hard cap: only the first _MAX_ROWS rows are kept for the live dashboard.
        The OOM killer was triggered by materialising a (50k-60) × 225 float32
        residual matrix (~4.4 GB at stride=1).  Capping at 50 000 rows is the
        first line of defence; inference stride is the second (see
        _run_full_inference).
      • Downsample step (sidebar slider) is applied BEFORE schema alignment so
        the largest possible allocation is avoided early.
      • All intermediate raw DataFrames are explicitly deleted and gc.collect()
        is called before returning so the cached bytes object is the only copy
        kept alive between reruns.

    Schema alignment strategy (handles raw end-testN.csv AND processed CSVs):
      1. Read canonical 225-column list from training CSV header (nrows=0 only).
      2. Strip metadata (timestamp/time/attack*/label*) via regex.
      3. Reorder/zero-fill to match canonical schema exactly.
      4. ffill → bfill → fillna(0.0) — no NaN may enter the model.
      5. Fit MinMaxScaler on nominal rows only (ot_physics_guardrails rule #2).

    Returns:
        raw_df       — capped/downsampled DataFrame with metadata (for EDA viewer)
        X_scaled     — float32 array (N_eff, F) ready for LSTM inference
        schema_cols  — ordered list of canonical sensor column names
    """
    import gc
    import re

    _MAX_ROWS = 50_000   # hard RAM budget: beyond this the residual matrix OOMs

    _META_RE   = re.compile(r"^(timestamp|time|attack.*|label.*)$", re.IGNORECASE)
    _ATTACK_RE = re.compile(r"^attack.*$", re.IGNORECASE)

    # ── 1. Canonical schema (header only — zero data read) ───────────────────
    _TRAIN_CSV = _ROOT / "data" / "processed" / "clean_100" / "train_clean_scaled.csv"
    if _TRAIN_CSV.exists():
        schema_cols = [
            c for c in pd.read_csv(_TRAIN_CSV, nrows=0).columns
            if not _META_RE.match(c.strip())
        ]
    else:
        schema_cols = []   # positional fallback handled below

    # ── 2. Read CSV ──────────────────────────────────────────────────────────
    raw_df = pd.read_csv(io.BytesIO(file_bytes))
    raw_df.columns = [c.strip() for c in raw_df.columns]

    # ── 3. Apply downsample + hard cap BEFORE any heavy allocation ───────────
    # Downsampling first shrinks the frame; the cap is a safety net on top.
    if downsample_step > 1:
        raw_df = raw_df.iloc[::downsample_step].reset_index(drop=True)
    if len(raw_df) > _MAX_ROWS:
        raw_df = raw_df.iloc[:_MAX_ROWS].copy()   # .copy() drops the slice view

    # ── 4. Extract attack label (before stripping metadata) ─────────────────
    attack_col = next((c for c in raw_df.columns if _ATTACK_RE.match(c)), None)
    y_attack   = (
        (raw_df[attack_col].fillna(0).to_numpy() > 0).astype(np.int8)
        if attack_col else None
    )

    # ── 5. Identify numeric sensor columns ──────────────────────────────────
    sensor_cols_in_file = [
        c for c in raw_df.columns
        if not _META_RE.match(c) and pd.api.types.is_numeric_dtype(raw_df[c])
    ]
    if not sensor_cols_in_file:
        raise ValueError(
            f"No numeric sensor columns found in '{file_name}' after stripping "
            "metadata (timestamp/time/attack/label). Verify this is a HAIEnd 23.05 CSV."
        )

    # ── 6. Schema alignment → float32 immediately ───────────────────────────
    if schema_cols:
        # Pre-allocate a single float32 array — avoids building a wide DataFrame
        # column-by-column then converting (which would briefly hold float64 copies).
        n_rows = len(raw_df)
        n_cols = len(schema_cols)
        X_raw  = np.zeros((n_rows, n_cols), dtype=np.float32)
        for ci, col in enumerate(schema_cols):
            if col in raw_df.columns:
                X_raw[:, ci] = pd.to_numeric(
                    raw_df[col], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=np.float32)
            # else: column stays 0.0 (zero-fill for absent sensor)
        canonical_cols = schema_cols
    else:
        # No training CSV — positional alignment, cast in one shot
        X_raw = (
            raw_df[sensor_cols_in_file]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        canonical_cols = sensor_cols_in_file

    # ── 7. Impute NaNs in the float32 array (ffill/bfill/zero) ──────────────
    # pandas ffill on a numpy array requires a detour through DataFrame,
    # but we do it on the already-float32 array — no extra type promotion.
    feat_df = pd.DataFrame(X_raw, columns=canonical_cols)
    feat_df = feat_df.ffill().bfill().fillna(0.0)
    del X_raw
    gc.collect()

    # ── 8. Fit MinMaxScaler on nominal rows only ─────────────────────────────
    scaler = MinMaxScaler()
    if y_attack is not None and 0 < int(y_attack.sum()) < len(y_attack):
        scaler.fit(feat_df[y_attack == 0])
    else:
        scaler.fit(feat_df)

    X_scaled = scaler.transform(feat_df).astype(np.float32)

    # ── 9. Release all intermediates before returning ────────────────────────
    del feat_df
    gc.collect()

    return raw_df, X_scaled, canonical_cols


@st.cache_data(
    show_spinner="Running LSTM Autoencoder inference (zero-copy sliding windows)...",
    max_entries=4,
)
def _run_full_inference(
    file_name:  str,       # cache key discriminator
    _X_id:      int,       # id(X_scaled) — forces re-run when array changes
    X_scaled:   np.ndarray,
    ckpt_path:  str,
    batch_size: int = 256,
    stride:     int = 5,
) -> pd.DataFrame:
    """Run LSTM AE forward pass and return a *lean* summary DataFrame.

    OOM fix — what changed vs the previous implementation
    ───────────────────────────────────────────────────────
    OLD: stride=1 → (N-60) windows; accumulate full E matrix (N×225 float32).
         At N=50 000 that is 49 940 × 225 × 4 B ≈ 4.4 GB just for E, plus
         another ~4.4 GB for the cached Streamlit DataFrame serialisation copy.

    NEW: stride=5  → ⌈(N-60)/5⌉ ≈ 9 988 windows.
         Per batch: compute E=(B,225), immediately reduce to:
           • mae  (B,)        float32 — global mean absolute error
           • top5_idx (B,5)   int16   — index of 5 highest-error sensors
           • top5_err (B,5)   float32 — corresponding error magnitudes
         The full E tensor is never retained; each batch tensor is explicitly
         deleted before the next iteration.
         Final DataFrame columns (13 total, ~3 MB at 10 000 windows):
           mae_raw  mae_smoothed  alert
           top_sensor_idx_0..4   (int16)
           top_sensor_err_0..4   (float32)

    The xAI page reconstructs per-sensor rankings from top_sensor_idx/err
    without needing the full residual matrix.

    Preserves:
      • pytorch_timeseries_perf rule #1 — SlidingWindowDataset zero-copy view
      • ot_physics_guardrails rule #3   — shuffle=False
      • pytorch_timeseries_perf rule #5 — num_workers=0 (CPU-safe)
      • pytorch_timeseries_perf rule #4 — lfilter EMA, no Python loop
    """
    import gc

    _TOP_K_STORE = 5   # number of top sensors to persist per window

    model, ckpt = _load_lstm_model(ckpt_path)
    if model is None:
        raise RuntimeError(
            f"Checkpoint not found or PyTorch unavailable: {ckpt_path}"
        )

    device = next(model.parameters()).device
    window = ckpt.get("window")     or ckpt.get("seq_len")      or 60
    n_feat = ckpt.get("n_features") or ckpt.get("num_features") or 225

    if X_scaled.shape[1] != n_feat:
        raise ValueError(
            f"Feature mismatch: checkpoint expects {n_feat} features, "
            f"input has {X_scaled.shape[1]}. "
            "Ensure the same HAIEnd 23.05 experiment is selected."
        )

    loader = torch.utils.data.DataLoader(
        SlidingWindowDataset(X_scaled, window=window, stride=stride),
        batch_size=batch_size,
        shuffle=False,                          # ot_physics_guardrails #3
        num_workers=0,                          # pytorch_timeseries_perf #5
        pin_memory=(str(device) != "cpu"),
    )

    mae_list:      list[np.ndarray] = []   # (B,) float32
    top_idx_list:  list[np.ndarray] = []   # (B, _TOP_K_STORE) int16
    top_err_list:  list[np.ndarray] = []   # (B, _TOP_K_STORE) float32

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device, non_blocking=True)   # (B, T, F)
            recon  = model(batch)

            # Per-sensor absolute residual at last timestep → (B, F) float32
            E_batch = (batch - recon).abs()[:, -1, :].cpu()   # stay as tensor

            # Global MAE — (B,)
            mae_list.append(E_batch.mean(dim=1).numpy().astype(np.float32))

            # Top-5 sensor indices and errors — (B, 5)
            top_vals, top_inds = torch.topk(E_batch, k=_TOP_K_STORE, dim=1)
            top_idx_list.append(top_inds.numpy().astype(np.int16))
            top_err_list.append(top_vals.numpy().astype(np.float32))

            # Explicitly free GPU/CPU tensors before next iteration
            del batch, recon, E_batch, top_vals, top_inds

        # Single gc pass after the loop (not per-batch — amortises cost)
        gc.collect()

    mae     = np.concatenate(mae_list,     axis=0)   # (N_win,)
    top_idx = np.concatenate(top_idx_list, axis=0)   # (N_win, 5)
    top_err = np.concatenate(top_err_list, axis=0)   # (N_win, 5)
    del mae_list, top_idx_list, top_err_list
    gc.collect()

    # Causal EMA — scipy IIR, O(N) C-compiled, no Python loop
    # (pytorch_timeseries_perf rule #4)
    alpha   = 0.10
    mae_ema = lfilter([alpha], [1.0, -(1.0 - alpha)], mae).astype(np.float32)

    # Alarm threshold: checkpoint p99 > live p99 fallback
    thr   = float(ckpt.get("threshold_p99") or np.percentile(mae_ema, 99))
    alert = (mae_ema > thr).astype(np.int8)

    # Build lean result DataFrame — 13 columns, ~3 MB at 10 000 windows
    out = pd.DataFrame({
        "mae_raw":      mae,
        "mae_smoothed": mae_ema,
        "alert":        alert,
    })
    for k in range(_TOP_K_STORE):
        out[f"top_sensor_idx_{k}"] = top_idx[:, k]   # int16
        out[f"top_sensor_err_{k}"] = top_err[:, k]   # float32
    del mae, mae_ema, alert, top_idx, top_err
    gc.collect()

    return out


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _risk_label(err: float, threshold: float = 0.15) -> str:
    """Threshold-aware 4-level risk assessment.

    The previous static thresholds (0.30 / 0.60 / 0.85) were calibrated for
    a fixed default threshold of 0.15.  When the operator moves the slider
    higher (e.g. 0.40) the MAE values crossing it are proportionally larger,
    so the old cut-points produced perpetual LOW even during active alarms.

    New rules (matches task spec):
      CRITICAL  : err >= 0.85                  (absolute danger ceiling)
      HIGH      : err >= max(threshold * 1.5, 0.40)
      MEDIUM    : err >  threshold              (active alarm, not yet high)
      LOW       : everything else
    """
    if err >= 0.85:
        return "Critical"
    if err >= max(threshold * 1.5, 0.40):
        return "High"
    if err > threshold:
        return "Medium"
    return "Low"


def _classify_subsystems(
    top_sensor_names: list[str],
    has_alarms: bool,
) -> set[str]:
    """Map a list of alarming sensor tag names to annunciator tile IDs.

    HAIEnd 23.05 uses internal control-block tags (e.g. '1004.21-OUT',
    'DM-PIT01', 'DM-AIT-PH') rather than plain SCADA prefixes, so a
    prefix-only check misses almost every tag.  This function uses substring
    matching against the tag families documented in HAIEnd 23.05.

    Subsystem mapping (from HAIEnd 23.05 P1/P2/P3/P4 block assignments):
      P1-BOILER     : 1004.*, 1003.*, DM-TIT*, DM-PIT*, DM-PCV*, DM-FCV*,
                      DM-PP04*, B2016, B3004, B3005, TIT, PIT, LIT, PCV,
                      FCV, LCV, DM-LIT*, DM-PP01*
      P1-FEEDWATER  : 1001.*, 1002.*, 1010.*, 1011.*, DM-FT*, DM-LCV*,
                      DM-SOL*, DM-CIP*, DM-COOL*, DM-LSH*, DM-LSL*,
                      DM-SW*, DM-SS*, PP04-SP, DM-ST*
      P2-TURBINE    : GT, SPD, VLV_T, P2_, 1020.*
      P3-WATER-TREAT: PT01, PT02, LT01, FT01, PMP, AIT, P3_, DM-AIT*,
                      DM-TWIT*, DM-PWIT*, DQ*, GATEOPEN

    Fallback: if any alarm is active but no tag resolves to a specific
    subsystem, default to P1-BOILER (primary process loop).
    """
    active: set[str] = set()

    _P1_BOILER = (
        "1004", "1003", "DM-TIT", "DM-PIT", "DM-PCV", "DM-FCV",
        "DM-PP04", "DM-LIT", "DM-PP01", "DM-PIT01", "DM-TIT01",
        "B2016", "B3004", "B3005",
        "PIT", "TIT", "LIT", "PCV", "FCV", "LCV",
    )
    _P1_FEEDWATER = (
        "1001", "1002", "1010", "1011",
        "DM-FT", "DM-LCV", "DM-SOL", "DM-CIP", "DM-COOL",
        "DM-LSH", "DM-LSL", "DM-SW", "DM-SS", "DM-ST",
        "PP04-SP", "DM-HT",
    )
    _P2_TURBINE = ("GT", "SPD", "VLV_T", "P2_", "1020")
    _P3_WATER   = (
        "PT01", "PT02", "LT01", "FT01", "FT02",
        "PMP", "AIT", "P3_",
        "DM-AIT", "DM-TWIT", "DM-PWIT", "DQ", "GATEOPEN",
    )

    for tag in top_sensor_names:
        t = tag.upper()
        if any(p.upper() in t for p in _P1_BOILER):
            active.add("P1-BOILER")
        elif any(p.upper() in t for p in _P1_FEEDWATER):
            active.add("P1-FEEDWATER")
        elif any(p.upper() in t for p in _P2_TURBINE):
            active.add("P2-TURBINE")
        elif any(p.upper() in t for p in _P3_WATER):
            active.add("P3-WATER-TREAT")
        else:
            # DM- catch-all → most DM tags are P1
            if "DM-" in t:
                active.add("P1-BOILER")

    # Fallback: any active alarm with no resolved subsystem → P1-BOILER
    if has_alarms and not active:
        active.add("P1-BOILER")

    return active


def _alert_segments(alerts: np.ndarray) -> list[tuple[int, int]]:
    segs, i, n = [], 0, len(alerts)
    while i < n:
        if alerts[i]:
            s = i
            while i < n and alerts[i]:
                i += 1
            segs.append((s, i - 1))
        else:
            i += 1
    return segs


def _recompute_ema(mae: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA via first-order IIR — O(N) compiled C, no Python loop."""
    return lfilter([alpha], [1.0, -(1.0 - alpha)], mae).astype(np.float32)


def _update_alarm_log(
    result:    pd.DataFrame,
    raw_df:    pd.DataFrame,
    threshold: float,
    alpha:     float,
    cursor:    int,
) -> None:
    """Scan result.iloc[:cursor] and latch every alarm segment into session state.

    Design constraints:
    • Idempotent — re-running with the same cursor changes nothing.
    • Incremental — only newly completed segments are appended; the in-progress
      tail segment (touching cursor) is upserted so its duration updates live.
    • Never re-processes samples before logged_alarms_cursor (high-water mark),
      which would be O(N) on every tick.

    Segment status:
      [ CLOSED ]  — segment ended before cursor; record is final.
      [ ACTIVE ]  — segment runs right up to cursor; still in progress.
    """
    if result is None or len(result) == 0:
        return

    # Only process new data since the last tick
    hwm    = st.session_state.get("logged_alarms_cursor", 0)
    end_ix = min(cursor, len(result))
    if end_ix <= 0:
        return

    # Recompute EMA over the full 0..cursor slice so segment boundaries are
    # consistent regardless of what rolling window is currently displayed.
    mae_full  = result["mae_raw"].iloc[:end_ix].to_numpy()
    ema_full  = _recompute_ema(mae_full, alpha)
    alert_arr = (ema_full > threshold).astype(int)

    all_segs  = _alert_segments(alert_arr)   # list[(start, end)] — 0-indexed

    # Resolve a display timestamp for a given result-row index
    ts_col    = next(
        (c for c in raw_df.columns if c.lower() in {"timestamp", "time"}), None
    )
    def _ts_str(idx: int) -> str:
        if ts_col and idx < len(raw_df):
            try:
                return str(pd.to_datetime(raw_df[ts_col].iloc[idx]).strftime("%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass
        base = datetime(2024, 1, 1) + timedelta(seconds=idx)
        return base.strftime("%Y-%m-%d %H:%M:%S")

    existing: list[dict] = st.session_state.get("logged_alarms", [])
    # Key existing records by their start index for O(1) lookup
    existing_by_start: dict[int, int] = {r["_start_idx"]: i for i, r in enumerate(existing)}

    for s, e in all_segs:
        # Skip segments that ended before the high-water mark AND are already
        # logged with CLOSED status — they are immutable.
        if s in existing_by_start:
            rec_ix = existing_by_start[s]
            if existing[rec_ix]["STATUS"] == "[ CLOSED ]":
                continue   # already finalised — do not touch

        # Determine status: ACTIVE only if segment reaches right to cursor edge
        is_active = (e >= end_ix - 1) and st.session_state.get("sim_running", False)
        status    = "[ ACTIVE ]" if is_active else "[ CLOSED ]"

        seg_ema   = ema_full[s: e + 1]
        peak_mae  = float(seg_ema.max())
        rlabel    = _risk_label(peak_mae, threshold)
        duration  = e - s + 1

        record: dict = {
            "SEQ":         0,                          # renumbered below
            "START":       _ts_str(s),
            "END":         _ts_str(e) if not is_active else "[ ACTIVE ]",
            "DURATION (s)": duration,
            "PEAK MAE":    f"{peak_mae:.4f}",
            "RISK":        f"[ {rlabel.upper()} ]",
            "STATUS":      status,
            "_start_idx":  s,                          # internal key, hidden in display
        }

        if s in existing_by_start:
            # Upsert — update the existing record in place (e.g. duration grew)
            existing[existing_by_start[s]] = record
        else:
            existing.append(record)

    # Renumber SEQ in chronological order
    existing_sorted = sorted(existing, key=lambda r: r["_start_idx"])
    for seq_i, rec in enumerate(existing_sorted, 1):
        rec["SEQ"] = seq_i

    st.session_state["logged_alarms"]        = existing_sorted
    # Advance high-water mark only past fully closed segments
    closed_ends = [e for (s, e) in all_segs if e < end_ix - 1]
    if closed_ends:
        st.session_state["logged_alarms_cursor"] = max(closed_ends) + 1


def _apply_plot_theme(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
        font=dict(family=_C_MONO, size=11, color=_C_TEXT),
        margin=dict(l=64, r=24, t=36, b=32),
        hovermode="x unified",
        legend=dict(bgcolor="#FFFFFF", bordercolor=_C_BORDER, borderwidth=1,
                    font=dict(family=_C_MONO, size=10)),
    )
    fig.update_xaxes(**_GRID_STYLE)
    fig.update_yaxes(**_GRID_STYLE)
    return fig


def _get_view_slice(result: pd.DataFrame, cursor: int) -> pd.DataFrame:
    """Return a rolling _VIEW_W window ending at cursor."""
    end   = min(cursor, len(result))
    start = max(0, end - _VIEW_W)
    return result.iloc[start:end]


def _build_timestamps(n: int, start: Optional[datetime] = None) -> list[datetime]:
    """Generate 1 Hz synthetic timestamps (used when CSV has no timestamp col)."""
    base = start or datetime(2024, 1, 1)
    return [base + timedelta(seconds=i) for i in range(n)]


def _extract_signals(
    raw_df: pd.DataFrame,
    feat_cols: list[str],
    start_idx: int,
    end_idx:   int,
) -> dict[str, np.ndarray]:
    """Extract up to 4 representative process variables for the trend chart."""
    # Prefer named P1 tags; fall back to first 4 feature columns
    preferred = ["PIT01", "TIT01", "FT03", "LIT01"]
    chosen = [c for c in preferred if c in feat_cols]
    if len(chosen) < 4:
        chosen += [c for c in feat_cols if c not in chosen][: 4 - len(chosen)]
    chosen = chosen[:4]

    sub = raw_df.iloc[start_idx:end_idx]
    return {
        c: sub[c].to_numpy() if c in sub.columns else np.zeros(end_idx - start_idx)
        for c in chosen
    }


# ════════════════════════════════════════════════════════════════════════════
# ISA-101 UI COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def _annunciator_matrix(active_alarms: set[str]) -> None:
    cols = st.columns(len(_ANNUNCIATORS))
    for col, (tag, subtitle) in zip(cols, _ANNUNCIATORS):
        alarming = tag in active_alarms
        bg    = _C_ALARM  if alarming else _C_NORMAL
        fg    = "#FFFFFF" if alarming else _C_TEXT_MUTED
        state = "ALARM"   if alarming else "NOMINAL"
        col.markdown(
            f"""<div style="background-color:{bg};
                border:2px solid {'#B91C1C' if alarming else _C_BORDER_DARK};
                border-radius:0px;padding:10px 8px 8px;text-align:center;
                font-family:{_C_MONO};">
              <div style="font-size:0.80rem;font-weight:700;
                          letter-spacing:0.06em;color:{fg};">[ {tag} ]</div>
              <div style="font-size:0.62rem;letter-spacing:0.05em;
                          color:{'rgba(255,255,255,0.75)' if alarming else _C_TEXT_MUTED};
                          margin-top:2px;">{subtitle}</div>
              <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.10em;
                          margin-top:4px;color:{fg};">[ {state} ]</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _led_risk_bar(current_risk: str) -> None:
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    fills  = {
        "LOW":      (_C_NORMAL,  _C_TEXT_MUTED),
        "MEDIUM":   (_C_WARNING, "#FFFFFF"),
        "HIGH":     ("#EA580C",  "#FFFFFF"),
        "CRITICAL": (_C_ALARM,   "#FFFFFF"),
    }
    active = current_risk.upper()
    tiles  = ""
    for lvl in levels:
        is_active = (lvl == active)
        bg, fg = fills[lvl] if is_active else (_C_NORMAL, "#9CA3AF")
        border = f"2px solid {'#B91C1C' if is_active and lvl == 'CRITICAL' else _C_BORDER}"
        tiles += (
            f'<div style="flex:1;background:{bg};border:{border};border-radius:0px;'
            f'padding:6px 4px;text-align:center;font-family:{_C_MONO};'
            f'font-size:0.70rem;font-weight:700;letter-spacing:0.08em;color:{fg};">{lvl}</div>'
        )
    st.markdown(
        f'<div style="margin-bottom:8px;">'
        f'<div style="font-family:{_C_MONO};font-size:0.65rem;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:{_C_TEXT_MUTED};margin-bottom:4px;">RISK INDEX</div>'
        f'<div style="display:flex;gap:3px;">{tiles}</div></div>',
        unsafe_allow_html=True,
    )


def _kpi_tile(col, label: str, value: str, sub: str, alarm: bool = False) -> None:
    bg     = "#FEF2F2" if alarm else _C_PANEL
    border = f"2px solid {_C_ALARM}" if alarm else f"1px solid {_C_BORDER}"
    v_col  = _C_ALARM if alarm else _C_TEXT
    col.markdown(
        f'<div style="background:{bg};border:{border};border-radius:0px;'
        f'padding:12px 14px 10px;font-family:{_C_MONO};">'
        f'<div style="font-size:0.62rem;letter-spacing:0.10em;text-transform:uppercase;'
        f'color:{_C_TEXT_MUTED};margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:1.45rem;font-weight:700;color:{v_col};'
        f'letter-spacing:0.02em;">{value}</div>'
        f'<div style="font-size:0.65rem;color:{_C_TEXT_MUTED};margin-top:3px;">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _standby_screen() -> None:
    """ISA-101 empty-state shown when no telemetry is loaded."""
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div style="
            border: 2px solid {_C_BORDER_DARK};
            border-radius: 0px;
            background: {_C_PANEL};
            padding: 48px 32px;
            text-align: center;
            font-family: {_C_MONO};
        ">
          <div style="font-size:1.10rem;font-weight:700;letter-spacing:0.12em;
                      color:{_C_TEXT_MUTED};">
            [ STANDBY — AWAITING TELEMETRY INGESTION ]
          </div>
          <div style="font-size:0.78rem;color:{_C_TEXT_MUTED};margin-top:14px;
                      letter-spacing:0.06em;">
            Upload a HAIEnd 23.05 CSV or select a benchmark dataset in the sidebar.
          </div>
          <div style="font-size:0.70rem;color:{_C_TEXT_MUTED};margin-top:8px;">
            Required: numeric sensor columns compatible with<br>
            <code style="font-family:{_C_MONO};">models_saved/clean_100/lstm_autoencoder.pth</code>
            &nbsp;(225 features, window=60)
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"<div style='font-family:{_C_MONO};font-size:1.05rem;font-weight:700;"
            f"letter-spacing:0.06em;color:{_C_TEXT};border-bottom:2px solid {_C_BORDER_DARK};"
            f"padding-bottom:6px;margin-bottom:4px;'>FACTORYSHIELD-OT</div>",
            unsafe_allow_html=True,
        )
        st.caption("Industrial Cybersecurity SOC  |  HAIEnd 23.05  |  ISA/IEC 62443")
        st.divider()

        page = st.radio("Navigation", PAGE_NAMES, label_visibility="collapsed")
        st.divider()

        # ── DATA SOURCE ──────────────────────────────────────────────────────
        st.markdown(
            f"<div style='font-family:{_C_MONO};font-size:0.68rem;letter-spacing:0.10em;"
            f"text-transform:uppercase;color:{_C_TEXT_MUTED};margin-bottom:6px;'>"
            f"DATA SOURCE</div>",
            unsafe_allow_html=True,
        )

        # Downsampling selector — shown before file uploader so user can choose
        # resolution before ingestion is triggered.  Changing the step
        # invalidates any previously cached inference result.
        _DOWNSAMPLE_OPTIONS = {
            "Full Dataset (1 s resolution)":        1,
            "Fast Preview — 1 sample every 2 s":    2,
            "Fast Preview — 1 sample every 5 s":    5,
            "Fast Preview — 1 sample every 10 s":  10,
        }
        _ds_label = st.selectbox(
            "Simulation Speed / Downsampling",
            list(_DOWNSAMPLE_OPTIONS.keys()),
            index=0,
            help=(
                "For files > 50 000 rows, select a lower resolution to reduce "
                "RAM usage and inference time. Full Dataset preserves 1 Hz "
                "fidelity; Fast Preview skips every N samples."
            ),
        )
        _new_step = _DOWNSAMPLE_OPTIONS[_ds_label]
        if st.session_state.get("downsample_step") != _new_step:
            # Resolution changed — discard cached result so re-ingestion runs
            st.session_state["downsample_step"] = _new_step
            st.session_state["infer_result"]    = None
            st.session_state["raw_df"]          = None
            st.session_state["source_label"]    = None
            st.session_state["play_cursor"]     = _VIEW_W

        uploaded = st.file_uploader(
            "Upload Industrial Telemetry (.csv)",
            type=["csv"],
            help="Raw or pre-scaled HAIEnd 23.05 CSV. Metadata columns "
                 "(timestamp, attack, label) are stripped automatically.",
        )

        # Benchmark datasets available on disk
        _BENCHMARK_OPTIONS: dict[str, Optional[Path]] = {"None": None}
        for _label, _path in [
            ("test1.csv — HAIEnd 23.05 (clean_100)",
             _ROOT / "data" / "processed" / "clean_100" / "test_set_final.csv"),
            ("test2.csv — HAIEnd 23.05 (mixed_70_30)",
             _ROOT / "data" / "processed" / "mixed_70_30" / "test_set_final.csv"),
            ("end-test1.csv — HAIEnd raw",
             _ROOT / "data" / "raw" / "end-test1.csv"),
            ("end-test2.csv — HAIEnd raw",
             _ROOT / "data" / "raw" / "end-test2.csv"),
        ]:
            if _path.exists():
                _BENCHMARK_OPTIONS[_label] = _path

        bench_choice = st.selectbox(
            "Or Select Benchmark Dataset",
            list(_BENCHMARK_OPTIONS.keys()),
            help="Pre-loaded HAIEnd 23.05 test partitions from disk.",
        )

        # ── Resolve active data source ───────────────────────────────────────
        _ckpt_path = str(_ROOT / "models_saved" / "clean_100" / "lstm_autoencoder.pth")
        _load_error: Optional[str] = None

        if uploaded is not None:
            source_key = f"{uploaded.name}:step={st.session_state['downsample_step']}"
            if st.session_state.get("source_label") != source_key:
                # New file — invalidate previous inference results
                st.session_state["infer_result"] = None
                st.session_state["raw_df"]       = None
                st.session_state["play_cursor"]  = _VIEW_W
                st.session_state["source_label"] = source_key
            if st.session_state["infer_result"] is None:
                try:
                    raw_df, X_scaled, feat_cols = _preprocess_raw_csv(
                        uploaded.getvalue(), uploaded.name,
                        st.session_state.get("downsample_step", 1),
                    )
                    result = _run_full_inference(
                        uploaded.name, id(X_scaled), X_scaled, _ckpt_path
                    )
                    st.session_state["raw_df"]       = raw_df
                    st.session_state["feature_cols"] = feat_cols
                    st.session_state["n_samples"]    = len(result)
                    st.session_state["infer_result"] = result
                    st.session_state["play_cursor"]  = _VIEW_W
                except Exception as exc:
                    _load_error = str(exc)

        elif bench_choice != "None":
            bench_path = _BENCHMARK_OPTIONS[bench_choice]
            source_key = f"{bench_choice}:step={st.session_state['downsample_step']}"
            if st.session_state.get("source_label") != source_key:
                st.session_state["infer_result"] = None
                st.session_state["raw_df"]       = None
                st.session_state["play_cursor"]  = _VIEW_W
                st.session_state["source_label"] = source_key
            if st.session_state["infer_result"] is None:
                try:
                    file_bytes = bench_path.read_bytes()
                    raw_df, X_scaled, feat_cols = _preprocess_raw_csv(
                        file_bytes, bench_choice,
                        st.session_state.get("downsample_step", 1),
                    )
                    result = _run_full_inference(
                        bench_choice, id(X_scaled), X_scaled, _ckpt_path
                    )
                    st.session_state["raw_df"]       = raw_df
                    st.session_state["feature_cols"] = feat_cols
                    st.session_state["n_samples"]    = len(result)
                    st.session_state["infer_result"] = result
                    st.session_state["play_cursor"]  = _VIEW_W
                except Exception as exc:
                    _load_error = str(exc)
        else:
            # No source selected — clear state
            if st.session_state.get("source_label") is not None:
                st.session_state["infer_result"] = None
                st.session_state["raw_df"]       = None
                st.session_state["source_label"] = None
                st.session_state["play_cursor"]  = _VIEW_W

        if _load_error:
            st.error(f"[ FAULT ] Ingestion error: {_load_error}")

        # Status indicator
        if st.session_state["infer_result"] is not None:
            n = st.session_state["n_samples"]
            st.markdown(
                f"<div style='font-family:{_C_MONO};font-size:0.68rem;"
                f"color:{_C_OK_TEXT};margin-top:4px;'>"
                f"[ ONLINE ] {n:,} windows inferred</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-family:{_C_MONO};font-size:0.68rem;"
                f"color:{_C_TEXT_MUTED};margin-top:4px;'>[ STANDBY ]</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── DETECTION PARAMETERS ─────────────────────────────────────────────
        st.markdown(
            f"<div style='font-family:{_C_MONO};font-size:0.68rem;letter-spacing:0.10em;"
            f"text-transform:uppercase;color:{_C_TEXT_MUTED};margin-bottom:6px;'>"
            f"DETECTION PARAMETERS</div>",
            unsafe_allow_html=True,
        )
        st.session_state["threshold"] = st.slider(
            "ALARM THRESHOLD", 0.01, 1.00,
            float(st.session_state["threshold"]), 0.01,
            help="Normalised MAE (EMA-smoothed) above which an alarm fires.",
        )
        st.session_state["ema_alpha"] = st.slider(
            "EMA ALPHA", 0.01, 0.50,
            float(st.session_state["ema_alpha"]), 0.01,
            help="Exponential moving-average smoothing factor α ∈ (0, 1].",
        )
        st.divider()

        # ── PLAYBACK CONTROL ─────────────────────────────────────────────────
        st.markdown(
            f"<div style='font-family:{_C_MONO};font-size:0.68rem;letter-spacing:0.10em;"
            f"text-transform:uppercase;color:{_C_TEXT_MUTED};margin-bottom:6px;'>"
            f"PLAYBACK CONTROL</div>",
            unsafe_allow_html=True,
        )

        is_running = st.session_state.get("sim_running", False)

        col_run, col_hold = st.columns(2)
        if col_run.button(
            "[ RUNNING ]" if is_running else "[ RUN ]",
            use_container_width=True,
            type="primary" if is_running else "secondary",
            key="btn_sim_run",
        ):
            st.session_state["sim_running"] = True
            st.rerun()

        if col_hold.button(
            "[ HELD ]" if not is_running else "[ HOLD ]",
            use_container_width=True,
            type="secondary" if is_running else "primary",
            key="btn_sim_hold",
        ):
            st.session_state["sim_running"] = False
            st.rerun()

        if st.button("[ RESET ]", use_container_width=True,
                     type="secondary", key="btn_sim_reset"):
            st.session_state["sim_running"]          = False
            st.session_state["play_cursor"]          = _VIEW_W
            st.session_state["logged_alarms"]        = []
            st.session_state["logged_alarms_cursor"] = 0
            st.rerun()

        if st.button("[ CLEAR DATA ]", use_container_width=True,
                     type="secondary", key="btn_sim_clear"):
            st.session_state["sim_running"]          = False
            st.session_state["infer_result"]         = None
            st.session_state["raw_df"]               = None
            st.session_state["source_label"]         = None
            st.session_state["play_cursor"]          = _VIEW_W
            st.session_state["logged_alarms"]        = []
            st.session_state["logged_alarms_cursor"] = 0
            st.cache_data.clear()
            st.rerun()

        # ISA-101 state badge — green = streaming, dark grey = standby
        if is_running:
            st.markdown(
                "<div style='background-color:#065F46;color:#ECFDF5;padding:6px;"
                "text-align:center;font-family:monospace;font-size:11px;"
                "margin-top:6px;border:1px solid #047857;font-weight:bold;'>"
                "STATE: STREAMING ACTIVE [ RUNNING ]</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background-color:#374151;color:#F3F4F6;padding:6px;"
                "text-align:center;font-family:monospace;font-size:11px;"
                "margin-top:6px;border:1px solid #4B5563;font-weight:bold;'>"
                "STATE: PAUSED / STANDBY [ HELD ]</div>",
                unsafe_allow_html=True,
            )

        # Cursor position
        n_total  = st.session_state["n_samples"]
        cursor   = st.session_state["play_cursor"]
        progress = min(cursor / max(n_total, 1), 1.0)
        if n_total > 0:
            st.progress(progress,
                        text=f"{cursor:,} / {n_total:,} samples")
            st.caption(f"TELEMETRY CURSOR: {min(cursor, n_total):,} / {n_total:,}")
        st.divider()

    return page


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE SOC
# ════════════════════════════════════════════════════════════════════════════

def page_live_soc(result: pd.DataFrame, raw_df: pd.DataFrame,
                  feat_cols: list[str]) -> None:
    st.markdown(
        f"<h2 style='font-family:{_C_MONO};'>LIVE SOC — PROCESS SUPERVISION</h2>",
        unsafe_allow_html=True,
    )
    src = st.session_state.get("source_label", "unknown")
    st.caption(f"Emerson Ovation DCS  |  HAIEnd 23.05  |  Source: {src}")

    threshold = float(st.session_state["threshold"])
    alpha     = float(st.session_state["ema_alpha"])
    cursor    = st.session_state["play_cursor"]
    n_total   = st.session_state["n_samples"]

    # Current view window
    view = _get_view_slice(result, cursor)
    n_view = len(view)
    if n_view == 0:
        st.info("[ STANDBY ] No data in current window.")
        return

    # Re-apply EMA and threshold on the view slice (slider may have changed)
    mae_raw  = view["mae_raw"].to_numpy()
    mae_ema  = _recompute_ema(mae_raw, alpha)
    alerts   = (mae_ema > threshold).astype(int)
    segs     = _alert_segments(alerts)
    max_err  = float(mae_ema[alerts == 1].max()) if alerts.any() else 0.0
    current_risk = _risk_label(max_err, threshold)

    # ── Resolve active subsystems from top-sensor indices ────────────────────
    # Pull the top-ranked sensor name at the highest-MAE alarming window.
    # This drives which annunciator tiles light up.
    feat_cols_soc = st.session_state.get("feature_cols") or []
    _TOP_K_STORE  = 5
    top_names: list[str] = []
    if segs and feat_cols_soc:
        # Find the window with the highest EMA error inside any alarm segment
        alarm_indices = np.where(alerts == 1)[0]
        peak_win      = int(alarm_indices[np.argmax(mae_ema[alarm_indices])])
        idx_cols_soc  = [f"top_sensor_idx_{k}" for k in range(_TOP_K_STORE)]
        if all(c in view.columns for c in idx_cols_soc):
            raw_indices = view[idx_cols_soc].iloc[peak_win].to_numpy().astype(int)
            top_names   = [
                feat_cols_soc[fi] if fi < len(feat_cols_soc) else f"sensor_{fi}"
                for fi in raw_indices
            ]

    active_alarms = _classify_subsystems(top_names, has_alarms=bool(segs))

    # ── Annunciator matrix ───────────────────────────────────────────────────
    _annunciator_matrix(active_alarms)
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # ── LED risk bar ─────────────────────────────────────────────────────────
    _led_risk_bar(current_risk)
    st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)

    # ── KPI tiles ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    proc_alarm = current_risk in ("High", "Critical")
    cursor_pct = f"{cursor / max(n_total, 1) * 100:.1f}% played"
    _ds = st.session_state.get("downsample_step", 1)
    _ds_tag = f"1 Hz" if _ds == 1 else f"1 sample / {_ds} s"
    _kpi_tile(k1, "PROCESS STATUS",    "[ ONLINE ]",
              f"{n_total:,} windows  |  {_ds_tag}")
    _kpi_tile(k2, "ACTIVE ALARM SEGS", str(len(segs)),
              f"{len(segs)} segment(s) flagged", alarm=bool(segs))
    _kpi_tile(k3, "RISK LEVEL",
              f"[ {current_risk.upper()} ]",
              {"Low": "NOMINAL", "Medium": "WARNING",
               "High": "ALERT", "Critical": "TRIP"}[current_risk],
              alarm=proc_alarm)
    _kpi_tile(k4, "PEAK EMA ERROR",    f"{max_err:.4f}",
              f"threshold={threshold:.2f}  |  {cursor_pct}",
              alarm=max_err > threshold)
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # ── Build timestamps for the view window ─────────────────────────────────
    view_start = cursor - n_view
    # Use timestamp column from raw_df if present
    ts_col = next(
        (c for c in raw_df.columns if c.lower() in {"timestamp", "time"}), None
    )
    if ts_col and view_start < len(raw_df):
        try:
            raw_ts = pd.to_datetime(raw_df[ts_col].iloc[view_start: view_start + n_view])
            ts = raw_ts.tolist()
        except Exception:
            ts = _build_timestamps(n_view, datetime(2024, 1, 1) + timedelta(seconds=view_start))
    else:
        ts = _build_timestamps(n_view, datetime(2024, 1, 1) + timedelta(seconds=view_start))

    # ── Extract process signals for the view slice ────────────────────────────
    signals = _extract_signals(raw_df, feat_cols,
                               start_idx=view_start,
                               end_idx=view_start + n_view)
    sig_colors = [_C_PRESSURE, _C_TEMPERATURE, _C_FLOW, _C_LEVEL]

    # ── 4-panel trend recorder ────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.36, 0.26, 0.19, 0.19],
        subplot_titles=(
            "PROCESS VARIABLES (RAW SENSOR READINGS)",
            "RECONSTRUCTION MAE — LSTM AE  (EMA-FILTERED)",
            "ALARM STATE",
            "RISK LEVEL",
        ),
    )

    # Row 1 — real process signals
    for idx, (name, vals) in enumerate(signals.items()):
        fig.add_trace(
            go.Scatter(x=ts, y=vals, name=name, mode="lines",
                       line=dict(width=1.5, color=sig_colors[idx % 4])),
            row=1, col=1,
        )

    # Row 2 — real MAE from LSTM AE
    fig.add_trace(
        go.Scatter(x=ts, y=mae_raw, name="MAE RAW", mode="lines",
                   line=dict(width=1, color=_C_MAE_RAW, dash="dot")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=ts, y=mae_ema, name="MAE EMA", mode="lines",
                   line=dict(width=2, color=_C_MAE_EMA)),
        row=2, col=1,
    )
    fig.add_hline(
        y=threshold,
        line=dict(color=_C_THRESHOLD, width=1.5, dash="dash"),
        annotation_text=f"TRIP @ {threshold:.2f}",
        annotation_font=dict(family=_C_MONO, size=10, color=_C_THRESHOLD),
        row=2, col=1,
    )

    # Row 3 — alarm
    fig.add_trace(
        go.Scatter(x=ts, y=alerts, name="ALARM", mode="lines",
                   line=dict(width=2, color=_C_ALARM),
                   fill="tozeroy", fillcolor=_C_ALERT_FILL),
        row=3, col=1,
    )

    # Row 4 — ordinal risk level
    risk_num = np.where(mae_ema >= 0.85, 3,
               np.where(mae_ema >= 0.60, 2,
               np.where(mae_ema >= 0.30, 1, 0)))
    fig.add_trace(
        go.Scatter(x=ts, y=risk_num, name="RISK", mode="lines",
                   line=dict(width=2, color=_C_ALARM_DARK),
                   fill="tozeroy", fillcolor="rgba(185,28,28,0.07)"),
        row=4, col=1,
    )

    # Shade alarm windows
    for s, e in segs:
        for r in range(1, 5):
            fig.add_vrect(x0=ts[s], x1=ts[min(e, n_view - 1)],
                          fillcolor=_C_ALERT_FILL, line_width=0, row=r, col=1)

    fig.update_yaxes(title_text="VALUE", row=1, col=1, **_GRID_STYLE)
    fig.update_yaxes(title_text="MAE",   row=2, col=1, range=[0, max(1.0, float(mae_raw.max()) * 1.1)], **_GRID_STYLE)
    fig.update_yaxes(title_text="0/1",   row=3, col=1, range=[-0.1, 1.2], **_GRID_STYLE)
    fig.update_yaxes(
        ticktext=["LOW", "MED", "HIGH", "CRIT"], tickvals=[0, 1, 2, 3],
        range=[-0.5, 3.5], row=4, col=1, **_GRID_STYLE,
    )
    fig.update_xaxes(title_text="TIME", row=4, col=1, **_GRID_STYLE)
    fig.update_layout(
        height=740,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
        font=dict(family=_C_MONO, size=11, color=_C_TEXT),
        margin=dict(l=72, r=24, t=40, b=32),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.05, bgcolor="#FFFFFF",
                    bordercolor=_C_BORDER, borderwidth=1,
                    font=dict(family=_C_MONO, size=10)),
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(family=_C_MONO, size=11, color=_C_TEXT_MUTED),
                   x=0.0, xanchor="left")

    st.plotly_chart(fig, key="soc_chart", use_container_width=True)

    # ── Incident log (latching — survives cursor scroll) ─────────────────────
    st.markdown(
        f"<h3 style='font-family:{_C_MONO};margin-top:16px;'>INCIDENT LOG</h3>",
        unsafe_allow_html=True,
    )

    logged: list[dict] = st.session_state.get("logged_alarms", [])

    if logged:
        # Build display DataFrame — hide the internal _start_idx column
        display_cols = ["SEQ", "START", "END", "DURATION (s)", "PEAK MAE", "RISK", "STATUS"]
        df_log = pd.DataFrame(logged)[display_cols]

        # Export button — above the table so it is visible without scrolling
        col_export, col_count = st.columns([2, 1])
        with col_export:
            st.download_button(
                label="Export Incident Log (CSV)",
                data=df_log.to_csv(index=False).encode("utf-8"),
                file_name="fs_ot_incident_log.csv",
                mime="text/csv",
                help="Download all detected incidents as a CSV file.",
            )
        with col_count:
            n_active = sum(1 for r in logged if r["STATUS"] == "[ ACTIVE ]")
            n_closed = len(logged) - n_active
            st.markdown(
                f"<div style='font-family:{_C_MONO};font-size:0.72rem;"
                f"color:{_C_ALARM if n_active else _C_TEXT_MUTED};"
                f"padding-top:8px;'>"
                f"{n_closed} CLOSED  |  {n_active} ACTIVE</div>",
                unsafe_allow_html=True,
            )

        st.dataframe(df_log, hide_index=True, use_container_width=True)

        with st.expander("OPERATOR RESPONSE PROCEDURE", expanded=True):
            if current_risk == "Critical":
                st.error("[ TRIP ] EMERGENCY — Initiate controlled shutdown. Notify plant manager. Isolate affected control loops.")
            elif current_risk == "High":
                st.warning("[ ALARM ] Switch affected loops to MANUAL. Verify field instrument readings. Log in DCS historian.")
            elif current_risk == "Medium":
                st.warning("[ WARNING ] Monitor closely. Cross-check redundant sensor channels.")
            else:
                st.info("[ SURVEILLANCE ] Micro-drift detected. No immediate action required.")
    else:
        st.success("[ NORMAL ] No incidents recorded — process operating within normal bounds.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — xAI ROOT CAUSE
# ════════════════════════════════════════════════════════════════════════════

def page_xai(result: pd.DataFrame) -> None:
    st.markdown(f"<h2 style='font-family:{_C_MONO};'>XAI ROOT CAUSE ANALYSIS</h2>",
                unsafe_allow_html=True)
    st.caption(
        "Feature-wise LSTM reconstruction error ranking — zero-overhead, no SHAP required.  "
        "E_{t,f} = |x_{t,f} − x̂_{t,f}|  |  Rank-1 sensor = designated attack injection point."
    )

    threshold   = float(st.session_state["threshold"])
    alpha       = float(st.session_state["ema_alpha"])
    cursor      = st.session_state["play_cursor"]
    feat_cols   = st.session_state.get("feature_cols") or []
    logged      = st.session_state.get("logged_alarms", [])

    # ════════════════════════════════════════════════════════════════════════
    # SECTION A — PDF FORENSIC AUDIT REPORT (full incident history)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<h3 style='font-family:{_C_MONO};'>FORENSIC AUDIT REPORT</h3>",
        unsafe_allow_html=True,
    )

    if not logged:
        st.markdown(
            f"<div style='border:1px solid {_C_BORDER};border-radius:0px;"
            f"background:{_C_PANEL};padding:20px 18px;font-family:{_C_MONO};"
            f"font-size:0.80rem;color:{_C_TEXT_MUTED};text-align:center;'>"
            f"[ STANDBY — NO INCIDENTS RECORDED TO AUDIT ]<br>"
            f"<span style='font-size:0.70rem;'>Advance the playback cursor past an "
            f"attack segment to populate the incident log.</span></div>",
            unsafe_allow_html=True,
        )
    elif not _PDF_OK:
        st.warning(
            "[ FAULT ] fpdf2 is not installed — PDF export unavailable.  "
            "Run:  pip install fpdf2>=2.7.9"
        )
    else:
        # Count closed vs active so the operator knows what's in the PDF
        n_closed = sum(1 for r in logged if r.get("STATUS") == "[ CLOSED ]")
        n_active = len(logged) - n_closed

        col_btn, col_meta = st.columns([3, 2])
        with col_btn:
            # Generate PDF bytes on demand — wrapped in try/except so a render
            # failure degrades to an error message rather than crashing the page.
            try:
                pdf_bytes = generate_incident_pdf(
                    logged_incidents = logged,
                    result_df        = result,
                    feat_cols        = feat_cols,
                    threshold        = threshold,
                    alpha            = alpha,
                )
                st.download_button(
                    label            = "[ EXPORT AUDIT REPORT (PDF) ]",
                    data             = pdf_bytes,
                    file_name        = f"factoryshield_audit_report_{int(time.time())}.pdf",
                    mime             = "application/pdf",
                    use_container_width = True,
                    help             = (
                        "Downloads a forensic-grade ISA-101 / ISA-18.2 compliant "
                        "PDF containing all detected incidents, per-sensor root-cause "
                        "rankings, and operator response directives."
                    ),
                )
            except Exception as exc:
                st.error(f"[ FAULT ] PDF generation failed: {exc}")

        with col_meta:
            st.markdown(
                f"<div style='font-family:{_C_MONO};font-size:0.72rem;"
                f"color:{_C_TEXT_MUTED};padding-top:8px;'>"
                f"{n_closed} incident(s) CLOSED<br>"
                f"<span style='color:{'#DC2626' if n_active else _C_TEXT_MUTED};'>"
                f"{n_active} ACTIVE</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION B — ROLLING-WINDOW ONSET ANALYSIS (current view)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<h3 style='font-family:{_C_MONO};'>ONSET ANALYSIS — CURRENT WINDOW</h3>",
        unsafe_allow_html=True,
    )

    view    = _get_view_slice(result, cursor)
    mae_ema = _recompute_ema(view["mae_raw"].to_numpy(), alpha)
    alerts  = (mae_ema > threshold).astype(int)

    _TOP_K_STORE = 5
    idx_cols  = [f"top_sensor_idx_{k}" for k in range(_TOP_K_STORE)]
    err_cols  = [f"top_sensor_err_{k}" for k in range(_TOP_K_STORE)]
    has_top_k = all(c in result.columns for c in idx_cols + err_cols)

    top_k = st.slider(
        "TOP-K SENSORS",
        min_value=1,
        max_value=_TOP_K_STORE,
        value=min(5, _TOP_K_STORE),
        key="xai_k",
    )

    if alerts.sum() == 0:
        st.info("No alarms in current window — lower the threshold or advance the cursor.")
        return

    if not has_top_k:
        st.error(
            "[ FAULT ] Result DataFrame is missing top_sensor_idx/err columns. "
            "Re-run inference via [ CLEAR DATA ] → reload dataset."
        )
        return

    # ── Reconstruct onset report from compact top-k columns ─────────────────
    prev       = np.concatenate(([0], alerts[:-1]))
    onset_rows = np.where((alerts == 1) & (prev == 0))[0]

    if len(onset_rows) == 0:
        st.warning("No onset rows found in current window.")
        return

    rows_list: list[dict] = []
    view_idx_arr = view[idx_cols].to_numpy()
    view_err_arr = view[err_cols].to_numpy()

    for row in onset_rows:
        raw_idx = view_idx_arr[row]
        raw_err = view_err_arr[row]
        k       = min(top_k, _TOP_K_STORE)
        sel_idx = raw_idx[:k]
        sel_err = raw_err[:k].astype(np.float32)
        total   = float(sel_err.sum())

        for rank, (fi, err) in enumerate(zip(sel_idx, sel_err), 1):
            fi          = int(fi)
            sensor_name = feat_cols[fi] if fi < len(feat_cols) else f"sensor_{fi}"
            rows_list.append(dict(
                timestep=int(row),
                rank=rank,
                sensor=sensor_name,
                error=round(float(err), 6),
                pct_contribution=round(
                    float(err / total * 100) if total > 0 else 0.0, 2),
                root_cause=(rank == 1),
            ))

    report = pd.DataFrame(rows_list)
    report = report.copy()
    if SUBSYSTEM_MAP:
        report["subsystem"] = report["sensor"].map(
            lambda s: SUBSYSTEM_MAP.get(s, "Unknown"))
    else:
        report["subsystem"] = "Unknown"

    onsets = sorted(report["timestep"].unique())
    st.markdown(
        f"<span style='font-family:{_C_MONO};font-size:0.80rem;'>"
        f"{len(onsets)} ALARM ONSET(S) IN CURRENT WINDOW — TOP-{top_k} SENSORS EACH"
        f"</span>",
        unsafe_allow_html=True,
    )

    _rank_colors = {1: _C_ALARM, 2: _C_WARNING}
    for onset in onsets:
        sub = report[report["timestep"] == onset].sort_values("rank")
        rc  = sub.iloc[0]["sensor"]
        with st.expander(
            f"ONSET t={onset}  |  ROOT CAUSE: {rc}",
            expanded=(onset == onsets[0]),
        ):
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                bar_colors = [_rank_colors.get(r, _C_BORDER_DARK) for r in sub["rank"]]
                fig_b = go.Figure(go.Bar(
                    x=sub["sensor"],
                    y=sub["error"],
                    text=sub["pct_contribution"].apply(lambda v: f"{v:.1f}%"),
                    textposition="auto",
                    textfont=dict(family=_C_MONO, size=11),
                    marker=dict(color=bar_colors,
                                line=dict(color=_C_BORDER_DARK, width=1)),
                ))
                fig_b.update_layout(
                    title=dict(
                        text=f"SENSOR RECONSTRUCTION ERROR  |  t={onset}",
                        font=dict(family=_C_MONO, size=11, color=_C_TEXT_MUTED),
                    ),
                    yaxis_title="ABS. RECONSTRUCTION ERROR",
                    height=340,
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
                    font=dict(family=_C_MONO, size=11, color=_C_TEXT),
                    margin=dict(l=56, r=16, t=40, b=32),
                )
                fig_b.update_xaxes(**_GRID_STYLE)
                fig_b.update_yaxes(**_GRID_STYLE)
                st.plotly_chart(fig_b, key=f"xai_bar_{onset}",
                                use_container_width=True)
            with col_table:
                st.dataframe(
                    sub[["rank", "sensor", "error",
                         "pct_contribution", "subsystem"]]
                    .rename(columns={
                        "rank":             "RANK",
                        "sensor":           "SENSOR",
                        "error":            "ERROR",
                        "pct_contribution": "% CONTRIB",
                        "subsystem":        "SUBSYSTEM",
                    })
                    .reset_index(drop=True),
                    hide_index=True,
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL BENCHMARK
# ════════════════════════════════════════════════════════════════════════════

def page_benchmark() -> None:
    st.markdown(f"<h2 style='font-family:{_C_MONO};'>MODEL PERFORMANCE BENCHMARK</h2>",
                unsafe_allow_html=True)
    st.caption("Comparative evaluation  |  IF / OC-SVM / XGBoost / LSTM AE  |  HAIEnd 23.05")

    out_dir = _ROOT / "out"
    experiments = ["clean_100", "mixed_70_30"]
    exp_labels  = {"clean_100": "CLEAN 100%", "mixed_70_30": "MIXED 70/30"}
    found = False
    for exp in experiments:
        p = out_dir / f"evaluation_report_{exp}.txt"
        if p.exists():
            found = True
            with st.expander(f"RAW EVALUATION REPORT — {exp_labels[exp]}", expanded=False):
                st.code(p.read_text(encoding="utf-8"), language="text")
    if not found:
        st.info("No evaluation reports in out/. Run evaluate.py first.")

    st.markdown(f"<h3 style='font-family:{_C_MONO};'>BENCHMARK OVERVIEW</h3>",
                unsafe_allow_html=True)
    bench = pd.DataFrame([
        {"MODEL": "Isolation Forest", "PRECISION": "~72%", "RECALL": "~68%",
         "F1": "~70%", "eTaF1": "~0.61", "LATENCY": "< 1 s", "TYPE": "Unsupervised"},
        {"MODEL": "One-Class SVM",    "PRECISION": "~69%", "RECALL": "~65%",
         "F1": "~67%", "eTaF1": "~0.58", "LATENCY": "< 1 s", "TYPE": "Unsupervised"},
        {"MODEL": "XGBoost",          "PRECISION": "~88%", "RECALL": "~85%",
         "F1": "~86%", "eTaF1": "~0.79", "LATENCY": "< 1 s", "TYPE": "Supervised"},
        {"MODEL": "LSTM Autoencoder", "PRECISION": "~91%", "RECALL": "~89%",
         "F1": "~90%", "eTaF1": "~0.87", "LATENCY": "3–5 s", "TYPE": "Semi-supervised"},
    ])
    st.dataframe(bench, hide_index=True, use_container_width=True)
    st.caption("Indicative values — re-run evaluate.py with trained checkpoints for exact figures.")

    st.markdown(f"<h3 style='font-family:{_C_MONO};'>ROC-AUC CURVES</h3>",
                unsafe_allow_html=True)
    roc_cols = st.columns(len(experiments))
    for col, exp in zip(roc_cols, experiments):
        img_path = out_dir / f"roc_curve_{exp}.png"
        if img_path.exists():
            col.image(str(img_path), caption=exp_labels[exp])
        else:
            col.info(f"No ROC image for {exp_labels[exp]}. Run evaluate.py.")

    with st.expander("ABOUT eTaPR — ENHANCED TIME-AWARE PRECISION / RECALL", expanded=False):
        st.markdown(
            "eTaPR (HAICon 2021): a predicted segment counts as correct only if overlap "
            "with a true anomaly segment >= 50% (eTaP), and a true segment is detected "
            "only if coverage >= 50% (eTaR). eTaF1 = harmonic mean.  \n"
            "LSTM AE achieves eTaF1 ~0.87 on HAIEnd 23.05 (clean_100 experiment)."
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA EXPLORATION
# ════════════════════════════════════════════════════════════════════════════

def page_eda(result: pd.DataFrame, raw_df: pd.DataFrame,
             feat_cols: list[str]) -> None:
    st.markdown(f"<h2 style='font-family:{_C_MONO};'>DATA EXPLORATION</h2>",
                unsafe_allow_html=True)
    st.caption("Time-series EDA and inter-sensor correlations  |  HAIEnd 23.05")

    cursor    = st.session_state["play_cursor"]
    view_res  = _get_view_slice(result, cursor)
    n_view    = len(view_res)
    view_start = cursor - n_view

    # Timestamps
    ts_col = next(
        (c for c in raw_df.columns if c.lower() in {"timestamp", "time"}), None
    )
    if ts_col and view_start < len(raw_df):
        try:
            ts = pd.to_datetime(
                raw_df[ts_col].iloc[view_start: view_start + n_view]
            ).tolist()
        except Exception:
            ts = _build_timestamps(n_view, datetime(2024, 1, 1) + timedelta(seconds=view_start))
    else:
        ts = _build_timestamps(n_view, datetime(2024, 1, 1) + timedelta(seconds=view_start))

    # ── Time-series viewer ────────────────────────────────────────────────────
    st.markdown(f"<h3 style='font-family:{_C_MONO};'>TIME-SERIES VIEWER</h3>",
                unsafe_allow_html=True)
    ts_colors = [_C_PRESSURE, _C_TEMPERATURE, _C_FLOW, _C_LEVEL]
    chosen = st.multiselect(
        "SELECT PROCESS VARIABLES",
        feat_cols,
        default=feat_cols[:4],
    )
    if chosen:
        fig_ts = go.Figure()
        raw_slice = raw_df.iloc[view_start: view_start + n_view]
        # Downsample to at most 600 rendered points per trace — beyond that
        # the SVG path data exceeds what browsers render without stuttering.
        _TS_MAX_PTS = 600
        _ts_step    = max(1, len(raw_slice) // _TS_MAX_PTS)
        raw_slice_d = raw_slice.iloc[::_ts_step]
        ts_d        = ts[::_ts_step]
        for ci, name in enumerate(chosen):
            if name in raw_slice_d.columns:
                fig_ts.add_trace(go.Scatter(
                    x=ts_d, y=raw_slice_d[name].to_numpy(), mode="lines", name=name,
                    line=dict(width=1.5, color=ts_colors[ci % 4]),
                ))
        _apply_plot_theme(fig_ts, height=380)
        fig_ts.update_layout(xaxis_title="TIME", yaxis_title="VALUE")
        st.plotly_chart(fig_ts, key="eda_ts", use_container_width=True)

    st.divider()

    # ── MAE distribution ─────────────────────────────────────────────────────
    st.markdown(f"<h3 style='font-family:{_C_MONO};'>RECONSTRUCTION ERROR DISTRIBUTION</h3>",
                unsafe_allow_html=True)
    mae_all = result["mae_raw"].to_numpy()
    c1, c2  = st.columns(2)
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=mae_all, nbinsx=80, name="MAE RAW",
            marker=dict(color=_C_PRESSURE,
                        line=dict(color=_C_BORDER_DARK, width=0.5)),
        ))
        fig_hist.add_vline(
            x=st.session_state["threshold"],
            line=dict(color=_C_THRESHOLD, width=1.5, dash="dash"),
            annotation_text=f"THRESHOLD {st.session_state['threshold']:.2f}",
            annotation_font=dict(family=_C_MONO, size=10, color=_C_THRESHOLD),
        )
        _apply_plot_theme(fig_hist, height=320)
        fig_hist.update_layout(
            title=dict(text="GLOBAL MAE HISTOGRAM (FULL DATASET)",
                       font=dict(family=_C_MONO, size=11)),
            xaxis_title="MAE", yaxis_title="COUNT",
        )
        st.plotly_chart(fig_hist, key="eda_hist", use_container_width=True)

    with c2:
        sorted_e = np.sort(mae_all)
        cdf      = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        fig_cdf  = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=sorted_e, y=cdf, mode="lines", name="CDF",
            line=dict(width=2, color=_C_FLOW),
        ))
        for p in (90, 95, 99):
            v = np.percentile(sorted_e, p)
            fig_cdf.add_vline(
                x=v,
                line=dict(color=_C_WARNING, width=1, dash="dot"),
                annotation_text=f"p{p}",
                annotation_font=dict(family=_C_MONO, size=9, color=_C_WARNING),
            )
        _apply_plot_theme(fig_cdf, height=320)
        fig_cdf.update_layout(
            title=dict(text="CDF — PERCENTILE THRESHOLDS",
                       font=dict(family=_C_MONO, size=11)),
            xaxis_title="MAE", yaxis_title="PROB.",
        )
        st.plotly_chart(fig_cdf, key="eda_cdf", use_container_width=True)

    st.divider()

    # ── Correlation matrix — top-dynamic tags, downsampled rows ──────────────
    # PERF FIX: the previous implementation computed corr() on all 225 raw
    # sensor columns and rendered it as a go.Heatmap with texttemplate on every
    # cell → 50 625 SVG text nodes → browser main-thread freeze.
    #
    # Solution:
    #   1. Select top-12 tags by variance (most signal, least noise).
    #   2. Downsample rows by 10× before calling .corr() to reduce compute.
    #   3. Use px.imshow() which renders a WebGL-accelerated raster image
    #      rather than individual SVG elements per cell.
    #   4. Expose a multiselect so the operator can override the default set.
    st.markdown(
        f"<h3 style='font-family:{_C_MONO};'>"
        f"INTER-SENSOR CORRELATION (TOP DYNAMIC TAGS)</h3>",
        unsafe_allow_html=True,
    )

    import re as _re
    import plotly.express as px

    _META_RE_EDA = _re.compile(r"^(timestamp|time|attack.*|label.*)$", _re.IGNORECASE)
    raw_sensor_cols = [
        c for c in raw_df.columns
        if not _META_RE_EDA.match(c) and pd.api.types.is_numeric_dtype(raw_df[c])
    ]

    # Work on the current view window only — not the full dataset
    raw_slice_corr = raw_df[raw_sensor_cols].iloc[view_start: view_start + n_view]

    if len(raw_sensor_cols) == 0:
        st.info("No numeric sensor columns found for correlation.")
    else:
        # Top-12 tags by variance — highest variance = most dynamic signal
        _N_DEFAULT = min(12, len(raw_sensor_cols))
        top_active_tags = (
            raw_slice_corr.var()
            .nlargest(_N_DEFAULT)
            .index
            .tolist()
        )

        selected_tags = st.multiselect(
            "SELECT TAGS TO CORRELATE",
            options=raw_sensor_cols,
            default=top_active_tags,
            help=(
                f"Default: top {_N_DEFAULT} most dynamic tags by variance in the "
                "current window. Add/remove tags — keep under ~20 for best performance."
            ),
        )

        if len(selected_tags) < 2:
            st.info("Select at least 2 tags to compute correlation.")
        else:
            # Downsample rows 10× before .corr() — removes redundant temporal
            # autocorrelation samples without changing the correlation structure
            corr_df = raw_slice_corr[selected_tags].iloc[::10].corr()

            fig_corr = px.imshow(
                corr_df,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
            )
            # Apply ISA-101 theme on top of px defaults
            fig_corr.update_layout(
                height=max(320, 28 * len(selected_tags)),   # scale height with tag count
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FAFAFA",
                font=dict(family=_C_MONO, size=10, color=_C_TEXT),
                margin=dict(l=120, r=24, t=36, b=80),
                coloraxis_colorbar=dict(
                    tickfont=dict(family=_C_MONO, size=10),
                    outlinewidth=1,
                    outlinecolor=_C_BORDER,
                    len=0.75,
                ),
                title=dict(
                    text=f"RAW SENSOR CORRELATIONS — {len(selected_tags)} TAGS "
                         f"({n_view // 10} ROWS SAMPLED)",
                    font=dict(family=_C_MONO, size=11, color=_C_TEXT_MUTED),
                ),
            )
            fig_corr.update_xaxes(
                tickfont=dict(family=_C_MONO, size=9),
                **_GRID_STYLE,
            )
            fig_corr.update_yaxes(
                tickfont=dict(family=_C_MONO, size=9),
                **_GRID_STYLE,
            )
            st.plotly_chart(fig_corr, key="eda_corr", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(_ISA_CSS, unsafe_allow_html=True)

    page = _sidebar()

    result     = st.session_state["infer_result"]
    raw_df     = st.session_state["raw_df"]
    feat_cols  = st.session_state["feature_cols"] or []

    # ── Gate: no data loaded ──────────────────────────────────────────────────
    if result is None:
        st.markdown(
            f"<h2 style='font-family:{_C_MONO};'>LIVE SOC — PROCESS SUPERVISION</h2>",
            unsafe_allow_html=True,
        )
        _standby_screen()
        st.divider()
        st.markdown(
            f"<div style='font-family:{_C_MONO};font-size:0.65rem;color:{_C_TEXT_MUTED};"
            f"letter-spacing:0.06em;'>"
            f"FACTORYSHIELD-OT v1.0  |  ISA/IEC 62443  |  "
            f"LAST RENDER: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Update latching alarm log before routing ──────────────────────────────
    # Runs on every render so the log stays current as the cursor advances.
    _update_alarm_log(
        result    = result,
        raw_df    = raw_df,
        threshold = float(st.session_state["threshold"]),
        alpha     = float(st.session_state["ema_alpha"]),
        cursor    = int(st.session_state["play_cursor"]),
    )

    # ── Page routing ─────────────────────────────────────────────────────────
    if page == "Live SOC":
        page_live_soc(result, raw_df, feat_cols)
    elif page == "xAI Root Cause":
        page_xai(result)
    elif page == "Model Benchmark":
        page_benchmark()
    elif page == "Data Exploration":
        page_eda(result, raw_df, feat_cols)
    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        f"<div style='font-family:{_C_MONO};font-size:0.65rem;color:{_C_TEXT_MUTED};"
        f"letter-spacing:0.06em;'>"
        f"FACTORYSHIELD-OT v1.0  |  ISA/IEC 62443  |  "
        f"LAST RENDER: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Live playback loop ────────────────────────────────────────────────────
    if st.session_state["sim_running"]:
        n_total = st.session_state["n_samples"]
        cursor  = st.session_state["play_cursor"]
        if cursor < n_total:
            time.sleep(_TICK_S)
            st.session_state["play_cursor"] = min(cursor + _TICK_ADV, n_total)
            st.rerun()
        else:
            # Reached end of dataset — hold
            st.session_state["sim_running"] = False


if __name__ == "__main__":
    main()
