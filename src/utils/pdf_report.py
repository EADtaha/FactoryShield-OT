"""
src/utils/pdf_report.py
=======================
Forensic-grade incident audit report generator for FactoryShield-OT.

Uses fpdf2 exclusively (no reportlab, no weasyprint, no browser rendering).
All output is written into an in-memory BytesIO buffer — no intermediate files
are ever written to disk.

Public API
----------
    generate_incident_pdf(incidents: list[dict], result_df: pd.DataFrame,
                          feat_cols: list[str], threshold: float,
                          alpha: float) -> bytes

    Returns a PDF as raw bytes suitable for st.download_button(data=...).

Design rules (ISA-101 / ISA-18.2 compliant)
--------------------------------------------
  • Helvetica for body text, Courier for telemetry values and tag names.
  • Black and mid-grey only — no decorative colour fill.
  • Safety-red rule lines for CRITICAL/HIGH severity banners.
  • Strict tabular layout; no images, no gradients, no rounded boxes.
  • Every page carries a header with document title and page number.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fpdf import FPDF
from scipy.signal import lfilter


# ── ISA-101 colour palette (greyscale + safety red) ─────────────────────────
_BLACK      = (17,  24,  39)    # #111827
_DARK_GREY  = (75,  85,  99)    # #4B5563
_MID_GREY   = (156, 163, 175)   # #9CA3AF
_LIGHT_GREY = (229, 231, 235)   # #E5E7EB
_WHITE      = (255, 255, 255)
_ALARM_RED  = (220, 38,  38)    # #DC2626
_WARN_AMBER = (217, 119, 6)     # #D97706

# Severity → border colour
_SEV_COLOR = {
    "CRITICAL": _ALARM_RED,
    "HIGH":     _ALARM_RED,
    "MEDIUM":   _WARN_AMBER,
    "LOW":      _MID_GREY,
}

# Column widths (mm) for the top-k sensor table
_COL_W = [14, 52, 48, 32, 30]   # Rank | Tag | Subsystem | MAE | % Contrib
_COL_H = 7                       # row height (mm)

# Subsystem classifier (mirrors _classify_subsystems in app.py but as a
# pure function — no Streamlit dependency)
_P1_BOILER_TOKENS = (
    "1004", "1003", "DM-TIT", "DM-PIT", "DM-PCV", "DM-FCV",
    "DM-PP04", "DM-LIT", "DM-PP01", "B2016", "B3004", "B3005",
    "PIT", "TIT", "LIT", "PCV", "FCV", "LCV",
)
_P1_FW_TOKENS = (
    "1001", "1002", "1010", "1011",
    "DM-FT", "DM-LCV", "DM-SOL", "DM-CIP", "DM-COOL",
    "DM-LSH", "DM-LSL", "DM-SW", "DM-SS", "DM-ST",
    "PP04-SP", "DM-HT",
)
_P2_TOKENS = ("GT", "SPD", "VLV_T", "P2_", "1020")
_P3_TOKENS = (
    "PT01", "PT02", "LT01", "FT01", "FT02",
    "PMP", "AIT", "P3_", "DM-AIT", "DM-TWIT", "DM-PWIT", "DQ", "GATEOPEN",
)

_SUBSYSTEM_LABELS = {
    "P1-BOILER":      "P1 - Boiler / Pressure Control (Emerson Ovation DCS)",
    "P1-FEEDWATER":   "P1 - Feedwater / Level Control",
    "P2-TURBINE":     "P2 - Turbine (GE Mark VIe)",
    "P3-WATER-TREAT": "P3 - Water Treatment (Siemens S7-300)",
    "UNKNOWN":        "Unknown / Cross-subsystem",
}

_DIRECTIVES = {
    "P1-BOILER": (
        "ACTION REQUIRED: Immediate isolation of Loop P1. Switch DCS function "
        "blocks to manual override. Verify physical transmitter calibration on "
        "PIT01/TIT01. Check emergency relief valve status. Log event in Ovation "
        "historian and notify process safety officer."
    ),
    "P1-FEEDWATER": (
        "ACTION REQUIRED: Switch feedwater control loop to manual mode. "
        "Cross-check LIT01 level reading against independent gauge. Inspect "
        "LCV01 valve position and actuator feedback. Verify DCS I/O module "
        "health on the affected fieldbus segment."
    ),
    "P2-TURBINE": (
        "ACTION REQUIRED: Cross-verify governor valve positions via GE Mark "
        "VIe HMI. Inspect trip relay status and vibration amplitude thresholds. "
        "If SPD sensor discrepancy is confirmed, engage turbine runback "
        "procedure and notify mechanical engineering."
    ),
    "P3-WATER-TREAT": (
        "ACTION REQUIRED: Inspect secondary pump line telemetry and verify "
        "Siemens S7-300 PLC bus integrity on the water treatment segment. "
        "Check AIT sensor readings for drift and confirm GATEOPEN state matches "
        "physical valve position."
    ),
    "UNKNOWN": (
        "Verify DCS historian data integrity for the affected time window. "
        "Isolate the fieldbus segment from Layer 2 routing and initiate "
        "physical loop inspection. Preserve raw tag logs as forensic evidence "
        "and escalate to the OT Security team."
    ),
}


# ────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Sanitise a string for FPDF core fonts (Helvetica / Courier).

    Core fonts in fpdf2 are limited to Latin-1 (ISO 8859-1).  Any character
    outside that range — em-dashes, curly quotes, ellipsis glyphs, mathematical
    symbols — raises a CharacterNotFound error at render time.

    Strategy:
      1. Replace the most common typographic substitutions with ASCII equivalents.
      2. Encode → decode through latin-1 with 'replace' as a safety net so that
         any character we didn't explicitly map becomes '?' rather than crashing.
    """
    if not isinstance(text, str):
        text = str(text)

    _REPLACEMENTS = {
        "\u2014": "-",    # em-dash  —
        "\u2013": "-",    # en-dash  –
        "\u2012": "-",    # figure dash
        "\u2011": "-",    # non-breaking hyphen
        "\u2010": "-",    # hyphen
        "\u201c": '"',    # left double quotation mark  "
        "\u201d": '"',    # right double quotation mark "
        "\u2018": "'",    # left single quotation mark  '
        "\u2019": "'",    # right single quotation mark '
        "\u2026": "...",  # horizontal ellipsis  …
        "\u2022": "*",    # bullet  •
        "\u2265": ">=",   # greater-than-or-equal  ≥
        "\u2264": "<=",   # less-than-or-equal  ≤
        "\u2248": "~",    # almost equal  ≈
        "\u00b0": "deg",  # degree sign  °
        "\u00b5": "u",    # micro sign  µ
        "\u03b1": "alpha",# α
        "\u03b2": "beta", # β
        "\u00d7": "x",    # multiplication sign  ×
        "\u00f7": "/",    # division sign  ÷
        "\u00a0": " ",    # non-breaking space
        "\u00ad": "-",    # soft hyphen
    }
    for orig, rep in _REPLACEMENTS.items():
        text = text.replace(orig, rep)

    # Final safety net: drop anything still outside latin-1
    return text.encode("latin-1", "replace").decode("latin-1")

def _classify_tag(tag: str) -> str:
    """Return the primary subsystem key for a single sensor tag."""
    t = tag.upper()
    if any(p.upper() in t for p in _P1_BOILER_TOKENS):
        return "P1-BOILER"
    if any(p.upper() in t for p in _P1_FW_TOKENS):
        return "P1-FEEDWATER"
    if any(p.upper() in t for p in _P2_TOKENS):
        return "P2-TURBINE"
    if any(p.upper() in t for p in _P3_TOKENS):
        return "P3-WATER-TREAT"
    if "DM-" in t:
        return "P1-BOILER"   # DM- catch-all
    return "UNKNOWN"


def _severity_from_risk(risk_str: str) -> str:
    """Normalise the stored [ HIGH ] / HIGH / High → uppercase bare word."""
    clean = risk_str.strip().strip("[]").strip().upper()
    return clean if clean in _SEV_COLOR else "LOW"


def _recompute_ema(mae: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    return lfilter([alpha], [1.0, -(1.0 - alpha)], mae).astype(np.float32)


def _build_sensor_rows(
    incident:   dict,
    result_df:  pd.DataFrame,
    feat_cols:  list[str],
    threshold:  float,
    alpha:      float,
    top_k:      int = 5,
) -> list[dict]:
    """Reconstruct per-sensor top-k rows for one incident from the lean result DF.

    Uses the compact top_sensor_idx / top_sensor_err columns stored during
    inference (no full 225-col residual matrix needed).
    Returns a list of dicts: rank, sensor, subsystem, error, pct_contribution.
    """
    _TOP_K_STORE = 5
    start_idx  = incident.get("_start_idx", 0)
    end_idx    = start_idx + incident.get("DURATION (s)", 1)
    end_idx    = min(end_idx, len(result_df))

    idx_cols = [f"top_sensor_idx_{k}" for k in range(_TOP_K_STORE)]
    err_cols = [f"top_sensor_err_{k}" for k in range(_TOP_K_STORE)]

    if not all(c in result_df.columns for c in idx_cols + err_cols):
        return []

    # Find the peak-MAE window within the segment
    mae_seg  = result_df["mae_raw"].iloc[start_idx:end_idx].to_numpy()
    if len(mae_seg) == 0:
        return []
    peak_off = int(np.argmax(mae_seg))
    peak_row = start_idx + peak_off

    raw_idx = result_df[idx_cols].iloc[peak_row].to_numpy().astype(int)
    raw_err = result_df[err_cols].iloc[peak_row].to_numpy().astype(np.float32)

    k        = min(top_k, _TOP_K_STORE)
    sel_idx  = raw_idx[:k]
    sel_err  = raw_err[:k]
    total    = float(sel_err.sum())

    rows = []
    for rank, (fi, err) in enumerate(zip(sel_idx, sel_err), 1):
        fi   = int(fi)
        name = feat_cols[fi] if fi < len(feat_cols) else f"sensor_{fi}"
        rows.append({
            "rank":             rank,
            "sensor":           name,
            "subsystem":        _SUBSYSTEM_LABELS.get(_classify_tag(name), "Unknown"),
            "error":            float(err),
            "pct_contribution": round(float(err / total * 100) if total > 0 else 0.0, 2),
        })
    return rows


# ────────────────────────────────────────────────────────────────────────────
# PDF ENGINE
# ────────────────────────────────────────────────────────────────────────────

class IncidentAuditPDF(FPDF):
    """ISA-101 / ISA-18.2 compliant forensic audit report.

    Typography:
      • Helvetica — all prose, headings, table headers
      • Courier   — sensor tags, timestamps, MAE values (monospaced telemetry)
    Colour: black + mid-grey + safety-red for severity banners only.
    No images, no decorative fills, no rounded boxes.
    """

    _TITLE     = "FACTORYSHIELD-OT"
    _SUBTITLE  = "FORENSIC INCIDENT AUDIT REPORT"
    _SYSTEM    = "Emerson Ovation DCS / GE Mark VIe / Siemens S7-300  |  HAIEnd 23.05"
    _TARGET    = "Level 1 Process Control (Purdue Model)  |  ISA/IEC 62443-3-3"
    _PAGE_W    = 210   # A4 mm
    _MARGIN    = 15
    _BODY_W    = 210 - 2 * 15   # 180 mm

    def __init__(self, generated_at: str, n_incidents: int, max_severity: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(self._MARGIN, self._MARGIN, self._MARGIN)
        self._generated_at  = generated_at
        self._n_incidents   = n_incidents
        self._max_severity  = max_severity

    # ── Sanitising wrappers — intercept ALL text before it reaches FPDF ──────
    # Every cell() / multi_cell() call in this class routes through these
    # overrides, so no individual call site needs a _clean_text() wrapper.
    def cell(self, w=0, h=0, txt="", **kwargs):
        return super().cell(w, h, _clean_text(txt), **kwargs)

    def multi_cell(self, w, h, txt="", **kwargs):
        return super().multi_cell(w, h, _clean_text(txt), **kwargs)

    # ── Running header (every page) ──────────────────────────────────────────
    def header(self) -> None:
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_DARK_GREY)
        self.cell(0, 5, self._TITLE + " - " + self._SUBTITLE, align="L")
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*_MID_GREY)
        self.set_line_width(0.3)
        self.line(self._MARGIN, self.get_y(), self._PAGE_W - self._MARGIN, self.get_y())
        self.ln(2)

    # ── Running footer ───────────────────────────────────────────────────────
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_MID_GREY)
        self.cell(0, 5,
                  f"CONFIDENTIAL — OT SECURITY OPERATIONS  |  {self._TARGET}",
                  align="C")

    # ── Section rule ─────────────────────────────────────────────────────────
    def _rule(self, color: tuple = _LIGHT_GREY, width: float = 0.4) -> None:
        self.set_draw_color(*color)
        self.set_line_width(width)
        self.line(self._MARGIN, self.get_y(),
                  self._PAGE_W - self._MARGIN, self.get_y())
        self.ln(2)

    # ── Key-value row ────────────────────────────────────────────────────────
    def _kv(self, key: str, value: str,
            value_font: str = "Helvetica") -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_DARK_GREY)
        self.cell(55, 6, key + ":", align="L")
        self.set_font(value_font, "", 9)
        self.set_text_color(*_BLACK)
        self.multi_cell(0, 6, value, align="L",
                        new_x="LMARGIN", new_y="NEXT")

    # ── Cover / metadata page ────────────────────────────────────────────────
    def build_cover(self) -> None:
        self.add_page()

        # Title block
        self.ln(6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*_BLACK)
        self.cell(0, 10, self._TITLE, align="C", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*_DARK_GREY)
        self.cell(0, 8, self._SUBTITLE, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

        self._rule(_MID_GREY, 0.6)

        # Metadata grid
        self.ln(4)
        self._kv("System",         self._SYSTEM)
        self._kv("Target",         self._TARGET)
        self._kv("Generated (UTC)", self._generated_at, value_font="Courier")
        self._kv("Total Incidents", str(self._n_incidents), value_font="Courier")

        sev_color = _SEV_COLOR.get(self._max_severity, _MID_GREY)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_DARK_GREY)
        self.cell(55, 6, "Max Severity Observed:", align="L")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*sev_color)
        self.cell(0, 6, f"[ {self._max_severity} ]",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*_BLACK)

        self.ln(4)
        self._rule(_LIGHT_GREY)
        self.ln(2)

        # Document purpose note
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_DARK_GREY)
        self.multi_cell(
            0, 5,
            "This document is automatically generated by the FactoryShield-OT "
            "LSTM Autoencoder anomaly detection system. It is intended as a "
            "first-response forensic aid for OT Security Operations personnel. "
            "All reconstruction errors are normalised MAE values (dimensionless, "
            "range [0, 1]). Timestamps refer to the HAIEnd 23.05 simulation clock "
            "unless a real-time data source is connected.",
            align="L",
        )

    # ── Incident section ─────────────────────────────────────────────────────
    def build_incident(
        self,
        incident:    dict,
        sensor_rows: list[dict],
    ) -> None:
        self.add_page()

        seq        = incident.get("SEQ", "?")
        start      = incident.get("START", "N/A")
        end        = incident.get("END",   "N/A")
        duration   = incident.get("DURATION (s)", 0)
        peak_mae   = incident.get("PEAK MAE", "N/A")
        risk_raw   = incident.get("RISK", "LOW")
        severity   = _severity_from_risk(risk_raw)
        sev_color  = _SEV_COLOR.get(severity, _MID_GREY)

        # Primary subsystem from the rank-1 sensor (if available)
        if sensor_rows:
            primary_tag = sensor_rows[0]["sensor"]
            primary_sub = _SUBSYSTEM_LABELS.get(
                _classify_tag(primary_tag), "Unknown"
            )
            directive_key = _classify_tag(primary_tag)
        else:
            primary_tag   = "N/A"
            primary_sub   = "N/A"
            directive_key = "UNKNOWN"

        # ── 1. Incident banner ───────────────────────────────────────────────
        self.set_draw_color(*sev_color)
        self.set_line_width(1.0)
        self.rect(self._MARGIN,
                  self.get_y(),
                  self._BODY_W, 12)
        self.set_fill_color(*_WHITE)
        self.rect(self._MARGIN, self.get_y(),
                  self._BODY_W, 12, style="F")

        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*sev_color)
        banner = (
            f"INCIDENT #{seq}  -  {start}  to  {end}"
            f"  |  DURATION: {duration}s  |  SEVERITY: {severity}"
        )
        self.set_xy(self._MARGIN + 2, self.get_y() + 2)
        self.cell(self._BODY_W - 4, 8, banner, align="L")
        self.set_xy(self._MARGIN, self.get_y() + 8)
        self.ln(2)

        # ── 2. Summary table ─────────────────────────────────────────────────
        self.ln(3)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_DARK_GREY)
        self.cell(0, 6, "INCIDENT SUMMARY", new_x="LMARGIN", new_y="NEXT")
        self._rule(_LIGHT_GREY)

        self._kv("Start Time",                start,       "Courier")
        self._kv("End Time",                  end,         "Courier")
        self._kv("Duration",                  f"{duration} s", "Courier")
        self._kv("Primary Subsystem",         primary_sub)
        self._kv("Primary Root-Cause Tag",    primary_tag, "Courier")
        self._kv("Peak Reconstruction MAE",   peak_mae,    "Courier")

        # ── 3. Top-K sensor table ────────────────────────────────────────────
        self.ln(4)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_DARK_GREY)
        self.cell(0, 6, "TOP ANOMALOUS SENSORS AT ONSET",
                  new_x="LMARGIN", new_y="NEXT")
        self._rule(_LIGHT_GREY)

        if sensor_rows:
            # Header row
            headers = ["Rank", "Sensor Tag", "Subsystem", "MAE", "% Contrib"]
            self.set_font("Helvetica", "B", 8)
            self.set_fill_color(*_LIGHT_GREY)
            self.set_text_color(*_BLACK)
            self.set_draw_color(*_MID_GREY)
            self.set_line_width(0.2)
            for h, w in zip(headers, _COL_W):
                self.cell(w, _COL_H, h, border=1, fill=True, align="C")
            self.ln()

            for sr in sensor_rows:
                rank_str  = str(sr["rank"])
                tag_str   = sr["sensor"]
                sub_str   = sr["subsystem"]
                err_str   = f"{sr['error']:.5f}"
                pct_str   = f"{sr['pct_contribution']:.1f}%"

                # Rank-1 row gets a subtle left-border highlight
                if sr["rank"] == 1:
                    self.set_text_color(*sev_color)
                    self.set_font("Helvetica", "B", 8)
                else:
                    self.set_text_color(*_BLACK)
                    self.set_font("Helvetica", "", 8)

                self.cell(_COL_W[0], _COL_H, rank_str,  border=1, align="C")
                self.set_font("Courier", "B" if sr["rank"] == 1 else "", 8)
                self.set_text_color(*sev_color if sr["rank"] == 1 else _BLACK)
                self.cell(_COL_W[1], _COL_H, tag_str,   border=1, align="L")
                self.set_font("Helvetica", "", 8)
                self.set_text_color(*_BLACK)
                # Subsystem may be long — truncate to fit column
                sub_disp = sub_str[:34] if len(sub_str) > 34 else sub_str
                self.cell(_COL_W[2], _COL_H, sub_disp,  border=1, align="L")
                self.set_font("Courier", "", 8)
                self.cell(_COL_W[3], _COL_H, err_str,   border=1, align="R")
                self.cell(_COL_W[4], _COL_H, pct_str,   border=1, align="R")
                self.ln()

            self.set_text_color(*_BLACK)
            self.set_font("Helvetica", "", 9)
        else:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*_DARK_GREY)
            self.cell(0, 6, "[ Sensor-level breakdown not available for this incident ]",
                      new_x="LMARGIN", new_y="NEXT")

        # ── 4. Operator directive ────────────────────────────────────────────
        self.ln(4)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_DARK_GREY)
        self.cell(0, 6, "ACTIONABLE DIRECTIVE — PLANT OPERATOR",
                  new_x="LMARGIN", new_y="NEXT")
        self._rule(_LIGHT_GREY)

        directive = _DIRECTIVES.get(directive_key, _DIRECTIVES["UNKNOWN"])
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_BLACK)
        self.multi_cell(0, 5, directive, align="L",
                        new_x="LMARGIN", new_y="NEXT")


# ────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION
# ────────────────────────────────────────────────────────────────────────────

def generate_incident_pdf(
    logged_incidents: list[dict],
    result_df:        pd.DataFrame,
    feat_cols:        list[str],
    threshold:        float,
    alpha:            float,
) -> bytes:
    """Generate a forensic incident audit PDF and return it as raw bytes.

    Parameters
    ----------
    logged_incidents : list[dict]
        Records from st.session_state["logged_alarms"]; each dict must contain
        the keys produced by _update_alarm_log: SEQ, START, END, DURATION (s),
        PEAK MAE, RISK, STATUS, _start_idx.
    result_df : pd.DataFrame
        The lean inference result (mae_raw, top_sensor_idx_0..4, etc.)
        from _run_full_inference, used to reconstruct per-sensor rankings.
    feat_cols : list[str]
        Canonical sensor column names in LSTM feature order — same list stored
        in st.session_state["feature_cols"].
    threshold : float
        Current alarm threshold (from st.session_state["threshold"]).
    alpha : float
        EMA smoothing factor (from st.session_state["ema_alpha"]).

    Returns
    -------
    bytes
        Raw PDF bytes ready for st.download_button(data=...).
    """
    if not logged_incidents:
        raise ValueError("No incidents to report — logged_incidents is empty.")

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Derive max severity across all closed + active incidents
    _sev_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    severities  = [_severity_from_risk(inc.get("RISK", "LOW"))
                   for inc in logged_incidents]
    max_sev     = max(severities, key=lambda s: _sev_order.get(s, 0))

    pdf = IncidentAuditPDF(
        generated_at = generated_at,
        n_incidents  = len(logged_incidents),
        max_severity = max_sev,
    )

    pdf.build_cover()

    for incident in logged_incidents:
        sensor_rows = _build_sensor_rows(
            incident  = incident,
            result_df = result_df,
            feat_cols = feat_cols,
            threshold = threshold,
            alpha     = alpha,
        )
        pdf.build_incident(incident, sensor_rows)

    # Output to in-memory bytes — no disk I/O
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
