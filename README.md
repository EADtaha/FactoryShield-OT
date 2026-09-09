# FactoryShield-OT

**Streamlit SOC Dashboard — User Guide & Technical Documentation**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/streamlit-%3E%3D1.35-red)]()
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.1-orange)]()
[![Standard](https://img.shields.io/badge/standard-ISA%2FIEC%2062443-lightgrey)]()

| | |
|---|---|
| **Version** | 1.0 |
| **Author** | EMSI Internship Project — OT/ICS Cybersecurity & AI Track |
| **Standards** | ISA/IEC 62443 (Industrial Cybersecurity) |
| **Dataset** | HAIEnd 23.05 / HAI 23.05 (KISA/NSHC) |
| **Target DCS** | Emerson Ovation (Boiler P1) · GE Mark VIe (Turbine P2) · Siemens S7-300 (Water Treatment P3) · dSPACE SCALEXIO HIL Simulator (P4) |

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Application Structure & Navigation](#application-structure--navigation)
  - [Page 1 — Live SOC](#page-1--live-soc-live-mini-soc-supervision)
  - [Page 2 — xAI Root Cause Analysis](#page-2--xai-root-cause-analysis)
  - [Page 3 — Model Benchmark](#page-3--model-benchmark)
  - [Page 4 — Data Exploration (EDA)](#page-4--data-exploration-eda)
- [Configuration Parameters](#configuration-parameters-configshyperparametersyaml)
- [Running the Full Pipeline (CLI Reference)](#running-the-full-pipeline-cli-reference)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Project Directory Layout](#project-directory-layout)
- [Scientific References](#scientific-references)
- [License & Academic Use](#license--academic-use)

---

## Overview

FactoryShield-OT is an intelligent **Cyber-Physical Anomaly Detection and Industrial Risk Scoring** platform built for Operational Technology (OT) and Industrial Control Systems (ICS) environments.

Unlike conventional IT cybersecurity tools, FactoryShield-OT monitors **physical process integrity** — thermodynamic cycles, control-loop correlations, and function-block logic — rather than network traffic or data confidentiality.

**Core capabilities of the Streamlit dashboard:**

- **Real-time SOC simulation** with live process signal curves and EMA-smoothed anomaly scores streamed window-by-window through the trained LSTM Autoencoder.
- **4-level industrial risk gauge** (Low / Medium / High / Critical) with operator safety procedure recommendations for each severity level.
- **Zero-overhead xAI root-cause analysis** — feature-wise reconstruction error ranking identifies the primary attack injection point without SHAP or any external explainability library.
- **Multi-model benchmark view** comparing Isolation Forest, One-Class SVM, XGBoost, and the LSTM Autoencoder across standard and time-aware metrics.
- **Exploratory Data Analysis page** with tag time-series, error distribution histograms, percentile markers, and inter-sensor correlation heatmaps.

---

## System Requirements

| Requirement | Detail |
|---|---|
| **Python** | 3.10+ (3.11 recommended) |
| **OS** | Linux / macOS / Windows (WSL2 recommended on Windows) |
| **GPU** | Optional — CUDA-capable GPU accelerates training only; dashboard runs fine on CPU |
| **RAM** | 8 GB minimum (16 GB recommended for the full 225-feature dataset) |
| **Disk** | ~2 GB for dataset CSVs + model checkpoints |

**Python packages** (see `requirements.txt` for pinned versions):

```
numpy, pandas, scipy
scikit-learn, xgboost, joblib
torch (PyTorch >= 2.1)
streamlit >= 1.35
plotly >= 5.18
PyYAML >= 6.0.1
fpdf2, tqdm, matplotlib, seaborn
```

---

## Installation

```bash
# 1. Clone the project
git clone <repo-url>
cd FactoryShield-OT

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**4. Place the HAIEnd 23.05 dataset CSVs in:**

```
data/raw/     ← original files (end-train*.csv, end-test*.csv, label-test*.csv)
```

```bash
# 5. Run preprocessing (generates scaled CSVs in data/processed/)
python src/data/preprocess.py

# 6. Train the LSTM Autoencoder (optional — pre-trained weights may exist)
python src/models/train.py \
    --train-csv data/processed/clean_100/train_clean_scaled.csv \
    --config    configs/hyperparameters.yaml \
    --epochs    50

# 7. Launch the Streamlit dashboard
streamlit run app/app.py
```

The app opens automatically at **http://localhost:8501**

---

## Application Structure & Navigation

The sidebar on the left contains:
- Page selector (radio buttons for the 4 dashboard pages)
- Detection settings — Alert threshold slider, EMA α slider
- Simulation controls — Start / Pause / Reset buttons
- Version and standards information

### Page 1 — Live SOC (Live Mini-SOC Supervision)

**Purpose:** Simulate real-time monitoring of the Emerson Ovation DCS by streaming pre-processed data through the LSTM Autoencoder window by window and displaying the resulting anomaly scores.

**KPI strip** (4 metric cards at the top)

| Metric | Description |
|---|---|
| Process status | Running / Stopped |
| Active alert segments | Count of contiguous alert windows currently firing |
| Current risk level | Low / Medium / High / Critical |
| Peak EMA error | Maximum smoothed MAE value vs. configured threshold |

**Industrial risk gauge** — a Plotly indicator gauge with four colour-coded zones:

| Zone | Range | Level | Meaning |
|---|---|---|---|
| 🟢 Green | 0–30 | Low | Micro-drifts, surveillance mode |
| 🟠 Amber | 30–60 | Medium | Moderate deviation, minor internal impact |
| 🔴 Red | 60–85 | High | Severe correlation breach, critical equipment |
| 🔺 Crimson | 85–100 | Critical | Confirmed cyber-attack or physical DoS threat |

**4-row synchronised time-series chart**

1. **Process signals** — PIT01 (pressure), TIT01 (temperature), FT03 (flow rate), LIT01 (tank level). Red shading marks time windows where alerts are active.
2. **Reconstruction MAE** — raw MAE (dotted grey) and EMA-smoothed MAE (purple, filled). Red dashed line = alert threshold.
3. **Alert binary** — 0/1 step function, red fill.
4. **Risk level** — 0 (Low) → 3 (Critical), colour-mapped.

**Alert segment table** — lists each contiguous alert burst with start time, end time, duration in seconds, peak MAE, and assigned risk level.

**Operator safety recommendations** (expandable panel), dynamically generated based on current risk level:

| Level | Recommendation |
|---|---|
| Critical | Initiate process shutdown. Notify plant manager. Isolate loops. |
| High | Switch to manual. Verify physical sensors. Log incident. |
| Medium | Monitor closely. Cross-check redundant sensors. |
| Low | Continue surveillance. |

**Sidebar controls that affect this page**

- **Alert threshold** — raise to reduce false positives; lower to increase sensitivity. Recommended starting point: 99th-percentile of normal training errors (saved automatically during training as `threshold_p99` inside the `.pth` checkpoint).
- **EMA α** — exponential moving-average smoothing factor. α = 0.10 means 10% weight on the newest sample. Smaller values smooth more aggressively and increase detection latency; larger values are more reactive but noisier.

### Page 2 — xAI Root Cause Analysis

**Purpose:** Identify which sensor tags (process variables) are driving each anomaly detection, enabling operators to locate the physical point of attack or fault without black-box explainability overhead.

**Methodology**

For every alert onset (first timestep of a new alert burst), the per-sensor absolute reconstruction error vector is computed:

```
E_{t,f} = | x_{t,f} - x̂_{t,f} |    for each feature f at time t
```

Sensors are ranked by `E_{t,f}` in descending order. Rank-1 sensor is designated the attack injection point / primary anomalous component.

The percentage contribution of each sensor to the total error at that timestep:

```
pct_contribution_f = E_{t,f} / Σ_f E_{t,f} × 100
```

Each sensor is mapped to its physical subsystem:

| Subsystem | Sensors |
|---|---|
| P1 – Pressure Control | PIT01, PCV01D, PCV02D, B2016 |
| P1 – Level Control | LIT01, LCV01D, B3004 |
| P1 – Flow Control | FT03, FCV03D, B3005 |
| P1 – Temperature Ctrl | TIT01, FCV01D, FCV02D, B4022 |
| P1 – Cooling Control | TIT03, PP04 |
| P2 – Turbine | GT01, GT02, SPD01, SPD02 |
| P3 – Water Treatment | FT01, FT02, PT01, LT01, PMP01 |
| P4 – HIL Simulator | HIL01, HIL02 |

**Components**

- **Methodology info box** — explains the zero-overhead xAI approach.
- **Top-K slider** — number of sensors to show per onset (3–10).
- **Per-onset expanders** — one collapsible panel per alert onset, labelled with the onset timestep, wall-clock time, and root-cause sensor name.

Inside each expander:
- **Bar chart** — sensors on X-axis, error magnitude on Y-axis. Rank-1 bar coloured Crimson, Rank-2 Red, remaining Grey. Percentage labels on bars.
- **Detail table** — `rank | sensor | error | % contrib | subsystem` columns.

### Page 3 — Model Benchmark

**Purpose:** Compare detection performance across all trained models to justify the semi-supervised LSTM Autoencoder approach and quantify its advantage over classical baselines.

**Raw evaluation reports** (collapsible) — if `out/evaluation_report_clean_100.txt` and/or `out/evaluation_report_mixed_70_30.txt` exist (produced by `python evaluate.py`), they are displayed as pre-formatted code blocks.

**Benchmark overview table** — static indicative metrics from the plan targets:

| Model | Precision | Recall | F1 | eTaF1 | Latency | Type |
|---|---|---|---|---|---|---|
| Isolation Forest | ~72% | ~68% | ~70% | ~0.61 | < 1s | Unsupervised |
| One-Class SVM | ~69% | ~65% | ~67% | ~0.58 | < 1s | Unsupervised |
| XGBoost | ~88% | ~85% | ~86% | ~0.79 | < 1s | Supervised |
| **LSTM Autoencoder** | **~91%** | **~89%** | **~90%** | **~0.87** | 3–5s | Semi-supervised |

> Run `python evaluate.py` after training for exact figures on your data.

**ROC-AUC curves** — displays `out/roc_curve_clean_100.png` and `out/roc_curve_mixed_70_30.png` side by side if they exist.

**eTaPR explainer** (collapsible) — describes the Enhanced Time-Aware Precision/Recall metric used in the HAICon 2021 competition and why it is more meaningful for ICS anomaly detection than point-wise F1.

### Page 4 — Data Exploration (EDA)

**Purpose:** Visually explore the process signals and reconstruction error distribution to support threshold selection, sensor selection, and data quality checks.

- **Time-series viewer** — multi-select widget to choose one or more process variables. Selected signals are plotted on a shared time axis with hover-linked tooltips.
- **MAE histogram** — distribution of raw reconstruction errors across all simulation windows. Red dashed vertical line shows the current alert threshold.
- **CDF with percentile markers** — cumulative distribution function of MAE. Dotted orange vertical lines mark the 90th, 95th, and 99th percentiles, helping operators calibrate the threshold to an acceptable false-positive rate.
- **Sensor correlation matrix** — heatmap of Pearson correlations between per-sensor reconstruction errors. High positive correlation indicates a shared physical cause; low correlation suggests sensor-specific faults.

---

## Configuration Parameters (`configs/hyperparameters.yaml`)

All parameters are documented inline in the YAML file. Key values:

| Parameter | Default | Description |
|---|---|---|
| `data.window_size` | 60 | Sliding window T (seconds) |
| `data.stride` | 1 | Stride between windows |
| `data.scaler_type` | minmax | MinMaxScaler fitted on normal training data ONLY |
| `lstm_autoencoder.hidden1` | 128 | Encoder/decoder hidden dim |
| `lstm_autoencoder.latent_dim` | 64 | Bottleneck latent vector size |
| `lstm_autoencoder.batch_size` | 64 | Training mini-batch size |
| `lstm_autoencoder.epochs` | 50 | Max training epochs |
| `lstm_autoencoder.lr` | 0.001 | Adam learning rate |
| `lstm_autoencoder.loss` | l1 | L1 (MAE) loss — matches metric |
| `detection.ema_alpha` | 0.10 | EMA smoothing factor α |
| `detection.percentile_thr` | 99.0 | p-th percentile for threshold |
| `risk_scoring.thresholds` | see YAML | Low/Med/High/Critical cutoffs |

**Threshold selection workflow**

1. Train with `--train-csv` pointing to the 100% clean dataset.
2. The training script computes the 99th-percentile MAE on training data and saves it as `threshold_p99` inside the `.pth` checkpoint.
3. The inference pipeline loads this value automatically.
4. Fine-tune via the sidebar slider in the dashboard if you observe too many false positives on normal operating data.

---

## Running the Full Pipeline (CLI Reference)

```bash
# 1. Preprocess raw CSV files
python src/data/preprocess.py

# 2. Train LSTM Autoencoder (clean_100 experiment)
python src/models/train.py \
    --train-csv data/processed/clean_100/train_clean_scaled.csv \
    --config    configs/hyperparameters.yaml \
    --epochs    50 \
    --patience  7

# 3. Run detection on test set (threshold loaded from checkpoint)
python src/detection/inference.py \
    --data   data/processed/clean_100/test_set_final.csv \
    --output out/scores_clean100.csv

# 4. Root-cause analysis on detection output
python src/detection/xai_root_cause.py \
    --errors-csv out/scores_clean100.csv \
    --top-k 5 \
    --output out/root_cause_clean100.csv

# 5. Evaluate all models (generates out/evaluation_report_*.txt + ROC PNGs)
python evaluate.py

# 6. Launch Streamlit dashboard
streamlit run app/app.py
```

---

## Troubleshooting & FAQ

<details>
<summary><strong>"No module named 'yaml'"</strong></summary>

PyYAML is not installed.

```bash
pip install PyYAML>=6.0.1
# or install all dependencies:
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>"Model checkpoint not found: models_saved/lstm_autoencoder_best.pth"</strong></summary>

The LSTM Autoencoder has not been trained yet. Run:

```bash
python src/models/train.py
```

Pre-trained baseline checkpoints (legacy architecture) are stored in `models_saved/clean_100/` and `models_saved/mixed_70_30/`.
</details>

<details>
<summary><strong>"Feature mismatch: checkpoint expects 225 features, but file has 86."</strong></summary>

You are mixing HAIEnd 23.05 (225 features) data with an HAI 23.05 (86 features) checkpoint or vice versa. Ensure the preprocessing CSV and the checkpoint were produced from the same dataset variant.
</details>

<details>
<summary><strong>The dashboard shows "No alerts in the current simulation window."</strong></summary>

Lower the Alert threshold slider in the sidebar. The simulation data contains three injected anomaly windows; a threshold above ~0.70 will suppress them all.
</details>

<details>
<summary><strong>Training is very slow on CPU.</strong></summary>

For 896,400 samples × 225 features × 60-step windows, a GPU is strongly recommended. On CPU, reduce `--epochs` to 10 and `--batch-size` to 128 for a quick smoke test. Set `--num-workers 0` to avoid DataLoader issues on some systems.
</details>

<details>
<summary><strong>CUDA out-of-memory error during training.</strong></summary>

Reduce `--batch-size` (try 32). The DataLoader uses `pin_memory=True` automatically when a GPU is detected; set `num_workers=0` if OOM persists.
</details>

<details>
<summary><strong>"No numeric feature columns found in &lt;path&gt;"</strong></summary>

The CSV passed to `inference.py` still contains only metadata columns (timestamp, attack, label). Preprocessing must produce a file where all columns except optionally `label` are scaled numeric features.
</details>

<details>
<summary><strong>The Streamlit app hot-reloads unexpectedly during simulation.</strong></summary>

This is normal — Streamlit detects file changes and reruns. Save source files only when the simulation is paused to avoid mid-run interruptions. Alternatively run with:

```bash
streamlit run app/app.py --server.runOnSave false
```
</details>

<details>
<summary><strong>How do I switch from the simulation data to real HAIEnd 23.05 data?</strong></summary>

Replace the call to `_make_sim_data()` in `app.py main()` with a function that loads your preprocessed test CSV, runs `inference.py`'s `run_inference()`, and returns the same dict structure:

```python
{timestamps, signals, mae, mae_ema, alerts, sensor_errors}
```
</details>

<details>
<summary><strong>Where are the per-model evaluation reports saved?</strong></summary>

```
out/evaluation_report_clean_100.txt
out/evaluation_report_mixed_70_30.txt
out/roc_curve_clean_100.png
out/roc_curve_mixed_70_30.png
```
</details>

---

## Project Directory Layout

```
FactoryShield-OT/
├── app/
│   └── app.py                   Streamlit entry point (this application)
├── configs/
│   ├── hyperparameters.yaml     All tunable parameters
│   └── config_loader.py         Typed dataclass loader for YAML config
├── data/
│   ├── raw/                     Original HAIEnd / HAI 23.05 CSV files
│   ├── processed/                Scaled train/test CSVs (not versioned)
│   │   ├── clean_100/
│   │   └── mixed_70_30/
│   └── external/                 Tag labels, NetworkX logic graphs
├── models_saved/
│   ├── clean_100/                Checkpoints for 100% clean experiment
│   └── mixed_70_30/              Checkpoints for 70/30 mixed experiment
├── notebooks/
│   ├── 01_exploration.ipynb      Week 1 EDA notebook
│   ├── legacy_train.py           Archived root-level training script
│   └── legacy_evaluate.py        Archived root-level evaluation script
├── out/                          Inference scores, reports, ROC images
├── src/
│   ├── data/
│   │   ├── dataloader.py         HAIDataLoader — CSV ingestion & inspection
│   │   ├── preprocess.py         HAIPreprocessor — scaling, sliding windows
│   │   └── make_splits.py        Train/test split utility
│   ├── detection/
│   │   ├── inference.py          Reconstruction → EMA → alert pipeline
│   │   ├── xai_root_cause.py     Feature-wise error ranking + subsystem map
│   │   └── risk_scoring.py       4-level industrial risk scoring engine
│   ├── models/
│   │   ├── lstm_autoencoder.py   LSTM AE architecture (225→128→64→128→225)
│   │   ├── train.py              Training loop with early stopping
│   │   └── inference.py          Model-level inference helpers
│   └── utils/
│       └── metrics.py            Precision/Recall/F1/eTaPR/latency metrics
├── logs/                         TensorBoard and run logs
├── plan.md                       Master project blueprint
├── README.md                     This file — Streamlit app user guide
└── requirements.txt              Pinned Python dependencies
```

---

## Scientific References

1. Shin, H. et al. "HAI Security Dataset" — KISA/NSHC, 2023. [github.com/icsdataset/hai](https://github.com/icsdataset/hai)
2. HAICon 2021 Competition — winning approach: semi-supervised normality learning on clean operational data. [github.com/icsdataset/haicon2021](https://github.com/icsdataset/haicon2021)
3. ISA/IEC 62443 — Security for Industrial Automation and Control Systems.
4. Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 9(8):1735–1780, 1997.

---

## License & Academic Use

This software is developed for the LPRI EMSI internship programme. Dataset usage is governed by the HAI 23.05 licence terms (KISA/NSHC). Please cite the HAI dataset if this work contributes to academic publications.
