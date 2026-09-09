# FactoryShield-OT: Master Project Blueprint & Execution Plan

---

## 1. Project Overview & Context

**FactoryShield-OT** is an intelligent Cyber-Physical Anomaly Detection and Industrial Risk Scoring Platform designed for Operational Technology (OT) and Industrial Control Systems (ICS) environments. Unlike traditional IT cybersecurity focused on data confidentiality and perimeter firewalls, FactoryShield-OT monitors physical process integrity, thermodynamic consistency, and internal controller logic behaviors.

### Key Context & References
* **Target Dataset:** HAI Security Dataset v23.05 & **HAIEnd 23.05** (HIL-based Augmented ICS Security Dataset released May 2023 by KISA/NSHC).
* **Industrial Environment:** Emulates a thermal power generation plant (Boiler process P1, Steam Turbine P2) and a pumped-storage hydropower plant (Water Treatment P3) interconnected with a dSPACE SCALEXIO Hardware-in-the-Loop (HIL) simulator (P4) and Siemens S7-1500 PLC.
* **Core Automation Target:** Emerson Ovation Distributed Control System (DCS) controlling the boiler process (225 data points including 35 SCADA I/O tags and 190 internal control logic function block edges/points).
* **Academic & Institutional Alignment:** Ecole Marocaine des Sciences de l'Ingénieur (EMSI) Internship Project / Cahier des Charges.
* **Standards Reference:** ISA/IEC 62443 (Security for Industrial Automation and Control Systems).

---

## 2. Core Strategic & Scientific Choices

1. **Rejection of Synthetic Oversampling (No SMOTE):**
   * *Rationale:* Mathematical generation of rare attack samples (e.g. SMOTE) produces physically and thermodynamically impossible system states (e.g., a valve at 100% position with 0 bar pressure). A model trained on invalid physics loses reliability in real industrial deployments.
2. **Semi-Supervised Learning ("Absolute Normality"):**
   * *Strategy:* Winning approach of the HAICon 2021 competition. The model does not learn specific attack signatures; instead, it is trained exclusively on **896,400 clean, normal operation samples**.
   * *Mechanism:* The model becomes an expert on valid thermodynamic cycles and control loop correlations. Any stealthy cyber-attack disrupts learned physical correlations, causing elevated reconstruction errors and triggering alarms.

---

## 3. The 5 Technical Layers (System Architecture)

### Layer 1: Ingestion & Temporal Preprocessing (`src/data/`)
* **Feature Scale:** 225 data features for HAIEnd 23.05 (or 86 tags for standard SCADA HAI 23.05).
* **Normalization:** `MinMaxScaler(0, 1)` fitted **strictly on the clean Train set** to prevent Data Leakage.
* **Temporal Sliding Window:**
  * Converts 2D tabular data $(N 	imes F)$ into 3D tensors $X \in \mathbb{R}^{B 	imes T 	imes F}$.
  * **Window Size ($T$):** 60 seconds (60 timesteps).
  * **Stride:** 1 second.
  * **Batch Size ($B$):** 64 or 128.
  * **Features ($F$):** 225 variables.

### Layer 2: AI Engine - LSTM Autoencoder (`src/models/`)
* **Architecture:** Deep Recurrent LSTM Autoencoder.
  * **Encoder:** Compresses the 60-second temporal matrix ($60 	imes 225$) into a low-dimensional latent vector $h$, capturing temporal dynamics and cross-sensor correlations.
  * **Decoder:** Reconstructs the original 60-second sequence $\hat{X}$ from $h$.
* **Loss Function:** Mean Squared Error (MSE) / Mean Absolute Error (MAE) computed during training on normal data.

### Layer 3: Detection & Smoothing Pipeline (`src/detection/`)
* **Absolute Mean Error per Timestep:**
  $$E_t = rac{1}{F} \sum_{f=1}^{F} |X_{t,f} - \hat{X}_{t,f}|$$
* **Error Smoothing (Moving Average / EMA):**
  * Inspired by top HAICon solutions. Applies Exponential Moving Average to $E_t$ to absorb sensor micro-noise and eliminate transient false positives.
* **Dynamic Thresholding (`src/detection/threshold.py`):**
  * Threshold set automatically using high percentiles (e.g., 99th or 99.5th percentile) or EVT (Extreme Value Theory) on validation/train reconstruction errors.

### Layer 4: Native Explainability (xAI) & Root Cause Analysis (`src/detection/xai_root_cause.py`)
* **Zero-Overhead Root Cause Identification:**
  * Avoids heavy external black-box explainers (like SHAP).
  * Computes feature-wise error vector $E_{t,f} = |X_{t,f} - \hat{X}_{t,f}|$ for all 225 variables.
  * Ranks variables by reconstruction error magnitude. The variable with the highest anomaly contribution is mathematically identified as the initial attack injection point or primary impacted component.

### Layer 5: Industrial Risk Scoring & Supervision (`app/`)
* **Risk Categorization Matrix:**
  * **Low (Surveillance):** Micro-drifts below alarm threshold.
  * **Medium (Warning):** Moderate deviation, minor internal logic point impacted.
  * **High (Alert):** Severe correlation breach targeting critical physical equipment (pumps, valves).
  * **Critical (Emergency):** Confirmed stealth cyber-attack or physical DoS threat (equipment damage risk, boiler overpressure, turbine trip).
* **Streamlit Dashboard:** Real-time mini-SOC interface displaying live process curves, risk status, root cause top-N features, and security recommendations.

---

## 4. Target Project Directory Structure

```
FactoryShield-OT/
├── data/                       # Operational Data (Excluded from Git)
│   ├── raw/                    # Original HAIEnd 23.05 / HAI 23.05 CSV files
│   ├── processed/              # Cleaned & scaled data (clean_scaled.csv)
│   └── external/               # Tag labels, descriptions, and NetworkX logic graphs
├── notebooks/                  # Exploration & Research
│   ├── 01_exploration.ipynb    # Semaine 1: EDA, tag distributions, temporal trends
│   └── 02_baseline_IF.ipynb    # Semaine 2-3: Baseline Isolation Forest, RF, XGBoost
├── configs/                    # Hyperparameters (batch_size, window_size, learning_rate)
├── logs/                       # TensorBoard & logging outputs
├── src/                        # Core Application Source Code
│   ├── __init__.py
│   ├── data/
│   │   ├── preprocess.py       # Cleaning, missing value handling, MinMaxScaler
│   │   └── dataloader.py       # PyTorch Sliding Window Dataset & DataLoader
│   ├── models/
│   │   ├── lstm_autoencoder.py # PyTorch LSTM Autoencoder Architecture
│   │   └── train.py            # PyTorch training loop & validation loss tracking
│   ├── detection/
│   │   ├── inference.py        # Reconstruction error calculation & threshold evaluation
│   │   └── xai_root_cause.py   # Top-K anomalous feature extraction
│   └── utils/
│       └── metrics.py          # Precision, Recall, F1-Score, eTaPR / Confusion Matrix
├── app/                        # Streamlit Web Application
│   ├── app.py                  # Streamlit Entry Point
│   ├── pages/                  # Dashboard multi-page views (EDA, Training, Live SOC, Audit Report)
│   └── assets/                 # Branding, diagrams, CSS
├── models_saved/               # Saved trained model checkpoints (.pth, .pkl, .joblib)
├── requirements.txt            # Python dependencies (torch, pandas, streamlit, scikit-learn, etc.)
├── README.md                   # Project Overview & Quickstart Guide
└── .gitignore                  # Git exclusions (data/, models_saved/*.pth, logs/)
```

---

## 5. Technical Specifications of HAIEnd 23.05 & Attack Scenarios

### Emerson Ovation DCS Process Control Loops (P1 Boiler)
1. **P1-PC (Pressure Control):** Controls pressure between main and return water tanks via valves `PCV01D` and `PCV02D` based on setpoint `B2016` and sensor `PIT01`.
2. **P1-LC (Level Control):** Controls return water tank water level `LIT01` via valve `LCV01D` based on setpoint `B3004`.
3. **P1-FC (Flowrate Control):** Controls outflow `FT03` via valve `FCV03D` based on setpoint `B3005`.
4. **P1-TC (Temperature Control):** Cascade/feedforward control for heat exchanger outlet temperature `TIT01` via valves `FCV01D` and `FCV02D` based on setpoint `B4022`.
5. **P1-CC (Cooling Control):** Controls cooling water pump `PP04` frequency based on main tank temperature `TIT03`.

### Attack Categories in HAIEnd 23.05
* **I/O Point Attacks (AP01 - AP47):** Direct manipulation of Setpoints (SP), Process Variables (PV), and Control Outputs (CV). Examples include short-term impulse attacks (ST), trapezoidal profile masking, and long-term bias attacks (LT).
* **Internal Point Attacks (AE01 - AE08):** Target DCS embedded function block algorithms inside Emerson Ovation:
  * *Artificial I/O functions (AE01, AE03):* Modulate initial state values sent to valves during auto/manual switching.
  * *Arithmetic functions (AE02, AE04, AE05, AE06, AE07):* Alter sensor calibration curves and output scaling limits (e.g. capping max valve opening command from 100% to 90%).
  * *Monitor functions (AE08):* Alter high/low threshold limits to cause missing or false safety trips.

---

## 6. 8-Week Roadmap (EMSII Internship Calendar)

| Week | Phase | Target Objectives | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **W1** | **Context & EDA** | Study OT/ICS context, inspect HAIEnd/HAI 23.05 CSV files, analyze tag distributions & time continuity. | `01_exploration.ipynb`, EDA summary report |
| **W2** | **Preprocessing** | Implement missing value handling, MinMaxScaler fitting on clean train data, PyTorch Sliding Window DataLoader. | `src/data/preprocess.py`, `src/data/dataloader.py` |
| **W3** | **Baselines** | Train classical tabular anomaly detection models: Isolation Forest, One-Class SVM, XGBoost/Random Forest. | `notebooks/02_baseline_IF.ipynb`, baseline metrics (F1, Recall) |
| **W4** | **Deep Learning** | Construct PyTorch LSTM Autoencoder (and/or GRU Autoencoder / 1D-CNN) with encoder-decoder architecture. | `src/models/lstm_autoencoder.py`, `src/models/train.py` |
| **W5** | **Evaluation & Comparison** | Evaluate LSTM AE vs Baselines on HAI test sets using Precision, Recall, F1, Confusion Matrix, and prediction latency. | `src/utils/metrics.py`, comparative performance table |
| **W6** | **Risk Scoring & xAI** | Develop Industrial Risk Scoring engine (4 levels: Low/Med/High/Critical) and Root-Cause feature error ranker. | `src/detection/inference.py`, `src/detection/xai_root_cause.py` |
| **W7** | **Streamlit App & SOC Sim** | Build multi-page Streamlit app with live streaming simulation (window-by-window processing), alerts, and curves. | `app/app.py`, interactive web dashboard |
| **W8** | **Reporting & Audit** | Implement automated audit report generator (PDF/HTML summary), complete project documentation, presentation slides. | PDF report exporter, final `README.md`, codebase clean-up |

---

## 7. Instructions for Kiro (AI Coding Agent)

### Mission
Kiro must act as an expert Lead OT/ICS Cybersecurity & AI Engineer. Your goal is to inspect the current state of the workspace directory (`FactoryShield-OT/`), identify completed vs missing modules according to this `plan.md`, and systematically implement or refine the project code.

### Step-by-Step Task Breakdown for Kiro

1. **Workspace Audit:**
   * Read and inspect all files inside `FactoryShield-OT/` (including `notebooks/`, `src/`, `configs/`, `app/`, `requirements.txt`).
   * Identify which week of the 8-week roadmap is currently reached and what components are missing or unoptimized.

2. **Data Pipeline Verification (`src/data/`):**
   * Ensure `preprocess.py` handles `MinMaxScaler` fitting ONLY on normal training data.
   * Verify that `dataloader.py` implements sliding windows $X \in \mathbb{R}^{B 	imes 60 	imes F}$ efficiently without RAM overflow.

3. **Model & Detection Pipeline (`src/models/` & `src/detection/`):**
   * Verify PyTorch LSTM Autoencoder structure in `lstm_autoencoder.py`.
   * Check training logic in `train.py` (loss logging, saving best model checkpoints to `models_saved/`).
   * Verify inference and reconstruction MAE calculation in `inference.py`.
   * Implement native xAI feature contribution ranking in `xai_root_cause.py`.

4. **Risk Scoring Engine & Dashboard (`app/`):**
   * Verify the risk classification logic based on reconstruction error magnitude, duration, and target tag criticality.
   * Ensure Streamlit `app.py` has pages for Data Exploration, Model Training/Evaluation, Live SOC Simulation, and Audit Report Generation.

---

## 8. Parameters to Pass to Kiro

When initializing or prompting Kiro, supply the following parameters and context flags:

```json
{
  "project_name": "FactoryShield-OT",
  "role": "Lead OT/ICS Cybersecurity & Machine Learning Engineer",
  "frameworks": ["PyTorch", "Streamlit", "Pandas", "Scikit-Learn", "Plotly", "WeasyPrint/FPDF"],
  "dataset": {
    "name": "HAIEnd 23.05 / HAI 23.05",
    "target_system": "Emerson Ovation DCS (Boiler Process P1)",
    "feature_count_haiend": 225,
    "feature_count_hai": 86,
    "clean_train_samples": 896400,
    "attack_ratio": "~4% in test sets"
  },
  "deep_learning_config": {
    "architecture": "LSTM Autoencoder",
    "window_size": 60,
    "stride": 1,
    "batch_size": 64,
    "learning_rate": 0.001,
    "loss_function": "MSELoss / L1Loss",
    "optimizer": "Adam"
  },
  "anomaly_detection": {
    "approach": "Semi-Supervised (Normality Learning)",
    "error_metric": "Mean Absolute Error (MAE) per timestep",
    "smoothing": "Moving Average / Exponential Moving Average",
    "threshold_method": "99th Percentile on Normal Train Error",
    "xai_method": "Feature-wise Reconstruction Error Ranking"
  },
  "risk_levels": ["Low", "Medium", "High", "Critical"],
  "execution_rules": [
    "Never apply SMOTE or synthetic data generation.",
    "Fit MinMaxScaler strictly on training set.",
    "Ensure memory-efficient sliding window generation using PyTorch Datasets.",
    "Maintain clean modular code under src/ and app/."
  ]
}
```
