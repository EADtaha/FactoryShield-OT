---
name: ot_physics_guardrails
description: Enforces physical laws, ISA/IEC 62443 Level 1 boundaries, and strict data leakage prevention for OT/ICS anomaly detection.
---
# Industrial OT & Physics Integrity Rules

## 1. Ban on Synthetic Data (SMOTE)
- **Never** use SMOTE or generate synthetic ICS anomalies[cite: 7]. Interpolating sensor tags creates states that mathematically violate thermodynamics and fluid dynamics (e.g., zero pressure with a 100% open valve)[cite: 7].

## 2. Zero Data Leakage
- `MinMaxScaler` must be fitted **exclusively on clean normal data** (`label == 0`)[cite: 7].
- Out-of-distribution values during attacks must be allowed to scale past 1.0 naturally to trigger reconstruction loss[cite: 7].

## 3. Temporal Sequence Integrity
- Train/test splits must strictly use `shuffle=False`[cite: 7]. Preserving continuous dynamic state transitions is required for LSTM context[cite: 7].

## 4. Subsystem Isolation
- Maintain tag-to-subsystem consistency[cite: 7]:
  - P1: Boiler (Emerson Ovation DCS)[cite: 7].
  - P2: Turbine (GE Mark VIe)[cite: 7].
  - P3: Water Treatment (Siemens S7)[cite: 7].
  - P4: HIL Simulator (dSPACE)[cite: 7].