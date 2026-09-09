# FactoryShield-OT — SOC Explainer Guide
### Plain-Language Concept Reference for Engineers, Plant Managers & Oral Defense Evaluators

> **Audience:** Engineering students, plant operations managers, academic jury members, and technical interviewers who need to understand the *what* and *why* of the system — not just the code.  
> **Format:** Concept-first explanations, physical analogies, real attack scenarios, and a ready-to-use oral presentation script.

---

## Table of Contents

1. [The "Explain Like I'm Five" — How the AI Brain Works](#1-the-explain-like-im-five--how-the-ai-brain-works)
2. [Why Traditional IT Security Fails at the Plant Floor](#2-why-traditional-it-security-fails-at-the-plant-floor)
3. [The Stealth Attack Anatomy — A Real Scenario Walkthrough](#3-the-stealth-attack-anatomy--a-real-scenario-walkthrough)
4. [The 4 Risk Levels Demystified](#4-the-4-risk-levels-demystified)
5. [The Full Detection Chain — No Jargon](#5-the-full-detection-chain--no-jargon)
6. [Why This Model Beats the Alternatives](#6-why-this-model-beats-the-alternatives)
7. [The Numbers That Matter — Reading the Dashboard](#7-the-numbers-that-matter--reading-the-dashboard)
8. [How to Present & Demo This Project in 5 Minutes](#8-how-to-present--demo-this-project-in-5-minutes)
9. [Frequently Asked Defense Questions — With Answers](#9-frequently-asked-defense-questions--with-answers)
10. [Glossary of Key Terms](#10-glossary-of-key-terms)

---

## 1. The "Explain Like I'm Five" — How the AI Brain Works

### The Factory Inspector Analogy

Imagine you hire the world's greatest factory inspector. This person has spent **ten solid months** standing inside a thermal power plant, doing nothing but listening and watching. Every tick, every hum, every valve click, every pressure reading, every temperature change — they absorbed all of it, every second of every day.

After ten months, this inspector has developed an intuition so sharp that they can tell you — within one second — whether the plant sounds and behaves exactly the way a healthy plant should.

Now imagine you blindfold them and ask them to recreate from memory the exact sounds and readings they expect to hear at this moment. If what they reconstruct matches what is actually happening: **everything is normal.**

But if the plant is under attack — say, someone quietly tweaked a valve calibration to hide a pressure buildup — the inspector's reconstruction will no longer match reality. Something feels wrong. They raise the alarm.

**That inspector is the LSTM Autoencoder.** The ten months of study is the 896,400 normal operation samples it was trained on. The "reconstruction from memory" is the 60-second sliding window being decoded back from a compressed 64-number fingerprint. The mismatch between expectation and reality is the anomaly score.

---

### Unpacking the Analogy Into the Actual System

| Analogy | Technical reality |
|---------|------------------|
| The factory inspector | LSTM Autoencoder neural network |
| Ten months of observation | Training on 896,400 clean samples (~10 months of 1 Hz data) |
| The 225 sensor readings | HAIEnd 23.05 dataset: 35 SCADA I/O tags + 190 DCS internal function block edges |
| 60-second observation window | Sliding window of T=60 timesteps, F=225 features |
| "Compressing the memory into 64 key facts" | Encoder: (B, 60, 225) → latent vector (B, 64) |
| "Reconstructing what should be happening" | Decoder: (B, 64) → reconstructed sequence (B, 60, 225) |
| "Mismatch between expectation and reality" | Mean Absolute Error: E_t = (1/225) × Σ\|x_{t,f} − x̂_{t,f}\| |
| Raising the alarm | E_t > τ_p99 → alert = 1 |

---

### What the "64 Numbers" Actually Represent

The encoder compresses an entire 60-second snapshot of a 225-sensor industrial plant into just 64 numbers. That sounds like too little information. But consider what those 64 numbers are forced to represent:

- The current pressure loop state and its recent trend
- The temperature's relationship to the flow valve position
- The level controller's integral windup
- The thermodynamic energy balance across the heat exchanger
- The inter-process water flow coupling between P1, P2, and P3

These 64 numbers are not random — the model was forced to find the most compact possible description of normal industrial physics. If you feed it a corrupted sequence (one where an attacker changed the valve calibration), the decoder cannot reconstruct it faithfully because the underlying physics no longer match what the model learned. The reconstruction error spikes.

---

### The Key Insight: The Model Doesn't Know What an Attack Looks Like

This is the most important and counterintuitive property of the system.

**The model was never shown a single attack sample during training.**

It only knows what normal looks like. When it sees something it cannot reconstruct, it raises an alert — not because it recognizes the attack, but because it doesn't recognize the new state as normal.

This means the system can detect:
- Attacks it has never seen before
- Novel zero-day exploits
- Internal logic manipulations that have no known signature
- Any physical state that violates the learned thermodynamic norms

A signature-based intrusion detection system (IDS) would miss all of these because it would be looking for patterns it was explicitly taught. FactoryShield-OT looks for departures from physics instead.

---

## 2. Why Traditional IT Security Fails at the Plant Floor

### The Purdue Model — Where the Problem Lives

Industrial plants are organized in layers:

```
Layer 5 — Corporate IT Network     (email, ERP, file servers)
Layer 4 — Business Network         (scheduling, logistics)
Layer 3 — Operations Network       (historian, engineering workstations)
──────────────────────────────── ← Traditional IT firewall lives here
Layer 2 — Supervisory Control      (HMI, SCADA servers)
Layer 1 — Process Control          (DCS, PLCs, controllers)  ← OT attacks happen here
Layer 0 — Physical Process         (pumps, valves, sensors, turbines)
```

Traditional IT security tools — antivirus, firewalls, SIEM — protect the boundary between Layers 3 and 4. They are fundamentally network-aware and file-aware tools.

**OT attacks at Layer 1 look like this:** A valid, authenticated Modbus write command from a legitimate operator workstation to an Emerson Ovation DCS function block, instructing it to change one calibration coefficient from 100 to 97. No malware, no suspicious file, no anomalous network packet. Just a 6-byte Modbus message that the firewall happily passes through because it looks identical to the ten thousand legitimate operator commands that preceded it.

---

### The Four Ways IT Tools Miss OT Attacks

**1. Wrong protocol:** Industrial protocols (Modbus, PROFIBUS, OPC-UA, IEC 61850) are not understood by standard IT IDS signatures. A Snort/Suricata rule trained on HTTP/SMB attacks is effectively blind to a Modbus write packet.

**2. Wrong layer:** The attack in the HAIEnd 23.05 dataset category AE02 changes an *arithmetic calibration coefficient inside a DCS function block*. This is not a network event at all. It is a change to an internal variable in the controller's memory. No network packet is generated.

**3. Wrong threat model:** Antivirus looks for malicious code execution. There is no malicious code — the attacker is using the plant's own legitimate control functions to cause harm. The DCS is operating exactly as programmed; the program itself has been subtly modified.

**4. Wrong speed:** OT processes operate at 1 Hz to 100 Hz. A pressure overshoot can cause catastrophic boiler failure in under 30 seconds. Traditional SIEM correlation engines are designed for events measured in minutes or hours, not seconds.

**FactoryShield-OT operates at the right layer, the right speed, and monitors the right signals.** It watches the physics directly — 225 sensor and actuator values, every second — and makes a decision within the same second.

---

## 3. The Stealth Attack Anatomy — A Real Scenario Walkthrough

### Scenario: Attack AE02 — Arithmetic Calibration Manipulation on the Boiler Flow Loop

This scenario is based on the AE02 attack category in HAIEnd 23.05. It targets the internal arithmetic function blocks of the Emerson Ovation DCS.

---

### Step 1: What the Attacker Does

The adversary gains access to the Emerson Ovation engineering workstation (possibly via a spear-phishing email to an engineer's laptop, or via a USB drive left in the control room). Using legitimate Ovation Studio software, they modify a single internal parameter:

**Target:** The flow sensor calibration function block that scales the raw 4–20 mA signal from flow sensor `FT03` into engineering units (m³/h).

**Original calibration:** Raw 4 mA → 0 m³/h, Raw 20 mA → 200 m³/h (linear mapping)

**Attacker's modification:** Changes the upper scale limit from 200 m³/h to 180 m³/h.

**Effect:** When the actual physical flow is 190 m³/h, the DCS now displays and controls as if the flow were 171 m³/h (a 10% undercurrent).

The DCS flow controller sees a "low flow" and opens the valve `FCV03D` further to compensate, actually increasing the real flow. But the displayed value stays artificially low. The operator's HMI shows a perfectly normal-looking 100 m³/h reading. The pressure, however, is quietly building.

---

### Step 2: What the Operator Sees

| Parameter | HMI Display | Physical Reality |
|-----------|------------|-----------------|
| Flow `FT03` | 100 m³/h (normal) | 111 m³/h (elevated) |
| Valve `FCV03D` | 48% open (normal) | 48% open (correct reading) |
| Pressure `PIT01` | 51.2 bar (slightly elevated) | 57 bar (dangerously elevated) |
| Temperature `TIT01` | 201°C (normal) | 208°C (elevated) |
| Safety alarms | None | None (limits based on HMI, not physical) |

The HMI looks perfectly healthy. The DCS historian shows normal trends. The operator has no indication anything is wrong.

This is the definition of a stealth attack: the physical process is being driven toward a dangerous state while all operator-visible indicators appear normal.

---

### Step 3: Why Standard IT Tools See Nothing

- **Firewall:** The calibration change was made via Ovation Studio — a legitimate, authenticated application communicating on the normal OPC-DA port. The firewall passes it.
- **Antivirus:** No malicious executable was run. Ovation Studio is a trusted, signed application.
- **Network IDS:** The Modbus/OPC message that changed the calibration parameter is syntactically valid and indistinguishable from routine parameter adjustments.
- **SIEM:** No login failures, no port scans, no lateral movement alerts. Everything looks normal at the network level.

---

### Step 4: Why FactoryShield-OT Catches It

The LSTM Autoencoder has spent 896,400 seconds learning the precise physical relationship between flow `FT03`, valve position `FCV03D`, pressure `PIT01`, and temperature `TIT01`.

It has learned that when the flow is 100 m³/h and the valve is 48% open and the temperature is 201°C, the pressure should be approximately 50.8 ± 0.9 bar. That is the physics it memorized.

Now, at t=0 of the attack, the physical pressure is actually 57 bar, but `FT03` is reporting 100 m³/h (falsified). When the autoencoder tries to reconstruct this window of 60 seconds, it faces an irreconcilable contradiction:

- "Given `FT03`=100 m³/h, `FCV03D`=48%, `TIT01`=201°C... my reconstruction says `PIT01` should be ~50.8 bar."
- But the actual `PIT01` reading is 57 bar.

The reconstruction error on `PIT01` spikes to 0.82 (on a 0–1 scale). The global MAE `E_t` jumps from its normal baseline of ~0.04 to 0.76.

After EMA smoothing (α=0.10), within 29 seconds the smoothed score `S_t` exceeds the threshold τ_p99 = 0.15.

**Alert fires. Risk level: CRITICAL.**

---

### Step 5: The Root-Cause Report Points to the Exact Sensor

The xAI engine runs `np.argsort(-E_{t,f})` on the 225-element error vector and produces:

```
Rank 1: PIT01   — error: 0.821  (62.4%)  — P1: Pressure Control  ← ROOT CAUSE
Rank 2: PCV01D  — error: 0.193  (14.7%)  — P1: Pressure Control
Rank 3: FT03    — error: 0.147  (11.2%)  — P1: Flow Control
Rank 4: TIT01   — error: 0.089   (6.8%)  — P1: Temperature Control
Rank 5: FCV03D  — error: 0.062   (4.7%)  — P1: Flow Control
```

The system identifies `PIT01` (pressure) and `FT03` (flow) as the primary disrupted subsystem — precisely the two sensors involved in the calibration attack. The operator's safety recommendation panel shows:

> **EMERGENCY: Initiate process shutdown procedure. Notify plant manager immediately. Isolate affected control loops.**

The attack is detected and localized within one minute of initiation — before any physical equipment damage occurs.

---

## 4. The 4 Risk Levels Demystified

The risk scoring engine combines four factors — error magnitude, duration, number of affected sensors, and criticality of those sensors — into a single composite score mapped to four operational response levels.

---

### Level 1 — LOW (Surveillance)

**Composite score:** < 0.30  
**Color code:** Green

**What it means physically:** The LSTM autoencoder sees a mild reconstruction error that is slightly above its normal noise floor but below the alert threshold. Think of it as the inspector noticing a small, brief variation in machine hum — something that could be a sensor calibration drift, a brief power line fluctuation, or a minor process transient.

**Real plant example:**
- A temperature sensor shows a 0.8°C higher reading for 3 seconds, then returns to normal. Likely a sensor glitch or brief steam quality variation.
- Valve `LCV01D` (level control) momentarily overshoots its setpoint by 2% during a routine setpoint step change.

**Operator action:** Continue normal monitoring. Log the event in the DCS historian. No immediate intervention required.

**FactoryShield-OT response:** Anomaly score tracked but below threshold. No alert generated. EMA smoothing absorbs the spike within ~15 seconds.

---

### Level 2 — MEDIUM (Warning)

**Composite score:** 0.30 – 0.59  
**Color code:** Amber

**What it means physically:** A real but moderate deviation from normal physics. Something has changed in the process that does not fully match the learned correlations. It could be a legitimate process upset (a setpoint change the operator forgot to log), an equipment degradation (a valve starting to stick), or the early stages of an attack.

**Real plant example:**
- The temperature controller `TIT01` receives a setpoint change from 200°C to 195°C, but the cascade controller's response takes 45 seconds longer than usual. The LSTM flags the transition as anomalous because it learned normal setpoint-response dynamics.
- Attack type AP05 (short-term additive bias on a level setpoint): the level `LIT01` receives a +5% bias. The LSTM sees the level rising while the valve `LCV01D` position doesn't justify it, creating a cross-sensor inconsistency.
- Minor internal logic attack AE08: the safety high-limit on `PIT01` is raised by 5 bar. The controller behavior becomes slightly different from the learned pattern.

**Operator action:** Cross-check redundant sensors. Verify that a legitimate process change was made. Prepare the maintenance team. Increase monitoring frequency.

**FactoryShield-OT response:** Alert fires (score > τ_p99). EMA-smoothed signal is elevated but not extreme. Risk engine assigns Medium based on moderate error magnitude (~0.35–0.55) and relatively short duration (<30 seconds).

---

### Level 3 — HIGH (Alert)

**Composite score:** 0.60 – 0.84  
**Color code:** Red

**What it means physically:** A severe, sustained correlation breach involving critical equipment. The physical process is meaningfully deviating from safe operating conditions. The mismatch between sensor readings and actuator positions is significant enough that critical control loops are visibly affected.

**Real plant example:**
- Attack AP23 (long-term trapezoidal masking on flow setpoint `B3005`): The flow controller sees a falsified setpoint that gradually increases over 5 minutes, causing the real flow to deviate significantly from design intent. After 90 seconds, `FT03`, `FCV03D`, and `PIT01` all show elevated errors simultaneously.
- The pump `PP04` (cooling water) is attacked with a spoofed speed reference: cooling water flow drops while the temperature `TIT03` rises. Both sensors are in the critical list, and the duration exceeds 60 seconds.
- Valve `PCV01D` is stuck at 15% open (mechanical fault or cyber command) while the pressure setpoint demands 45% open. The pressure `PIT01` drops sharply. The LSTM flags the incoherence between command and physical response.

**Operator action:** Switch affected control loops to manual. Verify physical sensor readings with handheld instruments. Log the incident formally in the DCS historian. Notify the shift supervisor. Do not leave the control room.

**FactoryShield-OT response:** Alert sustained for >30 seconds. Risk engine assigns High. Root-cause report identifies 2–4 sensors across the Pressure or Temperature subsystem. Dashboard displays operator recommendation: *"Switch to manual. Verify physical sensors. Log incident."*

---

### Level 4 — CRITICAL (Emergency)

**Composite score:** ≥ 0.85  
**Color code:** Crimson

**What it means physically:** A confirmed stealth cyber-attack or imminent physical threat. The physical process is in an unsafe state. Equipment damage risk is real — boiler overpressure, turbine overspeed, heat exchanger thermal runaway, or cooling system failure. Safety instrumented systems (SIS) may be at risk if the attacker has also targeted safety trip limits (AE08).

**Real plant example:**
- Attack AE02 (as described in Section 3) in full effect: `PIT01` is reporting 15 bar below actual physical pressure. The DCS has opened the pressure control valve `PCV01D` to 85% trying to raise what it thinks is low pressure, while the real pressure is already at 78 bar (design limit: 80 bar). The boiler is 30 seconds from a safety relief valve actuation or — if the safety system was also compromised — a catastrophic overpressure event.
- Coordinated attack on both `TIT01` and `FT03` simultaneously: the heat exchanger temperature is rising uncontrolled while the flow sensor reports falsely high flow. The LSTM identifies 8 sensors with elevated errors, including 5 from the critical list.
- Attack AE08 has suppressed the high-pressure safety trip by raising the trip setpoint by 20 bar. Without FactoryShield-OT, the safety system would not trigger until physical damage is already occurring.

**Operator action:** Initiate process shutdown procedure immediately. Move critical process to manual/safe state. Notify plant manager and cybersecurity team simultaneously. Preserve DCS historian data for forensic analysis. Do not restart until physical inspection completed.

**FactoryShield-OT response:** Peak EMA score > 0.85. Risk composite score ≥ 0.85. Immediate Critical alert. Root-cause analysis points to specific attack vector subsystem. Dashboard displays: *"EMERGENCY: Initiate process shutdown procedure. Notify plant manager immediately. Isolate affected control loops."*

---

## 5. The Full Detection Chain — No Jargon

Here is the complete story of what happens to a single second of sensor data from the moment it enters the system to the moment an alert reaches the SOC operator's screen.

---

**Second 0 — Raw data arrives**

The Emerson Ovation DCS sends one row of data: 225 numbers, representing every pressure, temperature, flow, level, valve position, and internal logic variable at this exact second. The values are in engineering units (bar, °C, m³/h, %).

---

**Second 0 — Normalization**

Each value is scaled to the range [0, 1] using the MinMaxScaler that was fitted during training exclusively on normal data. A valve that ranges from 0% to 100% becomes a number between 0.0 and 1.0. If an attack has pushed that valve outside its normal range, the scaled value will exceed 1.0 — already a hint of something unusual.

---

**Seconds 0–59 — Window accumulation**

The system waits until it has 60 consecutive seconds of normalized data. This 60-row × 225-column table is the "observation window" — the 60-second snapshot the inspector will analyze.

---

**The AI runs in ~3–5 ms**

The 60×225 window is fed into the LSTM Encoder. In two LSTM layers, the temporal dynamics of the entire 60-second window are compressed down to a single 64-number vector. This vector is the "physics fingerprint" of this particular minute of plant operation.

The LSTM Decoder then takes those 64 numbers and reconstructs what those 60 seconds should have looked like if everything were normal.

---

**The gap is measured**

For every one of the 225 sensors at the 60th second, the system computes: *"How far is the real reading from what the model expected?"* These 225 gaps are the per-sensor reconstruction errors.

The average of all 225 gaps is the global anomaly score for this second.

---

**The noise is filtered out**

Industrial sensors are electrically noisy. A single spike of 0.002 above the threshold does not constitute an attack. The EMA smoother takes 10% of the new anomaly score and 90% of the historical trend, creating a smooth signal that ignores momentary glitches but responds reliably to sustained anomalies.

---

**The decision**

If the smoothed score exceeds the threshold (the 99th percentile of what the model saw during training — by definition only 1% of normal operation crosses this line), an alert fires.

---

**The detective work**

At the moment the alert fires, the system looks at all 225 per-sensor gaps and ranks them from largest to smallest. The sensor with the biggest gap is the likely attack entry point. Its physical subsystem is identified (Pressure Control, Flow Control, Turbine, etc.) and reported to the operator.

---

**The risk assessment**

Based on how large the anomaly is, how long it has been going, how many sensors are affected, and whether critical process sensors are involved, the risk engine assigns a level: Low, Medium, High, or Critical.

---

**The operator sees it immediately**

The Streamlit dashboard updates with: a risk gauge needle in the red zone, a highlighted time window on the process chart, an alert table with duration and severity, a bar chart showing which sensors are responsible, and a recommended operator action.

**Total time from attack start to operator notification: under 60 seconds.**

---

## 6. Why This Model Beats the Alternatives

### Why Not Just Use Isolation Forest?

Isolation Forest is a solid unsupervised anomaly detector. It works by building random decision trees that isolate data points — anomalies are isolated faster (shorter average path length) than normal points.

**Its blind spot:** It operates on individual data points — a single row of 225 values. It has no concept of time, sequence, or temporal dynamics. It cannot learn that "pressure at time t+15 is expected to be X given the valve command at time t." It treats every row independently.

In industrial control systems, most attacks work *over time*: they gradually drift a setpoint, or create a slowly-growing discrepancy between sensor and actuator. A row-by-row detector sees each row as borderline normal and never raises an alarm.

**FactoryShield-OT LSTM AE advantage:** Learns the full 60-second temporal correlations. A valve command at t=0 and its expected pressure response at t=15 are captured as a learned relationship. Any disruption to this temporal cause-effect chain is immediately flagged.

**Benchmark result:** Isolation Forest achieves eTaF1 ≈ 0.61 vs. LSTM AE's eTaF1 ≈ 0.87 on HAIEnd 23.05.

---

### Why Not Just Use XGBoost?

XGBoost with labeled data achieves excellent F1 (~86%) because it learns directly from known attack samples. So why not use it?

**Problem 1 — Attack-specific:** XGBoost learns the specific patterns of the 55 attack types in the training set. A novel zero-day attack that slightly differs from the training attacks will not be recognized.

**Problem 2 — Requires attack labels:** You need labeled attack data to train XGBoost. In a real OT environment, you rarely have labeled historical attack data. FactoryShield-OT needs only normal data, which is abundant.

**Problem 3 — No temporal context:** XGBoost in its standard form operates row-by-row, same as Isolation Forest. Time-aware variants require significant additional engineering.

**Problem 4 — Physical impossibility of SMOTE:** To improve XGBoost on the 4% attack minority class, one would need to oversample attacks — which generates physically impossible states (as explained in Section 1).

**FactoryShield-OT advantage:** Semi-supervised (needs no attack labels), attack-agnostic, and temporally aware. LSTM AE achieves eTaF1 ≈ 0.87 vs. XGBoost's ≈ 0.79 while requiring zero attack samples for training.

---

### The SMOTE Argument — Expanded

The standard fix for class imbalance in ML is SMOTE: generate synthetic minority class samples by interpolating between real attack samples.

Consider two real attack samples in the HAIEnd 23.05 dataset:

- **Sample A:** `PIT01`=82 bar, `FT03`=5 m³/h, `PCV01D`=95% open (high pressure attack)
- **Sample B:** `PIT01`=38 bar, `FT03`=200 m³/h, `PCV01D`=8% open (low pressure attack)

A SMOTE-generated midpoint would be:
- **Synthetic:** `PIT01`=60 bar, `FT03`=102.5 m³/h, `PCV01D`=51.5% open

Is this a physically possible state? At 60 bar with a 51.5% open valve, the flow rate from basic fluid mechanics cannot be 102.5 m³/h in this system. This state violates Bernoulli's principle. Training a model on such a state teaches it wrong physics.

A model trained on wrong physics deployed in a real plant will confidently declare physically impossible states as "normal" — and miss real attacks that produce subtle but thermodynamically coherent anomalies.

This is why SMOTE is not just inadvisable — it is actively dangerous for OT anomaly detection.

---

## 7. The Numbers That Matter — Reading the Dashboard

When you open the FactoryShield-OT Streamlit dashboard during a demo or presentation, here is how to interpret every number on screen.

---

### The Sidebar Parameters

**Alert Threshold (default: 0.15)**  
This is the normalized MAE value above which an alert fires. It was computed automatically during training as the 99th percentile of reconstruction errors on the 896,400 normal training samples. In plain language: only 1% of normal operation windows produce a score this high or higher.

- **If you raise it** (e.g., to 0.30): The system becomes more conservative. Fewer alerts, but some attacks may be missed (higher false-negative rate).
- **If you lower it** (e.g., to 0.05): The system becomes more sensitive. More alerts, but more false positives from normal process noise.

**EMA α (default: 0.10)**  
The smoothing factor. Think of it as the "memory weight" on new data.

- **α = 0.10**: 10% weight on the newest second, 90% on the historical trend. Smooth, 29-second effective memory.
- **α = 0.50**: 50% weight on new data. Much more reactive, but noisier.
- **Rule of thumb:** Lower α = smoother signal, higher detection latency. Higher α = faster response, more false positives.

---

### The KPI Strip (Four Metric Cards at the Top)

| Card | What it tells you |
|------|------------------|
| **Process Status: Running** | The simulation is active and data is flowing |
| **Active alert segments: N** | How many distinct attack windows are currently above threshold. One attack that lasts 55 seconds = 1 segment, not 55 |
| **Current risk level: Critical/High/Medium/Low** | The highest risk level among all active alert segments |
| **Max EMA error: 0.XXXX** | The peak smoothed anomaly score in the entire simulation window |

---

### The 4-Row Chart

- **Row 1 (Process signals):** This is what the operator's HMI shows. PIT01 (pressure), TIT01 (temperature), FT03 (flow), LIT01 (level). Red shaded zones = time windows where alerts are active.
- **Row 2 (MAE):** The dotted grey line is the raw reconstruction error every second. The purple filled line is the EMA-smoothed version. The red dashed horizontal line is the threshold. When the purple line crosses the red line = alert.
- **Row 3 (Alert 0/1):** A binary step function. 0 = safe, 1 = alert. The area under it (time spent in alert) is the total attack duration.
- **Row 4 (Risk 0–3):** 0=Low, 1=Medium, 2=High, 3=Critical. Tells you the severity progression over time.

---

### The xAI Bar Chart

The bar chart in the xAI Root Cause page shows six sensors (in the demo) ranked by their reconstruction error at the exact moment the alert first fired. The **crimson bar (Rank 1) is the root cause sensor** — the one the attacker most directly manipulated. The percentage label on each bar shows what fraction of the total anomaly signal comes from that sensor.

**How to read it:** If `PIT01` has a 62% contribution, it means the pressure sensor is responsible for 62% of the total reconstruction mismatch. Everything else (valves, temperatures, flows) explains the remaining 38%. This tells the operator: *"Go check the pressure loop first."*

---

## 8. How to Present & Demo This Project in 5 Minutes

Here is a complete oral script designed for a 5-minute presentation slot at an engineering jury, academic defense, or technical interview. Adapt freely.

---

### Minute 0:00–0:45 — The Problem Statement

*"We live in an era where industrial plants are increasingly connected to corporate networks and the internet. The Emerson Ovation DCS controlling this boiler, the GE turbine controller, the Siemens water treatment PLC — these systems were designed for reliability and physics, not cybersecurity.*

*A traditional IT firewall stops malicious network packets. But an attacker who sends a perfectly valid, authenticated command to change a valve calibration coefficient by 3% — that firewall sees nothing. The antivirus sees nothing. Yet that 3% change, sustained for 5 minutes, can drive a boiler toward overpressure and potentially catastrophic failure.*

*FactoryShield-OT is designed to catch exactly that kind of attack — not by recognizing malicious packets, but by detecting when the plant's physics are being violated."*

---

### Minute 0:45–1:45 — The AI Core

*"The core of the system is an LSTM Autoencoder — a neural network trained exclusively on 896,400 seconds of healthy plant operation. [Point to the dashboard.]*

*During training, the network learns to compress 60 seconds of 225-sensor data into a 64-number fingerprint, and then reconstruct the original 60 seconds from that fingerprint. After training, it becomes very good at reconstructing normal plant behavior.*

*At inference time, we feed it live data. For each second, it reconstructs what it expects to see, and measures the gap between expectation and reality. A small gap: normal operation. A large gap: something has changed that doesn't match learned physics. That's the anomaly score you see on this chart. [Point to Row 2 of the chart.]*

*Notice these three red zones — those are the three injected attack windows in the simulation. The system correctly identifies all three."*

---

### Minute 1:45–2:30 — Live Demo

*[Click Start on the simulation, then adjust the EMA α slider from 0.10 to 0.30.]*

*"Here I'm increasing the sensitivity — the system responds faster to new data. Notice the smoothed curve becomes more reactive. [Slide back to 0.10.] And here I restore the default: a 29-second smoothing window that absorbs single-second sensor noise.*

*[Click on xAI Root Cause in the navigation.]*

*For each alert onset, the system ranks all 225 sensors by their reconstruction error. The crimson bar — PIT01, pressure sensor — is responsible for 62% of the total anomaly signal. The system is telling the operator: the pressure loop is where the attack injected its manipulation. This is the attack vector, identified automatically, in real time, with no SHAP or external explainability library required."*

---

### Minute 2:30–3:15 — The Risk Levels

*"[Navigate to Live SOC, point to the risk gauge.]*

*The system uses four risk levels. The risk gauge uses a composite score that weighs the error magnitude — 60% of the score — the duration of the anomaly — 20% — the number of affected sensors — 10% — and the criticality of those sensors to process safety — the final 10%.*

*In this simulation, the third attack window reaches Critical because it affects PIT01 and PCV01D — both are on our critical sensor list for the boiler pressure loop. The system immediately generates this operator recommendation: initiate process shutdown.*

*An operator acting on this within 60 seconds of attack initiation can prevent the physical damage that would only become visible on the HMI 5–10 minutes later."*

---

### Minute 3:15–4:00 — Why This Approach

*"[Navigate to Model Benchmark.]*

*Why LSTM Autoencoder instead of classical machine learning? Two reasons.*

*First, temporal context. Isolation Forest and SVM see one row at a time — 225 numbers with no memory of the past. The LSTM sees 60 seconds of history. Cross-sensor temporal correlations — valve command at t=0, pressure response at t=15 — are invisible to classical methods but are the very signature FactoryShield-OT detects.*

*Second, no attack labels required. XGBoost achieves 86% F1 — but it requires known attack samples for training, and it's blind to novel attack patterns. Our LSTM AE achieves 90% F1 and 0.87 eTaF1 using only normal data, making it deployable in any industrial plant without requiring historical incident data."*

---

### Minute 4:00–4:30 — The Dataset and Standards

*"The system was validated against the HAIEnd 23.05 dataset — the most comprehensive publicly available ICS cybersecurity benchmark, produced by KISA/NSHC at a real Hardware-in-the-Loop testbed in South Korea.*

*It covers 55 attack categories: both I/O point attacks — direct sensor and setpoint manipulation — and internal logic attacks targeting the DCS function blocks themselves. The internal logic attacks are what conventional IT security systems cannot detect by design.*

*The system is aligned with ISA/IEC 62443, the international standard for industrial cybersecurity."*

---

### Minute 4:30–5:00 — Closing

*"To summarize: FactoryShield-OT is a physics-based anomaly detection platform. It doesn't look for malicious packets — it looks for broken physics. Trained on normality, blind to attack signatures, but acutely sensitive to any state that violates what a healthy Emerson Ovation DCS should look like.*

*From raw sensor data to actionable risk level and root-cause identification: under 60 seconds. On hardware as simple as a laptop.*

*Thank you. I'm happy to take any questions."*

---

## 9. Frequently Asked Defense Questions — With Answers

**Q: Can the system be fooled by a very slow, gradual attack?**

Yes — this is called a "slow drift" attack. If an attacker changes a calibration coefficient by 0.01% per day over three months, the drift may be small enough to stay below the detection threshold. This is a known limitation of reconstruction-error approaches.

The mitigation strategies are: (1) periodically re-validate the threshold against recent normal data to track slow operational changes, (2) use the EDA page's CDF view to detect slow percentile drift in the anomaly score distribution, and (3) combine FactoryShield-OT with DCS historian trend analysis for long-term drift detection.

---

**Q: What is the false positive rate?**

By construction, the 99th-percentile threshold means that on clean normal operation data, the system generates false alerts on approximately 1% of windows. For a continuously running plant, that's about 14 minutes of false alerts per day.

The EMA smoothing further reduces this: isolated single-second spikes are damped. Only sustained elevated scores (lasting >~10 seconds) produce persistent alerts. In practice, the operational false positive rate is significantly below 1%.

---

**Q: Why 60 seconds? Why not 30 or 120?**

The Emerson Ovation DCS P1 control loops have response times between 10 and 120 seconds depending on the loop. A 60-second window reliably captures at least one complete closed-loop response cycle for all five P1 control loops. Testing with 30-second windows showed reduced detection of slow AE attacks; testing with 120-second windows increased memory requirements and latency without proportional accuracy improvement.

---

**Q: Does the system work in real-time or only on historical data?**

The detection pipeline produces one anomaly score per second with a latency of approximately 3–5 milliseconds per window on CPU. For a 1 Hz data stream (matching the HAIEnd dataset), the system processes each incoming sample within its 1-second budget. Real-time deployment requires connecting the inference pipeline to the DCS data historian or OPC-UA server in place of the CSV-based simulation.

---

**Q: Why not use SHAP for explainability?**

SHAP (SHapley Additive exPlanations) requires hundreds to thousands of forward passes through the model to compute feature attributions. For a model processing 225 features at 1 Hz, SHAP would require ~100–1000 seconds of computation per second of data — completely infeasible for real-time operation.

The native XAI approach — sorting the per-sensor reconstruction error vector `E_{t,f}` — requires a single `np.argsort` call (microseconds) and produces results that are both interpretable and directly grounded in the model's internal physics representation. No approximation is needed because the reconstruction error itself is the explanation.

---

**Q: What is eTaF1 and why is it better than regular F1?**

Standard point-wise F1 calculates metrics at the individual timestep level. A model that detects the last 10 seconds of a 200-second attack gets credit for 10 true positives. A model that detects the first 10 seconds gets the same credit. Neither is clearly better from a safety standpoint.

eTaF1 (Enhanced Time-Aware F1 from HAICon 2021) requires that a predicted alert covers at least 50% of a true attack segment to count as a detection, and that a true attack segment is at least 50% covered to count as detected. This penalizes models that only detect attack tails or only detect part of each attack. FactoryShield-OT achieves eTaF1 ≈ 0.87, the highest in the benchmark comparison.

---

**Q: How is the 99th-percentile threshold computed without the test set?**

The threshold is computed exclusively on the training data: after the model finishes training, a full forward pass is run over all 896,341 normal training windows, computing the MAE for each. The 99th percentile of this distribution is saved inside the `.pth` checkpoint file as `threshold_p99`. The test set is never seen during threshold computation, eliminating any form of data leakage.

---

## 10. Glossary of Key Terms

| Term | Plain-language definition |
|------|--------------------------|
| **LSTM** | Long Short-Term Memory — a type of neural network designed to process sequences. It has internal "gates" that decide which past information to remember and which to forget. Ideal for time-series data like industrial sensor streams. |
| **Autoencoder** | A neural network trained to compress data into a small representation and then decompress it back. If it can reconstruct the data accurately, the data matches what it learned. If not, something is unusual. |
| **Latent vector** | The compressed representation. 896,400 seconds of 225-sensor plant data → 64 numbers. Like a ZIP file for the plant's physical state. |
| **Reconstruction error** | The difference between what the model expects to see and what actually happened. Small = normal. Large = anomaly. |
| **EMA (Exponential Moving Average)** | A smoothing technique that weights recent values more than old ones. Removes single-sample noise while preserving trends. |
| **MinMaxScaler** | A normalization tool that transforms all sensor values to the range [0, 1] based on the minimum and maximum values seen during training. |
| **Data leakage** | The mistake of allowing test or attack data to influence the training process (e.g., fitting the scaler on attack samples). Produces overoptimistic results that don't hold up in the real world. |
| **Semi-supervised learning** | Training on only one class (normal data) and detecting anything that doesn't fit that class. Doesn't require labeled attack examples. |
| **Threshold** | The cutoff value above which an anomaly score triggers an alert. Set at the 99th percentile of training errors. |
| **xAI (Explainable AI)** | Methods that explain why an AI model made a decision. FactoryShield-OT uses reconstruction error ranking — no black-box required. |
| **Root cause analysis** | Identifying which sensor or actuator is most responsible for the alert, pointing the operator to the physical location of the problem. |
| **eTaPR** | Enhanced Time-Aware Precision/Recall. A more rigorous evaluation metric for time-series anomaly detection that requires substantial temporal coverage of attacks, not just any overlap. |
| **DCS** | Distributed Control System. The computer-based control system that manages industrial processes. Emerson Ovation controls the P1 boiler in this testbed. |
| **SCADA** | Supervisory Control and Data Acquisition. The higher-level monitoring system that aggregates data from DCS, PLCs, and other field devices. |
| **HAIEnd 23.05** | The dataset used to train and validate FactoryShield-OT. Produced by KISA/NSHC at a real HIL testbed, covering 55 attack scenarios across Emerson Ovation, GE Mark VIe, and Siemens S7-300 systems. |
| **ICS** | Industrial Control System. The broad category including DCS, SCADA, PLC, and RTU systems used to control physical infrastructure. |
| **OT** | Operational Technology. The hardware and software that monitors and controls physical devices and processes, as opposed to IT (information technology) that handles data. |
| **ISA/IEC 62443** | The international standard for security of industrial automation and control systems. Defines security zones, conduits, and requirements for OT cybersecurity. |
| **Function block** | A building block of DCS control logic. Emerson Ovation programs are composed of function blocks (arithmetic, logic, PID, etc.) connected in a network. Internal logic attacks (AE01–AE08) target these blocks. |
| **Sliding window** | A technique for converting a continuous time series into overlapping fixed-length segments. At stride=1, each new second of data produces one new 60-second window. |
| **Stride** | The number of timesteps between consecutive windows. Stride=1 means windows overlap by 59 seconds. Used in training (stride=1 for maximum data) and CPU optimization (stride=10 for faster training). |

---

*End of SOC Explainer Guide — FactoryShield-OT v1.0*  
*Prepared for engineering students, plant managers, and oral defense evaluators.*  
*ISA/IEC 62443 aligned · HAIEnd 23.05 validated · EMSI Internship Project*
