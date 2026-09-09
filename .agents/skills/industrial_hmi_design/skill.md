---
name: industrial_hmi_design
description: Enforces ISA-101 industrial HMI and SCADA UI standards; eliminates AI-generated SaaS aesthetics, rounded gradients, and cartoonish dashboards.
---

# Industrial HMI & Anti-AI-Aesthetic Design Rules

## 1. Eliminate the "AI SaaS" Clichés (Strict Bans)
- **NO Purple Gradients & Glowing Shadows:** Ban neon purple (`#7928CA`, `#8A2BE2`), cyan glows, drop shadows (`box-shadow: 0 10px 30px rgba(...)`), and translucent glassmorphism (`backdrop-filter: blur`).
- **NO Bubbly Rounded Radii:** Ban `border-radius: 12px` or higher. All containers, cards, and buttons must use `border-radius: 0px` or `2px` max (sharp, rectangular panels).
- **NO Casual SaaS Emojis:** Ban emojis in status pills or headers (`🚀`, `🔥`, `↑ Online`, `↑ +3`). Use strict industrial status tags: `[ ONLINE ]`, `[ FAULT ]`, `[ TRIP ]`, `[ STANDBY ]`.
- **NO Colorful Semicircular Gauges:** Replace toy-like speedometer widgets with horizontal bar progress meters, discrete LED step meters, or pure numerical digital readouts.

### 2. ISA-101 Light-Console Standard (Industrial Control Room)
- **The Neutral Light Canvas:** Use light industrial slate/concrete tones for the main surface (`#E5E7EB`, `#DDE1E6`, `#F3F4F6`). Panels should be bordered with darker steel gray (`#9CA3AF` or `#6B7280`).
- **Text & Telemetry:** Sharp high-contrast charcoal/black (`#111827`, `#1F2937`) using monospaced fonts (`Roboto Mono`, `Consolas`, `monospace`).
- **Reserved Alarm Colors (Maximum Contrast on Light Canvas):**
  - **Normal / Nominal:** Neutral off-black or dark slate. Muted green ONLY for running motors/actuators.
  - **Warning (Level 2):** Industrial Amber (`#D97706` / `#B45309`).
  - **Critical Alarm (Level 3):** High-contrast Safety Red (`#DC2626` / `#B91C1C`).
- **Plot & Trend Recorders:** Clean white or off-white background (`#FFFFFF` / `#F8FAFC`) with crisp thin gridlines (`#E2E8F0`) and solid black/navy/red line traces.

## 3. Industrial Telemetry Components
- **Alarm Annunciator Matrix:** Present system subsystems (P1-PC, P1-LC, P1-FC, P1-TC, P2-TURB) as a structured grid of physical-style light boxes.
- **Trend Recorders (Plots):**
  - Black/charcoal background (`#0e1117`).
  - Crisp, thin 1px plot traces (Cyan for Process Variable, White for Setpoint, Yellow for Filtered MAE, Red Dashed for Safety Trip Limit).
  - High-contrast technical gridlines (`#262a33`).