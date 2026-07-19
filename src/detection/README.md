FactoryShield-OT

Component: Real-Time Detection & Root-Cause Localization Pipeline
Files: src/detection/inference.py | src/detection/xai_root_cause.py
Dataset Target: HAIEnd 23.05 (ICS/OT End-device behavior)

-WHAT IT IS

This component is the active defense layer. It takes the pre-trained LSTM Autoencoder and deploys it over a simulated real-time data stream.

It performs two critical functions:

Inference: Continuously monitors the 225-sensor data stream, calculating an anomaly score for every passing second to detect deviations from the learned "normal" physics.

Explainability (xAI): When an alert is triggered, it instantly dissects the error vector to isolate the specific sensor or actuator that is the origin of the attack.

-THE DETECTION ENGINE (inference.py)

This script handles the real-time simulation and anomaly scoring.

1. Real-Time Semantics

To simulate a live ICS feed, we evaluate sliding windows of 60 seconds. However, we only care about the anomaly score at the very last second of that window. This represents the system's state right now.

The Geeky Part: Slicing the tensor.
We subtract the reconstructed batch from the real batch, take the absolute value, and isolate the last timestep (-1).

    # (batch - recon).abs() yields shape (Batch, Time, Features)
    chunks.append((batch - recon).abs()[:, -1, :].cpu().numpy())  # Extracts (B, F)


2. EMA Anti-Noise Filtering

Industrial environments are thermodynamically noisy. A single spike doesn't always mean an attack. We apply an Exponential Moving Average (EMA) to smooth the Mean Absolute Error (MAE) before comparing it to our threshold.

The Geeky Part: C-level execution.
Instead of a slow Python for loop, we use scipy.signal.lfilter to compute the Infinite Impulse Response (IIR) filter at C-speed.

    def ema_smooth(scores: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        return lfilter([alpha], [1.0, -(1.0 - alpha)], scores)


-THE LOCALIZATION ENGINE (xai_root_cause.py)

Once inference.py outputs a binary alert (1), this script jumps in to find the root cause.

1. Smart Triggers (Onset Detection)

If an attack lasts for 5 minutes, we don't need 300 identical alerts. We only want to analyze the exact moment the system transitioned from healthy (0) to under attack (1).

The Geeky Part: Transition tracking.
We shift the alert array by one position and compare it to the current array to find exactly where 0 -> 1 happens.

    def alert_onsets(df: pd.DataFrame) -> np.ndarray:
        alert = df["alert"].to_numpy()
        prev = np.concatenate(([0], alert[:-1]))
        return np.where((alert == 1) & (prev == 0))[0]


2. Vectorized Ranking

At the exact moment of the attack onset, we have an error vector of 225 sensors. We need to find the Top-5 worst reconstructors.

The Geeky Part: np.argsort
We sort the sensors by their error magnitude simultaneously for all alerts, without ever writing a Python loop over the 225 columns.

    # sub is the matrix of errors for the onset rows: shape (N_alerts, F)
    top_idx = np.argsort(-sub, axis=1)[:, :k]             
    top_sensors = np.array(feature_cols)[top_idx]         


-RUNNING THE PIPELINE

1. Run Inference (Generate anomaly scores):
Note: Use a lower batch size (e.g., 32) if processing large files on a CPU to avoid OOM (Killed) errors.

$ python src/detection/inference.py --data data/processed/test_mini.csv --threshold 0.08234 --batch-size 32 --output out/test_mini_results.csv


2. Isolate Root Causes (Generate xAI report):

$ python src/detection/xai_root_cause.py --errors-csv out/test_mini_results.csv --top-k 5 --output out/root_cause_report.csv
