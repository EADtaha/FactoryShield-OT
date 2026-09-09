---
name: pytorch_timeseries_perf
description: Enforces memory-efficient PyTorch and NumPy operations for sliding window time-series tensors to prevent RAM exhaustion.
---
# PyTorch & Time-Series Performance Rules

## 1. Zero-Copy Windowing (Strict RAM Limit)
- **Never** pre-materialize 3D window tensors in RAM[cite: 7].
- Always use `torch.as_tensor()` memory views or `np.lib.stride_tricks.sliding_window_view` inside `__getitem__`[cite: 7]. This is mandatory for running 896k rows on hardware with limited memory.

## 2. Tensor Shape & Type
- Enforce standard shape `(Batch, Sequence=60, Features=225)` with `batch_first=True`[cite: 7].
- Cast arrays to `float32` early to halve the memory footprint.

## 3. Loss & Metric Parity
- Train autoencoders using `nn.L1Loss()` (Mean Absolute Error)[cite: 7]. This maintains exact mathematical alignment with the inference anomaly reconstruction metrics[cite: 7].

## 4. Vectorized Processing
- Compute Exponential Moving Averages (EMA) strictly via `scipy.signal.lfilter`[cite: 7]. Never use Python `for` loops for time-series filtering.

## 5. DataLoader Constraints
- Use `pin_memory=True` and `non_blocking=True`[cite: 7].
- For CPU-only execution, set `num_workers=0` to prevent multiprocessing fork overhead[cite: 7].