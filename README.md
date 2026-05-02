# Bayesian Weather Mobile

Google slides for final presentation:
https://docs.google.com/presentation/d/1692lSZghK8dbuAyw8BiruqMwYcnnXMFiryBC4b0GOxM/edit?usp=sharing
Demo vid:
https://youtu.be/a9y7j-x6ASU

Real-time hyper-local weather forecasting on mobile devices using Bayesian Model Averaging (BMA) deployed at the edge.

## Overview

This project investigates the computational viability of Bayesian Model Averaging deployed on resource-constrained mobile hardware. It corrects GFS grid forecasts using real-time ERA5 analysis observations, computing a posterior distribution that reflects local micro-climate conditions.

Two architectural variants are benchmarked:

| Variant | Strategy | Description |
|---------|----------|-------------|
| A | Always-infer | Full neural network pass on every data ingestion event |
| B | Cache-optimized | SQLite persistence with configurable significance threshold to skip redundant inference |

---

## Model

**Type**: Bayesian Model Averaging (BMA) with amortized Stochastic Variational Inference (SVI), implemented in PyTorch + Pyro.

The network has two heads:

| Head | Architecture | Role |
|------|-------------|------|
| `bias_net` | Linear(8→64) → ReLU → Linear(64→64) → ReLU → Linear(64→6) | Learns the systematic error between GFS forecasts and ERA5 truth per variable |
| `noise_net` | Linear(8→32) → ReLU → Linear(32→6) → Softplus | Learns heteroscedastic uncertainty — how wrong GFS tends to be at each location and variable |

Input to both heads is 8 dimensions: 6 normalized GFS forecast variables + normalized latitude + normalized longitude.

At inference time the bias-corrected forecast is used as a Gaussian prior, and the current ERA5 analysis value is used as the observation. A closed-form conjugate Gaussian update then produces the posterior mean and standard deviation shown on screen.

**Forecast horizon**: 1 hour ahead. The app fetches the GFS seamless next-hour forecast and corrects it against the current ERA5 analysis reading.

**Variables predicted**: temperature (°C), surface pressure (hPa), u-wind (m/s), v-wind (m/s), precipitation (mm/hr), relative humidity (%).

---

## Training Data

| Property | Value |
|----------|-------|
| Source | [Open-Meteo](https://open-meteo.com/) historical archive — GFS seamless (forecast) paired with ERA5 (reanalysis truth) |
| Stations | 10 major CONUS airports: KSFO, KLAX, KORD, KJFK, KDEN, KDFW, KATL, KSEA, KBOS, KMIA |
| Period | 2023-03-01 → 2025-03-01 (2 years, hourly) |
| Raw rows | 175,680 paired (GFS, ERA5) hourly observations |
| After cleaning | 126,489 training samples / 31,623 validation samples |
| Training | 150 epochs, MPS GPU (Apple Silicon), Pyro SVI with `Trace_ELBO` |
| Best val ELBO | ~8,454 |

The same Open-Meteo API is used at runtime (GFS seamless next-hour + ERA5 current), so the live inputs are always in-distribution with respect to the training normalization statistics.

## Repository Structure

```
.
├── mobile/          # Flutter application
│   ├── lib/
│   │   ├── core/        # Models, services, utilities
│   │   ├── features/    # UI features (forecast, map, settings)
│   │   └── inference/   # BMA engine, GPU delegate, cache layer
│   └── pubspec.yaml
├── ml/              # Python training pipeline
│   ├── data/            # ERA5 ingestion scripts
│   ├── training/        # BMA model (PyTorch/Pyro)
│   └── export/          # ExecuTorch .pte export
├── scripts/         # NOAA NWS API integration
└── docs/            # Architecture and design docs
```

## Data Sources

- **Training & live forecasts**: [Open-Meteo](https://open-meteo.com/) — no API key required
  - `model=gfs_seamless` for the forecast prior
  - `model=era5` / current block for the observation evidence
- Both training and runtime use the same API, variables, and units (m/s, °C, hPa, mm, %) so there is no distribution shift

## Architecture

### Inference Pipeline

```
ERA5 Training Data
       │
       ▼
  BMA Model (Pyro)  ──→  Export (.pte)  ──→  ExecuTorch Runtime
                                                      │
NOAA GFS Forecast ──→ Prior P(θ)                     │
METAR Observations ──→ Evidence D         ──→  Posterior P(θ|D)
                                                      │
                                              flutter_map Overlay
```

### Variant A: GPU Path
```
New Observation → ExecuTorch (Vulkan/Metal) → Fresh Posterior → Display
```

### Variant B: Cache Path
```
New Observation → Δ > Threshold? ──Yes──→ GPU Inference → Cache → Display
                        │
                        No
                        ▼
                  SQLite Cache → Display
```

## Evaluation Metrics

- **Accuracy**: MAE of BMA posterior vs. raw GFS, validated against T+1h station observations
- **Latency**: GPU inference time vs. cache retrieval time (ms)
- **Efficiency**: Battery consumption and thermal load under continuous background updates

## Setup

### ML Training Pipeline

```bash
cd ml/
pip install torch pyro-ppl pandas requests tqdm

# 1. Download 2 years of paired GFS + ERA5 data (no API key needed, ~3 min)
python data/collect_training_data.py

# 2. Join, clean, normalize, split 80/20
python data/build_dataset.py

# 3. Train BMA model (MPS/CUDA/CPU auto-detected, ~6 min on Apple Silicon)
python -m training.train --epochs 150

# 4. Export trained weights to Flutter assets
python export/export_weights_json.py
```

### Mobile App

```bash
cd mobile/
flutter pub get
flutter run
```

Requires Flutter 3.x and a connected device or emulator.

## License

MIT
