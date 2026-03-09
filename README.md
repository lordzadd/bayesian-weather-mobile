# Bayesian Weather Mobile

Real-time hyper-local weather forecasting on mobile devices using Bayesian Model Averaging (BMA) deployed at the edge.

## Overview

This project investigates the computational viability of Bayesian Model Averaging deployed on resource-constrained mobile hardware. It corrects coarse NOAA GFS grid forecasts using real-time METAR/ASOS station observations, computing a posterior distribution that reflects local micro-climate conditions.

Two architectural variants are benchmarked:

| Variant | Strategy | Description |
|---------|----------|-------------|
| A | GPU-Accelerated | Full inference on every data ingestion event via ExecuTorch (Vulkan/Metal) |
| B | Cache-Optimized | SQLite persistence layer with configurable Significance Threshold to skip redundant inference |

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

- **Training**: [ERA5 reanalysis](https://cds.climate.copernicus.eu/) via Copernicus Climate Data Store (5+ years)
- **Live forecasts**: [NOAA NWS API](https://api.weather.gov/) for GFS grid forecasts
- **Observations**: METAR/ASOS station network for real-time evidence
- **Backup**: [Open-Meteo](https://open-meteo.com/) if NOAA API is unavailable

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

### ML Training Environment

```bash
cd ml/
pip install -r requirements.txt
python data/download_era5.py
python training/train.py
python export/export_to_pte.py
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
