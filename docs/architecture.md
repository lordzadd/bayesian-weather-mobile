# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Mobile App                           │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Forecast   │   │     Map      │   │    Settings    │  │
│  │   Screen    │   │    Screen    │   │    Screen      │  │
│  └──────┬──────┘   └──────┬───────┘   └────────┬───────┘  │
│         └─────────────────┼──────────────────────┘          │
│                           │                                  │
│                  ┌────────▼────────┐                        │
│                  │  ForecastNotifier│                        │
│                  │   (Riverpod)    │                        │
│                  └────────┬────────┘                        │
│                           │                                  │
│              ┌────────────┼────────────┐                    │
│              ▼            ▼            ▼                    │
│         ┌─────────┐ ┌─────────┐ ┌──────────┐              │
│         │  NOAA   │ │  BMA   │ │   Cache  │              │
│         │ Service │ │ Engine │ │  Service │              │
│         └────┬────┘ └────┬────┘ └────┬─────┘              │
│              │            │           │                      │
│         ┌────▼────────────▼───┐  ┌───▼──────┐             │
│         │   NOAA NWS API     │  │  SQLite  │             │
│         │   Open-Meteo (fallback)│  │   (Room) │             │
│         └────────────────────┘  └──────────┘             │
│                            │                                 │
│              ┌─────────────▼─────────────┐                 │
│              │    ExecuTorch Runtime      │                 │
│              │  Vulkan (Android)          │                 │
│              │  Metal  (iOS)              │                 │
│              └────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Inference Pipeline

### Variant A — GPU-Accelerated (No Cache)

```
User location resolved
        │
        ▼
NOAA NWS API ─────► GFS Feature Vector [6]
METAR Station ────► Observation Vector [6]
        │
        ▼
BmaEngine.infer()
        │
        ▼
ExecuTorch .pte
        │
        ├── Vulkan delegate (Android)
        └── Metal delegate (iOS)
        │
        ▼
Posterior (mean [6], std [6])
        │
        ▼
ForecastResult → UI + HeatmapLayer
```

Every observation update triggers a full GPU inference pass. Benchmark: `variant = 'A'`.

### Variant B — Cache-Optimized

```
User location resolved
        │
        ▼
ForecastCacheService.getForecast()
        │
        ├── getCachedPosterior(lat, lon, hour)
        │         │
        │   ┌─────▼─────┐
        │   │ Cache hit? │
        │   └─────┬─────┘
        │         │
        │   Yes: Δtemp < 0.2°C AND Δwind < 0.5 m/s?
        │         │
        │      Yes └──► Return cached ForecastResult (cache hit)
        │         │
        │      No  └──► BmaEngine.infer() → save to cache → return
        │
        └── No cache → BmaEngine.infer() → save to cache → return
```

Cache key: `(lat_4dp, lon_4dp, hour_bucket)`

## Data Flow: ML Pipeline

```
Copernicus CDS API
        │
        ▼
download_era5.py ──► data/raw/era5_{year}.nc
        │
        ▼
preprocess.py ────► data/processed/era5_tensors.pt
                 └─► data/processed/normalization_stats.pt
        │
        ▼
train.py (SVI on BMAModel) ──► checkpoints/bma_best.pt
        │
        ▼
export_to_pte.py ────────────► export/bma_model.pte
        │
        ▼
Bundle into Flutter app assets/models/
```

## Feature Vector Layout

All 6-element feature vectors follow this ordering:

| Index | Variable | Unit |
|-------|----------|------|
| 0 | 2m Temperature | °C (normalized) |
| 1 | Surface Pressure | hPa (normalized) |
| 2 | U-component wind (east) | m/s |
| 3 | V-component wind (north) | m/s |
| 4 | Total Precipitation | mm |
| 5 | Relative Humidity | % |

## Native FFI Interface

The `BmaEngine` calls into a native C library (`libbma_executorch.so` / `bma_executorch.framework`) that wraps the ExecuTorch Module API:

```c
// Load .pte model, returns opaque handle
void* bma_load(const char* model_path);

// Run inference: inputs gfs[6] + spatial[2], outputs mean[6] + std[6]
void bma_infer(void* handle,
               const float* gfs_input,
               const float* spatial_input,
               float* out_mean,
               float* out_std);

// Free model resources
void bma_free(void* handle);
```

See `mobile/android/app/src/main/cpp/` (to be implemented) for the C++ implementation.

## Benchmarking Schema

The `benchmark_log` table captures per-inference timing for comparative analysis:

```sql
CREATE TABLE benchmark_log (
    id          INTEGER PRIMARY KEY,
    variant     TEXT,    -- 'A' or 'B'
    inference_ms INTEGER, -- wall-clock latency
    cache_hit   INTEGER, -- 0 or 1
    timestamp   INTEGER  -- Unix ms
);
```

MAE validation runs against held-out METAR observations at T+1h, comparing:
- Raw GFS forecast MAE (baseline)
- BMA posterior mean MAE (primary metric)
