"""
Collects paired (GFS-seamless, ERA5) training data from Open-Meteo.

Uses two reanalysis/NWP model archives from the same API:
  - GFS seamless  — the forecast model we bias-correct at runtime
  - ERA5          — reanalysis used as the "truth" observation

Both are fetched at identical coordinates/times, so the difference
between them is exactly the systematic bias the BMA model must learn.

Stations: 10 major CONUS airports matching the app's location_service.dart.
Period:   2023-03-01 → 2025-03-01  (~17,520 hours × 10 stations = ~175K rows)

Output:
  ml/data/raw/gfs_proxy.csv   — GFS seamless (forecast side)
  ml/data/raw/metar_obs.csv   — ERA5 reanalysis (observation side)
"""

import time
import math
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = [
    # --- Original 10 major airports ---
    {"id": "KSFO", "lat": 37.619, "lon": -122.375},
    {"id": "KLAX", "lat": 33.942, "lon": -118.408},
    {"id": "KORD", "lat": 41.978, "lon":  -87.904},
    {"id": "KJFK", "lat": 40.639, "lon":  -73.778},
    {"id": "KDEN", "lat": 39.861, "lon": -104.673},
    {"id": "KDFW", "lat": 32.896, "lon":  -97.038},
    {"id": "KATL", "lat": 33.637, "lon":  -84.428},
    {"id": "KSEA", "lat": 47.449, "lon": -122.309},
    {"id": "KBOS", "lat": 42.364, "lon":  -71.005},
    {"id": "KMIA", "lat": 25.796, "lon":  -80.287},
    # --- 40 additional CONUS stations for geographic diversity ---
    {"id": "KPHX", "lat": 33.435, "lon": -112.007},  # Phoenix
    {"id": "KLAS", "lat": 36.080, "lon": -115.152},  # Las Vegas
    {"id": "KMSP", "lat": 44.883, "lon":  -93.229},  # Minneapolis
    {"id": "KDTW", "lat": 42.212, "lon":  -83.353},  # Detroit
    {"id": "KIAH", "lat": 29.984, "lon":  -95.341},  # Houston
    {"id": "KMCO", "lat": 28.429, "lon":  -81.309},  # Orlando
    {"id": "KPHL", "lat": 39.872, "lon":  -75.241},  # Philadelphia
    {"id": "KCLT", "lat": 35.214, "lon":  -80.943},  # Charlotte
    {"id": "KSLC", "lat": 40.789, "lon": -111.977},  # Salt Lake City
    {"id": "KPDX", "lat": 45.589, "lon": -122.597},  # Portland OR
    {"id": "KSTL", "lat": 38.749, "lon":  -90.370},  # St. Louis
    {"id": "KBNA", "lat": 36.126, "lon":  -86.681},  # Nashville
    {"id": "KMKE", "lat": 42.947, "lon":  -87.897},  # Milwaukee
    {"id": "KPIT", "lat": 40.492, "lon":  -80.233},  # Pittsburgh
    {"id": "KIND", "lat": 39.717, "lon":  -86.294},  # Indianapolis
    {"id": "KCVG", "lat": 39.048, "lon":  -84.667},  # Cincinnati
    {"id": "KMCI", "lat": 39.298, "lon":  -94.714},  # Kansas City
    {"id": "KAUS", "lat": 30.194, "lon":  -97.670},  # Austin
    {"id": "KSAN", "lat": 32.734, "lon": -117.190},  # San Diego
    {"id": "KSJC", "lat": 37.362, "lon": -121.929},  # San Jose
    {"id": "KRDU", "lat": 35.878, "lon":  -78.787},  # Raleigh-Durham
    {"id": "KCLE", "lat": 41.411, "lon":  -81.849},  # Cleveland
    {"id": "KABQ", "lat": 35.040, "lon": -106.609},  # Albuquerque
    {"id": "KTUL", "lat": 36.198, "lon":  -95.888},  # Tulsa
    {"id": "KOMA", "lat": 41.302, "lon":  -95.894},  # Omaha
    {"id": "KBOI", "lat": 43.564, "lon": -116.223},  # Boise
    {"id": "KBUF", "lat": 42.941, "lon":  -78.736},  # Buffalo
    {"id": "KPWM", "lat": 43.646, "lon":  -70.309},  # Portland ME
    {"id": "KJAX", "lat": 30.494, "lon":  -81.688},  # Jacksonville
    {"id": "KMSN", "lat": 43.140, "lon":  -89.337},  # Madison
    {"id": "KFAR", "lat": 46.921, "lon":  -96.816},  # Fargo (northern plains)
    {"id": "KRAP", "lat": 44.045, "lon": -103.054},  # Rapid City (plains)
    {"id": "KBIL", "lat": 45.808, "lon": -108.543},  # Billings (mountain)
    {"id": "KLIT", "lat": 34.729, "lon":  -92.224},  # Little Rock
    {"id": "KJAN", "lat": 32.311, "lon":  -90.076},  # Jackson MS
    {"id": "KELP", "lat": 31.807, "lon": -106.378},  # El Paso (desert)
    {"id": "KFAT", "lat": 36.776, "lon": -119.718},  # Fresno (central valley)
    {"id": "KGRR", "lat": 42.881, "lon":  -85.523},  # Grand Rapids
    {"id": "KDSM", "lat": 41.534, "lon":  -93.663},  # Des Moines
    {"id": "KSAT", "lat": 29.534, "lon":  -98.470},  # San Antonio
]

START_DATE = "2021-03-01"
END_DATE   = "2025-03-01"

HOURLY_VARS = [
    "temperature_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "relative_humidity_2m",
]


def _wind_uv(speed_series, dir_series):
    u = speed_series * dir_series.apply(lambda d: math.cos(math.radians(d)))
    v = speed_series * dir_series.apply(lambda d: math.sin(math.radians(d)))
    return u, v


def fetch_model(station: dict, model: str, prefix: str) -> pd.DataFrame:
    """Fetch hourly data for one model (gfs_seamless or era5)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   station["lat"],
        "longitude":  station["lon"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "hourly":     ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone":   "UTC",
        "models":     model,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    h = resp.json()["hourly"]

    speed = pd.Series(h["wind_speed_10m"], dtype=float)
    wdir  = pd.Series(h["wind_direction_10m"], dtype=float)
    u, v  = _wind_uv(speed, wdir)

    df = pd.DataFrame({
        "time":                   pd.to_datetime(h["time"]),
        f"{prefix}_temp_c":       pd.Series(h["temperature_2m"],    dtype=float),
        f"{prefix}_pres_hpa":     pd.Series(h["surface_pressure"],  dtype=float),
        f"{prefix}_u_ms":         u,
        f"{prefix}_v_ms":         v,
        f"{prefix}_precip_mm":    pd.Series(h["precipitation"],     dtype=float),
        f"{prefix}_rh_pct":       pd.Series(h["relative_humidity_2m"], dtype=float),
    })
    df["station"] = station["id"]
    df["lat"]     = station["lat"]
    df["lon"]     = station["lon"]

    # Temporal features — cyclic encoding of hour-of-day and day-of-year
    hours = df["time"].dt.hour + df["time"].dt.minute / 60.0
    doy = df["time"].dt.dayofyear
    df["sin_hour"] = (hours * 2 * math.pi / 24).apply(math.sin)
    df["cos_hour"] = (hours * 2 * math.pi / 24).apply(math.cos)
    df["sin_doy"]  = (doy * 2 * math.pi / 365.25).apply(math.sin)
    df["cos_doy"]  = (doy * 2 * math.pi / 365.25).apply(math.cos)

    return df


def main():
    gfs_frames, obs_frames = [], []

    for station in tqdm(STATIONS, desc="Stations"):
        sid = station["id"]
        try:
            gfs = fetch_model(station, model="gfs_seamless", prefix="gfs")
            gfs_frames.append(gfs)
            print(f"  {sid}: {len(gfs)} GFS rows")
        except Exception as e:
            print(f"  {sid} GFS failed: {e}")

        time.sleep(1)

        try:
            era5 = fetch_model(station, model="era5", prefix="obs")
            # Keep only time + obs columns (drop lat/lon/station, added by GFS side)
            era5 = era5[["time", "station",
                          "obs_temp_c", "obs_pres_hpa", "obs_u_ms", "obs_v_ms",
                          "obs_precip_mm", "obs_rh_pct"]]
            obs_frames.append(era5)
            print(f"  {sid}: {len(era5)} ERA5 rows")
        except Exception as e:
            print(f"  {sid} ERA5 failed: {e}")

        time.sleep(1)

    gfs_all = pd.concat(gfs_frames, ignore_index=True)
    obs_all = pd.concat(obs_frames, ignore_index=True)

    gfs_path = RAW_DIR / "gfs_proxy.csv"
    obs_path = RAW_DIR / "metar_obs.csv"
    gfs_all.to_csv(gfs_path, index=False)
    obs_all.to_csv(obs_path, index=False)
    print(f"\nSaved {len(gfs_all)} GFS rows  → {gfs_path}")
    print(f"Saved {len(obs_all)} ERA5/obs rows → {obs_path}")


if __name__ == "__main__":
    main()
