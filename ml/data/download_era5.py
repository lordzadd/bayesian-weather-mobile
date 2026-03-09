"""
ERA5 reanalysis data ingestion via Copernicus Climate Data Store.

Requires a CDS API key configured at ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api/v2
    key: <UID>:<API_KEY>
"""

import cdsapi
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Variables required for BMA training
ERA5_VARIABLES = [
    "2m_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "relative_humidity",
]

# Years for training corpus (adjust as needed)
TRAINING_YEARS = [str(y) for y in range(2018, 2024)]

# Bounding box: CONUS (adjust for target region)
AREA = [50, -125, 24, -66]  # [N, W, S, E]


def download_era5_year(year: str, client: cdsapi.Client) -> Path:
    out_path = RAW_DIR / f"era5_{year}.nc"
    if out_path.exists():
        logger.info(f"ERA5 {year} already downloaded, skipping.")
        return out_path

    logger.info(f"Downloading ERA5 data for {year}...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ERA5_VARIABLES,
            "year": year,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24)],
            "area": AREA,
        },
        str(out_path),
    )
    logger.info(f"Saved to {out_path}")
    return out_path


def main():
    client = cdsapi.Client()
    for year in TRAINING_YEARS:
        download_era5_year(year, client)
    logger.info("ERA5 download complete.")


if __name__ == "__main__":
    main()
