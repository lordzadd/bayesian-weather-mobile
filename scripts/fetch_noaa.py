"""
Fetches real-time GFS grid forecasts and METAR station observations from NOAA NWS API.

Endpoints used:
  - https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast
  - https://api.weather.gov/stations/{stationId}/observations/latest

Falls back to Open-Meteo if NOAA returns non-200 status.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOAA_BASE = "https://api.weather.gov"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT = 10  # seconds
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0


@dataclass
class GridForecast:
    temperature_c: float
    wind_speed_ms: float
    wind_direction_deg: float
    surface_pressure_hpa: float
    precipitation_mm: float
    relative_humidity_pct: float
    source: str  # "noaa" or "open-meteo"


@dataclass
class StationObservation:
    station_id: str
    temperature_c: float
    wind_speed_ms: float
    wind_direction_deg: float
    surface_pressure_hpa: float
    relative_humidity_pct: float
    timestamp: str


def _get_with_retry(url: str, headers: dict = None) -> Optional[requests.Response]:
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.ok:
                return resp
            logger.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            logger.warning(f"Request error: {e} (attempt {attempt + 1})")
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(RETRY_BACKOFF ** attempt)
    return None


def fetch_noaa_grid_forecast(office: str, grid_x: int, grid_y: int) -> Optional[GridForecast]:
    """
    Fetch NWS gridpoint forecast for the given office/grid coordinates.
    Use https://api.weather.gov/points/{lat},{lon} to resolve office/grid from coordinates.
    """
    url = f"{NOAA_BASE}/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly"
    headers = {"User-Agent": "bayesian-weather-mobile/1.0 (contact: see repo)"}

    resp = _get_with_retry(url, headers)
    if resp is None:
        logger.warning("NOAA API unavailable. Falling back to Open-Meteo.")
        return None

    data = resp.json()
    try:
        period = data["properties"]["periods"][0]
        return GridForecast(
            temperature_c=(period["temperature"] - 32) * 5 / 9 if period.get("temperatureUnit") == "F" else period["temperature"],
            wind_speed_ms=_parse_wind_speed(period.get("windSpeed", "0 mph")),
            wind_direction_deg=_parse_wind_direction(period.get("windDirection", "N")),
            surface_pressure_hpa=float(data["properties"].get("pressure", {}).get("values", [[None, 101325]])[0][1]) / 100,
            precipitation_mm=0.0,
            relative_humidity_pct=float(period.get("relativeHumidity", {}).get("value", 50)),
            source="noaa",
        )
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to parse NOAA response: {e}")
        return None


def fetch_open_meteo_forecast(lat: float, lon: float) -> Optional[GridForecast]:
    """Fallback: Open-Meteo current weather."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "relativehumidity_2m,surface_pressure,precipitation",
    }
    resp = _get_with_retry(OPEN_METEO_BASE, headers=None)
    if resp is None:
        return None

    data = resp.json()
    cw = data.get("current_weather", {})
    hourly = data.get("hourly", {})
    return GridForecast(
        temperature_c=cw.get("temperature", 15.0),
        wind_speed_ms=cw.get("windspeed", 0.0) / 3.6,
        wind_direction_deg=cw.get("winddirection", 0.0),
        surface_pressure_hpa=hourly.get("surface_pressure", [1013.25])[0],
        precipitation_mm=hourly.get("precipitation", [0.0])[0],
        relative_humidity_pct=hourly.get("relativehumidity_2m", [50])[0],
        source="open-meteo",
    )


def fetch_metar_observation(station_id: str) -> Optional[StationObservation]:
    """Fetch latest METAR observation from NWS for a given ICAO station."""
    url = f"{NOAA_BASE}/stations/{station_id}/observations/latest"
    headers = {"User-Agent": "bayesian-weather-mobile/1.0 (contact: see repo)"}

    resp = _get_with_retry(url, headers)
    if resp is None:
        return None

    props = resp.json().get("properties", {})
    try:
        return StationObservation(
            station_id=station_id,
            temperature_c=props["temperature"]["value"] or 0.0,
            wind_speed_ms=props["windSpeed"]["value"] or 0.0,
            wind_direction_deg=props["windDirection"]["value"] or 0.0,
            surface_pressure_hpa=(props["seaLevelPressure"]["value"] or 101325) / 100,
            relative_humidity_pct=props["relativeHumidity"]["value"] or 50.0,
            timestamp=props["timestamp"],
        )
    except (KeyError, TypeError) as e:
        logger.error(f"Failed to parse METAR observation for {station_id}: {e}")
        return None


def resolve_grid_point(lat: float, lon: float) -> Optional[tuple[str, int, int]]:
    """Returns (office, grid_x, grid_y) for a lat/lon coordinate."""
    url = f"{NOAA_BASE}/points/{lat:.4f},{lon:.4f}"
    headers = {"User-Agent": "bayesian-weather-mobile/1.0 (contact: see repo)"}
    resp = _get_with_retry(url, headers)
    if resp is None:
        return None
    props = resp.json().get("properties", {})
    return props.get("gridId"), props.get("gridX"), props.get("gridY")


def _parse_wind_speed(wind_str: str) -> float:
    """Parse '12 mph' -> m/s."""
    parts = wind_str.split()
    if not parts:
        return 0.0
    speed = float(parts[0])
    unit = parts[1].lower() if len(parts) > 1 else "mph"
    return speed * 0.44704 if unit == "mph" else speed


def _parse_wind_direction(direction: str) -> float:
    compass = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
               "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
               "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
               "WNW": 292.5, "NW": 315, "NNW": 337.5}
    return compass.get(direction.upper(), 0.0)
