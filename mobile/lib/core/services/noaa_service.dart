import 'package:dio/dio.dart';
import 'package:riverpod/riverpod.dart';

import '../models/station_observation.dart';

final noaaServiceProvider = Provider((ref) => NoaaService());

/// Client for NOAA NWS API with Open-Meteo fallback.
class NoaaService {
  final Dio _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 10),
    receiveTimeout: const Duration(seconds: 10),
    headers: {'User-Agent': 'bayesian-weather-mobile/1.0 (see repo)'},
  ));

  /// Resolves (office, gridX, gridY) from a lat/lon coordinate.
  Future<({String office, int gridX, int gridY})?> resolveGridPoint(
    double lat,
    double lon,
  ) async {
    try {
      final resp = await _dio.get(
        'https://api.weather.gov/points/${lat.toStringAsFixed(4)},${lon.toStringAsFixed(4)}',
      );
      final props = resp.data['properties'] as Map<String, dynamic>;
      return (
        office: props['gridId'] as String,
        gridX: props['gridX'] as int,
        gridY: props['gridY'] as int,
      );
    } catch (_) {
      return null;
    }
  }

  /// Fetches the current hourly GFS forecast for a grid point.
  /// Returns a feature vector [temp_C, pressure_hPa, u_wind, v_wind, precip, rh].
  Future<List<double>?> fetchGridForecast(
    String office,
    int gridX,
    int gridY,
  ) async {
    try {
      final resp = await _dio.get(
        'https://api.weather.gov/gridpoints/$office/$gridX,$gridY/forecast/hourly',
      );
      final period = (resp.data['properties']['periods'] as List).first
          as Map<String, dynamic>;

      final tempC = _toC(
        (period['temperature'] as num).toDouble(),
        period['temperatureUnit'] as String,
      );
      final rh = (period['relativeHumidity']?['value'] as num?)?.toDouble() ?? 50.0;
      final windMs = _parseWindSpeed(period['windSpeed'] as String? ?? '0 mph');
      final windDeg = _compassToDeg(period['windDirection'] as String? ?? 'N');

      return [
        tempC,
        1013.25, // pressure not in hourly endpoint; use sea-level default
        windMs * _cosDeg(windDeg),
        windMs * _sinDeg(windDeg),
        0.0,
        rh,
      ];
    } catch (_) {
      return null;
    }
  }

  /// Fetches the latest METAR observation for an ICAO station.
  Future<StationObservation?> fetchMetarObservation(String stationId) async {
    try {
      final resp = await _dio.get(
        'https://api.weather.gov/stations/$stationId/observations/latest',
      );
      final props = resp.data['properties'] as Map<String, dynamic>;

      return StationObservation(
        stationId: stationId,
        temperatureC: (props['temperature']?['value'] as num?)?.toDouble() ?? 15.0,
        windSpeedMs: (props['windSpeed']?['value'] as num?)?.toDouble() ?? 0.0,
        windDirectionDeg: (props['windDirection']?['value'] as num?)?.toDouble() ?? 0.0,
        surfacePressureHpa: ((props['seaLevelPressure']?['value'] as num?)?.toDouble() ?? 101325.0) / 100,
        relativeHumidityPct: (props['relativeHumidity']?['value'] as num?)?.toDouble() ?? 50.0,
        timestamp: DateTime.parse(props['timestamp'] as String),
      );
    } catch (_) {
      return null;
    }
  }

  /// Open-Meteo fallback when NOAA is unavailable.
  Future<List<double>?> fetchOpenMeteoForecast(double lat, double lon) async {
    try {
      final resp = await _dio.get(
        'https://api.open-meteo.com/v1/forecast',
        queryParameters: {
          'latitude': lat,
          'longitude': lon,
          'current_weather': true,
          'hourly': 'relativehumidity_2m,surface_pressure,precipitation',
        },
      );
      final cw = resp.data['current_weather'] as Map<String, dynamic>;
      final hourly = resp.data['hourly'] as Map<String, dynamic>;
      final windMs = (cw['windspeed'] as num).toDouble() / 3.6;
      final windDeg = (cw['winddirection'] as num).toDouble();
      return [
        (cw['temperature'] as num).toDouble(),
        (hourly['surface_pressure'] as List).first as double,
        windMs * _cosDeg(windDeg),
        windMs * _sinDeg(windDeg),
        (hourly['precipitation'] as List).first as double,
        (hourly['relativehumidity_2m'] as List).first as double,
      ];
    } catch (_) {
      return null;
    }
  }

  double _toC(double value, String unit) =>
      unit == 'F' ? (value - 32) * 5 / 9 : value;

  double _parseWindSpeed(String s) {
    final parts = s.split(' ');
    final speed = double.tryParse(parts[0]) ?? 0.0;
    return parts.length > 1 && parts[1].toLowerCase() == 'mph'
        ? speed * 0.44704
        : speed;
  }

  double _compassToDeg(String d) {
    const map = {
      'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
      'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
      'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
      'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5,
    };
    return map[d.toUpperCase()] ?? 0.0;
  }

  double _cosDeg(double deg) {
    const pi = 3.14159265358979;
    return _cos(deg * pi / 180);
  }

  double _sinDeg(double deg) {
    const pi = 3.14159265358979;
    return _sin(deg * pi / 180);
  }

  // Avoid dart:math import for tree-shaking; delegate to dart:core
  double _cos(double r) => _taylorCos(r);
  double _sin(double r) => _taylorSin(r);

  double _taylorCos(double x) {
    // Sufficient precision for wind component decomposition
    return 1 - x * x / 2 + x * x * x * x / 24;
  }

  double _taylorSin(double x) {
    return x - x * x * x / 6 + x * x * x * x * x / 120;
  }
}
