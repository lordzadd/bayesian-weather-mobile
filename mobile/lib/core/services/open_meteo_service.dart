import 'dart:math' as math;

import 'package:dio/dio.dart';
import 'package:riverpod/riverpod.dart';

final openMeteoServiceProvider = Provider((ref) => OpenMeteoService());

/// Fetches GFS forecast + current ERA5 observations from Open-Meteo.
///
/// Uses the same API, variables, and units as the training pipeline
/// (ml/data/collect_training_data.py), so inputs are always in-distribution:
///   - temperature_2m      → Celsius
///   - surface_pressure    → hPa
///   - wind_speed_10m      → m/s  (wind_speed_unit=ms)
///   - wind_direction_10m  → degrees
///   - precipitation       → mm
///   - relative_humidity_2m → %
///
/// Returns two feature vectors [6] each:
///   gfsForecast  — next-hour GFS seamless forecast  (prior in BMA)
///   observation  — current analysis/ERA5 values     (Bayesian evidence)
class OpenMeteoService {
  final Dio _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 10),
    receiveTimeout: const Duration(seconds: 15),
  ));

  static const _url = 'https://api.open-meteo.com/v1/forecast';

  static const _vars =
      'temperature_2m,surface_pressure,wind_speed_10m,wind_direction_10m,'
      'precipitation,relative_humidity_2m';

  /// Returns `(gfsForecast, observation)` feature vectors, or null on failure.
  Future<({List<double> gfs, List<double> obs})?> fetch(
      double lat, double lon) async {
    try {
      final resp = await _dio.get(_url, queryParameters: {
        'latitude': lat,
        'longitude': lon,
        'current': _vars,
        'hourly': _vars,
        'wind_speed_unit': 'ms',
        'timezone': 'UTC',
        'forecast_days': 1,
        'models': 'gfs_seamless',
      });

      final current = resp.data['current'] as Map<String, dynamic>;
      final hourly = resp.data['hourly'] as Map<String, dynamic>;

      // Current conditions = Bayesian evidence (observation)
      final obs = _toFeatureVector(
        tempC: (current['temperature_2m'] as num).toDouble(),
        pressHpa: (current['surface_pressure'] as num).toDouble(),
        windMs: (current['wind_speed_10m'] as num).toDouble(),
        windDeg: (current['wind_direction_10m'] as num).toDouble(),
        precipMm: (current['precipitation'] as num).toDouble(),
        rhPct: (current['relative_humidity_2m'] as num).toDouble(),
      );

      // Current time has sub-hour precision (e.g. "T17:45"); hourly slots are
      // on the hour ("T17:00"). Truncate to HH:00 to find the matching slot,
      // then take +1 for the next-hour GFS forecast.
      final currentTime = current['time'] as String;
      final currentHour = currentTime.length >= 13
          ? '${currentTime.substring(0, 13)}:00'
          : currentTime;
      final times = hourly['time'] as List;
      int idx = times.indexOf(currentHour);
      if (idx < 0) idx = 0;
      final fi = (idx + 1).clamp(0, times.length - 1);

      final gfs = _toFeatureVector(
        tempC: (hourly['temperature_2m'][fi] as num).toDouble(),
        pressHpa: (hourly['surface_pressure'][fi] as num).toDouble(),
        windMs: (hourly['wind_speed_10m'][fi] as num).toDouble(),
        windDeg: (hourly['wind_direction_10m'][fi] as num).toDouble(),
        precipMm: (hourly['precipitation'][fi] as num).toDouble(),
        rhPct: (hourly['relative_humidity_2m'][fi] as num).toDouble(),
      );

      return (gfs: gfs, obs: obs);
    } catch (_) {
      return null;
    }
  }

  /// Fetches a lookback pair for accuracy validation.
  ///
  /// Returns:
  ///   [gfsT0]    — GFS hourly slot at the current hour (what model uses as prior)
  ///   [obsT1hr]  — ERA5/hourly slot 1 hour ago (observation available at T-1hr)
  ///   [era5Now]  — current ERA5 analysis (ground truth at T-0)
  ///
  /// Simulates: "Given observation at T-1hr, predict T-0 → compare against actual T-0."
  Future<({List<double> gfsT0, List<double> obsT1hr, List<double> era5Now})?> fetchLookbackPair(
      double lat, double lon) async {
    try {
      final resp = await _dio.get(_url, queryParameters: {
        'latitude': lat,
        'longitude': lon,
        'current': _vars,
        'hourly': _vars,
        'wind_speed_unit': 'ms',
        'timezone': 'UTC',
        'forecast_days': 1,
        'past_hours': 2,
        'models': 'gfs_seamless',
      });

      final current = resp.data['current'] as Map<String, dynamic>;
      final hourly = resp.data['hourly'] as Map<String, dynamic>;

      // Ground truth: current ERA5 analysis
      final era5Now = _toFeatureVector(
        tempC: (current['temperature_2m'] as num).toDouble(),
        pressHpa: (current['surface_pressure'] as num).toDouble(),
        windMs: (current['wind_speed_10m'] as num).toDouble(),
        windDeg: (current['wind_direction_10m'] as num).toDouble(),
        precipMm: (current['precipitation'] as num).toDouble(),
        rhPct: (current['relative_humidity_2m'] as num).toDouble(),
      );

      final currentTime = current['time'] as String;
      final currentHour = currentTime.length >= 13
          ? '${currentTime.substring(0, 13)}:00'
          : currentTime;
      final times = hourly['time'] as List;
      int idx = times.indexOf(currentHour);
      if (idx < 0) idx = times.length - 1;

      // T-0 GFS slot: the model's prior for the current hour
      final gfsT0 = _slotVector(hourly, idx);

      // T-1hr observation: what was observed an hour ago
      final prevIdx = (idx - 1).clamp(0, times.length - 1);
      final obsT1hr = _slotVector(hourly, prevIdx);

      return (gfsT0: gfsT0, obsT1hr: obsT1hr, era5Now: era5Now);
    } catch (_) {
      return null;
    }
  }

  List<double> _slotVector(Map<String, dynamic> hourly, int idx) {
    return _toFeatureVector(
      tempC: (hourly['temperature_2m'][idx] as num).toDouble(),
      pressHpa: (hourly['surface_pressure'][idx] as num).toDouble(),
      windMs: (hourly['wind_speed_10m'][idx] as num).toDouble(),
      windDeg: (hourly['wind_direction_10m'][idx] as num).toDouble(),
      precipMm: (hourly['precipitation'][idx] as num).toDouble(),
      rhPct: (hourly['relative_humidity_2m'][idx] as num).toDouble(),
    );
  }

  // Decomposes wind speed+direction into (u, v) components —
  // same transform as collect_training_data.py.
  List<double> _toFeatureVector({
    required double tempC,
    required double pressHpa,
    required double windMs,
    required double windDeg,
    required double precipMm,
    required double rhPct,
  }) {
    final rad = windDeg * math.pi / 180.0;
    return [
      tempC,
      pressHpa,
      windMs * math.cos(rad), // u
      windMs * math.sin(rad), // v
      precipMm,
      rhPct,
    ];
  }
}
