import 'dart:math' as math;

import '../core/models/forecast_result.dart';
import '../core/services/database_service.dart';
import 'bma_engine.dart';

/// Significance Threshold logic for Variant B (cache-optimized inference).
///
/// Before triggering GPU inference, this service checks whether the incoming
/// observation differs from the cached state by more than the configured
/// threshold. If the delta is below the threshold, the cached posterior is
/// returned immediately, saving battery and compute.
class ForecastCacheService {
  final DatabaseService _db;
  final BmaEngine _engine;

  /// Temperature delta (°C) below which GPU inference is skipped.
  final double temperatureThresholdC;

  /// Wind speed delta (m/s) below which GPU inference is skipped.
  final double windSpeedThresholdMs;

  ForecastCacheService({
    DatabaseService? db,
    BmaEngine? engine,
    this.temperatureThresholdC = 0.2,
    this.windSpeedThresholdMs = 0.5,
  })  : _db = db ?? DatabaseService.instance,
        _engine = engine ?? BmaEngine.instance;

  /// Returns a [ForecastResult] either from cache or fresh GPU inference.
  ///
  /// Decision logic:
  ///   1. Fetch cached posterior for (lat, lon, now).
  ///   2. If no cache → run inference, store result.
  ///   3. If cache exists → compare [newObservation] to cached values.
  ///      - Delta below both thresholds → return cached result (source=cache).
  ///      - Delta exceeds any threshold → run inference, update cache.
  Future<ForecastResult> getForecast({
    required double lat,
    required double lon,
    required List<double> gfsForecast,
    required List<double> spatialEmbed,
    required ObservationSnapshot newObservation,
  }) async {
    final now = DateTime.now();
    final cached = await _db.getCachedPosterior(lat: lat, lon: lon, time: now);

    final stopwatch = Stopwatch()..start();

    if (cached != null && _belowThreshold(cached, newObservation)) {
      stopwatch.stop();
      await _db.logBenchmark(
        variant: 'B',
        inferenceMs: stopwatch.elapsedMilliseconds,
        cacheHit: true,
      );
      // Override source to reflect this came from cache, not fresh GPU inference
      return cached.copyWith(source: InferenceSource.cache);
    }

    // Run GPU inference
    final result = await _engine.infer(
      gfsForecast: gfsForecast,
      spatialEmbed: spatialEmbed,
    );

    stopwatch.stop();
    await Future.wait([
      _db.savePosterior(lat: lat, lon: lon, time: now, result: result),
      _db.logBenchmark(
        variant: 'B',
        inferenceMs: stopwatch.elapsedMilliseconds,
        cacheHit: false,
      ),
    ]);

    return result;
  }

  bool _belowThreshold(ForecastResult cached, ObservationSnapshot obs) {
    final tempDelta = (cached.temperatureC - obs.temperatureC).abs();
    final windDelta = (cached.windSpeedMs - obs.windSpeedMs).abs();
    return tempDelta < temperatureThresholdC && windDelta < windSpeedThresholdMs;
  }
}

/// Lightweight snapshot of observable weather values used for threshold comparison.
class ObservationSnapshot {
  final double temperatureC;
  final double windSpeedMs;

  const ObservationSnapshot({
    required this.temperatureC,
    required this.windSpeedMs,
  });
}

/// Convenience extension to build an [ObservationSnapshot] from a feature vector.
extension ObservationSnapshotX on List<double> {
  ObservationSnapshot toSnapshot() => ObservationSnapshot(
        temperatureC: this[0],
        windSpeedMs: math.sqrt(this[2] * this[2] + this[3] * this[3]),
      );
}
