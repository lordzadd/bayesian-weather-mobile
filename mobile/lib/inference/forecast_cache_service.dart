import 'dart:math' as math;

import '../core/models/forecast_result.dart';
import '../core/services/database_service.dart';
import 'bma_engine.dart';

/// Significance Threshold logic for Variant B (cache-optimized inference).
///
/// Before triggering Bayesian inference, checks whether the incoming
/// observation has changed enough to warrant recomputing the posterior.
/// If Δ < threshold on both temperature and wind, the cached result is served.
class ForecastCacheService {
  final DatabaseService _db;
  final BmaEngine _engine;

  final double temperatureThresholdC;
  final double windSpeedThresholdMs;

  ForecastCacheService({
    DatabaseService? db,
    BmaEngine? engine,
    this.temperatureThresholdC = 0.2,
    this.windSpeedThresholdMs = 0.5,
  })  : _db = db ?? DatabaseService.instance,
        _engine = engine ?? BmaEngine.instance;

  /// Returns a [ForecastResult] from cache or fresh inference.
  ///
  /// [obsFeatures] is the METAR observation vector — passed into the engine
  /// as Bayesian evidence so the posterior can be updated against real data.
  Future<ForecastResult> getForecast({
    required double lat,
    required double lon,
    required List<double> gfsForecast,
    required List<double>? obsFeatures,
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
      return cached.copyWith(source: InferenceSource.cache);
    }

    // Run Bayesian inference with METAR observation as evidence
    final result = await _engine.infer(
      gfsForecast: gfsForecast,
      obsFeatures: obsFeatures,
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

class ObservationSnapshot {
  final double temperatureC;
  final double windSpeedMs;

  const ObservationSnapshot({
    required this.temperatureC,
    required this.windSpeedMs,
  });
}

extension ObservationSnapshotX on List<double> {
  ObservationSnapshot toSnapshot() => ObservationSnapshot(
        temperatureC: this[0],
        windSpeedMs: math.sqrt(this[2] * this[2] + this[3] * this[3]),
      );
}
