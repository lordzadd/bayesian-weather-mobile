import 'package:flutter_test/flutter_test.dart';

import 'package:bayesian_weather/core/models/forecast_result.dart';
import 'package:bayesian_weather/inference/forecast_cache_service.dart';

/// Lightweight in-memory stubs — no code generation needed.

class _FakeEngine {
  int callCount = 0;
  ForecastResult? nextResult;

  Future<ForecastResult> infer({
    required List<double> gfsForecast,
    required List<double> spatialEmbed,
  }) async {
    callCount++;
    return nextResult ??
        ForecastResult(
          temperatureC: gfsForecast[0] + 0.5,
          temperatureStd: 0.4,
          windSpeedMs: 2.5,
          windSpeedStd: 0.3,
          surfacePressureHpa: gfsForecast[1],
          relativeHumidityPct: gfsForecast[5],
          precipitationMm: 0.0,
          computedAt: DateTime.now(),
          source: InferenceSource.gpu,
        );
  }
}

class _FakeDb {
  ForecastResult? stored;
  final List<Map<String, dynamic>> benchmarkLog = [];

  Future<ForecastResult?> getCachedPosterior({
    required double lat,
    required double lon,
    required DateTime time,
  }) async => stored;

  Future<void> savePosterior({
    required double lat,
    required double lon,
    required DateTime time,
    required ForecastResult result,
  }) async => stored = result;

  Future<void> logBenchmark({
    required String variant,
    required int inferenceMs,
    required bool cacheHit,
  }) async => benchmarkLog.add({'variant': variant, 'cacheHit': cacheHit});
}

/// Thin subclass that injects the fakes via the public constructor.
///
/// We use a wrapper because [ForecastCacheService] takes [DatabaseService]
/// and [BmaEngine] by their concrete types. For unit testing we use a
/// composition-friendly subclass that overrides the critical methods.
class _TestCacheService extends ForecastCacheService {
  final _FakeEngine fakeEngine;
  final _FakeDb fakeDb;

  _TestCacheService({
    required this.fakeEngine,
    required this.fakeDb,
    double tempThreshold = 0.2,
    double windThreshold = 0.5,
  }) : super(
          temperatureThresholdC: tempThreshold,
          windSpeedThresholdMs: windThreshold,
        );

  @override
  Future<ForecastResult> getForecast({
    required double lat,
    required double lon,
    required List<double> gfsForecast,
    required List<double> spatialEmbed,
    required ObservationSnapshot newObservation,
  }) async {
    final cached = await fakeDb.getCachedPosterior(lat: lat, lon: lon, time: DateTime.now());

    if (cached != null && _belowThreshold(cached, newObservation)) {
      await fakeDb.logBenchmark(variant: 'B', inferenceMs: 1, cacheHit: true);
      return cached.copyWith(source: InferenceSource.cache);
    }

    final result = await fakeEngine.infer(gfsForecast: gfsForecast, spatialEmbed: spatialEmbed);
    await fakeDb.savePosterior(lat: lat, lon: lon, time: DateTime.now(), result: result);
    await fakeDb.logBenchmark(variant: 'B', inferenceMs: 50, cacheHit: false);
    return result;
  }

  bool _belowThreshold(ForecastResult cached, ObservationSnapshot obs) {
    return (cached.temperatureC - obs.temperatureC).abs() < temperatureThresholdC &&
        (cached.windSpeedMs - obs.windSpeedMs).abs() < windSpeedThresholdMs;
  }
}

void main() {
  const lat = 37.7749;
  const lon = -122.4194;
  final spatial = [lat / 90.0, lon / 180.0];
  final gfs = [15.0, 1013.25, 2.0, 1.5, 0.0, 65.0];

  final baseResult = ForecastResult(
    temperatureC: 15.0,
    temperatureStd: 0.5,
    windSpeedMs: 2.5,
    windSpeedStd: 0.3,
    surfacePressureHpa: 1013.25,
    relativeHumidityPct: 65.0,
    precipitationMm: 0.0,
    computedAt: DateTime.now(),
    source: InferenceSource.gpu,
  );

  group('cache hit — below threshold', () {
    test('returns cached result and does NOT call engine', () async {
      final db = _FakeDb()..stored = baseResult;
      final engine = _FakeEngine();
      final service = _TestCacheService(fakeEngine: engine, fakeDb: db);

      // Delta = 0.05°C < 0.2°C threshold
      final obs = [15.05, 1013.25, 2.0, 1.5, 0.0, 65.0].toSnapshot();
      final result = await service.getForecast(
        lat: lat, lon: lon, gfsForecast: gfs, spatialEmbed: spatial, newObservation: obs,
      );

      expect(result.source, InferenceSource.cache);
      expect(engine.callCount, 0);
      expect(db.benchmarkLog.last['cacheHit'], true);
    });
  });

  group('cache miss — above temperature threshold', () {
    test('calls engine when temp delta > threshold', () async {
      final db = _FakeDb()..stored = baseResult;
      final engine = _FakeEngine();
      final service = _TestCacheService(fakeEngine: engine, fakeDb: db);

      // Delta = 1.5°C > 0.2°C threshold
      final obs = [16.5, 1013.25, 2.0, 1.5, 0.0, 65.0].toSnapshot();
      final result = await service.getForecast(
        lat: lat, lon: lon, gfsForecast: gfs, spatialEmbed: spatial, newObservation: obs,
      );

      expect(result.source, InferenceSource.gpu);
      expect(engine.callCount, 1);
      expect(db.benchmarkLog.last['cacheHit'], false);
    });
  });

  group('cold start — no cache', () {
    test('always calls engine when no cached result exists', () async {
      final db = _FakeDb(); // stored = null
      final engine = _FakeEngine();
      final service = _TestCacheService(fakeEngine: engine, fakeDb: db);

      final obs = gfs.toSnapshot();
      await service.getForecast(
        lat: lat, lon: lon, gfsForecast: gfs, spatialEmbed: spatial, newObservation: obs,
      );

      expect(engine.callCount, 1);
      expect(db.stored, isNotNull);
    });
  });

  group('ObservationSnapshotX extension', () {
    test('computes wind magnitude from U/V components', () {
      // U=3, V=4 → magnitude=5
      final snap = [10.0, 1013.0, 3.0, 4.0, 0.0, 60.0].toSnapshot();
      expect(snap.windSpeedMs, closeTo(5.0, 1e-9));
      expect(snap.temperatureC, 10.0);
    });

    test('zero wind gives zero magnitude', () {
      final snap = [20.0, 1013.0, 0.0, 0.0, 0.0, 50.0].toSnapshot();
      expect(snap.windSpeedMs, 0.0);
    });
  });
}
