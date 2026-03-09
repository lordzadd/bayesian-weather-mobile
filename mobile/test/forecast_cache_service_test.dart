import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';

import 'package:bayesian_weather/core/models/forecast_result.dart';
import 'package:bayesian_weather/core/services/database_service.dart';
import 'package:bayesian_weather/inference/bma_engine.dart';
import 'package:bayesian_weather/inference/forecast_cache_service.dart';

import 'forecast_cache_service_test.mocks.dart';

@GenerateMocks([DatabaseService, BmaEngine])
void main() {
  late MockDatabaseService mockDb;
  late MockBmaEngine mockEngine;
  late ForecastCacheService service;

  const lat = 37.7749;
  const lon = -122.4194;
  final spatial = [lat / 90.0, lon / 180.0];
  final gfs = [15.0, 1013.25, 2.0, 1.5, 0.0, 65.0];

  final cachedResult = ForecastResult(
    temperatureC: 15.0,
    temperatureStd: 0.5,
    windSpeedMs: 2.5,
    windSpeedStd: 0.3,
    surfacePressureHpa: 1013.25,
    relativeHumidityPct: 65.0,
    precipitationMm: 0.0,
    computedAt: DateTime.now().subtract(const Duration(minutes: 30)),
    source: InferenceSource.cache,
  );

  setUp(() {
    mockDb = MockDatabaseService();
    mockEngine = MockBmaEngine();
    service = ForecastCacheService(
      db: mockDb,
      engine: mockEngine,
      temperatureThresholdC: 0.2,
      windSpeedThresholdMs: 0.5,
    );
  });

  group('cache hit — below threshold', () {
    test('returns cached result without calling engine', () async {
      when(mockDb.getCachedPosterior(lat: lat, lon: lon, time: anyNamed('time')))
          .thenAnswer((_) async => cachedResult);
      when(mockDb.logBenchmark(
        variant: anyNamed('variant'),
        inferenceMs: anyNamed('inferenceMs'),
        cacheHit: anyNamed('cacheHit'),
      )).thenAnswer((_) async {});

      // Delta = 0.05°C < 0.2°C threshold → cache should be served
      final obs = [15.05, 1013.25, 2.0, 1.5, 0.0, 65.0].toSnapshot();

      final result = await service.getForecast(
        lat: lat,
        lon: lon,
        gfsForecast: gfs,
        spatialEmbed: spatial,
        newObservation: obs,
      );

      expect(result.temperatureC, cachedResult.temperatureC);
      verifyNever(mockEngine.infer(
        gfsForecast: anyNamed('gfsForecast'),
        spatialEmbed: anyNamed('spatialEmbed'),
      ));
    });
  });

  group('cache miss — above threshold', () {
    test('calls engine when temperature delta exceeds threshold', () async {
      when(mockDb.getCachedPosterior(lat: lat, lon: lon, time: anyNamed('time')))
          .thenAnswer((_) async => cachedResult);
      when(mockDb.logBenchmark(
        variant: anyNamed('variant'),
        inferenceMs: anyNamed('inferenceMs'),
        cacheHit: anyNamed('cacheHit'),
      )).thenAnswer((_) async {});

      final freshResult = cachedResult.copyWith(
        temperatureC: 16.5,
        source: InferenceSource.gpu,
        computedAt: DateTime.now(),
      );
      when(mockEngine.infer(
        gfsForecast: anyNamed('gfsForecast'),
        spatialEmbed: anyNamed('spatialEmbed'),
      )).thenAnswer((_) async => freshResult);
      when(mockDb.savePosterior(
        lat: anyNamed('lat'),
        lon: anyNamed('lon'),
        time: anyNamed('time'),
        result: anyNamed('result'),
      )).thenAnswer((_) async {});

      // Delta = 1.5°C > 0.2°C threshold → engine must be called
      final obs = [16.5, 1013.25, 2.0, 1.5, 0.0, 65.0].toSnapshot();

      final result = await service.getForecast(
        lat: lat,
        lon: lon,
        gfsForecast: gfs,
        spatialEmbed: spatial,
        newObservation: obs,
      );

      expect(result.source, InferenceSource.gpu);
      verify(mockEngine.infer(
        gfsForecast: anyNamed('gfsForecast'),
        spatialEmbed: anyNamed('spatialEmbed'),
      )).called(1);
    });
  });

  group('no cache — cold start', () {
    test('always calls engine on first request', () async {
      when(mockDb.getCachedPosterior(lat: lat, lon: lon, time: anyNamed('time')))
          .thenAnswer((_) async => null);

      final freshResult = cachedResult.copyWith(source: InferenceSource.gpu);
      when(mockEngine.infer(
        gfsForecast: anyNamed('gfsForecast'),
        spatialEmbed: anyNamed('spatialEmbed'),
      )).thenAnswer((_) async => freshResult);
      when(mockDb.savePosterior(
        lat: anyNamed('lat'),
        lon: anyNamed('lon'),
        time: anyNamed('time'),
        result: anyNamed('result'),
      )).thenAnswer((_) async {});
      when(mockDb.logBenchmark(
        variant: anyNamed('variant'),
        inferenceMs: anyNamed('inferenceMs'),
        cacheHit: anyNamed('cacheHit'),
      )).thenAnswer((_) async {});

      final obs = gfs.toSnapshot();
      await service.getForecast(
        lat: lat,
        lon: lon,
        gfsForecast: gfs,
        spatialEmbed: spatial,
        newObservation: obs,
      );

      verify(mockEngine.infer(
        gfsForecast: anyNamed('gfsForecast'),
        spatialEmbed: anyNamed('spatialEmbed'),
      )).called(1);
    });
  });
}
