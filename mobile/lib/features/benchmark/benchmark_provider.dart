import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/services/database_service.dart';
import '../../inference/bma_engine.dart';
import '../../inference/forecast_cache_service.dart';

class BenchmarkResult {
  final String variant;
  final List<int> latenciesMs;
  final int cacheHits;
  final int cacheMisses;

  const BenchmarkResult({
    required this.variant,
    required this.latenciesMs,
    required this.cacheHits,
    required this.cacheMisses,
  });

  double get meanMs =>
      latenciesMs.isEmpty ? 0 : latenciesMs.reduce((a, b) => a + b) / latenciesMs.length;

  int get p95Ms {
    if (latenciesMs.isEmpty) return 0;
    final sorted = List<int>.from(latenciesMs)..sort();
    return sorted[(sorted.length * 0.95).floor().clamp(0, sorted.length - 1)];
  }

  int get p99Ms {
    if (latenciesMs.isEmpty) return 0;
    final sorted = List<int>.from(latenciesMs)..sort();
    return sorted[(sorted.length * 0.99).floor().clamp(0, sorted.length - 1)];
  }

  double get cacheHitRate =>
      (cacheHits + cacheMisses) == 0 ? 0 : cacheHits / (cacheHits + cacheMisses);
}

class BenchmarkState {
  final bool running;
  final int progress;
  final int total;
  final BenchmarkResult? variantA;
  final BenchmarkResult? variantB;
  final String? error;

  const BenchmarkState({
    this.running = false,
    this.progress = 0,
    this.total = 0,
    this.variantA,
    this.variantB,
    this.error,
  });

  BenchmarkState copyWith({
    bool? running,
    int? progress,
    int? total,
    BenchmarkResult? variantA,
    BenchmarkResult? variantB,
    String? error,
  }) => BenchmarkState(
        running: running ?? this.running,
        progress: progress ?? this.progress,
        total: total ?? this.total,
        variantA: variantA ?? this.variantA,
        variantB: variantB ?? this.variantB,
        error: error ?? this.error,
      );
}

class BenchmarkNotifier extends Notifier<BenchmarkState> {
  static const int _cycles = 50;

  // Synthetic observation stream: simulates stable weather (small deltas)
  // interleaved with threshold-crossing events.
  static final List<List<double>> _syntheticGfs = List.generate(
    _cycles,
    (i) => [
      15.0 + (i % 5) * 0.05, // temp oscillates 0.05°C — below threshold
      1013.25,
      2.0,
      1.5,
      0.0,
      65.0,
    ],
  );

  @override
  BenchmarkState build() => const BenchmarkState();

  Future<void> run() async {
    if (state.running) return;
    state = BenchmarkState(running: true, total: _cycles * 2);

    try {
      final variantA = await _runVariantA();
      state = state.copyWith(variantA: variantA, progress: _cycles);

      final variantB = await _runVariantB();
      state = state.copyWith(variantB: variantB, progress: _cycles * 2, running: false);
    } catch (e) {
      state = state.copyWith(running: false, error: e.toString());
    }
  }

  Future<BenchmarkResult> _runVariantA() async {
    final latencies = <int>[];
    final engine = BmaEngine.instance;
    final spatial = [37.7749 / 90.0, -122.4194 / 180.0];

    for (int i = 0; i < _cycles; i++) {
      final sw = Stopwatch()..start();
      await engine.infer(gfsForecast: _syntheticGfs[i], spatialEmbed: spatial);
      latencies.add(sw.elapsedMilliseconds);

      await DatabaseService.instance.logBenchmark(
        variant: 'A',
        inferenceMs: latencies.last,
        cacheHit: false,
      );

      state = state.copyWith(progress: i + 1);
    }

    return BenchmarkResult(
      variant: 'A',
      latenciesMs: latencies,
      cacheHits: 0,
      cacheMisses: _cycles,
    );
  }

  Future<BenchmarkResult> _runVariantB() async {
    final latencies = <int>[];
    int cacheHits = 0;
    int cacheMisses = 0;

    final cacheService = ForecastCacheService(
      temperatureThresholdC: 0.2,
      windSpeedThresholdMs: 0.5,
    );
    final spatial = [37.7749 / 90.0, -122.4194 / 180.0];

    for (int i = 0; i < _cycles; i++) {
      final sw = Stopwatch()..start();
      final result = await cacheService.getForecast(
        lat: 37.7749,
        lon: -122.4194,
        gfsForecast: _syntheticGfs[i],
        spatialEmbed: spatial,
        newObservation: _syntheticGfs[i].toSnapshot(),
      );
      latencies.add(sw.elapsedMilliseconds);

      final fromCache = result.source == InferenceSource.cache;
      if (fromCache) cacheHits++; else cacheMisses++;

      state = state.copyWith(progress: _cycles + i + 1);
    }

    return BenchmarkResult(
      variant: 'B',
      latenciesMs: latencies,
      cacheHits: cacheHits,
      cacheMisses: cacheMisses,
    );
  }

  Future<String> exportCsv() async {
    final rows = await DatabaseService.instance.getBenchmarkLog();
    final buffer = StringBuffer('variant,inference_ms,cache_hit,timestamp\n');
    for (final row in rows) {
      buffer.writeln('${row['variant']},${row['inference_ms']},${row['cache_hit']},${row['timestamp']}');
    }
    return buffer.toString();
  }
}

final benchmarkProvider = NotifierProvider<BenchmarkNotifier, BenchmarkState>(
  BenchmarkNotifier.new,
);
