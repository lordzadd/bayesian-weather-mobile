import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/lookback_result.dart';
import '../../core/services/location_service.dart';
import '../../core/services/lookback_service.dart';
import '../../core/services/open_meteo_service.dart';

/// State for the lookback validation feature.
class LookbackState {
  final LookbackResult? lastResult;
  final AccuracyStats? rollingStats;
  final bool isLoading;
  final DateTime? lastRunAt;

  const LookbackState({
    this.lastResult,
    this.rollingStats,
    this.isLoading = false,
    this.lastRunAt,
  });

  LookbackState copyWith({
    LookbackResult? lastResult,
    AccuracyStats? rollingStats,
    bool? isLoading,
    DateTime? lastRunAt,
  }) {
    return LookbackState(
      lastResult: lastResult ?? this.lastResult,
      rollingStats: rollingStats ?? this.rollingStats,
      isLoading: isLoading ?? this.isLoading,
      lastRunAt: lastRunAt ?? this.lastRunAt,
    );
  }
}

class LookbackNotifier extends Notifier<LookbackState> {
  static const _cooldown = Duration(minutes: 30);

  late final LookbackService _service;
  late final LocationService _location;

  @override
  LookbackState build() {
    _service = LookbackService(
      weather: ref.read(openMeteoServiceProvider),
    );
    _location = ref.read(locationServiceProvider);

    // Auto-trigger on first build
    Future.microtask(() => runIfDue());

    return const LookbackState();
  }

  /// Runs lookback only if cooldown has elapsed.
  Future<void> runIfDue() async {
    final now = DateTime.now();
    final last = state.lastRunAt;
    if (last != null && now.difference(last) < _cooldown) return;
    await run();
  }

  /// Forces a lookback validation run regardless of cooldown.
  Future<void> run() async {
    state = state.copyWith(isLoading: true);

    try {
      final (:lat, :lon) = await _location.currentPosition();

      final result = await _service.runLookback(lat: lat, lon: lon);
      final stats = await _service.getRollingStats();

      state = LookbackState(
        lastResult: result,
        rollingStats: stats,
        isLoading: false,
        lastRunAt: DateTime.now(),
      );
    } catch (_) {
      state = state.copyWith(isLoading: false, lastRunAt: DateTime.now());
    }
  }
}

final lookbackProvider =
    NotifierProvider<LookbackNotifier, LookbackState>(LookbackNotifier.new);
