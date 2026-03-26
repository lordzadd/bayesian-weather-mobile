import 'dart:math' as math;

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/forecast_result.dart';
import '../../core/services/database_service.dart';
import '../../core/services/location_service.dart';
import '../../core/services/observation_buffer_service.dart';
import '../../core/services/open_meteo_service.dart';
import '../../inference/bma_engine.dart';
import '../../inference/linear_dart_engine.dart';
import '../../inference/lstm_dart_engine.dart';
import '../settings/settings_provider.dart';

// ── State ─────────────────────────────────────────────────────────────────────

enum ValidationStatus { idle, running, done, error }

class ValidationState {
  final ValidationStatus status;
  final String? errorMessage;

  /// Last single-run metrics (populated after a run completes).
  final Map<String, double>? lastMetrics;

  /// Aggregated summary rows from the DB, newest runs first.
  final List<Map<String, dynamic>> summary;

  const ValidationState({
    this.status = ValidationStatus.idle,
    this.errorMessage,
    this.lastMetrics,
    this.summary = const [],
  });

  ValidationState copyWith({
    ValidationStatus? status,
    String? errorMessage,
    Map<String, double>? lastMetrics,
    List<Map<String, dynamic>>? summary,
  }) {
    return ValidationState(
      status: status ?? this.status,
      errorMessage: errorMessage,
      lastMetrics: lastMetrics ?? this.lastMetrics,
      summary: summary ?? this.summary,
    );
  }
}

// ── Notifier ──────────────────────────────────────────────────────────────────

final validationProvider =
    AsyncNotifierProvider<ValidationNotifier, ValidationState>(
  ValidationNotifier.new,
);

class ValidationNotifier extends AsyncNotifier<ValidationState> {
  @override
  Future<ValidationState> build() async {
    final summary = await DatabaseService.instance.getLookbackSummary();
    return ValidationState(summary: summary);
  }

  Future<void> runLookback({ModelVariant? overrideVariant}) async {
    state = AsyncData(state.value!.copyWith(status: ValidationStatus.running));

    try {
      final location = ref.read(locationServiceProvider);
      final weather  = ref.read(openMeteoServiceProvider);
      final settings = ref.read(settingsProvider);
      final modelVariant = overrideVariant ?? settings.modelVariant;

      final (:lat, :lon) = await location.currentPosition();
      final spatial = [lat / 90.0, lon / 180.0];

      final pair = await weather.fetchLookbackPair(lat, lon);
      if (pair == null) throw Exception('Failed to fetch lookback data');

      final ForecastResult result;
      switch (modelVariant) {
        case ModelVariant.linear:
          result = LinearDartEngine.instance.infer(
            gfsForecast: pair.gfsT0,
            obsFeatures: pair.obsT1hr,
            spatialEmbed: spatial,
          );
        case ModelVariant.lstm:
          ObservationBufferService.instance.push(lat, lon, [...pair.obsT1hr, ...spatial]);
          final sequence = ObservationBufferService.instance.getSequence(lat, lon);
          result = await LstmDartEngine.instance.infer(sequence: sequence);
        case ModelVariant.bma:
          result = await BmaEngine.instance.infer(
            gfsForecast: pair.gfsT0,
            obsFeatures: pair.obsT1hr,
            spatialEmbed: spatial,
          );
      }

      final truth = pair.era5Now;
      // era5Now indices: [0]=temp, [1]=pressure, [2]=u-wind, [3]=v-wind, [4]=precip, [5]=humidity
      final truthWindSpeed =
          math.sqrt(truth[2] * truth[2] + truth[3] * truth[3]);

      final tempAe = (result.temperatureC - truth[0]).abs();
      final pressureAe = (result.surfacePressureHpa - truth[1]).abs();
      final windAe = (result.windSpeedMs - truthWindSpeed).abs();
      final humidityAe = (result.relativeHumidityPct - truth[5]).abs();
      final precipAe = (result.precipitationMm - truth[4]).abs();

      // Coverage: check if all primary variables fall within 2σ of truth
      final tempCovered = (truth[0] - result.temperatureC).abs() <=
          result.temperatureStd * 2;
      final windCovered =
          (truthWindSpeed - result.windSpeedMs).abs() <= result.windSpeedStd * 2;
      final within2Sigma = tempCovered && windCovered;

      await DatabaseService.instance.insertLookbackAccuracy(
        lat: lat,
        lon: lon,
        modelVariant: modelVariant.name,
        tempAe: tempAe,
        pressureAe: pressureAe,
        windSpeedAe: windAe,
        humidityAe: humidityAe,
        precipAe: precipAe,
        within2Sigma: within2Sigma,
      );

      final summary = await DatabaseService.instance.getLookbackSummary();

      state = AsyncData(state.value!.copyWith(
        status: ValidationStatus.done,
        lastMetrics: {
          'temp_ae': tempAe,
          'pressure_ae': pressureAe,
          'wind_ae': windAe,
          'humidity_ae': humidityAe,
          'precip_ae': precipAe,
        },
        summary: summary,
      ));
    } catch (e) {
      state = AsyncData(state.value!.copyWith(
        status: ValidationStatus.error,
        errorMessage: e.toString(),
      ));
    }
  }

  Future<void> refreshSummary() async {
    final summary = await DatabaseService.instance.getLookbackSummary();
    state = AsyncData(state.value!.copyWith(summary: summary));
  }
}
