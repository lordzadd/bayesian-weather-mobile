import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/forecast_result.dart';
import '../../core/services/connectivity_service.dart';
import '../../core/services/database_service.dart';
import '../../core/services/location_service.dart';
import '../../core/services/notification_service.dart';
import '../../core/services/observation_buffer_service.dart';
import '../../core/services/open_meteo_service.dart';
import '../../inference/bma_engine.dart';
import '../../inference/forecast_cache_service.dart';
import '../../inference/linear_dart_engine.dart';
import '../../inference/fusion_dart_engine.dart';
import '../../inference/lstm_dart_engine.dart';
import '../locations/saved_locations_provider.dart';
import '../settings/settings_provider.dart';

final forecastProvider =
    AsyncNotifierProvider<ForecastNotifier, ForecastResult>(
  ForecastNotifier.new,
);

class ForecastNotifier extends AsyncNotifier<ForecastResult> {
  late final OpenMeteoService _weather;
  late final LocationService  _location;

  @override
  Future<ForecastResult> build() async {
    _weather  = ref.read(openMeteoServiceProvider);
    _location = ref.read(locationServiceProvider);
    return _fetchForecast();
  }

  Future<void> refresh() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(_fetchForecast);
  }

  Future<ForecastResult> _fetchForecast() async {
    // ── Offline check ──────────────────────────────────────────────────────
    final connectivity = ref.read(connectivityProvider);
    final isOnline = connectivity.valueOrNull ?? true;

    if (!isOnline) {
      return _offlineFallback();
    }

    final settings = ref.read(settingsProvider);

    // ── Location (honour saved-location override) ──────────────────────────
    final override = ref.read(selectedLocationProvider);
    final LocationResult pos;
    if (override != null) {
      pos = LocationResult(lat: override.lat, lon: override.lon);
    } else {
      pos = await _location.currentPosition();
    }
    final (lat, lon) = (pos.lat, pos.lon);
    final spatial    = [lat / 90.0, lon / 180.0];

    final data        = await _weather.fetch(lat, lon);
    final gfsForecast = data?.gfs ?? [15.0, 1013.25, 0.0, 0.0, 0.0, 60.0];
    final obsFeatures = data?.obs;

    ForecastResult result;

    switch (settings.modelVariant) {
      case ModelVariant.linear:
        result = LinearDartEngine.instance.infer(
          gfsForecast: gfsForecast,
          obsFeatures: obsFeatures,
          spatialEmbed: spatial,
        );

      case ModelVariant.fusion:
        await FusionDartEngine.instance.load();
        // Build observation history from the buffer (raw obs values)
        ObservationBufferService.instance.push(lat, lon, [...(obsFeatures ?? gfsForecast), ...spatial]);
        final fusionSeq = ObservationBufferService.instance.getSequence(lat, lon);
        // Extract just the 6 weather vars from each buffered step (drop spatial)
        final obsHist = fusionSeq.map((step) => step.sublist(0, 6)).toList();
        result = FusionDartEngine.instance.infer(
          obsHistory: obsHist,
          gfsForecast: gfsForecast,
          spatialEmbed: spatial,
        );

      case ModelVariant.lstm:
        ObservationBufferService.instance.push(lat, lon, [...gfsForecast, ...spatial]);
        final sequence = ObservationBufferService.instance.getSequence(lat, lon);
        result = LstmDartEngine.instance.infer(sequence: sequence);

      case ModelVariant.bma:
        if (settings.variant == InferenceVariant.gpuAlways) {
          result = await BmaEngine.instance.infer(
            gfsForecast: gfsForecast,
            obsFeatures: obsFeatures,
            spatialEmbed: spatial,
          );
        } else {
          final cache = ForecastCacheService(
            temperatureThresholdC: settings.tempThresholdC,
            windSpeedThresholdMs:  settings.windThresholdMs,
          );
          final snapshot = (obsFeatures ?? gfsForecast).toSnapshot();
          result = await cache.getForecast(
            lat:          lat,
            lon:          lon,
            gfsForecast:  gfsForecast,
            obsFeatures:  obsFeatures,
            spatialEmbed: spatial,
            newObservation: snapshot,
          );
        }
    }

    _saveHistory(lat, lon, result, settings.modelVariant.name);
    _maybeNotify(result, settings);
    return result;
  }

  /// Returns the most-recent history row as a ForecastResult when offline.
  Future<ForecastResult> _offlineFallback() async {
    final rows = await DatabaseService.instance.getRecentHistory(limit: 1);
    if (rows.isEmpty) {
      throw Exception('No network connection and no cached forecast available.');
    }
    final r = rows.first;
    return ForecastResult(
      temperatureC:        (r['temp_mean']       as num).toDouble(),
      temperatureStd:      (r['temp_std']         as num).toDouble(),
      surfacePressureHpa:  (r['pressure_hpa']     as num).toDouble(),
      windSpeedMs:         (r['wind_speed_ms']    as num).toDouble(),
      windSpeedStd:        (r['wind_speed_std']   as num).toDouble(),
      precipitationMm:     (r['precip_mm']        as num).toDouble(),
      relativeHumidityPct: (r['humidity_pct']     as num).toDouble(),
      computedAt:          DateTime.fromMillisecondsSinceEpoch(r['timestamp'] as int),
      source:              InferenceSource.cache,
    );
  }

  void _saveHistory(double lat, double lon, ForecastResult result, String modelVariant) {
    DatabaseService.instance.insertForecastHistory(
      lat:          lat,
      lon:          lon,
      result:       result,
      modelVariant: modelVariant,
    );
  }

  void _maybeNotify(ForecastResult result, AppSettings settings) {
    if (!settings.notificationsEnabled) return;
    // Compare against the previous in-memory state if available
    final prev = state.valueOrNull;
    if (prev == null) return;
    NotificationService.instance.checkAndAlert(
      prevTempC:   prev.temperatureC,
      newTempC:    result.temperatureC,
      thresholdC:  settings.alertTempChangeC,
    );
  }
}
