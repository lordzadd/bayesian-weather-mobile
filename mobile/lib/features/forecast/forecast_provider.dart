import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/forecast_result.dart';
import '../../core/services/database_service.dart';
import '../../core/services/location_service.dart';
import '../../core/services/observation_buffer_service.dart';
import '../../core/services/open_meteo_service.dart';
import '../../inference/bma_engine.dart';
import '../../inference/forecast_cache_service.dart';
import '../../inference/linear_dart_engine.dart';
import '../../inference/lstm_dart_engine.dart';
import '../settings/settings_provider.dart';

final forecastProvider =
    AsyncNotifierProvider<ForecastNotifier, ForecastResult>(
  ForecastNotifier.new,
);

class ForecastNotifier extends AsyncNotifier<ForecastResult> {
  late final OpenMeteoService _weather;
  late final LocationService _location;

  @override
  Future<ForecastResult> build() async {
    _weather = ref.read(openMeteoServiceProvider);
    _location = ref.read(locationServiceProvider);
    return _fetchForecast();
  }

  Future<void> refresh() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(_fetchForecast);
  }

  Future<ForecastResult> _fetchForecast() async {
    final settings = ref.read(settingsProvider);

    final pos = await _location.currentPosition();
    final (lat, lon) = (pos.lat, pos.lon);
    final spatial = [lat / 90.0, lon / 180.0];

    final data = await _weather.fetch(lat, lon);
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

      case ModelVariant.lstm:
        // Push current observation into the sequence buffer, then infer
        ObservationBufferService.instance.push(lat, lon, [...gfsForecast, ...spatial]);
        final sequence = ObservationBufferService.instance.getSequence(lat, lon);
        result = await LstmDartEngine.instance.infer(sequence: sequence);

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
            windSpeedThresholdMs: settings.windThresholdMs,
          );
          final snapshot = (obsFeatures ?? gfsForecast).toSnapshot();
          result = await cache.getForecast(
            lat: lat,
            lon: lon,
            gfsForecast: gfsForecast,
            obsFeatures: obsFeatures,
            spatialEmbed: spatial,
            newObservation: snapshot,
          );
        }
    }

    _saveHistory(lat, lon, result, settings.modelVariant.name);
    return result;
  }

  void _saveHistory(double lat, double lon, ForecastResult result, String modelVariant) {
    DatabaseService.instance.insertForecastHistory(
      lat: lat,
      lon: lon,
      result: result,
      modelVariant: modelVariant,
    );
  }
}
