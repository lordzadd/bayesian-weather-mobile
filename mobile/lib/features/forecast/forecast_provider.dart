import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/forecast_result.dart';
import '../../core/services/location_service.dart';
import '../../core/services/open_meteo_service.dart';
import '../../inference/bma_engine.dart';
import '../../inference/forecast_cache_service.dart';
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
    // Read live settings so variant/threshold changes take effect immediately
    final settings = ref.read(settingsProvider);

    // 1. Real device GPS (falls back to San Francisco if denied)
    final (:lat, :lon) = await _location.currentPosition();
    final spatial = [lat / 90.0, lon / 180.0];

    // 2. Open-Meteo: same API + variables as training data.
    //    gfs = next-hour GFS seamless forecast  → prior
    //    obs = current ERA5 analysis conditions  → Bayesian evidence
    final data = await _weather.fetch(lat, lon);
    final gfsForecast = data?.gfs ?? [15.0, 1013.25, 0.0, 0.0, 0.0, 60.0];
    final obsFeatures = data?.obs;

    // 3. Variant A: always run inference, no caching
    if (settings.variant == InferenceVariant.gpuAlways) {
      return BmaEngine.instance.infer(
        gfsForecast: gfsForecast,
        obsFeatures: obsFeatures,
        spatialEmbed: spatial,
      );
    }

    // 4. Variant B: cache-gated inference using live threshold settings
    final cache = ForecastCacheService(
      temperatureThresholdC: settings.tempThresholdC,
      windSpeedThresholdMs: settings.windThresholdMs,
    );
    final snapshot = (obsFeatures ?? gfsForecast).toSnapshot();
    return cache.getForecast(
      lat: lat,
      lon: lon,
      gfsForecast: gfsForecast,
      obsFeatures: obsFeatures,
      spatialEmbed: spatial,
      newObservation: snapshot,
    );
  }
}
