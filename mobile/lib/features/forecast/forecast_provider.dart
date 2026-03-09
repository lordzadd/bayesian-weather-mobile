import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/forecast_result.dart';
import '../../core/services/noaa_service.dart';
import '../../inference/bma_engine.dart';
import '../../inference/forecast_cache_service.dart';
// ignore: unused_import — provides toSnapshot() extension on List<double>
export '../../inference/forecast_cache_service.dart' show ObservationSnapshotX;

// Current device location — replace with geolocator package in production
const _defaultLat = 37.7749; // San Francisco
const _defaultLon = -122.4194;
const _defaultStation = 'KSFO';

final forecastProvider =
    AsyncNotifierProvider<ForecastNotifier, ForecastResult>(
  ForecastNotifier.new,
);

class ForecastNotifier extends AsyncNotifier<ForecastResult> {
  late final NoaaService _noaa;
  late final ForecastCacheService _cache;

  @override
  Future<ForecastResult> build() async {
    _noaa = ref.read(noaaServiceProvider);
    _cache = ForecastCacheService();
    return _fetchForecast();
  }

  Future<void> refresh() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(_fetchForecast);
  }

  Future<ForecastResult> _fetchForecast() async {
    // 1. Resolve GFS grid point from location
    final gridPoint = await _noaa.resolveGridPoint(_defaultLat, _defaultLon);

    List<double>? gfsForecast;
    if (gridPoint != null) {
      gfsForecast = await _noaa.fetchGridForecast(
        gridPoint.office,
        gridPoint.gridX,
        gridPoint.gridY,
      );
    }

    // Fallback to Open-Meteo
    gfsForecast ??= await _noaa.fetchOpenMeteoForecast(_defaultLat, _defaultLon);
    gfsForecast ??= List.filled(6, 0.0); // last resort defaults

    // 2. Fetch latest METAR observation
    final obs = await _noaa.fetchMetarObservation(_defaultStation);
    final obsFeatures = obs?.toFeatureVector() ?? gfsForecast;

    // 3. Normalized spatial embedding
    final spatial = [_defaultLat / 90.0, _defaultLon / 180.0];

    // 4. Cache-gated inference (Variant B)
    return _cache.getForecast(
      lat: _defaultLat,
      lon: _defaultLon,
      gfsForecast: gfsForecast,
      spatialEmbed: spatial,
      newObservation: obsFeatures.toSnapshot(),
    );
  }
}
