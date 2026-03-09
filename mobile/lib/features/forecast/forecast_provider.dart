import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/forecast_result.dart';
import '../../core/services/location_service.dart';
import '../../core/services/noaa_service.dart';
import '../../inference/forecast_cache_service.dart';

final forecastProvider =
    AsyncNotifierProvider<ForecastNotifier, ForecastResult>(
  ForecastNotifier.new,
);

class ForecastNotifier extends AsyncNotifier<ForecastResult> {
  late final NoaaService _noaa;
  late final LocationService _location;
  late final ForecastCacheService _cache;

  @override
  Future<ForecastResult> build() async {
    _noaa = ref.read(noaaServiceProvider);
    _location = ref.read(locationServiceProvider);
    _cache = ForecastCacheService();
    return _fetchForecast();
  }

  Future<void> refresh() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(_fetchForecast);
  }

  Future<ForecastResult> _fetchForecast() async {
    // 1. Real device location (falls back to San Francisco if denied)
    final (:lat, :lon) = await _location.currentPosition();
    final stationId = nearestStation(lat, lon);
    final spatial = [lat / 90.0, lon / 180.0];

    // 2. GFS grid forecast — NOAA first, Open-Meteo fallback
    List<double>? gfsForecast;
    final gridPoint = await _noaa.resolveGridPoint(lat, lon);
    if (gridPoint != null) {
      gfsForecast = await _noaa.fetchGridForecast(
        gridPoint.office,
        gridPoint.gridX,
        gridPoint.gridY,
      );
    }
    gfsForecast ??= await _noaa.fetchOpenMeteoForecast(lat, lon);
    // Last resort: neutral defaults so the Bayesian update still runs
    gfsForecast ??= [15.0, 1013.25, 0.0, 0.0, 0.0, 60.0];

    // 3. METAR observation — the Bayesian evidence D in P(θ|D)
    final obs = await _noaa.fetchMetarObservation(stationId);
    final obsFeatures = obs?.toFeatureVector();
    // obsFeatures is null when station unreachable; engine uses prior-only mode

    // 4. Cache-gated Bayesian inference (Variant B by default)
    //    Pass obsFeatures so the posterior updates against real station data.
    final snapshot = (obsFeatures ?? gfsForecast).toSnapshot();
    return _cache.getForecast(
      lat: lat,
      lon: lon,
      gfsForecast: gfsForecast,
      obsFeatures: obsFeatures,
      spatialEmbed: spatial,
      newObservation: snapshot,
    );
  }
}
