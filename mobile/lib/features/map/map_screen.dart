import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

import '../../core/services/database_service.dart';
import '../../core/services/location_service.dart';
import '../forecast/forecast_provider.dart';
import '../locations/saved_locations_provider.dart';
import 'heatmap_painter.dart';

final _mapLocationProvider = FutureProvider<LocationResult>((ref) async {
  // 1. Chip-bar override (user tapped a saved location)
  final override = ref.watch(selectedLocationProvider);
  if (override != null) {
    return LocationResult(lat: override.lat, lon: override.lon);
  }
  // 2. First saved location as default (avoids GPS / simulator SF fallback)
  final savedRows = await DatabaseService.instance.getSavedLocations();
  if (savedRows.isNotEmpty) {
    final first = savedRows.first;
    return LocationResult(
      lat: (first['lat'] as num).toDouble(),
      lon: (first['lon'] as num).toDouble(),
    );
  }
  // 3. GPS fallback
  return ref.read(locationServiceProvider).currentPosition();
});

class MapScreen extends ConsumerStatefulWidget {
  const MapScreen({super.key});

  @override
  ConsumerState<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends ConsumerState<MapScreen> {
  final _mapController = MapController();

  @override
  Widget build(BuildContext context) {
    final forecast = ref.watch(forecastProvider);
    final locationAsync = ref.watch(_mapLocationProvider);

    if (!locationAsync.hasValue) {
      return Scaffold(
        appBar: AppBar(title: const Text('Probabilistic Heatmap')),
        body: const Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 12),
              Text('Waiting for GPS…'),
            ],
          ),
        ),
      );
    }

    final loc = locationAsync.value!;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Probabilistic Heatmap'),
        actions: [
          if (!loc.isFallback)
            Padding(
              padding: const EdgeInsets.only(right: 4),
              child: Chip(
                label: Text(
                  '${loc.lat.toStringAsFixed(3)}, ${loc.lon.toStringAsFixed(3)}',
                  style: const TextStyle(fontSize: 10),
                ),
                backgroundColor: Colors.green.shade900,
                side: BorderSide.none,
                visualDensity: VisualDensity.compact,
              ),
            ),
          IconButton(
            icon: const Icon(Icons.my_location),
            tooltip: 'Re-centre',
            onPressed: () => _mapController.move(LatLng(loc.lat, loc.lon), 9),
          ),
        ],
      ),
      body: Column(
        children: [
          if (loc.isFallback) _LocationWarningBanner(loc: loc, onRetry: () => ref.invalidate(_mapLocationProvider)),
          Expanded(
            child: Stack(
              children: [
                FlutterMap(
                  mapController: _mapController,
                  options: MapOptions(
                    initialCenter: LatLng(loc.lat, loc.lon),
                    initialZoom: 9,
                    minZoom: 5,
                    maxZoom: 15,
                  ),
                  children: [
                    TileLayer(
                      urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                      userAgentPackageName: 'com.example.bayesian_weather',
                    ),
                    forecast.when(
                      data: (result) => HeatmapLayer(
                        centerLat: loc.lat,
                        centerLon: loc.lon,
                        posteriorMean: result.temperatureC,
                        posteriorStd: result.temperatureStd,
                      ),
                      loading: () => const SizedBox.shrink(),
                      error: (_, __) => const SizedBox.shrink(),
                    ),
                  ],
                ),
                Positioned(
                  bottom: 16,
                  left: 16,
                  child: _Legend(),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _LocationWarningBanner extends StatelessWidget {
  final LocationResult loc;
  final VoidCallback onRetry;
  const _LocationWarningBanner({required this.loc, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.orange.shade900,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(
        children: [
          const Icon(Icons.location_off, size: 16, color: Colors.white),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              loc.failMessage,
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
          ),
          if (loc.canOpenSettings)
            TextButton(
              onPressed: () => Geolocator.openAppSettings(),
              child: const Text('Settings', style: TextStyle(color: Colors.white, fontSize: 12)),
            )
          else
            TextButton(
              onPressed: onRetry,
              child: const Text('Retry', style: TextStyle(color: Colors.white, fontSize: 12)),
            ),
        ],
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.black.withValues(alpha: 0.75),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Temperature (°C)',
                style: TextStyle(color: Colors.white, fontSize: 12)),
            const SizedBox(height: 6),
            SizedBox(
              width: 120,
              height: 12,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(4),
                  gradient: const LinearGradient(
                    colors: [Color(0xFF2196F3), Color(0xFFFFEB3B), Color(0xFFF44336)],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 4),
            const Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Cold', style: TextStyle(color: Colors.white70, fontSize: 10)),
                Text('Hot',  style: TextStyle(color: Colors.white70, fontSize: 10)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
