import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../../core/services/location_service.dart';
import '../forecast/forecast_provider.dart';
import 'heatmap_painter.dart';

final _mapLocationProvider =
    FutureProvider<({double lat, double lon})>((ref) async {
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
    final location = ref.watch(_mapLocationProvider);

    // Show spinner while waiting for GPS
    if (!location.hasValue) {
      return Scaffold(
        appBar: AppBar(title: const Text('Probabilistic Heatmap')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    final lat = location.value!.lat;
    final lon = location.value!.lon;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Probabilistic Heatmap'),
        actions: [
          IconButton(
            icon: const Icon(Icons.my_location),
            tooltip: 'Re-centre',
            onPressed: () => _mapController.move(LatLng(lat, lon), 9),
          ),
        ],
      ),
      body: Stack(
        children: [
          // FlutterMap is only built after location resolves, so
          // initialCenter is always the real GPS position.
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: LatLng(lat, lon),
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
                  centerLat: lat,
                  centerLon: lon,
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
                    colors: [
                      Color(0xFF2196F3),
                      Color(0xFFFFEB3B),
                      Color(0xFFF44336)
                    ],
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
