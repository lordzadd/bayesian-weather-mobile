import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../forecast/forecast_provider.dart';
import 'heatmap_painter.dart';

class MapScreen extends ConsumerWidget {
  const MapScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final forecast = ref.watch(forecastProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Probabilistic Heatmap')),
      body: Stack(
        children: [
          FlutterMap(
            options: const MapOptions(
              initialCenter: LatLng(37.7749, -122.4194),
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
                  centerLat: 37.7749,
                  centerLon: -122.4194,
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
            const Text('Temperature (°C)', style: TextStyle(color: Colors.white, fontSize: 12)),
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
                Text('Hot', style: TextStyle(color: Colors.white70, fontSize: 10)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
