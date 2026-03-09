import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

/// A [FlutterMap] layer that renders a Gaussian probability heatmap
/// centered on the inferred posterior location.
///
/// Color encodes the posterior mean temperature; opacity encodes 1/σ
/// (higher certainty → more opaque).
///
/// In flutter_map 6.x, custom layers are plain widgets added to the
/// [FlutterMap.children] list — no special layer base class needed.
class HeatmapLayer extends StatelessWidget {
  final double centerLat;
  final double centerLon;
  final double posteriorMean;
  final double posteriorStd;

  /// Approximate radius of the heatmap blob in degrees lat/lon.
  final double radiusDeg;

  const HeatmapLayer({
    super.key,
    required this.centerLat,
    required this.centerLon,
    required this.posteriorMean,
    required this.posteriorStd,
    this.radiusDeg = 0.15,
  });

  @override
  Widget build(BuildContext context) {
    // In flutter_map 6.x, use MapCamera to translate geo coords to screen coords.
    final camera = MapCamera.of(context);
    final center = camera.latLngToScreenPoint(LatLng(centerLat, centerLon));
    // Scale radius: 0.15° ≈ ~16 km; convert to pixels at current zoom
    final radiusPx = _degToPixels(radiusDeg, camera.zoom);

    return CustomPaint(
      painter: _HeatmapPainter(
        center: Offset(center.x, center.y),
        radiusPx: radiusPx,
        posteriorMean: posteriorMean,
        posteriorStd: posteriorStd,
      ),
      child: const SizedBox.expand(),
    );
  }

  /// Converts a degree delta to approximate pixels at the given zoom level.
  double _degToPixels(double deg, double zoom) {
    // At zoom 0, 360° = 256px. Each zoom level doubles.
    final tilesAtZoom = math.pow(2, zoom).toDouble();
    return deg / 360.0 * 256.0 * tilesAtZoom;
  }
}

class _HeatmapPainter extends CustomPainter {
  final Offset center;
  final double radiusPx;
  final double posteriorMean;
  final double posteriorStd;

  _HeatmapPainter({
    required this.center,
    required this.radiusPx,
    required this.posteriorMean,
    required this.posteriorStd,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Normalize temperature to [0, 1] for color mapping (-20°C to 45°C)
    final normalized = ((posteriorMean + 20) / 65).clamp(0.0, 1.0);
    final color = _tempToColor(normalized);

    // Uncertainty-based opacity: low std → high opacity
    final opacity = (1.0 - (posteriorStd / 5.0).clamp(0.0, 0.85)) * 0.65;

    final paint = Paint()
      ..shader = ui.Gradient.radial(
        center,
        radiusPx,
        [
          color.withValues(alpha: opacity),
          color.withValues(alpha: opacity * 0.4),
          Colors.transparent,
        ],
        [0.0, 0.5, 1.0],
      )
      ..blendMode = BlendMode.srcOver;

    canvas.drawCircle(center, radiusPx, paint);
  }

  Color _tempToColor(double t) {
    // Cold (blue) → Neutral (yellow) → Hot (red)
    if (t < 0.5) {
      return Color.lerp(const Color(0xFF2196F3), const Color(0xFFFFEB3B), t * 2)!;
    } else {
      return Color.lerp(const Color(0xFFFFEB3B), const Color(0xFFF44336), (t - 0.5) * 2)!;
    }
  }

  @override
  bool shouldRepaint(_HeatmapPainter old) =>
      old.posteriorMean != posteriorMean ||
      old.posteriorStd != posteriorStd ||
      old.center != center ||
      old.radiusPx != radiusPx;
}
