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
    return MobileLayerTransformer(
      child: CustomPaint(
        painter: _HeatmapPainter(
          centerLat: centerLat,
          centerLon: centerLon,
          posteriorMean: posteriorMean,
          posteriorStd: posteriorStd,
          radiusDeg: radiusDeg,
        ),
        child: const SizedBox.expand(),
      ),
    );
  }
}

class _HeatmapPainter extends CustomPainter {
  final double centerLat;
  final double centerLon;
  final double posteriorMean;
  final double posteriorStd;
  final double radiusDeg;

  _HeatmapPainter({
    required this.centerLat,
    required this.centerLon,
    required this.posteriorMean,
    required this.posteriorStd,
    required this.radiusDeg,
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
        Offset(size.width / 2, size.height / 2),
        size.width * 0.3,
        [
          color.withOpacity(opacity),
          color.withOpacity(opacity * 0.4),
          Colors.transparent,
        ],
        [0.0, 0.5, 1.0],
      )
      ..blendMode = BlendMode.srcOver;

    canvas.drawCircle(
      Offset(size.width / 2, size.height / 2),
      size.width * 0.3,
      paint,
    );
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
      old.posteriorMean != posteriorMean || old.posteriorStd != posteriorStd;
}
