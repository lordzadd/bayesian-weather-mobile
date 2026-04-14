import 'dart:math' as math;
import 'package:flutter/material.dart';

/// Animated compass arrow showing wind direction and speed.
///
/// The arrow points in the direction the wind is blowing TO.
/// [bearingDeg] is the meteorological bearing (direction wind comes FROM,
/// 0 = North, 90 = East). The arrow rotates 180° from that to show
/// the travel direction.
class WindArrow extends StatefulWidget {
  final double bearingDeg;
  final double speedMs;
  final double speedStd;

  const WindArrow({
    super.key,
    required this.bearingDeg,
    required this.speedMs,
    required this.speedStd,
  });

  @override
  State<WindArrow> createState() => _WindArrowState();
}

class _WindArrowState extends State<WindArrow>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _rotation;
  double _prevAngle = 0.0;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _prevAngle = _toRadians(widget.bearingDeg);
    _rotation = Tween<double>(begin: _prevAngle, end: _prevAngle)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOut));
  }

  @override
  void didUpdateWidget(WindArrow old) {
    super.didUpdateWidget(old);
    if (old.bearingDeg != widget.bearingDeg) {
      final target = _nearestAngle(_prevAngle, _toRadians(widget.bearingDeg));
      _rotation = Tween<double>(begin: _prevAngle, end: target)
          .animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOut));
      _prevAngle = target;
      _ctrl
        ..reset()
        ..forward();
    }
  }

  // Finds the closest equivalent angle to avoid spinning the long way around.
  static double _nearestAngle(double from, double to) {
    var diff = (to - from) % (2 * math.pi);
    if (diff > math.pi) diff -= 2 * math.pi;
    if (diff < -math.pi) diff += 2 * math.pi;
    return from + diff;
  }

  // Meteorological bearing → "blowing to" direction in radians for Transform.rotate
  // Icons.navigation points up (North). Rotating by `bearing` points it to
  // the wind-from direction; +π flips to wind-to direction.
  static double _toRadians(double bearingDeg) =>
      (bearingDeg + 180.0) * math.pi / 180.0;

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = _speedColor(widget.speedMs);

    return AnimatedBuilder(
      animation: _rotation,
      builder: (_, __) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Transform.rotate(
            angle: _rotation.value,
            child: Icon(Icons.navigation, size: 32, color: color),
          ),
          const SizedBox(height: 4),
          Text(
            '${widget.speedMs.toStringAsFixed(1)} ± ${widget.speedStd.toStringAsFixed(1)} m/s',
            style: theme.textTheme.bodySmall,
            textAlign: TextAlign.center,
          ),
          Text(
            '${widget.bearingDeg.toStringAsFixed(0)}° ${_compass(widget.bearingDeg)}',
            style: theme.textTheme.labelSmall?.copyWith(color: Colors.grey),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  static Color _speedColor(double ms) {
    if (ms < 2.0) return Colors.teal;
    if (ms < 7.0) return Colors.blue;
    if (ms < 15.0) return Colors.orange;
    return Colors.red;
  }

  static String _compass(double deg) {
    const pts = [
      'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',
    ];
    return pts[((deg + 11.25) / 22.5).floor() % 16];
  }
}
