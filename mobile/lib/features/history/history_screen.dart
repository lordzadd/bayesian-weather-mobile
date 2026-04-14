import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'history_provider.dart';

class HistoryScreen extends ConsumerStatefulWidget {
  const HistoryScreen({super.key});

  @override
  ConsumerState<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends ConsumerState<HistoryScreen> {
  bool _showChart = false;

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(historyProvider);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Forecast History'),
        actions: [
          state.maybeWhen(
            data: (rows) => rows.length >= 2
                ? IconButton(
                    icon: Icon(_showChart ? Icons.list : Icons.show_chart),
                    tooltip: _showChart ? 'Show list' : 'Show chart',
                    onPressed: () => setState(() => _showChart = !_showChart),
                  )
                : const SizedBox.shrink(),
            orElse: () => const SizedBox.shrink(),
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.invalidate(historyProvider),
          ),
        ],
      ),
      body: state.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('Error: $e')),
        data: (rows) {
          if (rows.isEmpty) {
            return const Center(
              child: Text('No history yet.\nRefresh the forecast to start recording.'),
            );
          }
          if (_showChart) {
            return _TemperatureChart(rows: rows);
          }
          return ListView.separated(
            itemCount: rows.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (context, i) => _HistoryTile(row: rows[i]),
          );
        },
      ),
    );
  }
}

// ── Temperature chart ─────────────────────────────────────────────────────────

class _TemperatureChart extends StatelessWidget {
  final List<Map<String, dynamic>> rows;
  const _TemperatureChart({required this.rows});

  @override
  Widget build(BuildContext context) {
    // Newest-first from DB; reverse for chronological left→right
    final ordered = rows.reversed.toList();
    final temps   = ordered.map((r) => (r['temp_mean'] as num).toDouble()).toList();
    final stds    = ordered.map((r) => (r['temp_std']  as num).toDouble()).toList();

    final minTemp = temps.map((t) => t).reduce(math.min);
    final maxTemp = temps.map((t) => t).reduce(math.max);
    final maxStd  = stds.reduce(math.max);
    final yMin    = minTemp - maxStd * 2 - 1;
    final yMax    = maxTemp + maxStd * 2 + 1;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Temperature °C  (±2σ band)',
              style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 4),
          Text('${ordered.length} readings  ·  newest right',
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Colors.grey)),
          const SizedBox(height: 12),
          Expanded(
            child: CustomPaint(
              painter: _ChartPainter(
                temps: temps,
                stds: stds,
                yMin: yMin,
                yMax: yMax,
              ),
              size: Size.infinite,
            ),
          ),
          const SizedBox(height: 8),
          _ChartLegend(),
        ],
      ),
    );
  }
}

class _ChartPainter extends CustomPainter {
  final List<double> temps;
  final List<double> stds;
  final double yMin;
  final double yMax;

  const _ChartPainter({
    required this.temps,
    required this.stds,
    required this.yMin,
    required this.yMax,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (temps.length < 2) return;

    const leftPad  = 40.0;
    const rightPad = 8.0;
    const topPad   = 8.0;
    const botPad   = 24.0;

    final chartW = size.width  - leftPad - rightPad;
    final chartH = size.height - topPad  - botPad;
    final n      = temps.length;

    double px(int i) => leftPad + (i / (n - 1)) * chartW;
    double py(double v) =>
        topPad + chartH * (1.0 - (v - yMin) / (yMax - yMin));

    // ── Grid lines + y-labels ──────────────────────────────────────────────
    final gridPaint = Paint()
      ..color  = Colors.white12
      ..strokeWidth = 1;
    final labelStyle = const TextStyle(color: Colors.grey, fontSize: 10);

    final yRange = yMax - yMin;
    final step   = _niceStep(yRange / 5);
    var yLine    = (yMin / step).ceil() * step;
    while (yLine <= yMax) {
      final y = py(yLine);
      canvas.drawLine(Offset(leftPad, y), Offset(size.width - rightPad, y), gridPaint);
      _drawText(canvas, '${yLine.toStringAsFixed(0)}°', Offset(0, y - 6), labelStyle);
      yLine += step;
    }

    // ── ±2σ band ───────────────────────────────────────────────────────────
    final bandPath = Path();
    // upper edge
    bandPath.moveTo(px(0), py(temps[0] + stds[0] * 2));
    for (int i = 1; i < n; i++) {
      bandPath.lineTo(px(i), py(temps[i] + stds[i] * 2));
    }
    // lower edge (reversed)
    for (int i = n - 1; i >= 0; i--) {
      bandPath.lineTo(px(i), py(temps[i] - stds[i] * 2));
    }
    bandPath.close();

    canvas.drawPath(
      bandPath,
      Paint()..color = Colors.blue.withValues(alpha: 0.15),
    );

    // ── Mean line ─────────────────────────────────────────────────────────
    final linePaint = Paint()
      ..color       = Colors.lightBlue.shade300
      ..strokeWidth = 2
      ..style       = PaintingStyle.stroke
      ..strokeCap   = StrokeCap.round
      ..strokeJoin  = StrokeJoin.round;

    final linePath = Path()..moveTo(px(0), py(temps[0]));
    for (int i = 1; i < n; i++) {
      linePath.lineTo(px(i), py(temps[i]));
    }
    canvas.drawPath(linePath, linePaint);

    // ── Dots at each reading ───────────────────────────────────────────────
    final dotPaint = Paint()..color = Colors.lightBlue.shade100;
    for (int i = 0; i < n; i++) {
      canvas.drawCircle(Offset(px(i), py(temps[i])), 3, dotPaint);
    }

    // ── x-axis baseline ───────────────────────────────────────────────────
    canvas.drawLine(
      Offset(leftPad, topPad + chartH),
      Offset(size.width - rightPad, topPad + chartH),
      gridPaint,
    );
  }

  void _drawText(Canvas canvas, String text, Offset offset, TextStyle style) {
    final tp = TextPainter(
      text: TextSpan(text: text, style: style),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, offset);
  }

  double _niceStep(double raw) {
    if (raw <= 0) return 1;
    final exp = math.pow(10, (math.log(raw) / math.ln10).floor());
    final frac = raw / exp;
    if (frac < 1.5) return 1 * exp.toDouble();
    if (frac < 3.5) return 2 * exp.toDouble();
    if (frac < 7.5) return 5 * exp.toDouble();
    return 10 * exp.toDouble();
  }

  @override
  bool shouldRepaint(_ChartPainter old) =>
      old.temps != temps || old.yMin != yMin || old.yMax != yMax;
}

class _ChartLegend extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(width: 20, height: 2, color: Colors.lightBlue.shade300),
        const SizedBox(width: 4),
        Text('Mean', style: Theme.of(context).textTheme.labelSmall),
        const SizedBox(width: 12),
        Container(
          width: 20,
          height: 10,
          color: Colors.blue.withValues(alpha: 0.2),
        ),
        const SizedBox(width: 4),
        Text('±2σ', style: Theme.of(context).textTheme.labelSmall),
      ],
    );
  }
}

// ── List view ─────────────────────────────────────────────────────────────────

class _HistoryTile extends StatelessWidget {
  final Map<String, dynamic> row;
  const _HistoryTile({required this.row});

  @override
  Widget build(BuildContext context) {
    final ts       = DateTime.fromMillisecondsSinceEpoch(row['timestamp'] as int);
    final tempMean = (row['temp_mean']   as num).toDouble();
    final tempStd  = (row['temp_std']    as num).toDouble();
    final windMs   = (row['wind_speed_ms'] as num).toDouble();
    final humidity = (row['humidity_pct']  as num).toDouble();
    final source   = row['inference_source'] as String;

    return ListTile(
      leading: _SourceDot(source: source),
      title: Text(
        '${tempMean.toStringAsFixed(1)}°C  ±${(tempStd * 2).toStringAsFixed(2)}°C',
        style: Theme.of(context).textTheme.bodyLarge,
      ),
      subtitle: Text(
        'Wind ${windMs.toStringAsFixed(1)} m/s · Humidity ${humidity.toStringAsFixed(0)}%',
        style: Theme.of(context).textTheme.bodySmall,
      ),
      trailing: Text(
        _formatTimestamp(ts),
        style: Theme.of(context).textTheme.labelSmall?.copyWith(color: Colors.grey),
      ),
      onTap: () => _showDetail(context, row),
    );
  }

  void _showDetail(BuildContext context, Map<String, dynamic> row) {
    final ts = DateTime.fromMillisecondsSinceEpoch(row['timestamp'] as int);
    showModalBottomSheet(
      context: context,
      builder: (_) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(_formatTimestamp(ts),
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 16),
            _DetailRow('Temperature',
                '${(row['temp_mean'] as num).toStringAsFixed(2)}°C  σ=${(row['temp_std'] as num).toStringAsFixed(3)}'),
            _DetailRow('Pressure',
                '${(row['pressure_hpa'] as num).toStringAsFixed(1)} hPa'),
            _DetailRow('Wind speed',
                '${(row['wind_speed_ms'] as num).toStringAsFixed(2)} m/s  σ=${(row['wind_speed_std'] as num).toStringAsFixed(3)}'),
            _DetailRow('Humidity',
                '${(row['humidity_pct'] as num).toStringAsFixed(1)}%'),
            _DetailRow('Precipitation',
                '${(row['precip_mm'] as num).toStringAsFixed(2)} mm'),
            _DetailRow('Source', row['inference_source'] as String),
            _DetailRow('Model', row['model_variant'] as String),
          ],
        ),
      ),
    );
  }

  String _formatTimestamp(DateTime t) {
    final date =
        '${t.year}-${t.month.toString().padLeft(2, '0')}-${t.day.toString().padLeft(2, '0')}';
    final time =
        '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';
    return '$date  $time';
  }
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String value;
  const _DetailRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Colors.grey)),
          Text(value, style: Theme.of(context).textTheme.bodyMedium),
        ],
      ),
    );
  }
}

class _SourceDot extends StatelessWidget {
  final String source;
  const _SourceDot({required this.source});

  @override
  Widget build(BuildContext context) {
    final color = switch (source) {
      'gpu'   => Colors.deepPurple,
      'dart'  => Colors.indigo,
      'cache' => Colors.teal,
      _       => Colors.grey,
    };
    return CircleAvatar(
      radius: 8,
      backgroundColor: color,
    );
  }
}
