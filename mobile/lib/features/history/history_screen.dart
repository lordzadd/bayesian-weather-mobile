import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'history_provider.dart';

class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(historyProvider);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Forecast History'),
        actions: [
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

class _HistoryTile extends StatelessWidget {
  final Map<String, dynamic> row;
  const _HistoryTile({required this.row});

  @override
  Widget build(BuildContext context) {
    final ts = DateTime.fromMillisecondsSinceEpoch(row['timestamp'] as int);
    final tempMean = (row['temp_mean'] as num).toDouble();
    final tempStd = (row['temp_std'] as num).toDouble();
    final windMs = (row['wind_speed_ms'] as num).toDouble();
    final humidity = (row['humidity_pct'] as num).toDouble();
    final source = row['inference_source'] as String;

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
      'gpu' => Colors.deepPurple,
      'dart' => Colors.indigo,
      'cache' => Colors.teal,
      _ => Colors.grey,
    };
    return CircleAvatar(
      radius: 8,
      backgroundColor: color,
    );
  }
}
