import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/services/prediction_history_dao.dart';
import 'history_provider.dart';

class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(historyProvider);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Prediction History'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.read(historyProvider.notifier).refresh(),
          ),
        ],
      ),
      body: state.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('Error: $e')),
        data: (rows) => rows.isEmpty
            ? const Center(child: Text('No predictions yet'))
            : ListView.builder(
                itemCount: rows.length,
                itemBuilder: (context, i) => _HistoryTile(row: rows[i]),
              ),
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  final PredictionHistoryRow row;
  const _HistoryTile({required this.row});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final time = row.timestamp.toLocal();
    final timeStr =
        '${time.month}/${time.day} '
        '${time.hour.toString().padLeft(2, '0')}:'
        '${time.minute.toString().padLeft(2, '0')}';

    return ExpansionTile(
      leading: _validationIcon(),
      title: Text(
        '${row.predTempC?.toStringAsFixed(1) ?? '?'}°C  •  $timeStr',
        style: theme.textTheme.bodyMedium,
      ),
      subtitle: Text(
        '${row.variant}  •  ${row.stationId ?? 'unknown'}',
        style: theme.textTheme.bodySmall?.copyWith(color: Colors.grey),
      ),
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
          child: Column(
            children: [
              _DetailRow('Temperature', '${row.predTempC?.toStringAsFixed(2)}°C',
                  obs: row.obsTempC, error: row.tempErrorC, unit: '°C'),
              _DetailRow('Wind Speed', '${row.predWindSpeedMs?.toStringAsFixed(2)} m/s',
                  obs: row.obsWindSpeedMs, error: row.windErrorMs, unit: 'm/s'),
              _DetailRow('Pressure', '${row.predPressureHpa?.toStringAsFixed(1)} hPa',
                  obs: row.obsPressureHpa, unit: 'hPa'),
              _DetailRow('Humidity', '${row.predHumidityPct?.toStringAsFixed(1)}%',
                  obs: row.obsHumidityPct, unit: '%'),
              _DetailRow('Precip', '${row.predPrecipMm?.toStringAsFixed(2)} mm',
                  obs: row.obsPrecipMm, unit: 'mm'),
              if (row.predTempStd != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    'Uncertainty: temp σ=${row.predTempStd?.toStringAsFixed(3)}, '
                    'wind σ=${row.predWindSpeedStd?.toStringAsFixed(3)}',
                    style: theme.textTheme.bodySmall?.copyWith(color: Colors.grey),
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _validationIcon() {
    if (!row.isValidated) {
      return const Icon(Icons.hourglass_empty, color: Colors.grey, size: 20);
    }
    final tempErr = row.tempErrorC ?? double.infinity;
    final color = tempErr < 1.0 ? Colors.green : (tempErr < 3.0 ? Colors.orange : Colors.red);
    return Icon(Icons.check_circle, color: color, size: 20);
  }
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String predicted;
  final double? obs;
  final double? error;
  final String unit;

  const _DetailRow(this.label, this.predicted, {this.obs, this.error, this.unit = ''});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(width: 90, child: Text(label, style: theme.textTheme.bodySmall)),
          Expanded(
            child: Text('Pred: $predicted', style: theme.textTheme.bodySmall),
          ),
          if (obs != null)
            Text(
              'Obs: ${obs!.toStringAsFixed(2)} $unit',
              style: theme.textTheme.bodySmall?.copyWith(color: Colors.grey),
            ),
          if (error != null) ...[
            const SizedBox(width: 8),
            Text(
              'Δ${error!.toStringAsFixed(2)}',
              style: theme.textTheme.bodySmall?.copyWith(
                color: error! < 1.0 ? Colors.green : Colors.orange,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
