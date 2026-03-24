import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/services/lookback_service.dart';
import '../lookback_provider.dart';

/// Displays the latest lookback validation result and rolling accuracy stats.
class AccuracyCard extends ConsumerWidget {
  const AccuracyCard({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final lookback = ref.watch(lookbackProvider);
    final theme = Theme.of(context);

    if (lookback.isLoading && lookback.lastResult == null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              const SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              const SizedBox(width: 12),
              Text('Running lookback validation...',
                  style: theme.textTheme.bodySmall),
            ],
          ),
        ),
      );
    }

    final result = lookback.lastResult;
    if (result == null) return const SizedBox.shrink();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.fact_check, size: 18, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text('Lookback Accuracy', style: theme.textTheme.titleSmall),
                const Spacer(),
                if (lookback.isLoading)
                  const SizedBox(
                    width: 12,
                    height: 12,
                    child: CircularProgressIndicator(strokeWidth: 1.5),
                  ),
              ],
            ),
            const SizedBox(height: 12),

            // Last lookback result
            _ErrorRow(
              label: 'Temp',
              predicted: '${result.prediction.temperatureC.toStringAsFixed(1)}°C',
              actual: '${result.obsTempC.toStringAsFixed(1)}°C',
              error: result.bmaTempErrorC,
              unit: '°C',
            ),
            const SizedBox(height: 4),
            _ErrorRow(
              label: 'Wind',
              predicted: '${result.prediction.windSpeedMs.toStringAsFixed(1)} m/s',
              actual: '${result.obsWindSpeedMs.toStringAsFixed(1)} m/s',
              error: result.bmaWindErrorMs,
              unit: 'm/s',
            ),

            // GFS baseline comparison
            const SizedBox(height: 12),
            _BaselineComparison(
              label: 'Temp',
              bmaError: result.bmaTempErrorC,
              gfsError: result.gfsTempErrorC,
              unit: '°C',
              improvementPct: result.tempImprovementPct,
            ),
            const SizedBox(height: 4),
            _BaselineComparison(
              label: 'Wind',
              bmaError: result.bmaWindErrorMs,
              gfsError: result.gfsWindErrorMs,
              unit: 'm/s',
              improvementPct: result.windImprovementPct,
            ),

            // Rolling stats
            if (lookback.rollingStats != null) ...[
              const Divider(height: 24),
              _RollingStatsRow(stats: lookback.rollingStats!),
            ],
          ],
        ),
      ),
    );
  }
}

class _ErrorRow extends StatelessWidget {
  final String label;
  final String predicted;
  final String actual;
  final double error;
  final String unit;

  const _ErrorRow({
    required this.label,
    required this.predicted,
    required this.actual,
    required this.error,
    required this.unit,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final errorColor = error < 1.0 ? Colors.green : (error < 3.0 ? Colors.orange : Colors.red);

    return Row(
      children: [
        SizedBox(width: 42, child: Text(label, style: theme.textTheme.bodySmall)),
        Expanded(
          child: Text(
            'Pred: $predicted  Actual: $actual',
            style: theme.textTheme.bodySmall?.copyWith(color: Colors.grey),
          ),
        ),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
          decoration: BoxDecoration(
            color: errorColor.withValues(alpha: 0.2),
            borderRadius: BorderRadius.circular(4),
          ),
          child: Text(
            '${error.toStringAsFixed(2)} $unit',
            style: theme.textTheme.bodySmall?.copyWith(
              color: errorColor,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ],
    );
  }
}

class _BaselineComparison extends StatelessWidget {
  final String label;
  final double bmaError;
  final double gfsError;
  final String unit;
  final double improvementPct;

  const _BaselineComparison({
    required this.label,
    required this.bmaError,
    required this.gfsError,
    required this.unit,
    required this.improvementPct,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isImproved = improvementPct > 0;

    return Row(
      children: [
        SizedBox(width: 42, child: Text(label, style: theme.textTheme.bodySmall)),
        Text(
          'BMA: ${bmaError.toStringAsFixed(2)} $unit  |  GFS: ${gfsError.toStringAsFixed(2)} $unit',
          style: theme.textTheme.bodySmall?.copyWith(color: Colors.grey),
        ),
        const Spacer(),
        Text(
          '${isImproved ? '+' : ''}${improvementPct.toStringAsFixed(0)}%',
          style: theme.textTheme.bodySmall?.copyWith(
            color: isImproved ? Colors.green : Colors.red,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}

class _RollingStatsRow extends StatelessWidget {
  final AccuracyStats stats;
  const _RollingStatsRow({required this.stats});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hours = stats.window.inHours;
    return Row(
      children: [
        const Icon(Icons.timeline, size: 14, color: Colors.grey),
        const SizedBox(width: 6),
        Text(
          'Last ${hours}h avg: '
          'temp ${stats.meanTempErrorC.toStringAsFixed(2)}°C, '
          'wind ${stats.meanWindErrorMs.toStringAsFixed(2)} m/s '
          '(${stats.sampleCount} samples)',
          style: theme.textTheme.bodySmall?.copyWith(color: Colors.grey),
        ),
      ],
    );
  }
}
