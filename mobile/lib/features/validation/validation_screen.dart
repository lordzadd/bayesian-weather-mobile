import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'validation_provider.dart';

class ValidationScreen extends ConsumerWidget {
  const ValidationScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(validationProvider);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Lookback Validation'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.read(validationProvider.notifier).refreshSummary(),
          ),
        ],
      ),
      body: state.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('Error: $e')),
        data: (vs) => ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _RunButton(vs: vs),
            if (vs.status == ValidationStatus.error && vs.errorMessage != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  vs.errorMessage!,
                  style: const TextStyle(color: Colors.redAccent),
                  textAlign: TextAlign.center,
                ),
              ),
            if (vs.lastMetrics != null) ...[
              const SizedBox(height: 16),
              _LastRunCard(metrics: vs.lastMetrics!),
            ],
            const SizedBox(height: 16),
            _ExplainerCard(),
            const SizedBox(height: 16),
            _SummaryTable(summary: vs.summary),
          ],
        ),
      ),
    );
  }
}

class _RunButton extends ConsumerWidget {
  final ValidationState vs;
  const _RunButton({required this.vs});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final running = vs.status == ValidationStatus.running;
    return FilledButton.icon(
      onPressed: running
          ? null
          : () => ref.read(validationProvider.notifier).runLookback(),
      icon: running
          ? const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
            )
          : const Icon(Icons.science_outlined),
      label: Text(running ? 'Running…' : 'Run 1-Hour Lookback'),
    );
  }
}

class _ExplainerCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('How it works',
                style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            Text(
              'Fetches the GFS forecast from T-0 and the ERA5 observation '
              'from T-1hr, runs the model as if it were an hour ago, then '
              'compares the prediction against today\'s actual ERA5 values.\n\n'
              'AE = absolute error  ·  Coverage = truth within predicted ±2σ',
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }
}

class _LastRunCard extends StatelessWidget {
  final Map<String, double> metrics;
  const _LastRunCard({required this.metrics});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Theme.of(context).colorScheme.primaryContainer.withOpacity(0.3),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Last run',
                style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            _MetricRow('Temperature AE',
                '${metrics['temp_ae']!.toStringAsFixed(2)} °C'),
            _MetricRow('Pressure AE',
                '${metrics['pressure_ae']!.toStringAsFixed(2)} hPa'),
            _MetricRow('Wind Speed AE',
                '${metrics['wind_ae']!.toStringAsFixed(2)} m/s'),
            _MetricRow('Humidity AE',
                '${metrics['humidity_ae']!.toStringAsFixed(1)} %'),
            _MetricRow('Precipitation AE',
                '${metrics['precip_ae']!.toStringAsFixed(3)} mm'),
          ],
        ),
      ),
    );
  }
}

class _MetricRow extends StatelessWidget {
  final String label;
  final String value;
  const _MetricRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: Theme.of(context).textTheme.bodySmall),
          Text(value, style: Theme.of(context).textTheme.bodyMedium),
        ],
      ),
    );
  }
}

class _SummaryTable extends StatelessWidget {
  final List<Map<String, dynamic>> summary;
  const _SummaryTable({required this.summary});

  @override
  Widget build(BuildContext context) {
    if (summary.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(32),
          child: Text(
            'No validation runs yet.\nTap "Run 1-Hour Lookback" to start.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Running averages (last 20 runs)',
                style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 12),
            ...summary.map((row) => _ModelSummaryRow(row: row)),
          ],
        ),
      ),
    );
  }
}

class _ModelSummaryRow extends StatelessWidget {
  final Map<String, dynamic> row;
  const _ModelSummaryRow({required this.row});

  @override
  Widget build(BuildContext context) {
    final model = row['model_variant'] as String;
    final runs = row['run_count'] as int;
    final coverage = ((row['coverage_rate'] as num).toDouble() * 100);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(model.toUpperCase(),
                style: Theme.of(context)
                    .textTheme
                    .labelMedium
                    ?.copyWith(fontWeight: FontWeight.bold)),
            const Spacer(),
            Text('$runs runs · ${coverage.toStringAsFixed(0)}% 2σ coverage',
                style: Theme.of(context)
                    .textTheme
                    .labelSmall
                    ?.copyWith(color: Colors.grey)),
          ],
        ),
        const SizedBox(height: 4),
        _MetricRow('Temp MAE',
            '${(row['avg_temp_ae'] as num).toStringAsFixed(2)} °C'),
        _MetricRow('Pressure MAE',
            '${(row['avg_pressure_ae'] as num).toStringAsFixed(2)} hPa'),
        _MetricRow('Wind MAE',
            '${(row['avg_wind_ae'] as num).toStringAsFixed(2)} m/s'),
        _MetricRow('Humidity MAE',
            '${(row['avg_humidity_ae'] as num).toStringAsFixed(1)} %'),
        const Divider(height: 16),
      ],
    );
  }
}
