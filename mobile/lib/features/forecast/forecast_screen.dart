import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/models/forecast_result.dart';
import 'forecast_provider.dart';

class ForecastScreen extends ConsumerWidget {
  const ForecastScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(forecastProvider);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Bayesian Forecast'),
        actions: [
          state.when(
            data: (r) => _SourceBadge(source: r.source),
            loading: () => const SizedBox.shrink(),
            error: (_, __) => const SizedBox.shrink(),
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.read(forecastProvider.notifier).refresh(),
          ),
        ],
      ),
      body: state.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => _ErrorView(error: e.toString()),
        data: (result) => _ForecastBody(result: result),
      ),
    );
  }
}

class _ForecastBody extends StatelessWidget {
  final ForecastResult result;
  const _ForecastBody({required this.result});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _PrimaryCard(result: result),
        const SizedBox(height: 12),
        _DetailGrid(result: result),
        const SizedBox(height: 12),
        _UncertaintyCard(result: result),
        const SizedBox(height: 12),
        Text(
          'Updated ${_formatTime(result.computedAt)}',
          style: theme.textTheme.bodySmall?.copyWith(color: Colors.grey),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  String _formatTime(DateTime t) =>
      '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';
}

class _PrimaryCard extends StatelessWidget {
  final ForecastResult result;
  const _PrimaryCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Text(
              '${result.temperatureC.toStringAsFixed(1)}°C',
              style: Theme.of(context).textTheme.displayLarge,
            ),
            Text(
              '${result.temperatureF.toStringAsFixed(1)}°F',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey),
            ),
            const SizedBox(height: 8),
            Text(
              '± ${result.tempConfidenceInterval.toStringAsFixed(2)}°C (95% CI)',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _DetailGrid extends StatelessWidget {
  final ForecastResult result;
  const _DetailGrid({required this.result});

  @override
  Widget build(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 2.5,
      crossAxisSpacing: 8,
      mainAxisSpacing: 8,
      children: [
        _Tile(label: 'Wind', value: '${result.windSpeedMs.toStringAsFixed(1)} m/s', icon: Icons.air),
        _Tile(label: 'Pressure', value: '${result.surfacePressureHpa.toStringAsFixed(0)} hPa', icon: Icons.compress),
        _Tile(label: 'Humidity', value: '${result.relativeHumidityPct.toStringAsFixed(0)}%', icon: Icons.water_drop),
        _Tile(label: 'Precip', value: '${result.precipitationMm.toStringAsFixed(1)} mm', icon: Icons.umbrella),
      ],
    );
  }
}

class _Tile extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  const _Tile({required this.label, required this.value, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          children: [
            Icon(icon, size: 18),
            const SizedBox(width: 8),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(label, style: Theme.of(context).textTheme.labelSmall),
                Text(value, style: Theme.of(context).textTheme.bodyMedium),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _UncertaintyCard extends StatelessWidget {
  final ForecastResult result;
  const _UncertaintyCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Posterior Uncertainty', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            _UncertaintyBar(
              label: 'Temperature',
              std: result.temperatureStd,
              maxStd: 5.0,
            ),
            _UncertaintyBar(
              label: 'Wind Speed',
              std: result.windSpeedStd,
              maxStd: 10.0,
            ),
          ],
        ),
      ),
    );
  }
}

class _UncertaintyBar extends StatelessWidget {
  final String label;
  final double std;
  final double maxStd;
  const _UncertaintyBar({required this.label, required this.std, required this.maxStd});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(width: 90, child: Text(label, style: Theme.of(context).textTheme.bodySmall)),
          Expanded(
            child: LinearProgressIndicator(
              value: (std / maxStd).clamp(0.0, 1.0),
              minHeight: 6,
              borderRadius: BorderRadius.circular(3),
            ),
          ),
          const SizedBox(width: 8),
          Text('σ=${std.toStringAsFixed(2)}', style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}

class _SourceBadge extends StatelessWidget {
  final InferenceSource source;
  const _SourceBadge({required this.source});

  @override
  Widget build(BuildContext context) {
    final isGpu = source == InferenceSource.gpu;
    return Padding(
      padding: const EdgeInsets.only(right: 4, top: 12, bottom: 12),
      child: Chip(
        label: Text(isGpu ? 'GPU' : 'Cache', style: const TextStyle(fontSize: 11)),
        backgroundColor: isGpu ? Colors.deepPurple.shade800 : Colors.teal.shade800,
        side: BorderSide.none,
        padding: EdgeInsets.zero,
        visualDensity: VisualDensity.compact,
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  final String error;
  const _ErrorView({required this.error});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, size: 48, color: Colors.red),
            const SizedBox(height: 16),
            Text(error, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}
