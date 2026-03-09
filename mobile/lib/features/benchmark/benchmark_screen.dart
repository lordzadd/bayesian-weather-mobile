import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'benchmark_provider.dart';

class BenchmarkScreen extends ConsumerWidget {
  const BenchmarkScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(benchmarkProvider);
    final notifier = ref.read(benchmarkProvider.notifier);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Benchmark Harness'),
        actions: [
          if (!state.running && (state.variantA != null || state.variantB != null))
            IconButton(
              icon: const Icon(Icons.download),
              tooltip: 'Export CSV',
              onPressed: () async {
                final csv = await notifier.exportCsv();
                if (context.mounted) {
                  _showCsvDialog(context, csv);
                }
              },
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _RunCard(state: state, onRun: () => notifier.run()),
          if (state.running) ...[
            const SizedBox(height: 16),
            _ProgressCard(state: state),
          ],
          if (state.error != null) ...[
            const SizedBox(height: 16),
            Card(
              color: Colors.red.shade900,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(state.error!, style: const TextStyle(color: Colors.white)),
              ),
            ),
          ],
          if (state.variantA != null) ...[
            const SizedBox(height: 16),
            _ResultCard(result: state.variantA!, color: Colors.deepPurple),
          ],
          if (state.variantB != null) ...[
            const SizedBox(height: 16),
            _ResultCard(result: state.variantB!, color: Colors.teal),
          ],
          if (state.variantA != null && state.variantB != null) ...[
            const SizedBox(height: 16),
            _ComparisonCard(a: state.variantA!, b: state.variantB!),
          ],
        ],
      ),
    );
  }

  void _showCsvDialog(BuildContext context, String csv) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Benchmark CSV'),
        content: SingleChildScrollView(
          child: SelectableText(csv, style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close')),
        ],
      ),
    );
  }
}

class _RunCard extends StatelessWidget {
  final BenchmarkState state;
  final VoidCallback onRun;
  const _RunCard({required this.state, required this.onRun});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Comparative Benchmark', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            const Text(
              'Runs 50 inference cycles for Variant A (GPU-always) and Variant B (cache-gated) '
              'on a synthetic observation stream. Records latency and cache hit rate.',
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: state.running ? null : onRun,
                icon: state.running
                    ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.play_arrow),
                label: Text(state.running ? 'Running…' : 'Run Benchmark'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ProgressCard extends StatelessWidget {
  final BenchmarkState state;
  const _ProgressCard({required this.state});

  @override
  Widget build(BuildContext context) {
    final progress = state.total > 0 ? state.progress / state.total : 0.0;
    final phase = state.progress <= 50 ? 'Variant A — GPU' : 'Variant B — Cache';
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(phase, style: Theme.of(context).textTheme.labelLarge),
            const SizedBox(height: 8),
            LinearProgressIndicator(value: progress, minHeight: 8, borderRadius: BorderRadius.circular(4)),
            const SizedBox(height: 4),
            Text('${state.progress} / ${state.total} cycles',
                style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final BenchmarkResult result;
  final Color color;
  const _ResultCard({required this.result, required this.color});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: color.withOpacity(0.25),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
            ),
            child: Text(
              'Variant ${result.variant}',
              style: Theme.of(context).textTheme.titleSmall,
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                _Stat(label: 'Mean', value: '${result.meanMs.toStringAsFixed(1)} ms'),
                _Stat(label: 'P95', value: '${result.p95Ms} ms'),
                _Stat(label: 'P99', value: '${result.p99Ms} ms'),
                if (result.variant == 'B')
                  _Stat(label: 'Cache rate', value: '${(result.cacheHitRate * 100).toStringAsFixed(0)}%'),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  final String label;
  final String value;
  const _Stat({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        children: [
          Text(value, style: Theme.of(context).textTheme.titleMedium),
          Text(label, style: Theme.of(context).textTheme.labelSmall?.copyWith(color: Colors.grey)),
        ],
      ),
    );
  }
}

class _ComparisonCard extends StatelessWidget {
  final BenchmarkResult a;
  final BenchmarkResult b;
  const _ComparisonCard({required this.a, required this.b});

  @override
  Widget build(BuildContext context) {
    final speedup = a.meanMs > 0 ? a.meanMs / b.meanMs : 1.0;
    final savings = a.meanMs > 0 ? (1 - b.meanMs / a.meanMs) * 100 : 0.0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Summary', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 12),
            _Row('Variant A mean latency', '${a.meanMs.toStringAsFixed(1)} ms'),
            _Row('Variant B mean latency', '${b.meanMs.toStringAsFixed(1)} ms'),
            _Row('Cache hit rate (B)', '${(b.cacheHitRate * 100).toStringAsFixed(0)}%'),
            _Row('Latency speedup (B vs A)', '${speedup.toStringAsFixed(2)}×'),
            _Row('Compute savings (B vs A)', '${savings.toStringAsFixed(1)}%'),
          ],
        ),
      ),
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  const _Row(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: Theme.of(context).textTheme.bodySmall),
          Text(value, style: Theme.of(context).textTheme.bodySmall?.copyWith(
                fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
