import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'settings_provider.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final settings = ref.watch(settingsProvider);
    final notifier = ref.read(settingsProvider.notifier);

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _SectionHeader('Inference Variant'),
          Card(
            child: Column(
              children: [
                RadioListTile<InferenceVariant>(
                  title: const Text('Variant A — GPU (always fresh)'),
                  subtitle: const Text('Full inference on every observation update'),
                  value: InferenceVariant.gpuAlways,
                  groupValue: settings.variant,
                  onChanged: (v) => notifier.setVariant(v!),
                ),
                RadioListTile<InferenceVariant>(
                  title: const Text('Variant B — Cache-optimized'),
                  subtitle: const Text('GPU gated by significance threshold'),
                  value: InferenceVariant.cacheOptimized,
                  groupValue: settings.variant,
                  onChanged: (v) => notifier.setVariant(v!),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          _SectionHeader('Significance Thresholds'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Temperature threshold: ${settings.tempThresholdC.toStringAsFixed(1)}°C',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  Slider(
                    value: settings.tempThresholdC,
                    min: 0.05,
                    max: 2.0,
                    divisions: 39,
                    label: '${settings.tempThresholdC.toStringAsFixed(2)}°C',
                    onChanged: settings.variant == InferenceVariant.cacheOptimized
                        ? (v) => notifier.setTempThreshold(v)
                        : null,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Wind threshold: ${settings.windThresholdMs.toStringAsFixed(1)} m/s',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  Slider(
                    value: settings.windThresholdMs,
                    min: 0.1,
                    max: 5.0,
                    divisions: 49,
                    label: '${settings.windThresholdMs.toStringAsFixed(1)} m/s',
                    onChanged: settings.variant == InferenceVariant.cacheOptimized
                        ? (v) => notifier.setWindThreshold(v)
                        : null,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          _SectionHeader('Benchmarking'),
          Card(
            child: ListTile(
              leading: const Icon(Icons.analytics_outlined),
              title: const Text('View benchmark log'),
              trailing: const Icon(Icons.chevron_right),
              onTap: () {
                // Navigate to benchmark screen
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  const _SectionHeader(this.title);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8, top: 4, left: 4),
      child: Text(
        title,
        style: Theme.of(context).textTheme.labelLarge?.copyWith(
              color: Theme.of(context).colorScheme.primary,
            ),
      ),
    );
  }
}
