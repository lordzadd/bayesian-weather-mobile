import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../benchmark/benchmark_screen.dart';
import '../forecast/forecast_provider.dart';
import '../locations/saved_locations_provider.dart';
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
                  title: const Text('Variant A — NN·Dart (always fresh)'),
                  subtitle: const Text('Full inference on every observation update'),
                  value: InferenceVariant.gpuAlways,
                  groupValue: settings.variant,
                  onChanged: (v) {
                    notifier.setVariant(v!);
                    ref.read(forecastProvider.notifier).refresh();
                  },
                ),
                RadioListTile<InferenceVariant>(
                  title: const Text('Variant B — Cache-optimized'),
                  subtitle: const Text('Inference gated by significance threshold'),
                  value: InferenceVariant.cacheOptimized,
                  groupValue: settings.variant,
                  onChanged: (v) {
                    notifier.setVariant(v!);
                    ref.read(forecastProvider.notifier).refresh();
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          _SectionHeader('Model Architecture'),
          Card(
            child: Column(
              children: [
                RadioListTile<ModelVariant>(
                  title: const Text('Fusion — LSTM + BayCal (best)'),
                  subtitle: const Text('39.5% over GFS, calibrated uncertainty'),
                  value: ModelVariant.fusion,
                  groupValue: settings.modelVariant,
                  onChanged: (v) {
                    notifier.setModelVariant(v!);
                    ref.read(forecastProvider.notifier).refresh();
                  },
                ),
                RadioListTile<ModelVariant>(
                  title: const Text('BMA — Bayesian Model Averaging'),
                  subtitle: const Text('SVI with heteroscedastic noise net'),
                  value: ModelVariant.bma,
                  groupValue: settings.modelVariant,
                  onChanged: (v) {
                    notifier.setModelVariant(v!);
                    ref.read(forecastProvider.notifier).refresh();
                  },
                ),
                RadioListTile<ModelVariant>(
                  title: const Text('Linear — Ridge regression baseline'),
                  subtitle: const Text('Fast bias-correction, no hidden layers'),
                  value: ModelVariant.linear,
                  groupValue: settings.modelVariant,
                  onChanged: (v) {
                    notifier.setModelVariant(v!);
                    ref.read(forecastProvider.notifier).refresh();
                  },
                ),
                RadioListTile<ModelVariant>(
                  title: const Text('LSTM — Sequence model'),
                  subtitle: const Text('Uses last 6 hourly observations'),
                  value: ModelVariant.lstm,
                  groupValue: settings.modelVariant,
                  onChanged: (v) {
                    notifier.setModelVariant(v!);
                    ref.read(forecastProvider.notifier).refresh();
                  },
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
          _SectionHeader('Notifications'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SwitchListTile(
                    title: const Text('Weather change alerts'),
                    subtitle: const Text(
                        'Alert when temperature shifts significantly'),
                    value: settings.notificationsEnabled,
                    onChanged: (v) => notifier.setNotificationsEnabled(v),
                    contentPadding: EdgeInsets.zero,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Alert threshold: ${settings.alertTempChangeC.toStringAsFixed(1)}°C',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  Slider(
                    value: settings.alertTempChangeC,
                    min: 1.0,
                    max: 10.0,
                    divisions: 18,
                    label: '${settings.alertTempChangeC.toStringAsFixed(1)}°C',
                    onChanged: settings.notificationsEnabled
                        ? (v) => notifier.setAlertTempChangeC(v)
                        : null,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          _SectionHeader('Saved Locations'),
          _SavedLocationsCard(),
          const SizedBox(height: 16),
          _SectionHeader('Benchmarking'),
          Card(
            child: ListTile(
              leading: const Icon(Icons.analytics_outlined),
              title: const Text('Open benchmark harness'),
              trailing: const Icon(Icons.chevron_right),
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const BenchmarkScreen()),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SavedLocationsCard extends ConsumerStatefulWidget {
  @override
  ConsumerState<_SavedLocationsCard> createState() =>
      _SavedLocationsCardState();
}

class _SavedLocationsCardState extends ConsumerState<_SavedLocationsCard> {
  final _nameController = TextEditingController();
  final _latController  = TextEditingController();
  final _lonController  = TextEditingController();

  @override
  void dispose() {
    _nameController.dispose();
    _latController.dispose();
    _lonController.dispose();
    super.dispose();
  }

  void _showAddDialog() {
    _nameController.clear();
    _latController.clear();
    _lonController.clear();
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Add Location'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(labelText: 'Name'),
            ),
            TextField(
              controller: _latController,
              decoration: const InputDecoration(labelText: 'Latitude'),
              keyboardType: const TextInputType.numberWithOptions(
                  signed: true, decimal: true),
            ),
            TextField(
              controller: _lonController,
              decoration: const InputDecoration(labelText: 'Longitude'),
              keyboardType: const TextInputType.numberWithOptions(
                  signed: true, decimal: true),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              final name = _nameController.text.trim();
              final lat  = double.tryParse(_latController.text);
              final lon  = double.tryParse(_lonController.text);
              if (name.isNotEmpty && lat != null && lon != null) {
                ref
                    .read(savedLocationsProvider.notifier)
                    .add(name: name, lat: lat, lon: lon);
                Navigator.pop(context);
              }
            },
            child: const Text('Add'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final locations = ref.watch(savedLocationsProvider).valueOrNull ?? [];

    return Card(
      child: Column(
        children: [
          if (locations.isEmpty)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Text(
                'No saved locations yet.',
                style: TextStyle(color: Colors.grey),
              ),
            )
          else
            ...locations.map((loc) => ListTile(
                  leading: const Icon(Icons.location_on_outlined, size: 18),
                  title: Text(loc.name),
                  subtitle: Text(
                    '${loc.lat.toStringAsFixed(3)}, ${loc.lon.toStringAsFixed(3)}',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                  trailing: IconButton(
                    icon: const Icon(Icons.delete_outline, size: 18),
                    onPressed: () =>
                        ref.read(savedLocationsProvider.notifier).remove(loc.id),
                  ),
                )),
          ListTile(
            leading: const Icon(Icons.add_location_alt_outlined),
            title: const Text('Add location'),
            onTap: _showAddDialog,
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
