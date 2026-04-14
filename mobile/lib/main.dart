import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'core/services/database_service.dart';
import 'core/services/notification_service.dart';
import 'inference/bma_engine.dart';
import 'inference/linear_dart_engine.dart';
import 'inference/fusion_dart_engine.dart';
import 'inference/lstm_dart_engine.dart';
import 'features/benchmark/benchmark_screen.dart';
import 'features/forecast/forecast_screen.dart';
import 'features/history/history_screen.dart';
import 'features/map/map_screen.dart';
import 'features/settings/settings_screen.dart';
import 'features/validation/validation_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await DatabaseService.instance.initialize();
  await BmaEngine.instance.initialize();
  await NotificationService.instance.initialize();
  // Pre-load lightweight model weights in parallel; failures are silent
  await Future.wait([
    LinearDartEngine.instance.load(),
    LstmDartEngine.instance.load(),
    FusionDartEngine.instance.load(),
  ]);
  runApp(const ProviderScope(child: WeatherApp()));
}

class WeatherApp extends StatelessWidget {
  const WeatherApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Bayesian Weather',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1565C0),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const MainShell(),
    );
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _selectedIndex = 0;

  static const List<Widget> _pages = [
    ForecastScreen(),
    MapScreen(),
    HistoryScreen(),
    ValidationScreen(),
    BenchmarkScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.cloud), label: 'Forecast'),
          NavigationDestination(icon: Icon(Icons.map), label: 'Map'),
          NavigationDestination(icon: Icon(Icons.history), label: 'History'),
          NavigationDestination(icon: Icon(Icons.science_outlined), label: 'Validate'),
          NavigationDestination(icon: Icon(Icons.analytics_outlined), label: 'Benchmark'),
          NavigationDestination(icon: Icon(Icons.settings), label: 'Settings'),
        ],
      ),
    );
  }
}
