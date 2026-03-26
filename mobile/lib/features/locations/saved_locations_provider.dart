import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/services/database_service.dart';

// ── Model ─────────────────────────────────────────────────────────────────────

class SavedLocation {
  final int id;
  final String name;
  final double lat;
  final double lon;

  const SavedLocation({
    required this.id,
    required this.name,
    required this.lat,
    required this.lon,
  });

  factory SavedLocation.fromRow(Map<String, dynamic> row) => SavedLocation(
        id:   row['id']   as int,
        name: row['name'] as String,
        lat:  (row['lat'] as num).toDouble(),
        lon:  (row['lon'] as num).toDouble(),
      );
}

// ── Selected location override ────────────────────────────────────────────────

/// When non-null, the forecast uses this location instead of GPS.
final selectedLocationProvider = StateProvider<SavedLocation?>((_) => null);

// ── Saved locations notifier ─────────────────────────────────────────────────

final savedLocationsProvider =
    AsyncNotifierProvider<SavedLocationsNotifier, List<SavedLocation>>(
  SavedLocationsNotifier.new,
);

class SavedLocationsNotifier extends AsyncNotifier<List<SavedLocation>> {
  @override
  Future<List<SavedLocation>> build() async {
    final rows = await DatabaseService.instance.getSavedLocations();
    return rows.map(SavedLocation.fromRow).toList();
  }

  Future<void> add({
    required String name,
    required double lat,
    required double lon,
  }) async {
    await DatabaseService.instance.insertSavedLocation(
      name: name,
      lat:  lat,
      lon:  lon,
    );
    ref.invalidateSelf();
  }

  Future<void> remove(int id) async {
    await DatabaseService.instance.deleteSavedLocation(id);
    // Clear selection if the deleted location was selected
    final selected = ref.read(selectedLocationProvider);
    if (selected?.id == id) {
      ref.read(selectedLocationProvider.notifier).state = null;
    }
    ref.invalidateSelf();
  }
}
