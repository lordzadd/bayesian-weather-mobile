import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Emits `true` when the device has any network connection, `false` when offline.
/// The first emission comes from an immediate check; subsequent ones from the
/// change stream, so the value is always up-to-date.
final connectivityProvider = StreamProvider<bool>((ref) async* {
  final connectivity = Connectivity();

  // Emit the current state immediately
  final initial = await connectivity.checkConnectivity();
  yield initial.any((r) => r != ConnectivityResult.none);

  // Then keep emitting on changes
  yield* connectivity.onConnectivityChanged
      .map((results) => results.any((r) => r != ConnectivityResult.none));
});
