import 'package:geolocator/geolocator.dart';
import 'package:riverpod/riverpod.dart';

const _stationIndex = [
  _Station('KSFO', 37.619, -122.375),
  _Station('KLAX', 33.942, -118.408),
  _Station('KORD', 41.978, -87.904),
  _Station('KJFK', 40.639, -73.778),
  _Station('KDEN', 39.861, -104.673),
  _Station('KDFW', 32.896, -97.038),
  _Station('KATL', 33.637, -84.428),
  _Station('KSEA', 47.449, -122.309),
  _Station('KBOS', 42.364, -71.005),
  _Station('KMIA', 25.796, -80.287),
  _Station('KPHX', 33.435, -112.007),
  _Station('KLAS', 36.080, -115.152),
];

class _Station {
  final String id;
  final double lat;
  final double lon;
  const _Station(this.id, this.lat, this.lon);
}

String nearestStation(double lat, double lon) {
  _Station best = _stationIndex.first;
  double bestDist = double.infinity;
  for (final s in _stationIndex) {
    final d = _dist(lat, lon, s.lat, s.lon);
    if (d < bestDist) { bestDist = d; best = s; }
  }
  return best.id;
}

double _dist(double lat1, double lon1, double lat2, double lon2) {
  final dlat = lat1 - lat2;
  final dlon = lon1 - lon2;
  return dlat * dlat + dlon * dlon;
}

enum LocationFailReason { permissionDenied, permissionDeniedForever, serviceDisabled, timeout, error }

class LocationResult {
  final double lat;
  final double lon;
  /// null = real GPS. Non-null = fallback with reason.
  final LocationFailReason? failReason;

  const LocationResult({required this.lat, required this.lon, this.failReason});

  bool get isFallback => failReason != null;

  String get failMessage => switch (failReason) {
    LocationFailReason.permissionDenied         => 'Location permission denied',
    LocationFailReason.permissionDeniedForever  => 'Location permission permanently denied — open Settings to fix',
    LocationFailReason.serviceDisabled          => 'Location services disabled on device',
    LocationFailReason.timeout                  => 'GPS timed out — no fix available',
    LocationFailReason.error                    => 'GPS unavailable',
    null                                        => '',
  };

  bool get canOpenSettings => failReason == LocationFailReason.permissionDeniedForever;
}

class LocationService {
  static const _fallbackLat = 37.7749;
  static const _fallbackLon = -122.4194;

  Future<LocationResult> currentPosition() async {
    try {
      final serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        return _fallback(LocationFailReason.serviceDisabled);
      }

      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.deniedForever) {
        return _fallback(LocationFailReason.permissionDeniedForever);
      }
      if (permission == LocationPermission.denied) {
        return _fallback(LocationFailReason.permissionDenied);
      }

      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.low,
          timeLimit: Duration(seconds: 15),
        ),
      );
      return LocationResult(lat: pos.latitude, lon: pos.longitude);
    } on PermissionDefinitionsNotFoundException catch (_) {
      return _fallback(LocationFailReason.error);
    } catch (e) {
      final isTimeout = e.toString().toLowerCase().contains('timeout') ||
                        e.toString().toLowerCase().contains('timedout');
      return _fallback(isTimeout ? LocationFailReason.timeout : LocationFailReason.error);
    }
  }

  LocationResult _fallback(LocationFailReason reason) =>
      LocationResult(lat: _fallbackLat, lon: _fallbackLon, failReason: reason);
}

final locationServiceProvider = Provider((_) => LocationService());
