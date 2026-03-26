import 'package:geolocator/geolocator.dart';
import 'package:riverpod/riverpod.dart';

/// Nearest METAR stations to look up observations for a given coordinate.
/// Mapped by approximate bounding box (lat range, lon range) → ICAO id.
const _stationIndex = [
  // CONUS major stations — expand as needed
  _Station('KSFO', 37.619, -122.375),  // San Francisco
  _Station('KLAX', 33.942, -118.408),  // Los Angeles
  _Station('KORD', 41.978, -87.904),   // Chicago O'Hare
  _Station('KJFK', 40.639, -73.778),   // New York JFK
  _Station('KDEN', 39.861, -104.673),  // Denver
  _Station('KDFW', 32.896, -97.038),   // Dallas/Fort Worth
  _Station('KATL', 33.637, -84.428),   // Atlanta
  _Station('KSEA', 47.449, -122.309),  // Seattle
  _Station('KBOS', 42.364, -71.005),   // Boston
  _Station('KMIA', 25.796, -80.287),   // Miami
  _Station('KPHX', 33.435, -112.007),  // Phoenix
  _Station('KLAS', 36.080, -115.152),  // Las Vegas
];

class _Station {
  final String id;
  final double lat;
  final double lon;
  const _Station(this.id, this.lat, this.lon);
}

/// Returns the ICAO station id of the closest station to [lat]/[lon].
String nearestStation(double lat, double lon) {
  _Station best = _stationIndex.first;
  double bestDist = double.infinity;
  for (final s in _stationIndex) {
    final d = _dist(lat, lon, s.lat, s.lon);
    if (d < bestDist) {
      bestDist = d;
      best = s;
    }
  }
  return best.id;
}

double _dist(double lat1, double lon1, double lat2, double lon2) {
  final dlat = lat1 - lat2;
  final dlon = lon1 - lon2;
  return dlat * dlat + dlon * dlon;
}

/// Current device position, falling back to San Francisco if permission denied.
class LocationService {
  static const _fallbackLat = 37.7749;
  static const _fallbackLon = -122.4194;

  Future<({double lat, double lon})> currentPosition() async {
    try {
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) return _fallback();

      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        return _fallback();
      }

      final pos = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.low, // low accuracy is sufficient for weather
          timeLimit: Duration(seconds: 15),
        ),
      );
      return (lat: pos.latitude, lon: pos.longitude);
    } catch (_) {
      return _fallback();
    }
  }

  ({double lat, double lon}) _fallback() =>
      (lat: _fallbackLat, lon: _fallbackLon);
}

final locationServiceProvider = Provider((_) => LocationService());
