import 'package:geolocator/geolocator.dart';
import 'package:riverpod/riverpod.dart';

/// Nearest METAR stations to look up observations for a given coordinate.
/// Mapped by approximate bounding box (lat range, lon range) → ICAO id.
const _stationIndex = [
  // Original 10 + 40 additional CONUS stations for geographic diversity
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
  _Station('KMSP', 44.883, -93.229),   // Minneapolis
  _Station('KDTW', 42.212, -83.353),   // Detroit
  _Station('KIAH', 29.984, -95.341),   // Houston
  _Station('KMCO', 28.429, -81.309),   // Orlando
  _Station('KPHL', 39.872, -75.241),   // Philadelphia
  _Station('KCLT', 35.214, -80.943),   // Charlotte
  _Station('KSLC', 40.789, -111.977),  // Salt Lake City
  _Station('KPDX', 45.589, -122.597),  // Portland OR
  _Station('KSTL', 38.749, -90.370),   // St. Louis
  _Station('KBNA', 36.126, -86.681),   // Nashville
  _Station('KMKE', 42.947, -87.897),   // Milwaukee
  _Station('KPIT', 40.492, -80.233),   // Pittsburgh
  _Station('KIND', 39.717, -86.294),   // Indianapolis
  _Station('KCVG', 39.048, -84.667),   // Cincinnati
  _Station('KMCI', 39.298, -94.714),   // Kansas City
  _Station('KAUS', 30.194, -97.670),   // Austin
  _Station('KSAN', 32.734, -117.190),  // San Diego
  _Station('KSJC', 37.362, -121.929),  // San Jose
  _Station('KRDU', 35.878, -78.787),   // Raleigh-Durham
  _Station('KCLE', 41.411, -81.849),   // Cleveland
  _Station('KABQ', 35.040, -106.609),  // Albuquerque
  _Station('KTUL', 36.198, -95.888),   // Tulsa
  _Station('KOMA', 41.302, -95.894),   // Omaha
  _Station('KBOI', 43.564, -116.223),  // Boise
  _Station('KBUF', 42.941, -78.736),   // Buffalo
  _Station('KPWM', 43.646, -70.309),   // Portland ME
  _Station('KJAX', 30.494, -81.688),   // Jacksonville
  _Station('KMSN', 43.140, -89.337),   // Madison
  _Station('KFAR', 46.921, -96.816),   // Fargo
  _Station('KRAP', 44.045, -103.054),  // Rapid City
  _Station('KBIL', 45.808, -108.543),  // Billings
  _Station('KLIT', 34.729, -92.224),   // Little Rock
  _Station('KJAN', 32.311, -90.076),   // Jackson MS
  _Station('KELP', 31.807, -106.378),  // El Paso
  _Station('KFAT', 36.776, -119.718),  // Fresno
  _Station('KGRR', 42.881, -85.523),   // Grand Rapids
  _Station('KDSM', 41.534, -93.663),   // Des Moines
  _Station('KSAT', 29.534, -98.470),   // San Antonio
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
          timeLimit: Duration(seconds: 5),
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
