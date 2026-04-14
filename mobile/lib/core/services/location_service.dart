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
  // Additional CONUS stations for geographic diversity
  _Station('KMSP', 44.883, -93.229),
  _Station('KDTW', 42.212, -83.353),
  _Station('KIAH', 29.984, -95.341),
  _Station('KMCO', 28.429, -81.309),
  _Station('KPHL', 39.872, -75.241),
  _Station('KCLT', 35.214, -80.943),
  _Station('KSLC', 40.789, -111.977),
  _Station('KPDX', 45.589, -122.597),
  _Station('KSTL', 38.749, -90.370),
  _Station('KBNA', 36.126, -86.681),
  _Station('KMKE', 42.947, -87.897),
  _Station('KPIT', 40.492, -80.233),
  _Station('KIND', 39.717, -86.294),
  _Station('KCVG', 39.048, -84.667),
  _Station('KMCI', 39.298, -94.714),
  _Station('KAUS', 30.194, -97.670),
  _Station('KSAN', 32.734, -117.190),
  _Station('KSJC', 37.362, -121.929),
  _Station('KRDU', 35.878, -78.787),
  _Station('KCLE', 41.411, -81.849),
  _Station('KABQ', 35.040, -106.609),
  _Station('KTUL', 36.198, -95.888),
  _Station('KOMA', 41.302, -95.894),
  _Station('KBOI', 43.564, -116.223),
  _Station('KBUF', 42.941, -78.736),
  _Station('KPWM', 43.646, -70.309),
  _Station('KJAX', 30.494, -81.688),
  _Station('KMSN', 43.140, -89.337),
  _Station('KFAR', 46.921, -96.816),
  _Station('KRAP', 44.045, -103.054),
  _Station('KBIL', 45.808, -108.543),
  _Station('KLIT', 34.729, -92.224),
  _Station('KJAN', 32.311, -90.076),
  _Station('KELP', 31.807, -106.378),
  _Station('KFAT', 36.776, -119.718),
  _Station('KGRR', 42.881, -85.523),
  _Station('KDSM', 41.534, -93.663),
  _Station('KSAT', 29.534, -98.470),
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
