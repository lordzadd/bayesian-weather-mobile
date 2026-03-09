import 'package:flutter_test/flutter_test.dart';

import 'package:bayesian_weather/core/models/station_observation.dart';

void main() {
  group('StationObservation.toFeatureVector()', () {
    test('decomposes wind into U and V components correctly', () {
      // Wind blowing due east (90°) at 10 m/s
      // U = 10 * cos(90°) ≈ 0, V = 10 * sin(90°) ≈ 10
      final obs = StationObservation(
        stationId: 'KSFO',
        temperatureC: 15.0,
        windSpeedMs: 10.0,
        windDirectionDeg: 90.0,
        surfacePressureHpa: 1013.25,
        relativeHumidityPct: 65.0,
        timestamp: DateTime.now(),
      );

      final vec = obs.toFeatureVector();
      expect(vec.length, 6);
      expect(vec[0], 15.0);            // temperature
      expect(vec[1], 1013.25);         // pressure
      expect(vec[2], closeTo(0.0, 1e-4));  // u-wind
      expect(vec[3], closeTo(10.0, 1e-4)); // v-wind
      expect(vec[4], 0.0);             // precipitation placeholder
      expect(vec[5], 65.0);            // humidity
    });

    test('north wind (0°) gives non-zero U, near-zero V', () {
      final obs = StationObservation(
        stationId: 'KSFO',
        temperatureC: 10.0,
        windSpeedMs: 5.0,
        windDirectionDeg: 0.0,
        surfacePressureHpa: 1013.0,
        relativeHumidityPct: 80.0,
        timestamp: DateTime.now(),
      );

      final vec = obs.toFeatureVector();
      expect(vec[2], closeTo(5.0, 1e-4));   // u = speed * cos(0) = speed
      expect(vec[3], closeTo(0.0, 1e-4));   // v = speed * sin(0) = 0
    });
  });

  group('StationObservation JSON round-trip', () {
    test('serializes and deserializes without data loss', () {
      final original = StationObservation(
        stationId: 'KOAK',
        temperatureC: 18.5,
        windSpeedMs: 3.2,
        windDirectionDeg: 270.0,
        surfacePressureHpa: 1015.0,
        relativeHumidityPct: 72.0,
        timestamp: DateTime.utc(2026, 3, 1, 12, 0),
      );

      final json = original.toJson();
      final restored = StationObservation.fromJson(json);

      expect(restored.stationId, original.stationId);
      expect(restored.temperatureC, original.temperatureC);
      expect(restored.windSpeedMs, original.windSpeedMs);
      expect(restored.relativeHumidityPct, original.relativeHumidityPct);
      expect(restored.timestamp, original.timestamp);
    });
  });
}
