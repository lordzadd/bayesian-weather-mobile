import 'dart:math' as math;
import 'package:json_annotation/json_annotation.dart';

part 'station_observation.g.dart';

@JsonSerializable()
class StationObservation {
  final String stationId;
  final double temperatureC;
  final double windSpeedMs;
  final double windDirectionDeg;
  final double surfacePressureHpa;
  final double relativeHumidityPct;
  final DateTime timestamp;

  const StationObservation({
    required this.stationId,
    required this.temperatureC,
    required this.windSpeedMs,
    required this.windDirectionDeg,
    required this.surfacePressureHpa,
    required this.relativeHumidityPct,
    required this.timestamp,
  });

  factory StationObservation.fromJson(Map<String, dynamic> json) =>
      _$StationObservationFromJson(json);

  Map<String, dynamic> toJson() => _$StationObservationToJson(this);

  /// Returns [temp, pressure, u_wind, v_wind, precip, rh] feature vector.
  List<double> toFeatureVector() => [
        temperatureC,
        surfacePressureHpa,
        windSpeedMs * math.cos(windDirectionDeg * math.pi / 180),
        windSpeedMs * math.sin(windDirectionDeg * math.pi / 180),
        0.0, // precipitation not in METAR; filled by GFS value downstream
        relativeHumidityPct,
      ];
}
