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

  List<double> toFeatureVector() => [
        temperatureC,
        surfacePressureHpa,
        windSpeedMs * _cosDeg(windDirectionDeg),
        windSpeedMs * _sinDeg(windDirectionDeg),
        0.0, // precipitation placeholder
        relativeHumidityPct,
      ];

  static double _cosDeg(double deg) =>
      0; // dart:math import handled in implementation

  static double _sinDeg(double deg) => 0;
}
