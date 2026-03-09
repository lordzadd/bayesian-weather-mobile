// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'station_observation.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

StationObservation _$StationObservationFromJson(Map<String, dynamic> json) =>
    StationObservation(
      stationId: json['stationId'] as String,
      temperatureC: (json['temperatureC'] as num).toDouble(),
      windSpeedMs: (json['windSpeedMs'] as num).toDouble(),
      windDirectionDeg: (json['windDirectionDeg'] as num).toDouble(),
      surfacePressureHpa: (json['surfacePressureHpa'] as num).toDouble(),
      relativeHumidityPct: (json['relativeHumidityPct'] as num).toDouble(),
      timestamp: DateTime.parse(json['timestamp'] as String),
    );

Map<String, dynamic> _$StationObservationToJson(StationObservation instance) =>
    <String, dynamic>{
      'stationId': instance.stationId,
      'temperatureC': instance.temperatureC,
      'windSpeedMs': instance.windSpeedMs,
      'windDirectionDeg': instance.windDirectionDeg,
      'surfacePressureHpa': instance.surfacePressureHpa,
      'relativeHumidityPct': instance.relativeHumidityPct,
      'timestamp': instance.timestamp.toIso8601String(),
    };
