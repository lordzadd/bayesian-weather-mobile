// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'forecast_result.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ForecastResult _$ForecastResultFromJson(Map<String, dynamic> json) =>
    ForecastResult(
      temperatureC: (json['temperatureC'] as num).toDouble(),
      temperatureStd: (json['temperatureStd'] as num).toDouble(),
      windSpeedMs: (json['windSpeedMs'] as num).toDouble(),
      windSpeedStd: (json['windSpeedStd'] as num).toDouble(),
      windBearingDeg: (json['windBearingDeg'] as num?)?.toDouble() ?? 0.0,
      surfacePressureHpa: (json['surfacePressureHpa'] as num).toDouble(),
      relativeHumidityPct: (json['relativeHumidityPct'] as num).toDouble(),
      precipitationMm: (json['precipitationMm'] as num).toDouble(),
      computedAt: DateTime.parse(json['computedAt'] as String),
      source: $enumDecode(_$InferenceSourceEnumMap, json['source']),
    );

Map<String, dynamic> _$ForecastResultToJson(ForecastResult instance) =>
    <String, dynamic>{
      'temperatureC': instance.temperatureC,
      'temperatureStd': instance.temperatureStd,
      'windSpeedMs': instance.windSpeedMs,
      'windSpeedStd': instance.windSpeedStd,
      'windBearingDeg': instance.windBearingDeg,
      'surfacePressureHpa': instance.surfacePressureHpa,
      'relativeHumidityPct': instance.relativeHumidityPct,
      'precipitationMm': instance.precipitationMm,
      'computedAt': instance.computedAt.toIso8601String(),
      'source': _$InferenceSourceEnumMap[instance.source]!,
    };

const _$InferenceSourceEnumMap = {
  InferenceSource.gpu: 'gpu',
  InferenceSource.dart: 'dart',
  InferenceSource.cache: 'cache',
};
