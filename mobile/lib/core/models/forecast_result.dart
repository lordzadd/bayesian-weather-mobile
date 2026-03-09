import 'package:json_annotation/json_annotation.dart';

part 'forecast_result.g.dart';

enum InferenceSource { gpu, cache }

@JsonSerializable()
class ForecastResult {
  final double temperatureC;
  final double temperatureStd;
  final double windSpeedMs;
  final double windSpeedStd;
  final double surfacePressureHpa;
  final double relativeHumidityPct;
  final double precipitationMm;
  final DateTime computedAt;
  final InferenceSource source;

  const ForecastResult({
    required this.temperatureC,
    required this.temperatureStd,
    required this.windSpeedMs,
    required this.windSpeedStd,
    required this.surfacePressureHpa,
    required this.relativeHumidityPct,
    required this.precipitationMm,
    required this.computedAt,
    required this.source,
  });

  factory ForecastResult.fromJson(Map<String, dynamic> json) =>
      _$ForecastResultFromJson(json);

  Map<String, dynamic> toJson() => _$ForecastResultToJson(this);

  double get temperatureF => temperatureC * 9 / 5 + 32;

  /// 95% confidence interval half-width (2σ)
  double get tempConfidenceInterval => temperatureStd * 2;

  ForecastResult copyWith({
    double? temperatureC,
    double? temperatureStd,
    double? windSpeedMs,
    double? windSpeedStd,
    double? surfacePressureHpa,
    double? relativeHumidityPct,
    double? precipitationMm,
    DateTime? computedAt,
    InferenceSource? source,
  }) {
    return ForecastResult(
      temperatureC: temperatureC ?? this.temperatureC,
      temperatureStd: temperatureStd ?? this.temperatureStd,
      windSpeedMs: windSpeedMs ?? this.windSpeedMs,
      windSpeedStd: windSpeedStd ?? this.windSpeedStd,
      surfacePressureHpa: surfacePressureHpa ?? this.surfacePressureHpa,
      relativeHumidityPct: relativeHumidityPct ?? this.relativeHumidityPct,
      precipitationMm: precipitationMm ?? this.precipitationMm,
      computedAt: computedAt ?? this.computedAt,
      source: source ?? this.source,
    );
  }
}
