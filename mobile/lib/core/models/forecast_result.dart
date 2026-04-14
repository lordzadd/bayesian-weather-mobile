import 'package:json_annotation/json_annotation.dart';

part 'forecast_result.g.dart';

/// gpu   — native ExecuTorch path (Vulkan/Metal hardware)
/// dart  — trained weights running in pure Dart (CPU)
/// cache — significance threshold hit; previous result reused
enum InferenceSource { gpu, dart, cache }

@JsonSerializable()
class ForecastResult {
  final double temperatureC;
  final double temperatureStd;
  final double windSpeedMs;
  final double windSpeedStd;
  /// Meteorological bearing: direction wind is coming FROM, 0–360° clockwise from North.
  final double windBearingDeg;
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
    this.windBearingDeg = 0.0,
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

  /// 16-point compass abbreviation for the wind bearing (e.g. "SW", "NNE").
  String get windCompassPoint {
    const points = [
      'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',
    ];
    final idx = ((windBearingDeg + 11.25) / 22.5).floor() % 16;
    return points[idx];
  }

  ForecastResult copyWith({
    double? temperatureC,
    double? temperatureStd,
    double? windSpeedMs,
    double? windSpeedStd,
    double? windBearingDeg,
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
      windBearingDeg: windBearingDeg ?? this.windBearingDeg,
      surfacePressureHpa: surfacePressureHpa ?? this.surfacePressureHpa,
      relativeHumidityPct: relativeHumidityPct ?? this.relativeHumidityPct,
      precipitationMm: precipitationMm ?? this.precipitationMm,
      computedAt: computedAt ?? this.computedAt,
      source: source ?? this.source,
    );
  }
}
