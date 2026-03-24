import 'dart:math' as math;

import 'forecast_result.dart';

/// Result of a single lookback validation cycle.
///
/// Captures the BMA prediction for T-1h, the actual observation (ground truth),
/// the raw GFS forecast (baseline), and per-variable absolute errors.
class LookbackResult {
  final DateTime targetTime;
  final ForecastResult prediction;

  // Ground truth observation (from current conditions)
  final double obsTempC;
  final double obsPressureHpa;
  final double obsWindSpeedMs;
  final double obsPrecipMm;
  final double obsHumidityPct;

  // Raw GFS values (for baseline comparison)
  final double gfsTempC;
  final double gfsPressureHpa;
  final double gfsWindSpeedMs;
  final double gfsPrecipMm;
  final double gfsHumidityPct;

  const LookbackResult({
    required this.targetTime,
    required this.prediction,
    required this.obsTempC,
    required this.obsPressureHpa,
    required this.obsWindSpeedMs,
    required this.obsPrecipMm,
    required this.obsHumidityPct,
    required this.gfsTempC,
    required this.gfsPressureHpa,
    required this.gfsWindSpeedMs,
    required this.gfsPrecipMm,
    required this.gfsHumidityPct,
  });

  // --- BMA errors (prediction vs observation) ---

  double get bmaTempErrorC => (prediction.temperatureC - obsTempC).abs();
  double get bmaWindErrorMs => (prediction.windSpeedMs - obsWindSpeedMs).abs();
  double get bmaPressureErrorHpa =>
      (prediction.surfacePressureHpa - obsPressureHpa).abs();
  double get bmaPrecipErrorMm => (prediction.precipitationMm - obsPrecipMm).abs();
  double get bmaHumidityErrorPct =>
      (prediction.relativeHumidityPct - obsHumidityPct).abs();

  // --- Raw GFS errors (baseline vs observation) ---

  double get gfsTempErrorC => (gfsTempC - obsTempC).abs();
  double get gfsWindErrorMs => (gfsWindSpeedMs - obsWindSpeedMs).abs();
  double get gfsPressureErrorHpa => (gfsPressureHpa - obsPressureHpa).abs();
  double get gfsPrecipErrorMm => (gfsPrecipMm - obsPrecipMm).abs();
  double get gfsHumidityErrorPct => (gfsHumidityPct - obsHumidityPct).abs();

  /// Temperature improvement: positive means BMA is better.
  double get tempImprovementPct => gfsTempErrorC > 0
      ? (1 - bmaTempErrorC / gfsTempErrorC) * 100
      : 0;

  /// Wind improvement: positive means BMA is better.
  double get windImprovementPct => gfsWindErrorMs > 0
      ? (1 - bmaWindErrorMs / gfsWindErrorMs) * 100
      : 0;

  /// Constructs a LookbackResult from raw feature vectors.
  ///
  /// [gfsFeatures] and [obsFeatures] are [temp, pressure, u, v, precip, rh].
  factory LookbackResult.fromVectors({
    required DateTime targetTime,
    required ForecastResult prediction,
    required List<double> obsFeatures,
    required List<double> gfsFeatures,
  }) {
    return LookbackResult(
      targetTime: targetTime,
      prediction: prediction,
      obsTempC: obsFeatures[0],
      obsPressureHpa: obsFeatures[1],
      obsWindSpeedMs: math.sqrt(
          obsFeatures[2] * obsFeatures[2] + obsFeatures[3] * obsFeatures[3]),
      obsPrecipMm: obsFeatures[4],
      obsHumidityPct: obsFeatures[5],
      gfsTempC: gfsFeatures[0],
      gfsPressureHpa: gfsFeatures[1],
      gfsWindSpeedMs: math.sqrt(
          gfsFeatures[2] * gfsFeatures[2] + gfsFeatures[3] * gfsFeatures[3]),
      gfsPrecipMm: gfsFeatures[4],
      gfsHumidityPct: gfsFeatures[5],
    );
  }
}
