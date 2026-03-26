import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart';

import '../core/models/forecast_result.dart';

/// Ridge regression bias-correction baseline running entirely in Dart.
///
/// Trained on the same paired (GFS, ERA5) data as the BMA model.
/// Input: [gfs_norm(6), lat/90, lon/180] — 8 features
/// Output: per-variable (mean, σ) where σ = validation residual std
///
/// Weights loaded from assets/models/linear_weights.json.
/// Normalization stats loaded from assets/models/bma_stats.json (shared).
class LinearDartEngine {
  static final LinearDartEngine instance = LinearDartEngine._();
  LinearDartEngine._();

  static const int _nIn  = 8;
  static const int _nOut = 6;

  bool _loaded = false;

  // [_nOut][_nIn] coefficient matrix and [_nOut] intercept vector
  late List<List<double>> _coefs;
  late List<double> _intercepts;
  // Per-variable validation residual std → used as prediction uncertainty
  late List<double> _residualStds;

  late Map<String, Map<String, double>> _stats;

  static const List<String> _gfsCols = [
    'gfs_temp_c', 'gfs_pres_hpa', 'gfs_u_ms', 'gfs_v_ms',
    'gfs_precip_mm', 'gfs_rh_pct',
  ];
  static const List<String> _obsCols = [
    'obs_temp_c', 'obs_pres_hpa', 'obs_u_ms', 'obs_v_ms',
    'obs_precip_mm', 'obs_rh_pct',
  ];

  Future<void> load() async {
    if (_loaded) return;
    try {
      final wJson = await rootBundle.loadString('assets/models/linear_weights.json');
      final sJson = await rootBundle.loadString('assets/models/bma_stats.json');

      final w = jsonDecode(wJson) as Map<String, dynamic>;
      final s = jsonDecode(sJson) as Map<String, dynamic>;

      _coefs = (w['coefficients'] as List)
          .map((row) => (row as List).map((v) => (v as num).toDouble()).toList())
          .toList();
      _intercepts = (w['intercepts'] as List)
          .map((v) => (v as num).toDouble())
          .toList();
      _residualStds = (w['residual_stds'] as List)
          .map((v) => (v as num).toDouble())
          .toList();
      _stats = s.map((key, val) {
        final m = val as Map<String, dynamic>;
        return MapEntry(key, {
          'mean': (m['mean'] as num).toDouble(),
          'std':  (m['std']  as num).toDouble(),
        });
      });
      _loaded = true;
    } catch (_) {
      // Weights not available yet — fall through, infer() uses GFS passthrough.
    }
  }

  /// Runs linear forward pass.
  ///
  /// [gfsForecast]  — raw 6-element GFS values
  /// [obsFeatures]  — raw 6-element ERA5 observation (used to sharpen mean if present)
  /// [spatialEmbed] — [lat/90, lon/180]
  ForecastResult infer({
    required List<double> gfsForecast,
    required List<double>? obsFeatures,
    required List<double> spatialEmbed,
  }) {
    if (!_loaded) return _passthrough(gfsForecast);

    // Normalize GFS input
    final gfsNorm = _normalize(gfsForecast, _gfsCols);
    final input = [...gfsNorm, ...spatialEmbed]; // [8]

    // y_norm = W * x + b
    final yNorm = List<double>.filled(_nOut, 0.0);
    for (int i = 0; i < _nOut; i++) {
      double v = _intercepts[i];
      for (int j = 0; j < _nIn; j++) {
        v += _coefs[i][j] * input[j];
      }
      yNorm[i] = v;
    }

    // If observation available, blend linear prediction with observation
    // using a simple precision-weighted average (linear has ~4× less precision than obs)
    if (obsFeatures != null) {
      final obsNorm = _normalize(obsFeatures, _obsCols);
      for (int i = 0; i < _nOut; i++) {
        final sigmaLin = _residualStds[i];
        final sigmaObs = sigmaLin * 0.5; // observations ~2× more precise
        final varLin   = sigmaLin * sigmaLin;
        final varObs   = sigmaObs * sigmaObs;
        final postVar  = 1.0 / (1.0 / varLin + 1.0 / varObs);
        yNorm[i] = postVar * (yNorm[i] / varLin + obsNorm[i] / varObs);
      }
    }

    final mean = _denormalize(yNorm, _obsCols);

    // σ in physical units: residual_std × feature std
    final std = List.generate(_nOut, (i) {
      return _residualStds[i] * _stats[_obsCols[i]]!['std']!;
    });

    return _toResult(mean, std);
  }

  List<double> _normalize(List<double> raw, List<String> cols) {
    return List.generate(cols.length, (i) {
      final s = _stats[cols[i]]!;
      return (raw[i] - s['mean']!) / s['std']!;
    });
  }

  List<double> _denormalize(List<double> normed, List<String> cols) {
    return List.generate(cols.length, (i) {
      final s = _stats[cols[i]]!;
      return normed[i] * s['std']! + s['mean']!;
    });
  }

  ForecastResult _toResult(List<double> mean, List<double> std) {
    return ForecastResult(
      temperatureC:       mean[0],
      temperatureStd:     std[0],
      surfacePressureHpa: mean[1],
      windSpeedMs:        math.sqrt(mean[2] * mean[2] + mean[3] * mean[3]),
      windSpeedStd:       math.sqrt(std[2] * std[2] + std[3] * std[3]),
      precipitationMm:    mean[4].clamp(0, double.infinity),
      relativeHumidityPct: mean[5].clamp(0, 100),
      computedAt:         DateTime.now(),
      source:             InferenceSource.dart,
    );
  }

  ForecastResult _passthrough(List<double> gfs) {
    return ForecastResult(
      temperatureC:        gfs[0],
      temperatureStd:      2.0,
      surfacePressureHpa:  gfs[1],
      windSpeedMs:         0.0,
      windSpeedStd:        3.0,
      precipitationMm:     gfs[4].clamp(0, double.infinity),
      relativeHumidityPct: gfs[5].clamp(0, 100),
      computedAt:          DateTime.now(),
      source:              InferenceSource.dart,
    );
  }
}
