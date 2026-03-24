import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart';

/// Runs the trained BMA neural network entirely in Dart.
///
/// The network was trained on paired (GFS seamless, ERA5) data for 10 CONUS
/// airport stations over 2023-03-01 → 2025-03-01 (~126K samples, 150 epochs,
/// MPS GPU, best val ELBO ≈ 8454).
///
/// Architecture (matches ml/training/bma_model.py):
///   bias_net:  Linear(8→64) → ReLU → Linear(64→64) → ReLU → Linear(64→6)
///   noise_net: Linear(8→32) → ReLU → Linear(32→6)  → Softplus
///
/// Input: [gfs_features(6), lat_norm, lon_norm]  → 8 dims
/// Output: (posterior_mean[6], posterior_std[6])
///
/// The trained bias corrects systematic GFS error patterns learned from 2 years
/// of ERA5 reanalysis vs GFS seamless divergence across 10 CONUS airports.
class BmaDartEngine {
  static const int nFeatures = 6;

  // Ensemble: list of (biasNet, noiseNet) weight sets
  final List<({List<_Layer> biasNet, List<_Layer> noiseNet})> _ensembleModels = [];
  Map<String, Map<String, double>>? _stats;
  bool _loaded = false;

  /// Load weights from assets. Call once before first inference.
  Future<void> load() async {
    if (_loaded) return;

    final weightsJson =
        await rootBundle.loadString('assets/models/bma_weights.json');
    final statsJson =
        await rootBundle.loadString('assets/models/bma_stats.json');

    final w = jsonDecode(weightsJson) as Map<String, dynamic>;
    final s = jsonDecode(statsJson) as Map<String, dynamic>;

    // Load ensemble models if available, otherwise single model
    if (w.containsKey('ensemble')) {
      final ensemble = w['ensemble'] as List;
      for (final m in ensemble) {
        final model = m as Map<String, dynamic>;
        _ensembleModels.add((
          biasNet: _parseLayers(model['bias_net'] as List),
          noiseNet: _parseLayers(model['noise_net'] as List),
        ));
      }
    } else {
      _ensembleModels.add((
        biasNet: _parseLayers(w['bias_net'] as List),
        noiseNet: _parseLayers(w['noise_net'] as List),
      ));
    }

    _stats = s.map((key, val) {
      final m = val as Map<String, dynamic>;
      return MapEntry(
        key,
        {'mean': (m['mean'] as num).toDouble(), 'std': (m['std'] as num).toDouble()},
      );
    });
    _loaded = true;
  }

  List<_Layer> _parseLayers(List raw) {
    return raw.map((l) {
      final layer = l as Map<String, dynamic>;
      final w = (layer['w'] as List)
          .map((row) => (row as List).map((v) => (v as num).toDouble()).toList())
          .toList();
      final b = (layer['b'] as List).map((v) => (v as num).toDouble()).toList();
      return _Layer(w, b);
    }).toList();
  }

  // ── Normalize/denormalize using training stats ──────────────────────────────

  static const List<String> _gfsCols = [
    'gfs_temp_c', 'gfs_pres_hpa', 'gfs_u_ms', 'gfs_v_ms',
    'gfs_precip_mm', 'gfs_rh_pct',
  ];
  static const List<String> _obsCols = [
    'obs_temp_c', 'obs_pres_hpa', 'obs_u_ms', 'obs_v_ms',
    'obs_precip_mm', 'obs_rh_pct',
  ];

  List<double> _normalize(List<double> raw, List<String> cols) {
    final stats = _stats!;
    return List.generate(cols.length, (i) {
      final s = stats[cols[i]]!;
      return (raw[i] - s['mean']!) / s['std']!;
    });
  }

  List<double> _denormalize(List<double> normed, List<String> cols) {
    final stats = _stats!;
    return List.generate(cols.length, (i) {
      final s = stats[cols[i]]!;
      return normed[i] * s['std']! + s['mean']!;
    });
  }

  /// Cyclic encoding of hour-of-day and day-of-year — matches training pipeline.
  static List<double> _temporalFeatures(DateTime t) {
    final hour = t.hour + t.minute / 60.0;
    final doy = t.difference(DateTime(t.year, 1, 1)).inDays + 1;
    return [
      math.sin(hour * 2 * math.pi / 24),
      math.cos(hour * 2 * math.pi / 24),
      math.sin(doy * 2 * math.pi / 365.25),
      math.cos(doy * 2 * math.pi / 365.25),
    ];
  }

  // ── Neural network forward passes ───────────────────────────────────────────

  /// [gfsForecast]  — raw (denormalized) GFS values [6]
  /// [obsFeatures]  — raw METAR/ERA5 observation [6], nullable
  /// [spatialEmbed] — [lat/90, lon/180]
  ///
  /// Returns ({mean, std}) in original physical units.
  ({List<double> mean, List<double> std}) update({
    required List<double> gfsForecast,
    required List<double>? obsFeatures,
    required List<double> spatialEmbed,
  }) {
    if (!_loaded) {
      // Fallback to conjugate prior if weights not yet loaded
      return _conjugateFallback(gfsForecast, obsFeatures);
    }

    final gfsNorm = _normalize(gfsForecast, _gfsCols);
    final temporal = _temporalFeatures(DateTime.now());
    final input = [...gfsNorm, ...spatialEmbed, ...temporal]; // [12]
    final obsNorm = obsFeatures != null ? _normalize(obsFeatures, _obsCols) : null;

    // Run each ensemble member and accumulate
    final ensembleMeans = List.generate(nFeatures, (_) => 0.0);
    final ensembleStds = List.generate(nFeatures, (_) => 0.0);
    final nModels = _ensembleModels.length;

    for (final model in _ensembleModels) {
      final biasNorm = _forward(model.biasNet, input, activation: _relu);
      final noiseNorm = _forward(model.noiseNet, input, activation: _relu,
          outputActivation: _softplus);

      // posterior_mean (normalized) = gfs_norm + bias
      final meanNorm = List.generate(nFeatures, (i) => gfsNorm[i] + biasNorm[i]);

      // If observation available, do a conjugate update in normalized space
      final postMeanNorm = List<double>.filled(nFeatures, 0.0);
      if (obsNorm != null) {
        for (int i = 0; i < nFeatures; i++) {
          final priorVar = noiseNorm[i] * noiseNorm[i];
          final obsVar = priorVar * 0.25; // METAR ~2× more precise than GFS
          final postVar = 1.0 / (1.0 / priorVar + 1.0 / obsVar);
          postMeanNorm[i] =
              postVar * (meanNorm[i] / priorVar + obsNorm[i] / obsVar);
        }
      } else {
        for (int i = 0; i < nFeatures; i++) {
          postMeanNorm[i] = meanNorm[i];
        }
      }

      for (int i = 0; i < nFeatures; i++) {
        ensembleMeans[i] += postMeanNorm[i];
        ensembleStds[i] += noiseNorm[i];
      }
    }

    // Average across ensemble members
    final avgMeanNorm = List.generate(nFeatures, (i) => ensembleMeans[i] / nModels);
    final avgNoiseNorm = List.generate(nFeatures, (i) => ensembleStds[i] / nModels);

    final postMean = _denormalize(avgMeanNorm, _obsCols);
    final stats = _stats!;
    final postStd = List.generate(nFeatures, (i) {
      return avgNoiseNorm[i] * stats[_obsCols[i]]!['std']!;
    });

    return (mean: postMean, std: postStd);
  }

  // ── Network primitives ──────────────────────────────────────────────────────

  List<double> _forward(
    List<_Layer> layers,
    List<double> x, {
    double Function(double) activation = _relu,
    double Function(double)? outputActivation,
  }) {
    List<double> h = x;
    for (int li = 0; li < layers.length; li++) {
      final layer = layers[li];
      final out = List<double>.filled(layer.bias.length, 0.0);
      for (int j = 0; j < layer.bias.length; j++) {
        double sum = layer.bias[j];
        for (int k = 0; k < h.length; k++) {
          sum += layer.weights[j][k] * h[k];
        }
        final isLast = (li == layers.length - 1);
        out[j] = isLast
            ? (outputActivation != null ? outputActivation(sum) : sum)
            : activation(sum);
      }
      h = out;
    }
    return h;
  }

  static double _relu(double x) => x < 0 ? 0.0 : x;

  static double _softplus(double x) {
    // log(1 + exp(x)) — numerically stable
    if (x > 20) return x;
    return math.log(1.0 + math.exp(x));
  }

  // ── Conjugate fallback (used before weights load) ───────────────────────────

  static const List<double> _priorStd = [2.0, 2.0, 3.0, 3.0, 2.0, 8.0];
  static const List<double> _obsStd   = [0.5, 0.5, 1.0, 1.0, 0.5, 3.0];

  ({List<double> mean, List<double> std}) _conjugateFallback(
    List<double> gfsForecast,
    List<double>? obsFeatures,
  ) {
    final postMean = List<double>.filled(nFeatures, 0.0);
    final postStd = List<double>.filled(nFeatures, 0.0);
    for (int i = 0; i < nFeatures; i++) {
      final pv = _priorStd[i] * _priorStd[i];
      final ov = _obsStd[i] * _obsStd[i];
      final postVar = 1.0 / (1.0 / pv + 1.0 / ov);
      postStd[i] = math.sqrt(postVar);
      postMean[i] = obsFeatures != null
          ? postVar * (gfsForecast[i] / pv + obsFeatures[i] / ov)
          : gfsForecast[i];
    }
    return (mean: postMean, std: postStd);
  }

  List<double> disagreement({
    required List<double> gfsForecast,
    required List<double> obsFeatures,
  }) =>
      List.generate(nFeatures, (i) => (gfsForecast[i] - obsFeatures[i]).abs());
}

class _Layer {
  final List<List<double>> weights;
  final List<double> bias;
  const _Layer(this.weights, this.bias);
}
