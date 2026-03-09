import 'dart:math' as math;

/// Pure-Dart Gaussian conjugate Bayesian update.
///
/// Implements the BMA core math that the trained ExecuTorch model will also
/// perform — so the app produces real posterior estimates even before the
/// compiled native library is available.
///
/// Model per variable i:
///   Prior:       θ_i  ~ N(gfs_i,  σ_prior_i²)
///   Likelihood:  obs_i ~ N(θ_i,  σ_obs_i²)
///   Posterior:   θ_i | obs_i ~ N(μ_post_i, σ_post_i²)
///
/// Closed-form update (conjugate Gaussian):
///   σ_post_i² = 1 / (1/σ_prior_i² + 1/σ_obs_i²)
///   μ_post_i  = σ_post_i² * (gfs_i/σ_prior_i² + obs_i/σ_obs_i²)
///
/// When obs == gfs (fallback case), the posterior equals the prior.
/// When they disagree, the posterior is pulled toward the more precise source.
class BmaDartEngine {
  /// GFS forecast error std dev per variable, estimated from ERA5 validation.
  /// Order: [temp_C, pressure_hPa, u_wind_ms, v_wind_ms, precip_mm, rh_pct]
  static const List<double> _priorStd = [
    2.0,   // temperature  — typical GFS MAE ~1.5–2.5°C
    2.0,   // pressure     — GFS SLP error ~2 hPa
    3.0,   // u-wind       — wind component error ~2–4 m/s
    3.0,   // v-wind
    2.0,   // precipitation — high uncertainty
    8.0,   // relative humidity — GFS RH error ~8–12%
  ];

  /// METAR/ASOS instrument accuracy per variable.
  static const List<double> _obsStd = [
    0.5,   // temperature  — WMO Class 1 sensor: ±0.3°C
    0.5,   // pressure     — digital barometer: ±0.5 hPa
    1.0,   // u-wind       — cup anemometer: ±1 m/s
    1.0,   // v-wind
    0.5,   // precipitation — tipping bucket: ±0.5 mm
    3.0,   // humidity     — capacitive sensor: ±3%
  ];

  static const int nFeatures = 6;

  /// Computes the Gaussian conjugate posterior.
  ///
  /// [gfsForecast]   — prior mean, shape [6]
  /// [obsFeatures]   — METAR observation, shape [6]; if null, returns prior
  ///
  /// Returns ({mean, std}) with shape [6] each.
  ({List<double> mean, List<double> std}) update({
    required List<double> gfsForecast,
    required List<double>? obsFeatures,
  }) {
    final postMean = List<double>.filled(nFeatures, 0.0);
    final postStd = List<double>.filled(nFeatures, 0.0);

    for (int i = 0; i < nFeatures; i++) {
      final priorVar = _priorStd[i] * _priorStd[i];
      final obsVar = _obsStd[i] * _obsStd[i];

      final postVar = 1.0 / (1.0 / priorVar + 1.0 / obsVar);
      postStd[i] = math.sqrt(postVar);

      if (obsFeatures != null) {
        postMean[i] =
            postVar * (gfsForecast[i] / priorVar + obsFeatures[i] / obsVar);
      } else {
        // No observation: posterior collapses to prior
        postMean[i] = gfsForecast[i];
      }
    }

    return (mean: postMean, std: postStd);
  }

  /// Posterior predictive: uncertainty increases when GFS and obs disagree.
  ///
  /// Returns an additional "disagreement" term added to the posterior std —
  /// useful for visualizing heatmap opacity (high disagreement → more uncertain).
  List<double> disagreement({
    required List<double> gfsForecast,
    required List<double> obsFeatures,
  }) {
    return List.generate(
      nFeatures,
      (i) => (gfsForecast[i] - obsFeatures[i]).abs(),
    );
  }
}
