import '../models/lookback_result.dart';
import '../services/open_meteo_service.dart';
import '../services/prediction_history_dao.dart';
import '../../inference/bma_engine.dart';

/// Implements "predict the past, then check" validation.
///
/// 1. Fetch the GFS forecast from 1 hour ago
/// 2. Run BMA inference → prediction for T-1
/// 3. Fetch current observation → ground truth
/// 4. Compare prediction vs reality, compute per-variable MAE
/// 5. Persist to prediction_history for rolling accuracy stats
class LookbackService {
  final OpenMeteoService _weather;
  final BmaEngine _engine;
  final PredictionHistoryDao _historyDao;

  LookbackService({
    OpenMeteoService? weather,
    BmaEngine? engine,
    PredictionHistoryDao? historyDao,
  })  : _weather = weather ?? OpenMeteoService(),
        _engine = engine ?? BmaEngine.instance,
        _historyDao = historyDao ?? PredictionHistoryDao();

  /// Runs a single lookback validation cycle for the given location.
  ///
  /// Returns null if the API call or inference fails.
  Future<LookbackResult?> runLookback({
    required double lat,
    required double lon,
  }) async {
    // 1. Fetch GFS data from 1hr ago + current observation
    final data = await _weather.fetchForLookback(lat, lon, hoursAgo: 1);
    if (data == null) return null;

    final gfsFeatures = data.gfs;
    final obsFeatures = data.obs;
    final targetTime = data.targetTime;
    final spatial = [lat / 90.0, lon / 180.0];

    // 2. Run BMA inference on the historical GFS data
    final prediction = await _engine.infer(
      gfsForecast: gfsFeatures,
      obsFeatures: null, // no observation — we're predicting blind
      spatialEmbed: spatial,
    );

    // 3. Build the lookback result with errors
    final result = LookbackResult.fromVectors(
      targetTime: targetTime,
      prediction: prediction,
      obsFeatures: obsFeatures,
      gfsFeatures: gfsFeatures,
    );

    // 4. Persist to prediction_history with observation data
    final rowId = await _historyDao.insert(
      timestamp: DateTime.now(),
      targetTimestamp: targetTime,
      latitude: lat,
      longitude: lon,
      result: prediction,
    );

    await _historyDao.updateWithObservation(
      id: rowId,
      obsTemperatureC: result.obsTempC,
      obsPressureHpa: result.obsPressureHpa,
      obsWindSpeedMs: result.obsWindSpeedMs,
      obsPrecipMm: result.obsPrecipMm,
      obsHumidityPct: result.obsHumidityPct,
    );

    return result;
  }

  /// Computes rolling accuracy stats from validated prediction history.
  Future<AccuracyStats?> getRollingStats({
    Duration window = const Duration(hours: 24),
  }) async {
    final rows = await _historyDao.getValidated(window: window);
    if (rows.isEmpty) return null;

    double tempErrorSum = 0;
    double windErrorSum = 0;
    int count = 0;

    for (final row in rows) {
      final tempErr = row.tempErrorC;
      final windErr = row.windErrorMs;
      if (tempErr != null && windErr != null) {
        tempErrorSum += tempErr;
        windErrorSum += windErr;
        count++;
      }
    }

    if (count == 0) return null;

    return AccuracyStats(
      meanTempErrorC: tempErrorSum / count,
      meanWindErrorMs: windErrorSum / count,
      sampleCount: count,
      window: window,
    );
  }
}

/// Aggregated accuracy statistics over a time window.
class AccuracyStats {
  final double meanTempErrorC;
  final double meanWindErrorMs;
  final int sampleCount;
  final Duration window;

  const AccuracyStats({
    required this.meanTempErrorC,
    required this.meanWindErrorMs,
    required this.sampleCount,
    required this.window,
  });
}
