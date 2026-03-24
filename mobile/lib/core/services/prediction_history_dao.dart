import '../models/forecast_result.dart';
import 'database_service.dart';

/// Data access object for the prediction_history table.
///
/// Persists every forecast the app generates and later pairs predictions
/// with actual observations via the lookback validation service.
class PredictionHistoryDao {
  final DatabaseService _db;

  PredictionHistoryDao([DatabaseService? db])
      : _db = db ?? DatabaseService.instance;

  /// Inserts a new prediction record. Returns the row id.
  Future<int> insert({
    required DateTime timestamp,
    required DateTime targetTimestamp,
    required double latitude,
    required double longitude,
    String? stationId,
    required ForecastResult result,
  }) {
    return _db.database.insert('prediction_history', {
      'timestamp': timestamp.toUtc().toIso8601String(),
      'target_timestamp': targetTimestamp.toUtc().toIso8601String(),
      'latitude': latitude,
      'longitude': longitude,
      'station_id': stationId,
      'variant': result.source.name,
      'pred_temp_c': result.temperatureC,
      'pred_temp_std': result.temperatureStd,
      'pred_wind_speed_ms': result.windSpeedMs,
      'pred_wind_speed_std': result.windSpeedStd,
      'pred_pressure_hpa': result.surfacePressureHpa,
      'pred_humidity_pct': result.relativeHumidityPct,
      'pred_precip_mm': result.precipitationMm,
    });
  }

  /// Returns the most recent [limit] predictions, newest first.
  Future<List<PredictionHistoryRow>> getRecent({int limit = 50}) async {
    final rows = await _db.database.query(
      'prediction_history',
      orderBy: 'timestamp DESC',
      limit: limit,
    );
    return rows.map(PredictionHistoryRow.fromMap).toList();
  }

  /// Returns predictions that haven't been validated yet (no observation data).
  Future<List<PredictionHistoryRow>> getUnvalidated() async {
    final rows = await _db.database.query(
      'prediction_history',
      where: 'validated_at IS NULL',
      orderBy: 'timestamp DESC',
    );
    return rows.map(PredictionHistoryRow.fromMap).toList();
  }

  /// Returns validated predictions within the given duration, newest first.
  Future<List<PredictionHistoryRow>> getValidated({
    Duration window = const Duration(hours: 24),
    int limit = 100,
  }) async {
    final cutoff =
        DateTime.now().subtract(window).toUtc().toIso8601String();
    final rows = await _db.database.query(
      'prediction_history',
      where: 'validated_at IS NOT NULL AND timestamp > ?',
      whereArgs: [cutoff],
      orderBy: 'timestamp DESC',
      limit: limit,
    );
    return rows.map(PredictionHistoryRow.fromMap).toList();
  }

  /// Fills in the observation columns for a prediction after lookback validation.
  Future<void> updateWithObservation({
    required int id,
    required double obsTemperatureC,
    required double obsPressureHpa,
    required double obsWindSpeedMs,
    required double obsPrecipMm,
    required double obsHumidityPct,
  }) {
    return _db.database.update(
      'prediction_history',
      {
        'obs_temp_c': obsTemperatureC,
        'obs_pressure_hpa': obsPressureHpa,
        'obs_wind_speed_ms': obsWindSpeedMs,
        'obs_precip_mm': obsPrecipMm,
        'obs_humidity_pct': obsHumidityPct,
        'validated_at': DateTime.now().toUtc().toIso8601String(),
      },
      where: 'id = ?',
      whereArgs: [id],
    );
  }
}

/// Read-only row from the prediction_history table.
class PredictionHistoryRow {
  final int id;
  final DateTime timestamp;
  final DateTime targetTimestamp;
  final double latitude;
  final double longitude;
  final String? stationId;
  final String variant;

  // Predicted values
  final double? predTempC;
  final double? predTempStd;
  final double? predWindSpeedMs;
  final double? predWindSpeedStd;
  final double? predPressureHpa;
  final double? predHumidityPct;
  final double? predPrecipMm;

  // Observation values (null until validated)
  final double? obsTempC;
  final double? obsPressureHpa;
  final double? obsWindSpeedMs;
  final double? obsPrecipMm;
  final double? obsHumidityPct;
  final DateTime? validatedAt;

  const PredictionHistoryRow({
    required this.id,
    required this.timestamp,
    required this.targetTimestamp,
    required this.latitude,
    required this.longitude,
    this.stationId,
    required this.variant,
    this.predTempC,
    this.predTempStd,
    this.predWindSpeedMs,
    this.predWindSpeedStd,
    this.predPressureHpa,
    this.predHumidityPct,
    this.predPrecipMm,
    this.obsTempC,
    this.obsPressureHpa,
    this.obsWindSpeedMs,
    this.obsPrecipMm,
    this.obsHumidityPct,
    this.validatedAt,
  });

  bool get isValidated => validatedAt != null;

  /// Absolute temperature error in °C (null if not validated).
  double? get tempErrorC =>
      (predTempC != null && obsTempC != null)
          ? (predTempC! - obsTempC!).abs()
          : null;

  /// Absolute wind speed error in m/s (null if not validated).
  double? get windErrorMs =>
      (predWindSpeedMs != null && obsWindSpeedMs != null)
          ? (predWindSpeedMs! - obsWindSpeedMs!).abs()
          : null;

  factory PredictionHistoryRow.fromMap(Map<String, dynamic> m) {
    return PredictionHistoryRow(
      id: m['id'] as int,
      timestamp: DateTime.parse(m['timestamp'] as String),
      targetTimestamp: DateTime.parse(m['target_timestamp'] as String),
      latitude: (m['latitude'] as num).toDouble(),
      longitude: (m['longitude'] as num).toDouble(),
      stationId: m['station_id'] as String?,
      variant: m['variant'] as String,
      predTempC: (m['pred_temp_c'] as num?)?.toDouble(),
      predTempStd: (m['pred_temp_std'] as num?)?.toDouble(),
      predWindSpeedMs: (m['pred_wind_speed_ms'] as num?)?.toDouble(),
      predWindSpeedStd: (m['pred_wind_speed_std'] as num?)?.toDouble(),
      predPressureHpa: (m['pred_pressure_hpa'] as num?)?.toDouble(),
      predHumidityPct: (m['pred_humidity_pct'] as num?)?.toDouble(),
      predPrecipMm: (m['pred_precip_mm'] as num?)?.toDouble(),
      obsTempC: (m['obs_temp_c'] as num?)?.toDouble(),
      obsPressureHpa: (m['obs_pressure_hpa'] as num?)?.toDouble(),
      obsWindSpeedMs: (m['obs_wind_speed_ms'] as num?)?.toDouble(),
      obsPrecipMm: (m['obs_precip_mm'] as num?)?.toDouble(),
      obsHumidityPct: (m['obs_humidity_pct'] as num?)?.toDouble(),
      validatedAt: m['validated_at'] != null
          ? DateTime.parse(m['validated_at'] as String)
          : null,
    );
  }
}
