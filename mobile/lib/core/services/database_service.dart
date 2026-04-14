import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as p;
import 'dart:convert';

import '../models/forecast_result.dart';

/// SQLite-backed persistence layer.
///
/// Tables:
///   posterior_cache   — Variant B cache-gated inference (keyed by location+hour)
///   benchmark_log     — per-inference latency records
///   forecast_history  — every ForecastResult ever produced (for history screen)
///   lookback_accuracy — results of 1-hour lookback validation runs
class DatabaseService {
  DatabaseService._();
  static final DatabaseService instance = DatabaseService._();

  static const _dbName = 'bayesian_weather.db';
  static const _dbVersion = 4;

  Database? _db;

  Future<void> initialize() async {
    final dbPath = p.join(await getDatabasesPath(), _dbName);
    _db = await openDatabase(
      dbPath,
      version: _dbVersion,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE posterior_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        time_bucket INTEGER NOT NULL,
        result_json TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        UNIQUE(lat, lon, time_bucket)
      )
    ''');

    await db.execute('''
      CREATE INDEX idx_location_time ON posterior_cache(lat, lon, time_bucket)
    ''');

    await db.execute('''
      CREATE TABLE benchmark_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        variant TEXT NOT NULL,
        inference_ms INTEGER NOT NULL,
        cache_hit INTEGER NOT NULL,
        timestamp INTEGER NOT NULL
      )
    ''');

    await _createHistoryTable(db);
    await _createLookbackTable(db);
    await _createSavedLocationsTable(db);
  }

  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    if (oldVersion < 2) {
      await _createHistoryTable(db);
      await _createLookbackTable(db);
    }
    if (oldVersion < 3) {
      await _createSavedLocationsTable(db);
    }
    if (oldVersion < 4) {
      // Repair: version 2 from the old branch created prediction_history
      // instead of forecast_history + lookback_accuracy. Create any
      // missing tables idempotently.
      await _createHistoryTable(db);
      await _createLookbackTable(db);
      await _createSavedLocationsTable(db);
    }
  }

  Future<void> _createHistoryTable(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS forecast_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        temp_mean REAL NOT NULL,
        temp_std REAL NOT NULL,
        pressure_hpa REAL NOT NULL,
        wind_speed_ms REAL NOT NULL,
        wind_speed_std REAL NOT NULL,
        precip_mm REAL NOT NULL,
        humidity_pct REAL NOT NULL,
        inference_source TEXT NOT NULL,
        model_variant TEXT NOT NULL DEFAULT 'bma'
      )
    ''');
    await db.execute(
        'CREATE INDEX IF NOT EXISTS idx_history_ts ON forecast_history(timestamp DESC)');
  }

  Future<void> _createLookbackTable(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS lookback_accuracy (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        model_variant TEXT NOT NULL,
        temp_ae REAL NOT NULL,
        pressure_ae REAL NOT NULL,
        wind_speed_ae REAL NOT NULL,
        humidity_ae REAL NOT NULL,
        precip_ae REAL NOT NULL,
        within_2sigma INTEGER NOT NULL
      )
    ''');
  }

  Database get _database {
    if (_db == null) throw StateError('DatabaseService not initialized');
    return _db!;
  }

  /// Stores a posterior result for the given location and time bucket (hour).
  Future<void> savePosterior({
    required double lat,
    required double lon,
    required DateTime time,
    required ForecastResult result,
  }) async {
    final bucket = _timeBucket(time);
    await _database.insert(
      'posterior_cache',
      {
        'lat': lat,
        'lon': lon,
        'time_bucket': bucket,
        'result_json': jsonEncode(result.toJson()),
        'created_at': DateTime.now().millisecondsSinceEpoch,
      },
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  /// Returns cached posterior if available for the given location + hour, else null.
  Future<ForecastResult?> getCachedPosterior({
    required double lat,
    required double lon,
    required DateTime time,
  }) async {
    final bucket = _timeBucket(time);
    final rows = await _database.query(
      'posterior_cache',
      where: 'lat = ? AND lon = ? AND time_bucket = ?',
      whereArgs: [lat, lon, bucket],
      limit: 1,
    );
    if (rows.isEmpty) return null;
    return ForecastResult.fromJson(
      jsonDecode(rows.first['result_json'] as String) as Map<String, dynamic>,
    );
  }

  /// Logs a benchmark sample for latency analysis.
  Future<void> logBenchmark({
    required String variant,
    required int inferenceMs,
    required bool cacheHit,
  }) async {
    await _database.insert('benchmark_log', {
      'variant': variant,
      'inference_ms': inferenceMs,
      'cache_hit': cacheHit ? 1 : 0,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    });
  }

  Future<List<Map<String, dynamic>>> getBenchmarkLog() async {
    return _database.query('benchmark_log', orderBy: 'timestamp DESC');
  }

  Future<void> clearExpiredCache({Duration maxAge = const Duration(hours: 6)}) async {
    final cutoff = DateTime.now().subtract(maxAge).millisecondsSinceEpoch;
    await _database.delete(
      'posterior_cache',
      where: 'created_at < ?',
      whereArgs: [cutoff],
    );
  }

  // ── Forecast history ──────────────────────────────────────────────────────

  /// Appends a forecast to the history table and prunes oldest rows above [limit].
  Future<void> insertForecastHistory({
    required double lat,
    required double lon,
    required ForecastResult result,
    String modelVariant = 'bma',
    int limit = 500,
  }) async {
    await _database.insert('forecast_history', {
      'timestamp': result.computedAt.millisecondsSinceEpoch,
      'lat': lat,
      'lon': lon,
      'temp_mean': result.temperatureC,
      'temp_std': result.temperatureStd,
      'pressure_hpa': result.surfacePressureHpa,
      'wind_speed_ms': result.windSpeedMs,
      'wind_speed_std': result.windSpeedStd,
      'precip_mm': result.precipitationMm,
      'humidity_pct': result.relativeHumidityPct,
      'inference_source': result.source.name,
      'model_variant': modelVariant,
    });
    // Purge oldest rows beyond the retention limit
    await _database.rawDelete('''
      DELETE FROM forecast_history
      WHERE id NOT IN (
        SELECT id FROM forecast_history ORDER BY timestamp DESC LIMIT $limit
      )
    ''');
  }

  /// Returns up to [limit] most-recent history rows, newest first.
  Future<List<Map<String, dynamic>>> getRecentHistory({int limit = 100}) async {
    return _database.query(
      'forecast_history',
      orderBy: 'timestamp DESC',
      limit: limit,
    );
  }

  // ── Lookback accuracy ─────────────────────────────────────────────────────

  Future<void> insertLookbackAccuracy({
    required double lat,
    required double lon,
    required String modelVariant,
    required double tempAe,
    required double pressureAe,
    required double windSpeedAe,
    required double humidityAe,
    required double precipAe,
    required bool within2Sigma,
  }) async {
    await _database.insert('lookback_accuracy', {
      'timestamp': DateTime.now().millisecondsSinceEpoch,
      'lat': lat,
      'lon': lon,
      'model_variant': modelVariant,
      'temp_ae': tempAe,
      'pressure_ae': pressureAe,
      'wind_speed_ae': windSpeedAe,
      'humidity_ae': humidityAe,
      'precip_ae': precipAe,
      'within_2sigma': within2Sigma ? 1 : 0,
    });
  }

  /// Returns lookback rows newest first, grouped summary per [modelVariant].
  Future<List<Map<String, dynamic>>> getLookbackSummary({int limit = 20}) async {
    return _database.rawQuery('''
      SELECT
        model_variant,
        AVG(temp_ae)       AS avg_temp_ae,
        AVG(pressure_ae)   AS avg_pressure_ae,
        AVG(wind_speed_ae) AS avg_wind_ae,
        AVG(humidity_ae)   AS avg_humidity_ae,
        AVG(precip_ae)     AS avg_precip_ae,
        AVG(within_2sigma) AS coverage_rate,
        COUNT(*)           AS run_count
      FROM (
        SELECT * FROM lookback_accuracy ORDER BY timestamp DESC LIMIT $limit
      )
      GROUP BY model_variant
    ''');
  }

  Future<List<Map<String, dynamic>>> getLookbackRuns({int limit = 50}) async {
    return _database.query(
      'lookback_accuracy',
      orderBy: 'timestamp DESC',
      limit: limit,
    );
  }

  // ── Saved locations ───────────────────────────────────────────────────────

  Future<void> _createSavedLocationsTable(Database db) async {
    await db.execute('''
      CREATE TABLE IF NOT EXISTS saved_locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        created_at INTEGER NOT NULL
      )
    ''');
  }

  Future<int> insertSavedLocation({
    required String name,
    required double lat,
    required double lon,
  }) async {
    return _database.insert('saved_locations', {
      'name': name,
      'lat': lat,
      'lon': lon,
      'created_at': DateTime.now().millisecondsSinceEpoch,
    });
  }

  Future<List<Map<String, dynamic>>> getSavedLocations() async {
    return _database.query('saved_locations', orderBy: 'created_at ASC');
  }

  Future<void> deleteSavedLocation(int id) async {
    await _database.delete(
      'saved_locations',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  /// Buckets a DateTime to the nearest hour for cache keying.
  int _timeBucket(DateTime t) =>
      DateTime(t.year, t.month, t.day, t.hour).millisecondsSinceEpoch;
}
