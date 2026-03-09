import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as p;
import 'dart:convert';

import '../models/forecast_result.dart';

/// SQLite-backed persistence layer for Variant B (cache-optimized) inference.
///
/// Stores serialized [ForecastResult] records keyed by (lat, lon, timestamp_bucket).
/// The significance threshold check is performed by [ForecastCacheService].
class DatabaseService {
  DatabaseService._();
  static final DatabaseService instance = DatabaseService._();

  static const _dbName = 'bayesian_weather.db';
  static const _dbVersion = 1;

  Database? _db;

  Future<void> initialize() async {
    final dbPath = p.join(await getDatabasesPath(), _dbName);
    _db = await openDatabase(
      dbPath,
      version: _dbVersion,
      onCreate: _onCreate,
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

  /// Buckets a DateTime to the nearest hour for cache keying.
  int _timeBucket(DateTime t) =>
      DateTime(t.year, t.month, t.day, t.hour).millisecondsSinceEpoch;
}
