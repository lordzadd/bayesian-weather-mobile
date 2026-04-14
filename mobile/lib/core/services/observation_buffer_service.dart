/// In-memory ring buffer that accumulates hourly feature vectors per location.
///
/// Used by the LSTM engine to build input sequences. Each push adds one
/// [gfs_norm(6), lat/90, lon/180] vector. Sequences are keyed by a
/// coarse location bucket (2 decimal places ≈ 1.1 km resolution).
///
/// If fewer than [seqLen] observations are available for a location, the
/// oldest available observation is repeated to pad the front of the sequence.
class ObservationBufferService {
  static final ObservationBufferService instance = ObservationBufferService._();
  ObservationBufferService._();

  static const int defaultSeqLen = 6;

  // location_key → circular list of feature vectors
  final Map<String, List<List<double>>> _buffers = {};

  /// Records a new observation for the given location.
  ///
  /// [features] should be [gfs_features(6), lat/90, lon/180] — 8 elements.
  void push(double lat, double lon, List<double> features) {
    final key = _key(lat, lon);
    final buf = _buffers.putIfAbsent(key, () => []);
    buf.add(List<double>.from(features));
    // Keep only the last defaultSeqLen observations
    if (buf.length > defaultSeqLen) {
      buf.removeAt(0);
    }
  }

  /// Returns a [seqLen]-step sequence for [lat]/[lon], newest observation last.
  ///
  /// If fewer than [seqLen] observations exist, pads the front by repeating
  /// the earliest available observation.
  List<List<double>> getSequence(double lat, double lon,
      {int seqLen = defaultSeqLen}) {
    final key = _key(lat, lon);
    final buf = _buffers[key] ?? [];

    if (buf.isEmpty) {
      // No history at all — return zero vectors
      return List.generate(seqLen, (_) => List<double>.filled(8, 0.0));
    }

    if (buf.length >= seqLen) {
      return buf.sublist(buf.length - seqLen);
    }

    // Pad front with the oldest available observation
    final pad = List.generate(
      seqLen - buf.length,
      (_) => List<double>.from(buf.first),
    );
    return [...pad, ...buf];
  }

  /// Clears the buffer for a given location (e.g. when location changes).
  void clear(double lat, double lon) {
    _buffers.remove(_key(lat, lon));
  }

  String _key(double lat, double lon) {
    final la = (lat  * 100).round() / 100;
    final lo = (lon  * 100).round() / 100;
    return '${la}_$lo';
  }
}
