import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import '../core/models/forecast_result.dart';

/// Native ExecuTorch inference bridge via FFI.
///
/// On first call, loads the .pte model from app assets into the ExecuTorch
/// runtime. Subsequent calls reuse the loaded module.
///
/// Variant A: GPU-accelerated path — calls [infer] on every update.
/// Variant B: Cache-optimized path — [ForecastCacheService] gates calls here.
class BmaEngine {
  static final BmaEngine _instance = BmaEngine._();
  static BmaEngine get instance => _instance;
  BmaEngine._();

  bool _initialized = false;

  // FFI bindings — populated after _loadNativeLibrary()
  late final DynamicLibrary _lib;
  late final _BmaLoadFn _bmaLoad;
  late final _BmaInferFn _bmaInfer;
  late final _BmaFreeFn _bmaFree;
  late final Pointer<Void> _moduleHandle;

  static const int _nFeatures = 6;

  Future<void> initialize() async {
    if (_initialized) return;
    _loadNativeLibrary();
    final modelPath = await _resolveModelPath();
    _moduleHandle = _bmaLoad(modelPath.toNativeUtf8().cast());
    if (_moduleHandle == nullptr) {
      throw StateError('ExecuTorch failed to load BMA model from $modelPath');
    }
    _initialized = true;
  }

  void _loadNativeLibrary() {
    final libName = Platform.isAndroid ? 'libbma_executorch.so'
        : Platform.isIOS ? 'bma_executorch.framework/bma_executorch'
        : throw UnsupportedError('Unsupported platform');
    _lib = DynamicLibrary.open(libName);

    _bmaLoad = _lib.lookupFunction<
        Pointer<Void> Function(Pointer<Char>),
        Pointer<Void> Function(Pointer<Char>)>('bma_load');

    _bmaInfer = _lib.lookupFunction<
        Void Function(Pointer<Void>, Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Float>),
        void Function(Pointer<Void>, Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Float>)>('bma_infer');

    _bmaFree = _lib.lookupFunction<
        Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('bma_free');
  }

  Future<String> _resolveModelPath() async {
    final dir = await getApplicationDocumentsDirectory();
    return p.join(dir.path, 'models', 'bma_model.pte');
  }

  /// Runs a full posterior inference pass.
  ///
  /// [gfsForecast]  — 6-element feature vector from GFS grid forecast
  /// [spatialEmbed] — 2-element [lat, lon] normalized to [-1, 1]
  ///
  /// Returns [ForecastResult] with posterior mean and std dev per variable.
  Future<ForecastResult> infer({
    required List<double> gfsForecast,
    required List<double> spatialEmbed,
  }) async {
    if (!_initialized) await initialize();

    final gfsPtr = _toFloatPointer(gfsForecast);
    final spatialPtr = _toFloatPointer(spatialEmbed);
    final meanPtr = calloc<Float>(_nFeatures);
    final stdPtr = calloc<Float>(_nFeatures);

    final stopwatch = Stopwatch()..start();
    _bmaInfer(_moduleHandle, gfsPtr, spatialPtr, meanPtr, stdPtr);
    stopwatch.stop();

    final mean = List.generate(_nFeatures, (i) => meanPtr[i].toDouble());
    final std = List.generate(_nFeatures, (i) => stdPtr[i].toDouble());

    calloc.free(gfsPtr);
    calloc.free(spatialPtr);
    calloc.free(meanPtr);
    calloc.free(stdPtr);

    return ForecastResult(
      temperatureC: mean[0],
      temperatureStd: std[0],
      windSpeedMs: _magnitude(mean[2], mean[3]),
      windSpeedStd: _magnitude(std[2], std[3]),
      surfacePressureHpa: mean[1],
      relativeHumidityPct: mean[5],
      precipitationMm: mean[4],
      computedAt: DateTime.now(),
      source: InferenceSource.gpu,
    );
  }

  Pointer<Float> _toFloatPointer(List<double> values) {
    final ptr = calloc<Float>(values.length);
    for (int i = 0; i < values.length; i++) {
      ptr[i] = values[i];
    }
    return ptr;
  }

  double _magnitude(double u, double v) => (u * u + v * v) < 0 ? 0 : _sqrt(u * u + v * v);

  double _sqrt(double x) {
    // Newton-Raphson for tree-shaking; replace with dart:math in production
    if (x <= 0) return 0;
    double g = x / 2;
    for (int i = 0; i < 10; i++) {
      g = (g + x / g) / 2;
    }
    return g;
  }

  void dispose() {
    if (_initialized) {
      _bmaFree(_moduleHandle);
      _initialized = false;
    }
  }
}

// FFI type aliases
typedef _BmaLoadFn = Pointer<Void> Function(Pointer<Char>);
typedef _BmaInferFn = void Function(Pointer<Void>, Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Float>);
typedef _BmaFreeFn = void Function(Pointer<Void>);
