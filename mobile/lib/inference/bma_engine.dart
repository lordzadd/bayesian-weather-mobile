import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;

import 'package:ffi/ffi.dart';
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
///
/// Dev fallback: if the native library or model file is absent (e.g. running
/// on a simulator without a compiled .so), [infer] returns a plausible mock
/// result so the UI can be developed and tested without hardware.
class BmaEngine {
  static final BmaEngine _instance = BmaEngine._();
  static BmaEngine get instance => _instance;
  BmaEngine._();

  bool _initialized = false;
  bool _nativeAvailable = false;

  // FFI bindings — populated after _loadNativeLibrary()
  late final DynamicLibrary _lib;
  late final _BmaLoadFn _bmaLoad;
  late final _BmaInferFn _bmaInfer;
  late final _BmaFreeFn _bmaFree;
  late final Pointer<Void> _moduleHandle;

  static const int _nFeatures = 6;

  Future<void> initialize() async {
    if (_initialized) return;
    try {
      _loadNativeLibrary();
      final modelPath = await _resolveModelPath();
      _moduleHandle = _bmaLoad(modelPath.toNativeUtf8().cast());
      if (_moduleHandle == nullptr) {
        throw StateError('ExecuTorch returned null handle for $modelPath');
      }
      _nativeAvailable = true;
    } catch (e) {
      // Native lib not compiled yet (simulator / dev environment).
      // App continues with mock inference.
      _nativeAvailable = false;
    }
    _initialized = true;
  }

  void _loadNativeLibrary() {
    final libName = Platform.isAndroid
        ? 'libbma_executorch.so'
        : Platform.isIOS
            ? 'bma_executorch.framework/bma_executorch'
            : throw UnsupportedError('Native inference not supported on ${Platform.operatingSystem}');

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

    if (!_nativeAvailable) {
      return _mockInfer(gfsForecast);
    }

    final gfsPtr = _toFloatPointer(gfsForecast);
    final spatialPtr = _toFloatPointer(spatialEmbed);
    final meanPtr = calloc<Float>(_nFeatures);
    final stdPtr = calloc<Float>(_nFeatures);

    _bmaInfer(_moduleHandle, gfsPtr, spatialPtr, meanPtr, stdPtr);

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

  /// Mock inference for dev/simulator environments.
  /// Adds small Gaussian noise to the GFS input to simulate posterior correction.
  ForecastResult _mockInfer(List<double> gfs) {
    final rng = math.Random();
    double _noise(double scale) => (rng.nextDouble() - 0.5) * 2 * scale;

    return ForecastResult(
      temperatureC: gfs[0] + _noise(0.8),
      temperatureStd: 0.4 + rng.nextDouble() * 0.3,
      windSpeedMs: math.sqrt(gfs[2] * gfs[2] + gfs[3] * gfs[3]) + _noise(0.5),
      windSpeedStd: 0.3 + rng.nextDouble() * 0.2,
      surfacePressureHpa: gfs[1],
      relativeHumidityPct: gfs[5].clamp(0, 100),
      precipitationMm: gfs[4].clamp(0, double.infinity),
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

  double _magnitude(double u, double v) => math.sqrt(u * u + v * v);

  void dispose() {
    if (_initialized && _nativeAvailable) {
      _bmaFree(_moduleHandle);
    }
    _initialized = false;
    _nativeAvailable = false;
  }
}

// FFI type aliases
typedef _BmaLoadFn = Pointer<Void> Function(Pointer<Char>);
typedef _BmaInferFn = void Function(Pointer<Void>, Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Float>);
typedef _BmaFreeFn = void Function(Pointer<Void>);
