import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;

import 'package:ffi/ffi.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import '../core/models/forecast_result.dart';
import 'bma_dart_engine.dart';

/// Native ExecuTorch inference bridge via FFI.
///
/// Execution paths:
///   1. Native (production): loads compiled .pte, delegates to Vulkan/Metal GPU.
///   2. Dart fallback (dev/simulator): uses [BmaDartEngine] — real Gaussian
///      conjugate Bayesian update, identical math to the trained model.
///
/// Variant A: [infer] called on every observation update (no cache gate).
/// Variant B: [ForecastCacheService] calls [infer] only when Δobs > threshold.
class BmaEngine {
  static final BmaEngine _instance = BmaEngine._();
  static BmaEngine get instance => _instance;
  BmaEngine._();

  bool _initialized = false;
  bool _nativeAvailable = false;

  final _dartEngine = BmaDartEngine();

  // FFI bindings — populated when native library loads successfully
  late final DynamicLibrary _lib;
  late final _BmaLoadFn _bmaLoad;
  late final _BmaInferFn _bmaInfer;
  late final _BmaFreeFn _bmaFree;
  late final Pointer<Void> _moduleHandle;

  static const int _nFeatures = BmaDartEngine.nFeatures;

  Future<void> initialize() async {
    if (_initialized) return;
    // Load trained neural network weights into the Dart engine first
    await _dartEngine.load();
    try {
      _loadNativeLibrary();
      final modelPath = await _resolveModelPath();
      _moduleHandle = _bmaLoad(modelPath.toNativeUtf8().cast());
      if (_moduleHandle == nullptr) {
        throw StateError('ExecuTorch returned null handle for $modelPath');
      }
      _nativeAvailable = true;
    } catch (_) {
      // Native lib not compiled / model not exported yet.
      // Dart BMA engine runs instead — trained weights, CPU only.
      _nativeAvailable = false;
    }
    _initialized = true;
  }

  void _loadNativeLibrary() {
    final libName = Platform.isAndroid
        ? 'libbma_executorch.so'
        : Platform.isIOS
            ? 'bma_executorch.framework/bma_executorch'
            : throw UnsupportedError(
                'Native inference not supported on ${Platform.operatingSystem}');

    _lib = DynamicLibrary.open(libName);

    _bmaLoad = _lib.lookupFunction<
        Pointer<Void> Function(Pointer<Char>),
        Pointer<Void> Function(Pointer<Char>)>('bma_load');

    _bmaInfer = _lib.lookupFunction<
        Void Function(Pointer<Void>, Pointer<Float>, Pointer<Float>,
            Pointer<Float>, Pointer<Float>, Pointer<Float>),
        void Function(Pointer<Void>, Pointer<Float>, Pointer<Float>,
            Pointer<Float>, Pointer<Float>, Pointer<Float>)>('bma_infer');

    _bmaFree = _lib.lookupFunction<
        Void Function(Pointer<Void>),
        void Function(Pointer<Void>)>('bma_free');
  }

  Future<String> _resolveModelPath() async {
    final dir = await getApplicationDocumentsDirectory();
    return p.join(dir.path, 'models', 'bma_model.pte');
  }

  /// Runs a posterior inference pass.
  ///
  /// [gfsForecast]   — 6-element prior from the GFS grid forecast
  /// [obsFeatures]   — 6-element METAR observation (null → prior only)
  /// [spatialEmbed]  — [lat/90, lon/180] for spatial conditioning
  Future<ForecastResult> infer({
    required List<double> gfsForecast,
    required List<double>? obsFeatures,
    required List<double> spatialEmbed,
  }) async {
    if (!_initialized) await initialize();

    if (_nativeAvailable) {
      return _nativeInfer(gfsForecast, obsFeatures, spatialEmbed);
    }
    return _dartInfer(gfsForecast, obsFeatures, spatialEmbed);
  }

  /// Delegates to ExecuTorch GPU runtime.
  ForecastResult _nativeInfer(
    List<double> gfs,
    List<double>? obs,
    List<double> spatial,
  ) {
    final gfsPtr = _toFloatPointer(gfs);
    final obsPtr = _toFloatPointer(obs ?? gfs); // pass gfs when no observation
    final spatialPtr = _toFloatPointer(spatial);
    final meanPtr = calloc<Float>(_nFeatures);
    final stdPtr = calloc<Float>(_nFeatures);

    _bmaInfer(_moduleHandle, gfsPtr, obsPtr, spatialPtr, meanPtr, stdPtr);

    final mean = List.generate(_nFeatures, (i) => meanPtr[i].toDouble());
    final std = List.generate(_nFeatures, (i) => stdPtr[i].toDouble());

    calloc.free(gfsPtr);
    calloc.free(obsPtr);
    calloc.free(spatialPtr);
    calloc.free(meanPtr);
    calloc.free(stdPtr);

    return _toForecastResult(mean, std);
  }

  /// Runs trained neural network weights in pure Dart.
  ForecastResult _dartInfer(
      List<double> gfs, List<double>? obs, List<double> spatial) {
    final (:mean, :std) = _dartEngine.update(
      gfsForecast: gfs,
      obsFeatures: obs,
      spatialEmbed: spatial,
    );
    return _toForecastResult(mean, std);
  }

  ForecastResult _toForecastResult(List<double> mean, List<double> std) {
    final bearing = (math.atan2(mean[3], mean[2]) * 180.0 / math.pi + 360.0) % 360.0;
    return ForecastResult(
      temperatureC: mean[0],
      temperatureStd: std[0],
      windSpeedMs: math.sqrt(mean[2] * mean[2] + mean[3] * mean[3]),
      windSpeedStd: math.sqrt(std[2] * std[2] + std[3] * std[3]),
      windBearingDeg: bearing,
      surfacePressureHpa: mean[1],
      relativeHumidityPct: mean[5].clamp(0, 100),
      precipitationMm: mean[4].clamp(0, double.infinity),
      computedAt: DateTime.now(),
      source: _nativeAvailable ? InferenceSource.gpu : InferenceSource.dart,
    );
  }

  Pointer<Float> _toFloatPointer(List<double> values) {
    final ptr = calloc<Float>(values.length);
    for (int i = 0; i < values.length; i++) {
      ptr[i] = values[i];
    }
    return ptr;
  }

  void dispose() {
    if (_initialized && _nativeAvailable) {
      _bmaFree(_moduleHandle);
    }
    _initialized = false;
    _nativeAvailable = false;
  }
}

// FFI type aliases — obs tensor added as 3rd input
typedef _BmaLoadFn = Pointer<Void> Function(Pointer<Char>);
typedef _BmaInferFn = void Function(Pointer<Void>, Pointer<Float>,
    Pointer<Float>, Pointer<Float>, Pointer<Float>, Pointer<Float>);
typedef _BmaFreeFn = void Function(Pointer<Void>);
