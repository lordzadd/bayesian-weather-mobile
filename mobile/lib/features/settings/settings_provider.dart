import 'package:flutter_riverpod/flutter_riverpod.dart';

enum InferenceVariant { gpuAlways, cacheOptimized }

/// Which neural architecture to use for bias correction.
enum ModelVariant { bma, linear, lstm, fusion }

class AppSettings {
  final InferenceVariant variant;
  final ModelVariant modelVariant;
  final double tempThresholdC;
  final double windThresholdMs;
  final bool notificationsEnabled;
  final double alertTempChangeC;

  const AppSettings({
    this.variant = InferenceVariant.cacheOptimized,
    this.modelVariant = ModelVariant.fusion,
    this.tempThresholdC = 0.2,
    this.windThresholdMs = 0.5,
    this.notificationsEnabled = false,
    this.alertTempChangeC = 3.0,
  });

  AppSettings copyWith({
    InferenceVariant? variant,
    ModelVariant? modelVariant,
    double? tempThresholdC,
    double? windThresholdMs,
    bool? notificationsEnabled,
    double? alertTempChangeC,
  }) {
    return AppSettings(
      variant: variant ?? this.variant,
      modelVariant: modelVariant ?? this.modelVariant,
      tempThresholdC: tempThresholdC ?? this.tempThresholdC,
      windThresholdMs: windThresholdMs ?? this.windThresholdMs,
      notificationsEnabled: notificationsEnabled ?? this.notificationsEnabled,
      alertTempChangeC: alertTempChangeC ?? this.alertTempChangeC,
    );
  }
}

class SettingsNotifier extends Notifier<AppSettings> {
  @override
  AppSettings build() => const AppSettings();

  void setVariant(InferenceVariant v) =>
      state = state.copyWith(variant: v);

  void setModelVariant(ModelVariant v) =>
      state = state.copyWith(modelVariant: v);

  void setTempThreshold(double v) =>
      state = state.copyWith(tempThresholdC: v);

  void setWindThreshold(double v) =>
      state = state.copyWith(windThresholdMs: v);

  void setNotificationsEnabled(bool v) =>
      state = state.copyWith(notificationsEnabled: v);

  void setAlertTempChangeC(double v) =>
      state = state.copyWith(alertTempChangeC: v);
}

final settingsProvider = NotifierProvider<SettingsNotifier, AppSettings>(
  SettingsNotifier.new,
);
