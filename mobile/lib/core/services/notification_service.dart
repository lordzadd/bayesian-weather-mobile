import 'dart:math' as math;

import 'package:flutter_local_notifications/flutter_local_notifications.dart';

/// Wraps flutter_local_notifications for in-app weather change alerts.
///
/// Usage:
///   await NotificationService.instance.initialize();
///   NotificationService.instance.checkAndAlert(prevTemp, newTemp, threshold);
class NotificationService {
  static final NotificationService instance = NotificationService._();
  NotificationService._();

  final _plugin = FlutterLocalNotificationsPlugin();
  bool _initialized = false;

  Future<void> initialize() async {
    if (_initialized) return;
    const androidInit = AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosInit     = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: false,
      requestSoundPermission: false,
    );
    await _plugin.initialize(
      const InitializationSettings(android: androidInit, iOS: iosInit),
    );
    _initialized = true;
  }

  /// Sends a notification if |newTemp - prevTemp| >= [thresholdC].
  Future<void> checkAndAlert({
    required double prevTempC,
    required double newTempC,
    required double thresholdC,
  }) async {
    if (!_initialized) return;
    final delta = (newTempC - prevTempC).abs();
    if (delta < thresholdC) return;

    final direction = newTempC > prevTempC ? 'risen' : 'dropped';
    final body =
        'Temperature has $direction by ${delta.toStringAsFixed(1)}°C  '
        '(${prevTempC.toStringAsFixed(1)}° → ${newTempC.toStringAsFixed(1)}°C)';

    await _plugin.show(
      math.Random().nextInt(10000),
      'Weather change detected',
      body,
      const NotificationDetails(
        iOS: DarwinNotificationDetails(
          presentAlert: true,
          presentBadge: false,
          presentSound: false,
        ),
        android: AndroidNotificationDetails(
          'weather_alerts',
          'Weather Alerts',
          importance: Importance.defaultImportance,
          priority: Priority.defaultPriority,
        ),
      ),
    );
  }
}
