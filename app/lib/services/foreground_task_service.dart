import 'package:flutter_foreground_task/flutter_foreground_task.dart';

/// Keeps the app alive during long-running inference on Android.
///
/// Uses a foreground service with a persistent notification so that
/// Doze mode does not suspend llama-server mid-OCR.
class ForegroundTaskService {
  static bool _initialized = false;

  /// Start the foreground service with a processing notification.
  static Future<void> start({String? detail}) async {
    if (await FlutterForegroundTask.isRunningService) return;

    if (!_initialized) {
      FlutterForegroundTask.init(
        androidNotificationOptions: AndroidNotificationOptions(
          channelId: 'ocr_inference',
          channelName: 'OCR Inference',
          channelDescription:
              'Keeps inference running while the app is backgrounded',
          channelImportance: NotificationChannelImportance.LOW,
          priority: NotificationPriority.LOW,
        ),
        iosNotificationOptions: const IOSNotificationOptions(
          showNotification: false,
        ),
        foregroundTaskOptions: ForegroundTaskOptions(
          eventAction: ForegroundTaskEventAction.nothing(),
          allowWakeLock: true,
          allowWifiLock: true,
        ),
      );
      _initialized = true;
    }

    await FlutterForegroundTask.startService(
      notificationTitle: 'OCR to Anki — Processing',
      notificationText: detail ?? 'Running AI inference…',
      callback: _taskCallback,
    );
  }

  /// Update the notification text (e.g., to show progress).
  static Future<void> updateNotification(String text) async {
    if (!await FlutterForegroundTask.isRunningService) return;
    await FlutterForegroundTask.updateService(
      notificationText: text,
    );
  }

  /// Stop the foreground service.
  static Future<void> stop() async {
    if (!await FlutterForegroundTask.isRunningService) return;
    await FlutterForegroundTask.stopService();
  }
}

/// Top-level callback required by flutter_foreground_task.
@pragma('vm:entry-point')
void _taskCallback() {
  FlutterForegroundTask.setTaskHandler(_OcrTaskHandler());
}

class _OcrTaskHandler extends TaskHandler {
  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {
    // Nothing to do — the service exists only to keep the process alive.
  }

  @override
  void onRepeatEvent(DateTime timestamp) {
    // Not called because eventAction is set to nothing().
  }

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {
    // Clean up if needed.
  }

  @override
  void onNotificationButtonPressed(String id) {}

  @override
  void onNotificationPressed() {
    FlutterForegroundTask.launchApp('/');
  }
}
