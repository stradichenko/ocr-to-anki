import 'dart:io';

import 'package:flutter/services.dart';

/// Thin Dart wrapper around the `com.ocrtoanki.ocr_to_anki/system` Kotlin
/// MethodChannel.  Surfaces Android-only system calls (notifications,
/// battery optimisation, storage info, share-intent extraction) without
/// pulling in heavyweight permission plugins.
///
/// All methods are no-ops / sensible defaults on non-Android platforms so
/// callers don't need to platform-gate every call site.
class SystemChannel {
  SystemChannel._();

  static const _method = MethodChannel('com.ocrtoanki.ocr_to_anki/system');
  static const _shareEvents =
      EventChannel('com.ocrtoanki.ocr_to_anki/share_events');

  // ---------------------------------------------------------------------------
  // Notifications
  // ---------------------------------------------------------------------------

  /// Whether POST_NOTIFICATIONS is granted (always true on API < 33 and on
  /// non-Android platforms).
  static Future<bool> isPostNotificationsGranted() async {
    if (!Platform.isAndroid) return true;
    final v = await _method.invokeMethod<bool>('isPostNotificationsGranted');
    return v ?? false;
  }

  /// Show the system POST_NOTIFICATIONS prompt.  Returns the granted state.
  static Future<bool> requestPostNotifications() async {
    if (!Platform.isAndroid) return true;
    try {
      final v = await _method.invokeMethod<bool>('requestPostNotifications');
      return v ?? false;
    } on PlatformException {
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Battery optimisation
  // ---------------------------------------------------------------------------

  /// Whether the app is whitelisted from battery optimisation.
  static Future<bool> isBatteryOptimizationDisabled() async {
    if (!Platform.isAndroid) return true;
    final v = await _method.invokeMethod<bool>('isBatteryOptimizationDisabled');
    return v ?? false;
  }

  /// Open the system dialog asking the user to opt this app out of battery
  /// optimisation.  Resolves immediately — the user's choice is observed by
  /// re-checking [isBatteryOptimizationDisabled] after app resume.
  static Future<void> requestIgnoreBatteryOptimizations() async {
    if (!Platform.isAndroid) return;
    await _method.invokeMethod<void>('requestIgnoreBatteryOptimizations');
  }

  // ---------------------------------------------------------------------------
  // Settings deep-link
  // ---------------------------------------------------------------------------

  /// Open the OS-level "App info" settings page for this app.
  static Future<void> openAppDetailsSettings() async {
    if (!Platform.isAndroid) return;
    await _method.invokeMethod<void>('openAppDetailsSettings');
  }

  // ---------------------------------------------------------------------------
  // Storage
  // ---------------------------------------------------------------------------

  /// Free bytes available on the partition holding app-private storage.
  /// Returns -1 on platforms / errors where the value can't be determined.
  static Future<int> getAvailableStorageBytes() async {
    if (!Platform.isAndroid) return -1;
    final v = await _method.invokeMethod<int>('getAvailableStorageBytes');
    return v ?? -1;
  }

  // ---------------------------------------------------------------------------
  // Inbound share intents
  // ---------------------------------------------------------------------------

  /// Cache paths for any images this activity was launched with via an
  /// inbound share intent.  Idempotent: a second call within the same
  /// activity instance returns an empty list.
  static Future<List<String>> getInitialSharedImages() async {
    if (!Platform.isAndroid) return const [];
    final raw = await _method.invokeMethod<List<dynamic>>(
      'getInitialSharedImages',
    );
    return raw?.cast<String>() ?? const [];
  }

  /// Stream of cache-path lists for share intents received while the
  /// activity was already running (`onNewIntent` flow).
  static Stream<List<String>> shareEvents() {
    if (!Platform.isAndroid) return const Stream<List<String>>.empty();
    return _shareEvents.receiveBroadcastStream().map(
          (event) => (event as List<dynamic>).cast<String>(),
        );
  }

  // ---------------------------------------------------------------------------
  // Camera permission
  // ---------------------------------------------------------------------------

  /// Whether the CAMERA permission is granted.
  static Future<bool> isCameraGranted() async {
    if (!Platform.isAndroid) return false;
    final v = await _method.invokeMethod<bool>('isCameraGranted');
    return v ?? false;
  }

  /// Request the CAMERA permission. Returns optimistically after the
  /// system dialog is shown; callers should re-check on app resume.
  static Future<bool> requestCameraPermission() async {
    if (!Platform.isAndroid) return false;
    try {
      final v = await _method.invokeMethod<bool>('requestCameraPermission');
      return v ?? false;
    } on PlatformException {
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // AnkiDroid
  // ---------------------------------------------------------------------------

  /// Whether AnkiDroid is installed on this device.
  static Future<bool> isAnkiDroidInstalled() async {
    if (!Platform.isAndroid) return false;
    final v = await _method.invokeMethod<bool>('isAnkiDroidInstalled');
    return v ?? false;
  }

  /// Request the AnkiDroid READ_WRITE_PERMISSION. Returns true if granted.
  static Future<bool> requestAnkiDroidPermission() async {
    if (!Platform.isAndroid) return false;
    try {
      final v = await _method.invokeMethod<bool>('requestAnkiDroidPermission');
      return v ?? false;
    } on PlatformException {
      return false;
    }
  }

  /// List available AnkiDroid decks as `{id, name}` maps.
  static Future<List<Map<String, dynamic>>> getAnkiDroidDecks() async {
    if (!Platform.isAndroid) return const [];
    final raw = await _method.invokeMethod<List<dynamic>>('getAnkiDroidDecks');
    return raw?.cast<Map<String, dynamic>>() ?? const [];
  }

  /// Add notes directly to AnkiDroid. Returns the number of cards added.
  static Future<int> addNotesToAnkiDroid(
    List<Map<String, dynamic>> notes,
    int deckId,
  ) async {
    if (!Platform.isAndroid) return 0;
    final v = await _method.invokeMethod<int>(
      'addNotesToAnkiDroid',
      {'notes': notes, 'deckId': deckId},
    );
    return v ?? 0;
  }
}
