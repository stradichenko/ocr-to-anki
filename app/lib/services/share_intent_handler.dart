import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'system_channel.dart';

/// One entry in a batch of shared images: in-memory bytes plus the
/// originating filename (used for display + Anki card naming).
typedef SharedImage = ({Uint8List bytes, String name});

/// Receives image-share intents from the Android system and surfaces them
/// as in-memory byte batches.
///
/// Two paths are supported:
///   * Cold launch — activity was started by the share intent.  The
///     paths are pulled once via [SystemChannel.getInitialSharedImages]
///     when the handler is constructed.
///   * Warm launch — activity was already running and `onNewIntent`
///     fired.  We subscribe to [SystemChannel.shareEvents] for the
///     lifetime of the handler.
///
/// Shared images that arrive before any UI subscriber is attached are
/// buffered in [pendingDrainable]; consumers must call [drainPending]
/// once during their first build to pick them up.
class ShareIntentHandler {
  ShareIntentHandler() {
    if (!Platform.isAndroid) return;
    _subscription = SystemChannel.shareEvents().listen((paths) async {
      final entries = await _readPaths(paths);
      if (entries.isEmpty) return;
      _pending.addAll(entries);
      if (!_controller.isClosed) _controller.add(entries);
    });
    _consumeInitial();
  }

  final StreamController<List<SharedImage>> _controller =
      StreamController<List<SharedImage>>.broadcast();
  final List<SharedImage> _pending = [];
  StreamSubscription<List<String>>? _subscription;

  /// Stream of shared-image batches.  One event per inbound intent.
  Stream<List<SharedImage>> get stream => _controller.stream;

  /// Whether any shared images are waiting for a UI consumer.
  bool get pendingDrainable => _pending.isNotEmpty;

  /// Returns and clears any shared images that arrived before the UI
  /// was ready to handle them.  Call once on first build.
  List<SharedImage> drainPending() {
    final out = List<SharedImage>.from(_pending);
    _pending.clear();
    return out;
  }

  Future<void> _consumeInitial() async {
    try {
      final paths = await SystemChannel.getInitialSharedImages();
      final entries = await _readPaths(paths);
      if (entries.isEmpty) return;
      _pending.addAll(entries);
      if (!_controller.isClosed) _controller.add(entries);
    } catch (_) {
      // Best-effort — a missing or unreadable share intent should never
      // bring the app down.
    }
  }

  Future<List<SharedImage>> _readPaths(List<String> paths) async {
    final out = <SharedImage>[];
    for (final path in paths) {
      try {
        final file = File(path);
        if (!await file.exists()) continue;
        final bytes = await file.readAsBytes();
        if (bytes.isEmpty) continue;
        out.add((bytes: bytes, name: path.split('/').last));
        // Best-effort cleanup — the file lives in our own cache and
        // serves no purpose once the bytes are in memory.
        try {
          await file.delete();
        } catch (_) {}
      } catch (_) {
        // Skip unreadable paths but continue with the rest of the batch.
      }
    }
    return out;
  }

  void dispose() {
    _subscription?.cancel();
    _subscription = null;
    if (!_controller.isClosed) _controller.close();
  }
}
