import 'dart:async';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import '../models/model_info.dart';
import 'foreground_task_service.dart';

/// Downloads large model files with resume support, automatic retry on
/// transient network failures, and live foreground-service notification
/// updates.
class ModelDownloadService {
  /// Max attempts before giving up on a single file.
  static const _maxAttempts = 5;

  /// Cancel token — set to true to abort an in-flight download.
  bool _cancelled = false;

  /// Last time a foreground-service notification update was pushed.  Used
  /// to throttle the OS notification system to ~1 update per second.
  DateTime _lastNotificationAt = DateTime.fromMillisecondsSinceEpoch(0);

  void cancel() => _cancelled = true;

  /// Download both files for [model] if missing, with progress callbacks.
  ///
  /// [onProgress] receives (downloadedBytes, totalBytes, currentFile).
  Future<void> downloadModel(
    ModelInfo model, {
    required void Function(int downloaded, int total, String file) onProgress,
  }) async {
    _cancelled = false;
    final appDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${appDir.path}/llama_cpp/models');
    await modelDir.create(recursive: true);

    final modelPath = '${modelDir.path}/${model.modelFilename}';
    final mmprojPath = '${modelDir.path}/${model.mmprojFilename}';

    if (!await File(modelPath).exists()) {
      await _downloadWithRetry(
        url: model.modelUrl,
        destPath: modelPath,
        onProgress: onProgress,
      );
    }

    if (!await File(mmprojPath).exists()) {
      await _downloadWithRetry(
        url: model.mmprojUrl,
        destPath: mmprojPath,
        onProgress: onProgress,
      );
    }
  }

  /// Wraps [_downloadFile] with exponential-backoff retry on transient
  /// network errors.  The .part file persists across attempts so each
  /// retry resumes via the Range header instead of restarting at zero.
  Future<void> _downloadWithRetry({
    required String url,
    required String destPath,
    required void Function(int downloaded, int total, String file) onProgress,
  }) async {
    var attempt = 0;
    while (true) {
      attempt++;
      try {
        await _downloadFile(
          url: url,
          destPath: destPath,
          onProgress: onProgress,
        );
        return;
      } on SocketException catch (_) {
        if (attempt >= _maxAttempts || _cancelled) rethrow;
      } on http.ClientException catch (_) {
        if (attempt >= _maxAttempts || _cancelled) rethrow;
      } on TimeoutException catch (_) {
        if (attempt >= _maxAttempts || _cancelled) rethrow;
      } on HandshakeException catch (_) {
        if (attempt >= _maxAttempts || _cancelled) rethrow;
      }
      // Exponential backoff: 1, 2, 4, 8 seconds between attempts 1-2, 2-3, ...
      final delaySeconds = 1 << (attempt - 1);
      await ForegroundTaskService.updateNotification(
        'Network hiccup — retrying in ${delaySeconds}s (attempt '
        '${attempt + 1}/$_maxAttempts)…',
      );
      await Future<void>.delayed(Duration(seconds: delaySeconds));
      if (_cancelled) {
        throw Exception('Download cancelled');
      }
    }
  }

  /// Download a single file with resume support.
  Future<void> _downloadFile({
    required String url,
    required String destPath,
    required void Function(int downloaded, int total, String file) onProgress,
  }) async {
    final partialPath = '$destPath.part';
    final partialFile = File(partialPath);

    var startByte = 0;
    if (await partialFile.exists()) {
      startByte = await partialFile.length();
    }

    final request = http.Request('GET', Uri.parse(url));
    if (startByte > 0) {
      request.headers['Range'] = 'bytes=$startByte-';
    }

    final client = http.Client();
    try {
      final streamed = await client.send(request);

      if (streamed.statusCode != 200 && streamed.statusCode != 206) {
        throw Exception('HTTP ${streamed.statusCode}');
      }

      final total = streamed.contentLength != null
          ? startByte + streamed.contentLength!
          : 0;
      final sink = partialFile.openWrite(
        mode: startByte > 0
            ? FileMode.writeOnlyAppend
            : FileMode.writeOnly,
      );

      var downloaded = startByte;
      final fileName = url.split('/').last;

      try {
        await for (final chunk in streamed.stream) {
          if (_cancelled) {
            await sink.close();
            throw Exception('Download cancelled');
          }
          sink.add(chunk);
          downloaded += chunk.length;
          onProgress(downloaded, total, fileName);
          _maybeUpdateNotification(downloaded, total, fileName);
        }
      } finally {
        await sink.close();
      }

      // Rename partial to final.
      await partialFile.rename(destPath);
    } finally {
      client.close();
    }
  }

  /// Push a progress update to the foreground-service notification at
  /// most once per second.  Spamming `updateService` is rate-limited by
  /// Android and burns CPU re-rendering the notification panel.
  void _maybeUpdateNotification(int downloaded, int total, String file) {
    final now = DateTime.now();
    if (now.difference(_lastNotificationAt).inMilliseconds < 1000) return;
    _lastNotificationAt = now;

    if (total > 0) {
      final pct = (downloaded * 100 ~/ total).clamp(0, 100);
      final mb = (downloaded / (1024 * 1024)).toStringAsFixed(0);
      final totalMb = (total / (1024 * 1024)).toStringAsFixed(0);
      ForegroundTaskService.updateNotification(
        'Downloading $file — $pct% ($mb / $totalMb MB)',
      );
    } else {
      final mb = (downloaded / (1024 * 1024)).toStringAsFixed(0);
      ForegroundTaskService.updateNotification(
        'Downloading $file — $mb MB',
      );
    }
  }
}
