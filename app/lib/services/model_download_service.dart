import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

/// Downloads large model files with resume support and progress callbacks.
class ModelDownloadService {
  static const _modelUrl =
      'https://huggingface.co/stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small/resolve/main/gemma-3-4b-it-q4_0_s.gguf';
  static const _mmprojUrl =
      'https://huggingface.co/stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small/resolve/main/mmproj-google_gemma-3-4b-it-f16.gguf';

  static const _modelFile = 'gemma-3-4b-it-q4_0_s.gguf';
  static const _mmprojFile = 'mmproj-model-f16-4B.gguf';

  /// Cancel token — set to true to abort an in-flight download.
  bool _cancelled = false;

  void cancel() => _cancelled = true;

  /// Download both models if missing, with progress callbacks.
  ///
  /// [onProgress] receives (downloadedBytes, totalBytes, currentFile).
  Future<void> downloadAll({
    required void Function(int downloaded, int total, String file) onProgress,
  }) async {
    _cancelled = false;
    final appDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${appDir.path}/llama_cpp/models');
    await modelDir.create(recursive: true);

    final modelPath = '${modelDir.path}/$_modelFile';
    final mmprojPath = '${modelDir.path}/$_mmprojFile';

    if (!await File(modelPath).exists()) {
      await _downloadFile(
        url: _modelUrl,
        destPath: modelPath,
        onProgress: onProgress,
      );
    }

    if (!await File(mmprojPath).exists()) {
      await _downloadFile(
        url: _mmprojUrl,
        destPath: mmprojPath,
        onProgress: onProgress,
      );
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
      final sink = partialFile.openWrite(mode: startByte > 0
          ? FileMode.writeOnlyAppend
          : FileMode.writeOnly);

      var downloaded = startByte;
      final fileName = url.split('/').last;

      try {
        await for (final chunk in streamed.stream) {
          if (_cancelled) {
            sink.close();
            throw Exception('Download cancelled');
          }
          sink.add(chunk);
          downloaded += chunk.length;
          onProgress(downloaded, total, fileName);
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
}
