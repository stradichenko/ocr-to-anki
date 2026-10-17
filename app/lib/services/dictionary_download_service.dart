import 'dart:async';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;

import '../models/dictionary_info.dart';
import 'dictionary_registry_service.dart';

/// Downloads and installs offline dictionary indices.
///
/// Implemented in Dart rather than in the Python backend because Android has no
/// Python: the app runs llama.cpp directly there, so a backend-only path would
/// leave Android without dictionaries.
class DictionaryDownloadService {
  DictionaryDownloadService(this._registry);

  final DictionaryRegistryService _registry;

  static const int _maxAttempts = 5;

  /// Download [info], verify it, and install it.
  ///
  /// [onProgress] receives bytes downloaded and the expected total, either of
  /// which may be zero when the server does not report a length.
  Future<void> download(
    DictionaryInfo info, {
    required String baseUrl,
    void Function(int received, int total)? onProgress,
  }) async {
    final dest = await _registry.dictionaryPath(info);
    final part = File('$dest.part');
    final url = Uri.parse(info.urlFor(baseUrl));

    Object? lastError;
    for (var attempt = 1; attempt <= _maxAttempts; attempt++) {
      try {
        final existing = await part.exists() ? await part.length() : 0;
        final client = http.Client();
        try {
          final request = http.Request('GET', url);
          if (existing > 0) {
            request.headers['Range'] = 'bytes=$existing-';
          }
          final response = await client.send(request);

          if (response.statusCode != 200 && response.statusCode != 206) {
            throw HttpException('HTTP ${response.statusCode} for $url');
          }

          final remaining = response.contentLength ?? 0;
          final total = existing + remaining;

          final sink = part.openWrite(mode: FileMode.append);
          var received = existing;
          try {
            await for (final chunk in response.stream) {
              sink.add(chunk);
              received += chunk.length;
              onProgress?.call(received, total);
            }
          } finally {
            await sink.close();
          }
        } finally {
          client.close();
        }

        await _verifyAndInstall(info, part, File(dest));
        return;
      } catch (e) {
        lastError = e;
        if (attempt == _maxAttempts) break;
        // The .part file is kept so the next attempt resumes rather than
        // restarting a download that may be tens of megabytes.
        await Future.delayed(Duration(seconds: 2 * attempt));
      }
    }
    throw Exception('Dictionary download failed after $_maxAttempts attempts: $lastError');
  }

  Future<void> _verifyAndInstall(
    DictionaryInfo info,
    File part,
    File dest,
  ) async {
    if (info.sha256.isNotEmpty) {
      final digest = await _sha256Of(part);
      if (digest != info.sha256.toLowerCase()) {
        // Discard rather than install: a wrong index would answer some lookups
        // and silently miss others.
        await part.delete();
        throw Exception(
          'Checksum mismatch for ${info.fileName}: '
          'expected ${info.sha256}, got $digest',
        );
      }
    }
    if (info.sizeBytes > 0 && await part.length() != info.sizeBytes) {
      await part.delete();
      throw Exception(
        'Size mismatch for ${info.fileName}: '
        'expected ${info.sizeBytes}, got ${await part.length()}',
      );
    }
    if (await dest.exists()) await dest.delete();
    await part.rename(dest.path);
  }

  Future<String> _sha256Of(File file) async {
    final sink = _DigestSink();
    final input = sha256.startChunkedConversion(sink);
    await for (final chunk in file.openRead()) {
      input.add(chunk);
    }
    input.close();
    return sink.digest.toString();
  }

  /// Remove a partially downloaded index, e.g. when the user cancels.
  Future<void> discardPartial(DictionaryInfo info) async {
    final dest = await _registry.dictionaryPath(info);
    final part = File('$dest.part');
    if (await part.exists()) await part.delete();
  }
}

/// Accumulates a hash without exposing the crypto types to callers.
class _DigestSink implements Sink<Digest> {
  Digest? _digest;

  @override
  void add(Digest data) => _digest = data;

  @override
  void close() {}

  Digest get digest => _digest!;
}