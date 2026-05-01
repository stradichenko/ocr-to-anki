import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

/// Manages llama.cpp native binaries on Android.
///
/// On first use, copies the bundled ARM64 binaries from Flutter assets
/// to the app's private storage and sets executable permissions.
/// Then spawns llama-server and runs llama-mtmd-cli for vision OCR.
class LlamaCppAndroidService {
  LlamaCppAndroidService();

  Process? _serverProcess;
  String? _serverUrl;

  /// Directory where binaries and models are stored.
  Directory? _appDir;

  /// Whether binaries have been extracted from assets.
  bool _binariesReady = false;

  /// URL of the running llama-server, or null if not running.
  String? get serverUrl => _serverUrl;

  /// Whether llama-server is currently running.
  bool get isServerRunning => _serverProcess != null;

  // ---------------------------------------------------------------------------
  // Binary extraction
  // ---------------------------------------------------------------------------

  /// Copy native binaries from Flutter assets to app storage.
  Future<void> ensureBinaries() async {
    if (_binariesReady) return;

    final appDir = await getApplicationDocumentsDirectory();
    _appDir = Directory('${appDir.path}/llama_cpp');
    await _appDir!.create(recursive: true);

    final binDir = Directory('${_appDir!.path}/bin');
    await binDir.create(recursive: true);

    const binaries = ['llama-server', 'llama-mtmd-cli'];
    for (final name in binaries) {
      final assetPath = 'assets/llama-binaries/arm64-v8a/$name';
      final destPath = '${binDir.path}/$name';

      final destFile = File(destPath);
      if (await destFile.exists()) {
        // Already extracted — verify it's executable.
        await _chmod(destPath, '755');
        continue;
      }

      final byteData = await rootBundle.load(assetPath);
      final bytes = byteData.buffer.asUint8List();
      await destFile.writeAsBytes(bytes);
      await _chmod(destPath, '755');
    }

    // Copy libc++_shared.so if bundled.
    final stlAsset = 'assets/llama-binaries/arm64-v8a/libc++_shared.so';
    try {
      final byteData = await rootBundle.load(stlAsset);
      final libDir = Directory('${_appDir!.path}/lib');
      await libDir.create(recursive: true);
      final destFile = File('${libDir.path}/libc++_shared.so');
      if (!await destFile.exists()) {
        await destFile.writeAsBytes(byteData.buffer.asUint8List());
      }
    } catch (_) {
      // libc++_shared.so may not be bundled if statically linked.
    }

    _binariesReady = true;
  }

  Future<void> _chmod(String path, String mode) async {
    try {
      await Process.run('chmod', [mode, path]);
    } catch (_) {
      // Best effort — may fail on some devices.
    }
  }

  String get _binDir => '${_appDir!.path}/bin';
  String get _libDir => '${_appDir!.path}/lib';

  // ---------------------------------------------------------------------------
  // Model paths
  // ---------------------------------------------------------------------------

  /// Directory where GGUF models are stored.
  Directory get modelDir => Directory('${_appDir!.path}/models');

  String get modelPath => '${modelDir.path}/gemma-3-4b-it-q4_0_s.gguf';
  String get mmprojPath => '${modelDir.path}/mmproj-model-f16-4B.gguf';

  Future<bool> get modelsExist async {
    return File(modelPath).existsSync() && File(mmprojPath).existsSync();
  }

  // ---------------------------------------------------------------------------
  // llama-server lifecycle
  // ---------------------------------------------------------------------------

  /// Start llama-server with the text-generation model.
  Future<void> startServer() async {
    await ensureBinaries();

    if (_serverProcess != null) return;

    final model = modelPath;
    if (!File(model).existsSync()) {
      throw StateError('Model not found: $model');
    }

    final port = 8090;
    final env = Map<String, String>.from(Platform.environment);
    // Help the dynamic linker find libc++_shared.so if needed.
    if (Directory(_libDir).existsSync()) {
      env['LD_LIBRARY_PATH'] = _libDir;
    }

    _serverProcess = await Process.start(
      '$_binDir/llama-server',
      [
        '-m', model,
        '--host', '127.0.0.1',
        '--port', '$port',
        '-c', '4096',
        '-np', '1',
        '--cache-type-k', 'q4_0',
        '--slots',
      ],
      environment: env,
    );

    _serverUrl = 'http://127.0.0.1:$port';

    // Wait for server to be ready with detailed diagnostics.
    String? lastError;
    for (var i = 0; i < 60; i++) {
      await Future<void>.delayed(const Duration(milliseconds: 500));
      try {
        final resp = await http
            .get(Uri.parse('$_serverUrl/health'))
            .timeout(const Duration(seconds: 2));
        if (resp.statusCode == 200) return;
      } catch (e) {
        lastError = e.toString();
        // Not ready yet.
      }

      // Check if process died early.
      if (_serverProcess != null) {
        final exited = await _serverProcess!.exitCode
            .then((_) => true)
            .catchError((_) => false);
        if (exited) {
          final stderr = _serverProcess?.stderr;
          var errText = '';
          if (stderr != null) {
            try {
              errText = await stderr
                  .transform(const Utf8Decoder())
                  .take(50)
                  .join('\n');
            } catch (_) {}
          }
          _serverProcess = null;
          _serverUrl = null;
          throw StateError(
            'llama-server exited prematurely.\n'
            'Stderr: ${errText.isNotEmpty ? errText : "(empty)"}',
          );
        }
      }
    }

    throw StateError(
      'llama-server failed to start within 30s. Last error: $lastError',
    );
  }

  /// Check whether llama-server is responding.
  Future<bool> checkHealth() async {
    if (_serverUrl == null) return false;
    try {
      final resp = await http
          .get(Uri.parse('$_serverUrl/health'))
          .timeout(const Duration(seconds: 3));
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Ensure llama-server is running, restarting it if necessary.
  ///
  /// Call this before inference on Android to handle Doze-mode kills.
  Future<void> ensureServerRunning() async {
    if (await checkHealth()) return;
    // Server is down — restart it.
    await stopServer();
    await startServer();
  }

  /// Stop llama-server.
  Future<void> stopServer() async {
    if (_serverProcess == null) return;

    // Try graceful SIGTERM first.
    _serverProcess!.kill(ProcessSignal.sigterm);

    // Wait up to 5s for clean exit.
    try {
      await _serverProcess!.exitCode.timeout(const Duration(seconds: 5));
    } catch (_) {
      // Force kill if still running.
      _serverProcess!.kill(ProcessSignal.sigkill);
    }

    _serverProcess = null;
    _serverUrl = null;
  }

  // ---------------------------------------------------------------------------
  // Vision OCR via llama-mtmd-cli
  // ---------------------------------------------------------------------------

  /// Run vision OCR on an image using llama-mtmd-cli.
  ///
  /// [imageBytes] — raw JPEG / PNG bytes.
  /// [prompt] — vision prompt for the model.
  Future<String> runVisionOcr({
    required Uint8List imageBytes,
    String prompt =
        'List every word visible in this image. Output ONLY the words, one per line. No bullet points, no numbering, no descriptions, no commentary.',
  }) async {
    await ensureBinaries();

    if (!File(modelPath).existsSync()) {
      throw StateError('Model not found: $modelPath');
    }
    if (!File(mmprojPath).existsSync()) {
      throw StateError('Vision projector not found: $mmprojPath');
    }

    // Write image to a temp file — llama-mtmd-cli needs a file path.
    final tmpDir = Directory('${_appDir!.path}/tmp');
    await tmpDir.create(recursive: true);
    final tmpFile =
        File('${tmpDir.path}/vision_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await tmpFile.writeAsBytes(imageBytes);

    try {
      final env = Map<String, String>.from(Platform.environment);
      if (Directory(_libDir).existsSync()) {
        env['LD_LIBRARY_PATH'] = _libDir;
      }

      final result = await Process.run(
        '$_binDir/llama-mtmd-cli',
        [
          '-m', modelPath,
          '--mmproj', mmprojPath,
          '--image', tmpFile.path,
          '-p', prompt,
          '-n', '512',
          '--temp', '0.1',
          '--no-display-prompt',
        ],
        environment: env,
        stdoutEncoding: utf8,
        stderrEncoding: utf8,
      );

      if (result.exitCode != 0) {
        throw Exception(
          'llama-mtmd-cli failed (exit ${result.exitCode}): ${result.stderr}',
        );
      }

      // Parse output: llama-mtmd-cli prints the generated text to stdout.
      // There may be log lines mixed in; we filter for the response after
      // the prompt echo.
      final output = result.stdout as String;
      return _extractVisionText(output);
    } finally {
      // Clean up temp file.
      try {
        await tmpFile.delete();
      } catch (_) {}
    }
  }

  /// Extract the generated text from llama-mtmd-cli stdout.
  ///
  /// The CLI prints various log lines and then the model response.
  /// Uses multiple heuristics to find the actual generated text.
  String _extractVisionText(String raw) {
    final lines = raw.split('\n');

    // Heuristic 1: Known log prefixes that appear during model loading.
    final logPrefixes = [
      RegExp(r'^clip_model_load'),
      RegExp(r'^llama_model_loader'),
      RegExp(r'^load_tensors'),
      RegExp(r'^ggml_'),
      RegExp(r'^build:\s'),
      RegExp(r'^system_info:\s'),
      RegExp(r'^srv\s'),           // server log lines
      RegExp(r'^main:\s'),         // main function logs
      RegExp(r'^\s*$'),            // empty lines
    ];

    bool isLogLine(String line) {
      for (final prefix in logPrefixes) {
        if (prefix.hasMatch(line)) return true;
      }
      return false;
    }

    // Find the transition point: the first contiguous block of non-log lines.
    final candidateBlocks = <List<String>>[];
    var currentBlock = <String>[];

    for (final line in lines) {
      if (isLogLine(line)) {
        if (currentBlock.isNotEmpty) {
          candidateBlocks.add(List.from(currentBlock));
          currentBlock.clear();
        }
      } else {
        currentBlock.add(line);
      }
    }
    if (currentBlock.isNotEmpty) {
      candidateBlocks.add(currentBlock);
    }

    // The longest contiguous block of non-log lines is most likely the response.
    if (candidateBlocks.isEmpty) return '';
    candidateBlocks.sort((a, b) => b.length.compareTo(a.length));
    final bestBlock = candidateBlocks.first;

    // Heuristic 2: If the block starts with common artefacts, trim them.
    var startIdx = 0;
    for (var i = 0; i < bestBlock.length && i < 3; i++) {
      final trimmed = bestBlock[i].trim();
      // Skip single-character lines or obvious separators at the start.
      if (trimmed.length <= 1 && !trimmed.contains(RegExp(r'[a-zA-Z]'))) {
        startIdx = i + 1;
      } else {
        break;
      }
    }

    final result = bestBlock.sublist(startIdx);
    return result.join('\n').trim();
  }

  // ---------------------------------------------------------------------------
  // Text generation via llama-server
  // ---------------------------------------------------------------------------

  /// Generate text via llama-server's /completion endpoint.
  Future<String> generate({
    required String prompt,
    int maxTokens = 256,
    double temperature = 0.7,
  }) async {
    if (_serverUrl == null) {
      throw StateError('llama-server is not running');
    }

    final uri = Uri.parse('$_serverUrl/completion');
    final response = await http
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'prompt': prompt,
            'n_predict': maxTokens,
            'temperature': temperature,
            'stop': ['</s>', 'User:', 'Assistant:'],
          }),
        )
        .timeout(const Duration(minutes: 5));

    if (response.statusCode != 200) {
      throw Exception(
        'Generation failed (${response.statusCode}): ${response.body}',
      );
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    return body['content'] as String? ?? '';
  }
}
