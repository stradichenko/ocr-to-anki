import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

/// Manages llama.cpp native binaries on Android.
///
/// The binaries ship as `lib*.so` files under `jniLibs/arm64-v8a/` and are
/// extracted by Android to `applicationInfo.nativeLibraryDir` at install
/// time. App-private storage (`/data/user/0/<pkg>/`) is mounted `noexec`
/// on modern Android, so files copied there cannot be `execve`'d — only
/// the native lib dir is exec-allowed.
///
/// We invoke `llama-server` (renamed `libllama-server.so`) and
/// `llama-mtmd-cli` (renamed `libllama-mtmd-cli.so`) directly from that dir.
class LlamaCppAndroidService {
  LlamaCppAndroidService();

  static const _channel =
      MethodChannel('com.ocrtoanki.ocr_to_anki/native_lib_dir');

  Process? _serverProcess;
  String? _serverUrl;

  /// Directory for downloaded GGUF models and temp files.
  Directory? _appDir;

  /// Path to the Android nativeLibraryDir, where lib*.so files are extracted.
  String? _nativeLibDir;

  /// Whether binaries have been located.
  bool _binariesReady = false;

  /// Rolling buffer of llama-server stderr lines, kept while the process
  /// is alive so we can surface them on health-check failure.
  final List<String> _serverStderrLines = [];

  /// URL of the running llama-server, or null if not running.
  String? get serverUrl => _serverUrl;

  /// Whether llama-server is currently running.
  bool get isServerRunning => _serverProcess != null;

  // ---------------------------------------------------------------------------
  // Native lib dir lookup
  // ---------------------------------------------------------------------------

  /// Resolve the nativeLibraryDir from Android and verify the binaries are there.
  Future<void> ensureBinaries() async {
    if (_binariesReady) return;

    // App docs dir is still used for downloaded models and temp files.
    final appDir = await getApplicationDocumentsDirectory();
    _appDir = Directory('${appDir.path}/llama_cpp');
    await _appDir!.create(recursive: true);

    // Ask Android for nativeLibraryDir via the method channel registered in
    // MainActivity.kt. This is the only directory on modern Android where
    // app-shipped binaries can be exec'd.
    final dir = await _channel.invokeMethod<String>('getNativeLibDir');
    if (dir == null || dir.isEmpty) {
      throw StateError(
        'Android nativeLibraryDir is unavailable. '
        'Method channel returned null/empty.',
      );
    }
    _nativeLibDir = dir;

    // Verify the expected lib*.so files were extracted by the installer.
    const expected = ['libllama-server.so', 'libllama-mtmd-cli.so'];
    for (final name in expected) {
      final f = File('$_nativeLibDir/$name');
      if (!f.existsSync()) {
        throw StateError(
          'Native binary missing: $_nativeLibDir/$name\n'
          'The APK may have been built without llama.cpp jniLibs, '
          'or Android did not extract them at install (useLegacyPackaging=false?).',
        );
      }
    }

    _binariesReady = true;
  }

  String get _libDir => _nativeLibDir!;

  String get _serverBinary => '$_nativeLibDir/libllama-server.so';
  String get _mtmdBinary => '$_nativeLibDir/libllama-mtmd-cli.so';

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
    final binaryPath = _serverBinary;
    final env = Map<String, String>.from(Platform.environment);
    // Help the dynamic linker find libc++_shared.so — it's in the same dir.
    env['LD_LIBRARY_PATH'] = _libDir;

    // Diagnostics about the binary itself — captured up front so error
    // paths can attach them.
    final binFile = File(binaryPath);
    final binExists = binFile.existsSync();
    final binSize = binExists ? binFile.lengthSync() : 0;
    String binStat = '(stat unavailable)';
    try {
      final r = await Process.run('ls', ['-l', binaryPath]);
      if (r.exitCode == 0) binStat = (r.stdout as String).trim();
    } catch (_) {}

    String binDiagnostics() => [
          'binary:   $binaryPath',
          'exists:   $binExists',
          'size:     $binSize bytes',
          'ls -l:    $binStat',
          'libDir:   $_libDir (exists=${Directory(_libDir).existsSync()})',
          'model:    $model (size=${File(model).existsSync() ? File(model).lengthSync() : 0} bytes)',
        ].join('\n');

    _serverStderrLines.clear();
    try {
      _serverProcess = await Process.start(
        binaryPath,
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
    } on ProcessException catch (pe) {
      throw StateError(
        'Failed to launch llama-server.\n'
        'ProcessException: ${pe.message}\n'
        'errno:    ${pe.errorCode}\n'
        'executable: ${pe.executable}\n'
        '${binDiagnostics()}',
      );
    } catch (e) {
      throw StateError(
        'Failed to launch llama-server: $e\n${binDiagnostics()}',
      );
    }

    _serverUrl = 'http://127.0.0.1:$port';

    // Capture the exit code asynchronously so the polling loop can check
    // for early termination without blocking on the exitCode Future.
    // Awaiting `process.exitCode` directly would block until the process
    // exits, which defeats the point of a polling loop while the process
    // is still alive but loading a multi-GB model.
    int? earlyExitCode;
    unawaited(_serverProcess!.exitCode.then((code) {
      earlyExitCode = code;
    }));

    // Tee stderr into a rolling buffer so we have context even when
    // the process is alive but failing health checks.
    _serverProcess!.stderr
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen((line) {
      _serverStderrLines.add(line);
      if (_serverStderrLines.length > 200) {
        _serverStderrLines.removeAt(0);
      }
    }, onError: (_) {});

    // Drain stdout to avoid the OS pipe buffer filling up and blocking
    // the child process. We don't need the content.
    _serverProcess!.stdout.drain<void>().catchError((_) {});

    // Wait up to 120 s for the server to be healthy. A 2.4 GB Q4_0 model
    // mmap'd from app-private storage can take 30-60 s to be ready on
    // mid-range Android phones, especially on first launch when the file
    // is still in flash and not in the page cache.
    String? lastError;
    for (var i = 0; i < 240; i++) {
      await Future<void>.delayed(const Duration(milliseconds: 500));
      try {
        final resp = await http
            .get(Uri.parse('$_serverUrl/health'))
            .timeout(const Duration(seconds: 2));
        if (resp.statusCode == 200) return;
      } catch (e) {
        lastError = e.toString();
      }

      // Non-blocking check: did the process exit?
      if (earlyExitCode != null) {
        final errText = _serverStderrLines.isNotEmpty
            ? _serverStderrLines.join('\n')
            : '(empty)';
        _serverProcess = null;
        _serverUrl = null;
        throw StateError(
          'llama-server exited prematurely (exit code $earlyExitCode).\n'
          '${binDiagnostics()}\n\n'
          '--- stderr ---\n$errText',
        );
      }
    }

    final errText = _serverStderrLines.isNotEmpty
        ? _serverStderrLines.join('\n')
        : '(empty)';
    throw StateError(
      'llama-server did not become healthy within 120s.\n'
      'Last health-check error: $lastError\n'
      '${binDiagnostics()}\n\n'
      '--- stderr ---\n$errText',
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
        _mtmdBinary,
        [
          '-m', modelPath,
          '--mmproj', mmprojPath,
          '--image', tmpFile.path,
          '-p', prompt,
          '-n', '512',
          '--temp', '0.1',
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
