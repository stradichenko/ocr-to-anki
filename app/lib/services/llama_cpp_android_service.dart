import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import '../models/model_info.dart';
import 'backend_probe.dart';
import 'foreground_task_service.dart';

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
/// Simple [Sink<Digest>] for the crypto package's chunked hashing.
class _DigestSink implements Sink<Digest> {
  Digest? _digest;
  @override
  void add(Digest digest) => _digest = digest;
  @override
  void close() {}
  Digest get digest => _digest!;
}

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

  /// Rolling buffer of debug logs (prompts, responses, errors) for
  /// enrichment troubleshooting.
  final List<String> _debugLogs = [];

  /// Available GPU backends detected at runtime (always contains 'cpu').
  Set<String> _availableBackends = {'cpu'};

  /// Resolved backend after applying user preference ('cpu', 'vulkan', or 'opencl').
  String _resolvedBackend = 'cpu';

  /// User GPU preference: 'auto', 'vulkan', 'opencl', or 'cpu'.
  String _gpuMode = 'auto';

  /// Number of GPU layers to offload. 999 means all layers.
  int _nGpuLayers = 999;

  /// Per-backend notes from the last detection run, for the diagnostics
  /// screen. One line per probed backend.
  final List<String> _backendProbe = [];

  /// Full probe output per backend, capped, for the diagnostics screen.
  /// The short [_backendProbe] notes truncate linker and driver errors at
  /// exactly the wrong place, which cost a round of diagnosis on the
  /// OpenCL path; the raw text is small and worth keeping.
  final List<String> _backendProbeOutput = [];

  /// Why the resolved GPU backend was abandoned for CPU at server start,
  /// when that happened. Null while the chosen backend is still trusted.
  String? _backendFallbackReason;

  /// Device properties fetched once at startup (model, SoC, SDK level).
  final Map<String, String> _deviceProps = {};

  /// Memory figures from /proc/meminfo in kB, or null when unreadable.
  int? _memTotalKb;
  int? _memAvailableKb;

  /// The currently selected model. Must be set before [startServer] or
  /// [runVisionOcr] are called.
  ModelInfo? _activeModel;

  /// URL of the running llama-server, or null if not running.
  String? get serverUrl => _serverUrl;

  /// Whether llama-server is currently running.
  bool get isServerRunning => _serverProcess != null;

  /// HTTP client for the current generation request; can be closed to abort.
  http.Client? _generateClient;

  // ---------------------------------------------------------------------------
  // Active model
  // ---------------------------------------------------------------------------

  /// Set the model that will be used for inference. This must be called
  /// before [startServer] or [runVisionOcr].
  void setActiveModel(ModelInfo model) {
    _activeModel = model;
  }

  ModelInfo get _model {
    if (_activeModel == null) {
      throw StateError(
        'No active model set. Call setActiveModel() before using the server.',
      );
    }
    return _activeModel!;
  }

  void _logDebug(String tag, String message) {
    final line = '[${DateTime.now().toIso8601String()}] $tag: $message';
    _debugLogs.add(line);
    if (_debugLogs.length > 500) {
      _debugLogs.removeAt(0);
    }
  }

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

    // Detect available GPU backends and pick the best one.
    _availableBackends = await _detectAvailableBackends();
    _resolvedBackend = _resolveBackend();

    // Best-effort device facts for the diagnostics screen.
    await _collectDeviceInfo();

    // Verify the variant-specific binaries were extracted by the installer.
    // Fall back to CPU if the resolved backend binaries are missing (e.g.
    // the APK was built without Vulkan support).
    final variants = [_resolvedBackend, 'cpu'];
    var backendFound = false;
    for (final variant in variants) {
      final expected = [
        'libllama-server-$variant.so',
        'libllama-mtmd-cli-$variant.so',
      ];
      var allExist = true;
      for (final name in expected) {
        if (!File('$_nativeLibDir/$name').existsSync()) {
          allExist = false;
          break;
        }
      }
      if (allExist) {
        _resolvedBackend = variant;
        backendFound = true;
        break;
      }
    }

    if (!backendFound) {
      throw StateError(
        'Native binaries missing: no libllama-server-*.so or '
        'libllama-mtmd-cli-*.so found in $_nativeLibDir.\n'
        'The APK may have been built without llama.cpp jniLibs, '
        'or Android did not extract them at install (useLegacyPackaging=false?).',
      );
    }

    _binariesReady = true;
  }

  /// Configure GPU mode before [ensureBinaries] or [startServer] are called.
  void setGpuConfig({String? gpuMode, int? nGpuLayers}) {
    if (gpuMode != null) _gpuMode = gpuMode;
    if (nGpuLayers != null) _nGpuLayers = nGpuLayers;
  }

  /// Directories that hold vendor GPU driver libraries on Android. The
  /// OpenCL runtime (Adreno, Mali) lives here rather than in /system/lib64.
  ///
  /// A spawned binary resolves its DT_NEEDED `libOpenCL.so` through
  /// LD_LIBRARY_PATH, and these directories are where it can actually be
  /// found — verified on a Galaxy Note10+ (Adreno 640, Android 12), where a
  /// test binary linked against `libOpenCL.so` and run with
  /// LD_LIBRARY_PATH=/vendor/lib64 enumerated the Adreno device. LD_PRELOAD
  /// of the same library is rejected by the linker namespace, and Dart-side
  /// dlopen is useless: vendor public libraries are not visible to the app
  /// namespace at all without a manifest declaration, and loading a library
  /// says nothing about whether a backend can use it.
  static const _vendorLibDirs = [
    '/vendor/lib64',
    '/odm/lib64',
    '/system/vendor/lib64',
    '/system/vendor/lib',
  ];

  /// LD_LIBRARY_PATH for spawned llama.cpp processes: our own native lib
  /// dir first (libc++_shared.so), then the vendor driver directories.
  String get _childLdLibraryPath => [_libDir, ..._vendorLibDirs].join(':');

  /// Detect which GPU backends actually work on this device by asking the
  /// shipped binaries themselves. Always returns a set containing at least
  /// 'cpu'.
  ///
  /// Each variant's `llama-mtmd-cli` is run with `--list-devices`, which
  /// initializes the backend for real — including any runtime driver
  /// interaction a static check cannot see. A variant counts as available
  /// only if the binary lists a device.
  Future<Set<String>> _detectAvailableBackends() async {
    final backends = <String>{'cpu'};
    _backendProbe.clear();
    _backendProbeOutput.clear();

    for (final variant in const ['vulkan', 'opencl']) {
      final bin = '$_nativeLibDir/libllama-mtmd-cli-$variant.so';
      if (!File(bin).existsSync()) {
        _backendProbe.add('$variant: binary not in APK');
        _backendProbeOutput.add('===== $variant =====\n(binary not in APK)');
        continue;
      }

      try {
        final env = Map<String, String>.from(Platform.environment);
        env['LD_LIBRARY_PATH'] = _childLdLibraryPath;
        final result = await Process.run(bin, ['--list-devices'],
                environment: env, stdoutEncoding: utf8, stderrEncoding: utf8)
            .timeout(const Duration(seconds: 30));
        final stdout = result.stdout as String;
        final output = '$stdout\n${result.stderr}';
        _backendProbeOutput.add(
            '===== $variant (exit ${result.exitCode}) =====\n'
            '${output.length > 4000 ? '${output.substring(0, 4000)}…' : output}');
        // The device list goes to stdout and llama.cpp's log lines to stderr;
        // parsing the combined buffer counts a trailing error line as a
        // device, which is how Vulkan 1.1 hardware once passed as capable.
        if (listDevicesFoundDevice(stdout)) {
          _backendProbe.add('$variant: device listed');
          backends.add(variant);
        } else {
          final forErrors = output.replaceAll(_nativeLibDir ?? '', '<lib>');
          _backendProbe.add('$variant: no device — ${firstBackendErrorLine(forErrors)}');
        }
      } catch (e) {
        _backendProbe.add('$variant: probe failed — $e');
        _backendProbeOutput.add('===== $variant =====\n$e');
      }
    }

    return backends;
  }

  /// Fetch device properties and memory figures for the diagnostics screen.
  /// Every lookup is best-effort; a failure just leaves the field empty.
  Future<void> _collectDeviceInfo() async {
    for (final key in const [
      'ro.product.model',
      'ro.product.brand',
      'ro.soc.model',
      'ro.board.platform',
      'ro.build.version.release',
      'ro.build.version.sdk',
    ]) {
      try {
        final r = await Process.run('getprop', [key]);
        final value = (r.stdout as String).trim();
        if (value.isNotEmpty) _deviceProps[key] = value;
      } catch (_) {}
    }

    try {
      final lines = await File('/proc/meminfo').readAsLines();
      for (final line in lines.take(3)) {
        final parts = line.split(RegExp(r'\s+'));
        if (parts.length >= 2) {
          final kb = int.tryParse(parts[1]);
          if (kb != null) {
            if (parts[0] == 'MemTotal:') _memTotalKb = kb;
            if (parts[0] == 'MemAvailable:') _memAvailableKb = kb;
          }
        }
      }
    } catch (_) {}
  }

  /// Resolve the backend to use given the user preference and what's
  /// actually available on the device. Falls back to CPU if the requested
  /// backend is unavailable.
  String _resolveBackend() {
    final available = _availableBackends;
    switch (_gpuMode) {
      case 'vulkan':
        return available.contains('vulkan') ? 'vulkan' : 'cpu';
      case 'opencl':
        return available.contains('opencl') ? 'opencl' : 'cpu';
      case 'cpu':
        return 'cpu';
      case 'auto':
      default:
        // Prefer Vulkan > OpenCL > CPU.
        if (available.contains('vulkan')) return 'vulkan';
        if (available.contains('opencl')) return 'opencl';
        return 'cpu';
    }
  }

  String get _libDir => _nativeLibDir!;

  String get _serverBinary =>
      '$_nativeLibDir/libllama-server-$_resolvedBackend.so';
  String get _mtmdBinary =>
      '$_nativeLibDir/libllama-mtmd-cli-$_resolvedBackend.so';

  // ---------------------------------------------------------------------------
  // Model paths
  // ---------------------------------------------------------------------------

  /// Directory where GGUF models are stored.
  Directory get modelDir => Directory('${_appDir!.path}/models');

  String get modelPath => '${modelDir.path}/${_model.modelFilename}';
  String get mmprojPath => '${modelDir.path}/${_model.mmprojFilename}';

  Future<bool> get modelsExist async {
    return File(modelPath).existsSync() && File(mmprojPath).existsSync();
  }

  /// Where a successful verification is remembered.
  ///
  /// Hashing the model costs a full read of every weight file - over 4 GB with
  /// the projector - in pure Dart, which on a phone is minutes of a frozen
  /// screen. Doing that on every launch to re-prove something that cannot have
  /// changed is what made startup look like it had hung.
  File get _verificationMarker =>
      File('${modelDir.path}/.verified.json');

  /// Verify model files against the active model's SHA-256 hashes.
  ///
  /// Files are hashed once and then remembered against their size and
  /// modification time, so a launch that has not touched them skips straight
  /// past. A download that replaces a file changes both, so it is hashed again.
  ///
  /// Returns `true` if all files match. Returns `false` if any file is missing,
  /// the wrong size, or has a mismatched hash — the caller should trigger
  /// re-download in this case.
  ///
  /// [onProgress] receives the fraction of the work done, which is dominated by
  /// hashing, so a first-run verification can at least show movement.
  Future<bool> verifyModels({void Function(double)? onProgress}) async {
    await ensureBinaries();

    final model = _model;

    final files = <(String path, int size, String hash)>[
      (modelPath, model.modelSizeBytes, model.sha256Model),
      if (model.supportsVision)
        (mmprojPath, model.mmprojSizeBytes, model.sha256Mmproj),
    ];

    final remembered = await _readVerificationMarker();
    final updated = <String, Map<String, int>>{};
    var index = 0;

    for (final (path, expectedSize, expectedHash) in files) {
      final file = File(path);
      if (!file.existsSync()) return false;

      final stat = file.statSync();
      final actualSize = stat.size;
      if (actualSize != expectedSize) return false;

      final name = path.split('/').last;
      final mtime = stat.modified.millisecondsSinceEpoch;
      final previous = remembered[name];

      final unchanged = previous != null &&
          previous['size'] == actualSize &&
          previous['mtime'] == mtime;

      if (!unchanged) {
        final sink = _DigestSink();
        final input = sha256.startChunkedConversion(sink);
        await for (final chunk in file.openRead()) {
          input.add(chunk);
        }
        input.close();
        final actualHash = sink.digest.toString();
        if (actualHash != expectedHash) return false;
      }

      updated[name] = {'size': actualSize, 'mtime': mtime};
      index++;
      onProgress?.call(index / files.length);
    }

    await _writeVerificationMarker(updated);
    return true;
  }

  Future<Map<String, Map<String, int>>> _readVerificationMarker() async {
    try {
      if (!await _verificationMarker.exists()) return {};
      final decoded = jsonDecode(await _verificationMarker.readAsString());
      final out = <String, Map<String, int>>{};
      (decoded as Map<String, dynamic>).forEach((key, value) {
        final entry = value as Map<String, dynamic>;
        out[key] = {
          'size': (entry['size'] as num).toInt(),
          'mtime': (entry['mtime'] as num).toInt(),
        };
      });
      return out;
    } catch (_) {
      // An unreadable marker only costs a re-hash, so ignore it.
      return {};
    }
  }

  Future<void> _writeVerificationMarker(Map<String, Map<String, int>> data) async {
    try {
      await _verificationMarker.writeAsString(jsonEncode(data));
    } catch (_) {
      // Likewise: failing to record it just means hashing again next launch.
    }
  }

  // ---------------------------------------------------------------------------
  // llama-server lifecycle
  // ---------------------------------------------------------------------------

  /// Start llama-server with the text-generation model.
  ///
  /// A GPU backend can pass detection and still fail when the server
  /// actually initializes it (driver quirks the probe did not exercise).
  /// In that case the server is retried once on CPU, and the reason is
  /// kept for the diagnostics screen rather than surfaced as a fatal
  /// error — slow-but-working beats a wall of text.
  Future<void> startServer() async {
    try {
      await _startServerWithBackend();
    } catch (e) {
      if (_resolvedBackend == 'cpu') rethrow;
      _backendFallbackReason = e.toString();
      _resolvedBackend = 'cpu';
      await stopServer();
      await _startServerWithBackend();
    }
  }

  Future<void> _startServerWithBackend() async {
    await ensureBinaries();

    if (_serverProcess != null) return;

    // Start a foreground service so Android (especially aggressive OEM
    // skins like Samsung One UI) does not SIGTERM the child process
    // during the multi-GB model load. The service stays alive as long as
    // the server is running.
    try {
      await ForegroundTaskService.start(
        detail: 'Loading AI model — this may take up to a minute…',
      );
    } catch (_) {}

    final model = modelPath;
    if (!File(model).existsSync()) {
      throw StateError('Model not found: $model');
    }

    final port = 8090;
    final binaryPath = _serverBinary;
    final env = Map<String, String>.from(Platform.environment);
    // libc++_shared.so comes from our own lib dir; GPU driver libraries
    // (OpenCL on Adreno/Mali) come from the vendor partitions. Our dir
    // first: it never contains a driver library, so the vendor one wins.
    env['LD_LIBRARY_PATH'] = _childLdLibraryPath;

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

    // Quick sanity check: run the binary with --version to verify it
    // executes at all and to capture any dynamic-linker errors.
    try {
      final versionResult = await Process.run(
        binaryPath,
        ['--version'],
        environment: env,
        stdoutEncoding: utf8,
        stderrEncoding: utf8,
      );
      if (versionResult.exitCode != 0) {
        throw StateError(
          'llama-server --version failed (exit ${versionResult.exitCode}).\n'
          '${binDiagnostics()}\n\n'
          '--- stdout ---\n${versionResult.stdout}\n'
          '--- stderr ---\n${versionResult.stderr}',
        );
      }
    } on ProcessException catch (pe) {
      throw StateError(
        'Failed to launch llama-server (binary sanity check).\n'
        'ProcessException: ${pe.message}\n'
        'errno:    ${pe.errorCode}\n'
        'executable: ${pe.executable}\n'
        '${binDiagnostics()}',
      );
    }

    _serverStderrLines.clear();
    try {
      final args = [
        '-m', model,
        '--host', '127.0.0.1',
        '--port', '$port',
        '--ctx-size', '${_model.contextSize}',
        '--parallel', '1',
        '--jinja',
        '--cache-ram', '0',
        '--cache-type-k', 'q4_0',
        '--no-webui',
        // Phones are heterogeneous: e.g. a Snapdragon 855 is 4 big + 4 little
        // cores. The default of one thread per core saturates the little
        // cluster too, which slows token generation AND starves the Flutter
        // UI thread — a rotation or scroll visibly janks the whole app while
        // a chunk generates. Four threads track the big cores, which is
        // where llama.cpp wants to be anyway on big.LITTLE.
        if (_resolvedBackend == 'cpu') ...[
          '--threads', '4',
        ],
        if (_resolvedBackend != 'cpu') ...[
          '-ngl', '$_nGpuLayers',
        ],
      ];

      _serverProcess = await Process.start(
        binaryPath,
        args,
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
      _serverStderrLines.add('[stderr] $line');
      if (_serverStderrLines.length > 200) {
        _serverStderrLines.removeAt(0);
      }
    }, onError: (_) {});

    // Tee stdout into the same buffer — llama.cpp logs INFO messages to
    // stdout and ERROR messages to stderr. We need both for diagnostics.
    _serverProcess!.stdout
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen((line) {
      _serverStderrLines.add('[stdout] $line');
      if (_serverStderrLines.length > 200) {
        _serverStderrLines.removeAt(0);
      }
    }, onError: (_) {});

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
        if (resp.statusCode == 200) {
          try {
            await ForegroundTaskService.update(
              detail: 'AI model ready',
            );
          } catch (_) {}
          return;
        }
      } catch (e) {
        lastError = e.toString();
      }

      // Non-blocking check: did the process exit?
      if (earlyExitCode != null) {
        // Give the stdout/stderr stream listeners a moment to process
        // any pending data on the Dart event loop before reading the buffer.
        await Future<void>.delayed(const Duration(milliseconds: 800));
        final errText = _serverStderrLines.isNotEmpty
            ? _serverStderrLines.join('\n')
            : '(empty)';
        _serverProcess = null;
        _serverUrl = null;
        throw StateError(
          'llama-server exited prematurely (exit code $earlyExitCode).\n'
          '${binDiagnostics()}\n\n'
          '--- output ---\n$errText',
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
      '--- output ---\n$errText',
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
    if (_serverProcess != null) {
      // Try graceful SIGTERM first.
      _serverProcess!.kill(ProcessSignal.sigterm);

      // Wait up to 5s for clean exit.
      try {
        await _serverProcess!.exitCode.timeout(const Duration(seconds: 5));
      } catch (_) {
        // Force kill if still running.
        _serverProcess!.kill(ProcessSignal.sigkill);
      }
    }

    _serverProcess = null;
    _serverUrl = null;

    // Stop the foreground service now that the server is down. Unconditional
    // so a failed start attempt also cleans up before a fallback retry.
    try {
      await ForegroundTaskService.stop();
    } catch (_) {}
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
        'Transcribe every visible word in the image. Output exactly one word per line. '
        'Plain text only. No line numbers, no markdown, no headers, no explanations, no commentary.',
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
        env['LD_LIBRARY_PATH'] = _childLdLibraryPath;
      }

      final args = [
        '-m', modelPath,
        '--mmproj', mmprojPath,
        '--image', tmpFile.path,
        '-p', prompt,
        '-n', '512',
        '--temp', '0.1',
        '--jinja',
        if (_resolvedBackend != 'cpu') ...[
          '-ngl', '$_nGpuLayers',
        ],
      ];

      final result = await Process.run(
        _mtmdBinary,
        args,
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
  /// Uses multiple heuristics to find the actual generated text and
  /// filters out instruction artifacts produced by chat-templated models.
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
      RegExp(r'^srv\s'),
      RegExp(r'^main:\s'),
      RegExp(r'^\s*$'),
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
      if (trimmed.length <= 1 && !trimmed.contains(RegExp(r'[a-zA-Z]'))) {
        startIdx = i + 1;
      } else {
        break;
      }
    }

    var result = bestBlock.sublist(startIdx);

    // Heuristic 3: Strip instruction artifacts from chat-templated models.
    final cleaned = <String>[];
    for (final line in result) {
      var text = line.trim();
      if (text.isEmpty) continue;

      // Skip markdown headers / bold markers (e.g. **List all identified words:**)
      if (text.startsWith('**') && text.endsWith('**')) continue;
      if (text.startsWith('*') && text.endsWith('*')) continue;
      if (text.startsWith('#')) continue;

      // Skip lines that are pure punctuation or separators.
      if (!text.contains(RegExp(r'[a-zA-Z0-9À-ɏЀ-ӿ]'))) {
        continue;
      }

      // Remove "Line N:" prefixes (e.g. "Line 1: NO" → "NO").
      text = text.replaceFirst(RegExp(r'^Line\s+\d+[:\.)\-]?\s*', caseSensitive: false), '');

      // Remove leading bullet markers and numbering.
      text = text.replaceFirst(RegExp(r'^[\s*\-•·\d+\.\)]*\s*'), '');

      // Skip commentary / parenthetical explanations.
      if (RegExp(r'\b(not distinct|identified words|constraints|review the|list all)\b', caseSensitive: false).hasMatch(text)) {
        continue;
      }

      // Skip if after stripping it became empty or just punctuation.
      text = text.trim();
      if (text.isEmpty) continue;
      if (!text.contains(RegExp(r'[a-zA-Z0-9À-ɏЀ-ӿ]'))) {
        continue;
      }

      // Skip pure numbers (e.g. "42", "123").
      if (RegExp(r'^\d+$').hasMatch(text)) {
        continue;
      }

      cleaned.add(text);
    }

    return cleaned.join('\n').trim();
  }

  // ---------------------------------------------------------------------------
  // Text generation via llama-server
  // ---------------------------------------------------------------------------

  /// Generate text via llama-server's chat endpoint.
  ///
  /// /v1/chat/completions rather than /completion: the [system] message is
  /// applied by the model's own chat template (--jinja is set at server
  /// start), and instruction-tuned models follow format instructions given
  /// that way far more reliably than as a raw continuation. Gemma 4 sent a
  /// bare completion drifted into stray bullet fragments and occasionally
  /// echoed the word list instead of answering.
  Future<String> generate({
    required String system,
    required String prompt,
    int maxTokens = 256,
    double temperature = 0.7,
  }) async {
    if (_serverUrl == null) {
      throw StateError('llama-server is not running');
    }

    final uri = Uri.parse('$_serverUrl/v1/chat/completions');
    _generateClient = http.Client();

    // Gemma 3 uses <end_of_turn> and <eos> as stop tokens.
    // Include them first so generation stops at the right place.
    final stopTokens = <String>[
      '<end_of_turn>',
      '<eos>',
      '</s>',
      'User:',
      'Assistant:',
    ];

    try {
      final payload = {
        'messages': [
          {'role': 'system', 'content': system},
          {'role': 'user', 'content': prompt},
        ],
        'max_tokens': maxTokens,
        'temperature': temperature,
        'stop': stopTokens,
        'cache_prompt': false,
      };

      _logDebug('generate', 'system=${system.substring(0, system.length.clamp(0, 120))}... '
          'prompt=${prompt.substring(0, prompt.length.clamp(0, 200))}... '
          'n_predict=$maxTokens temp=$temperature');

      final response = await _generateClient!
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(minutes: 10));

      if (response.statusCode != 200) {
        _logDebug('generate', 'HTTP ${response.statusCode}: ${response.body}');
        throw Exception(
          'Generation failed (${response.statusCode}): ${response.body}',
        );
      }

      final body = jsonDecode(response.body) as Map<String, dynamic>;
      final choices = body['choices'] as List<dynamic>? ?? const [];
      String content = '';
      if (choices.isNotEmpty) {
        final message = choices.first['message'] as Map<String, dynamic>?;
        content = message?['content'] as String? ?? '';
      }
      _logDebug('generate', 'content=${content.substring(0, content.length.clamp(0, 200))}...');
      return content;
    } finally {
      _generateClient = null;
    }
  }

  /// Cancel any in-flight generation request.
  ///
  /// Closes the HTTP client to abort the current request to llama-server.
  void cancelGeneration() {
    _generateClient?.close();
    _generateClient = null;
  }

  // ---------------------------------------------------------------------------
  // Diagnostics
  // ---------------------------------------------------------------------------

  /// Returns a snapshot of the current server state for debugging.
  Map<String, dynamic> getDiagnostics() {
    final model = _activeModel;
    final modelFile = model != null ? File(modelPath) : null;
    final mmprojFile = model != null && model.supportsVision
        ? File(mmprojPath)
        : null;

    final libVariants = <String, bool>{};
    for (final variant in const ['cpu', 'vulkan', 'opencl']) {
      libVariants[variant] = _nativeLibDir != null &&
          File('$_nativeLibDir/libllama-server-$variant.so').existsSync() &&
          File('$_nativeLibDir/libllama-mtmd-cli-$variant.so').existsSync();
    }

    return {
      'serverRunning': isServerRunning,
      'serverUrl': _serverUrl,
      'resolvedBackend': _resolvedBackend,
      'availableBackends': _availableBackends.toList(),
      'backendProbe': _backendProbe.toList(),
      'backendProbeOutput': _backendProbeOutput.toList(),
      'backendFallbackReason': _backendFallbackReason,
      'nativeLibVariants': libVariants,
      'ldLibraryPath': _nativeLibDir != null ? _childLdLibraryPath : null,
      'device': _deviceProps.isNotEmpty ? Map<String, String>.of(_deviceProps) : null,
      'memTotalKb': _memTotalKb,
      'memAvailableKb': _memAvailableKb,
      'gpuMode': _gpuMode,
      'nGpuLayers': _nGpuLayers,
      'nativeLibDir': _nativeLibDir,
      'modelPath': modelFile?.path,
      'modelExists': modelFile?.existsSync() ?? false,
      'modelSize': modelFile?.lengthSync() ?? 0,
      'mmprojPath': mmprojFile?.path,
      'mmprojExists': mmprojFile?.existsSync() ?? false,
      'mmprojSize': mmprojFile?.lengthSync() ?? 0,
      'activeModelId': model?.id,
      'activeModelName': model?.name,
      'recentLogs': _serverStderrLines.toList(),
      'debugLogs': _debugLogs.toList(),
    };
  }
}
