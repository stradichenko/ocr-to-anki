import 'dart:async';
import 'dart:io';

import 'package:http/http.dart' as http;

/// Manages the lifecycle of the FastAPI backend server (uvicorn).
///
/// Spawns the server as a child process when the app starts and kills it
/// when the app closes, so the user never has to think about it.
class BackendServerService {
  BackendServerService({
    this.host = '0.0.0.0',
    this.port = 8000,
  });

  final String host;
  final int port;

  Process? _process;
  final _logLines = <String>[];

  /// Whether the server process has been spawned.
  bool get isRunning => _process != null;

  /// Recent log lines from the server (kept capped at 200).
  List<String> get logs => List.unmodifiable(_logLines);

  /// The URL of the running backend.
  String get url => 'http://127.0.0.1:$port';

  /// Start the backend server.
  ///
  /// Resolves `python` from the current PATH (set up by `nix develop`).
  /// Returns once the server responds to `/health`, or throws on timeout.
  Future<void> start({Duration timeout = const Duration(seconds: 30)}) async {
    if (_process != null) return; // already running

    // Find the project root.  The Flutter app lives in <root>/app/, so we
    // go one level up from the executable's directory.  When running via
    // `flutter run -d linux` the working directory is already correct, but
    // to be safe we also check Platform.environment for an override.
    final projectRoot = Platform.environment['OCR_TO_ANKI_ROOT'] ??
        _findProjectRoot();

    // Resolve python from PATH (nix develop puts it there).
    final python = _which('python3') ?? _which('python');
    if (python == null) {
      throw StateError(
        'Cannot find python3 or python in PATH.\n'
        'Make sure the app is launched inside the nix develop shell.',
      );
    }

    _logLines.clear();
    _log('Starting backend: $python -m uvicorn src.api.app:app');
    _log('Project root: $projectRoot');

    final env = <String, String>{
      ...Platform.environment,
      // Ensure ~/.local/bin is in PATH so llama-server-vulkan is found.
      'PATH': '${Platform.environment['HOME']}/.local/bin:'
          '${Platform.environment['PATH'] ?? ''}',
    };

    _process = await Process.start(
      python,
      [
        '-m',
        'uvicorn',
        'src.api.app:app',
        '--host',
        host,
        '--port',
        '$port',
      ],
      workingDirectory: projectRoot,
      environment: env,
    );

    // Drain stdout / stderr into our log buffer.
    _process!.stdout.transform(const SystemEncoding().decoder).listen((data) {
      for (final line in data.split('\n')) {
        if (line.trim().isNotEmpty) _log('[server] $line');
      }
    });
    _process!.stderr.transform(const SystemEncoding().decoder).listen((data) {
      for (final line in data.split('\n')) {
        if (line.trim().isNotEmpty) _log('[server] $line');
      }
    });

    // Handle unexpected exit.
    _process!.exitCode.then((code) {
      _log('Backend process exited with code $code');
      _process = null;
    });

    // Poll /health until the server is ready.
    final deadline = DateTime.now().add(timeout);
    var ready = false;
    while (DateTime.now().isBefore(deadline)) {
      try {
        final resp = await http
            .get(Uri.parse('$url/health'))
            .timeout(const Duration(seconds: 2));
        if (resp.statusCode == 200) {
          ready = true;
          break;
        }
      } catch (_) {
        // Server not ready yet.
      }
      await Future.delayed(const Duration(milliseconds: 500));
    }

    if (!ready) {
      // Clean up the process if we never got healthy.
      await stop();
      throw TimeoutException(
        'Backend server did not become ready within ${timeout.inSeconds}s.\n'
        'Recent logs:\n${_logLines.take(20).join('\n')}',
      );
    }

    _log('Backend ready at $url');
  }

  /// Stop the backend server (and any child llama-server it spawned).
  Future<void> stop() async {
    final proc = _process;
    if (proc == null) return;

    _log('Stopping backend server (PID ${proc.pid})...');
    _process = null;

    // Send SIGTERM so uvicorn runs its shutdown hooks (which stop llama-server).
    proc.kill(ProcessSignal.sigterm);

    // Give it a couple seconds to shut down gracefully.
    try {
      await proc.exitCode.timeout(const Duration(seconds: 5));
    } on TimeoutException {
      _log('Graceful shutdown timed out, sending SIGKILL');
      proc.kill(ProcessSignal.sigkill);
    }

    // Belt-and-suspenders: kill any orphaned llama-server on the port.
    try {
      await Process.run('fuser', ['-k', '$port/tcp']);
    } catch (_) {}

    _log('Backend stopped.');
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  void _log(String message) {
    _logLines.add(message);
    if (_logLines.length > 200) _logLines.removeAt(0);
  }

  /// Walk up from the current working directory looking for `flake.nix`.
  String _findProjectRoot() {
    var dir = Directory.current;
    // If we're inside the app/ folder, go up.
    if (dir.path.endsWith('/app')) {
      dir = dir.parent;
    }
    for (var d = dir; d.path != d.parent.path; d = d.parent) {
      if (File('${d.path}/flake.nix').existsSync()) return d.path;
    }
    // Fallback: assume cwd is fine.
    return dir.path;
  }

  String? _which(String cmd) {
    try {
      final result = Process.runSync('which', [cmd]);
      if (result.exitCode == 0) {
        return (result.stdout as String).trim();
      }
    } catch (_) {}
    return null;
  }
}
