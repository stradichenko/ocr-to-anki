import 'dart:async';
import 'dart:io';

import 'package:http/http.dart' as http;

/// Manages the lifecycle of the FastAPI backend server (uvicorn).
///
/// Spawns the server as a child process when the app starts and kills it
/// when the app closes, so the user never has to think about it.
///
/// When running from a release bundle the backend source lives in a `backend/`
/// directory next to the executable.  In that case we create (or reuse) a
/// Python virtual environment there and install the requirements before
/// starting uvicorn.  This means users only need a system Python 3 -- they do
/// not have to install any pip packages by hand.
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
  Future<void> start({Duration timeout = const Duration(seconds: 60)}) async {
    if (_process != null) return; // already running

    // Resolve the project / backend root using this priority:
    //
    //   1. OCR_TO_ANKI_ROOT env var  (set by run.sh or the user)
    //   2. <exe_dir>/backend/        (release bundle, user ran ./ocr_to_anki)
    //   3. Walk up from CWD looking for flake.nix  (nix develop / dev mode)
    //
    // In a release bundle the layout is:
    //   <dir>/ocr_to_anki            (Flutter binary)
    //   <dir>/backend/src/           (Python source)
    //   <dir>/backend/config/        (settings.yaml)
    //   <dir>/backend/requirements.txt
    //   <dir>/run.sh                 (optional launcher)
    final projectRoot = _resolveProjectRoot();

    // Check whether projectRoot contains backend source directly
    // (OCR_TO_ANKI_ROOT pointed at the backend/ dir, or dev mode repo root).
    final isBundled = File('$projectRoot/requirements.txt').existsSync() &&
        Directory('$projectRoot/src').existsSync();

    String python;
    if (isBundled) {
      python = await _ensureVenv(projectRoot);
    } else {
      final found = _which('python3') ?? _which('python');
      if (found == null) {
        throw StateError(
          'Cannot find python3 or python in PATH.\n'
          'Make sure the app is launched inside the nix develop shell,\n'
          'or run from the release bundle which sets up a venv automatically.',
        );
      }
      python = found;
    }

    _logLines.clear();
    _log('Starting backend: $python -m uvicorn src.api.app:app');
    _log('Project root: $projectRoot');
    _log('Bundled mode: $isBundled');

    // Resolve the executable's directory for bundled shared libraries.
    final exeDir = File(Platform.resolvedExecutable).parent.path;

    final env = <String, String>{
      ...Platform.environment,
      // Ensure ~/.local/bin is in PATH so llama-server-vulkan is found.
      'PATH': '${Platform.environment['HOME']}/.local/bin:'
          '${Platform.environment['PATH'] ?? ''}',
      // The app uses bare `from api.models import …` and
      // `from backends.… import …` (without the `src.` prefix).
      // In dev mode the Nix shell adds src/ to PYTHONPATH; replicate that here.
      'PYTHONPATH': '$projectRoot/src:'
          '${Platform.environment['PYTHONPATH'] ?? ''}',
      // Include bundled shared libs (libstdc++ etc.) so pip-installed native
      // extensions (numpy, etc.) work on systems without FHS layout (NixOS).
      if (isBundled)
        'LD_LIBRARY_PATH': '$exeDir/lib:'
            '${Platform.environment['LD_LIBRARY_PATH'] ?? ''}',
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

  /// Ensure a Python virtual environment exists inside [backendRoot] and
  /// that the requirements are installed.  Returns the path to the venv
  /// python binary.
  Future<String> _ensureVenv(String backendRoot) async {
    final venvDir = '$backendRoot/.venv';
    final isWindows = Platform.isWindows;
    final venvPython = isWindows
        ? '$venvDir/Scripts/python.exe'
        : '$venvDir/bin/python';

    // Find system python.
    final sysPython = _which('python3') ?? _which('python');
    if (sysPython == null) {
      throw StateError(
        'Cannot find python3 or python in PATH.\n'
        'Please install Python 3.10 or newer.',
      );
    }

    // Create venv if it doesn't exist.
    if (!File(venvPython).existsSync()) {
      _log('Creating Python virtual environment in $venvDir ...');
      final createResult = await Process.run(
        sysPython,
        ['-m', 'venv', venvDir],
        workingDirectory: backendRoot,
      );
      if (createResult.exitCode != 0) {
        throw StateError(
          'Failed to create Python venv:\n${createResult.stderr}',
        );
      }
    }

    // Install / update requirements if the marker is stale or missing.
    final marker = File('$venvDir/.deps_installed');
    final reqFile = File('$backendRoot/requirements.txt');
    final needsInstall = !marker.existsSync() ||
        marker.lastModifiedSync().isBefore(reqFile.lastModifiedSync());

    if (needsInstall) {
      _log('Installing Python dependencies (first run may take a moment)...');
      final pipResult = await Process.run(
        venvPython,
        ['-m', 'pip', 'install', '--quiet', '-r', 'requirements.txt'],
        workingDirectory: backendRoot,
      );
      if (pipResult.exitCode != 0) {
        throw StateError(
          'pip install failed:\n${pipResult.stderr}',
        );
      }
      // Write marker so we skip next time.
      marker.writeAsStringSync(DateTime.now().toIso8601String());
      _log('Dependencies installed.');
    }

    return venvPython;
  }

  /// Resolve the backend root directory.
  ///
  /// Priority:
  ///   1. `OCR_TO_ANKI_ROOT` environment variable (set by run.sh or user).
  ///   2. `backend/` directory next to the running executable (release bundle).
  ///   3. Walk up from CWD looking for `flake.nix` (development mode).
  String _resolveProjectRoot() {
    // 1. Explicit env var.
    final envRoot = Platform.environment['OCR_TO_ANKI_ROOT'];
    if (envRoot != null && envRoot.isNotEmpty) return envRoot;

    // 2. Release bundle: look for backend/ next to the executable.
    //    Platform.resolvedExecutable gives the absolute path to ocr_to_anki.
    final exeDir = File(Platform.resolvedExecutable).parent.path;
    final bundleBackend = '$exeDir/backend';
    if (File('$bundleBackend/requirements.txt').existsSync() &&
        Directory('$bundleBackend/src').existsSync()) {
      return bundleBackend;
    }

    // 3. Dev mode: walk up from CWD looking for flake.nix.
    var dir = Directory.current;
    if (dir.path.endsWith('/app')) {
      dir = dir.parent;
    }
    for (var d = dir; d.path != d.parent.path; d = d.parent) {
      if (File('${d.path}/flake.nix').existsSync()) return d.path;
    }

    // Fallback: assume CWD is fine.
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
