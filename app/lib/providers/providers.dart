import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:drift/drift.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';

import '../database/database.dart';
import '../models/models.dart';
import '../models/pending_batch.dart';
import '../services/services.dart';

// ---------------------------------------------------------------------------
// Database -- singleton
// ---------------------------------------------------------------------------

final databaseProvider = Provider<AppDatabase>((ref) {
  final db = AppDatabase();
  ref.onDispose(() => db.close());
  return db;
});

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

/// Derived provider so MaterialApp only rebuilds on theme changes.
final themeModeProvider = Provider<ThemeMode>((ref) {
  return ref.watch(settingsProvider).themeMode;
});

/// Derived provider for the color scheme seed.
final colorSeedProvider = Provider<Color>((ref) {
  final settings = ref.watch(settingsProvider);
  return resolveColorSeed(settings.colorSchemeSeed, settings.customColorHex);
});

/// Maps a named scheme or custom hex to a Color.
Color resolveColorSeed(String schemeName, String customHex) {
  switch (schemeName) {
    case 'deepOrange':
      return Colors.deepOrange;
    case 'blue':
      return Colors.blue;
    case 'teal':
      return Colors.teal;
    case 'purple':
      return Colors.deepPurple;
    case 'green':
      return Colors.green;
    case 'red':
      return Colors.red;
    case 'indigo':
      return Colors.indigo;
    case 'amber':
      return Colors.amber;
    case 'cyan':
      return Colors.cyan;
    case 'pink':
      return Colors.pink;
    case 'lime':
      return Colors.lime;
    case 'brown':
      return Colors.brown;
    case 'custom':
      if (customHex.isNotEmpty) {
        final hex = customHex.replaceFirst('#', '');
        if (hex.length == 6) {
          final value = int.tryParse(hex, radix: 16);
          if (value != null) return Color(0xFF000000 | value);
        }
      }
      return Colors.deepOrange;
    default:
      return Colors.deepOrange;
  }
}

/// Named color schemes available in the picker.
const kColorSchemes = <String, String>{
  'deepOrange': 'Deep Orange',
  'blue': 'Blue',
  'teal': 'Teal',
  'purple': 'Purple',
  'green': 'Green',
  'red': 'Red',
  'indigo': 'Indigo',
  'amber': 'Amber',
  'cyan': 'Cyan',
  'pink': 'Pink',
  'lime': 'Lime',
  'brown': 'Brown',
  'custom': 'Custom',
};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

final settingsProvider =
    NotifierProvider<SettingsNotifier, AppSettings>(SettingsNotifier.new);

class SettingsNotifier extends Notifier<AppSettings> {
  late AppDatabase _db;

  @override
  AppSettings build() {
    _db = ref.watch(databaseProvider);
    _load();
    return AppSettings();
  }

  Future<void> _load() async {
    final json = await _db.getSetting('app_settings');
    if (json != null) {
      try {
        var loaded = AppSettings.fromJson(
          jsonDecode(json) as Map<String, dynamic>,
        );
        // Migrations:
        var dirty = false;
        // 1) "localhost" resolves to IPv6 ::1 on some systems while the
        //    server only listens on IPv4, causing connection failures.
        if (loaded.serverUrl.contains('://localhost:')) {
          loaded.serverUrl =
              loaded.serverUrl.replaceFirst('://localhost:', '://127.0.0.1:');
          dirty = true;
        }
        state = loaded; // always assign a fresh object so listeners fire
        if (dirty) {
          await _db.setSetting('app_settings', jsonEncode(state.toJson()));
        }
      } catch (_) {
        // Corrupted settings -- keep defaults.
      }
    }
  }

  Future<void> update(AppSettings Function(AppSettings) updater) async {
    // Clone via JSON round-trip so the updater mutates a FRESH instance.
    // StateNotifier only notifies when the object reference changes;
    // cascade operators (s..field = x) return the same instance so
    // without this clone the UI would never rebuild.
    final copy = AppSettings.fromJson(state.toJson());
    state = updater(copy);
    await _db.setSetting('app_settings', jsonEncode(state.toJson()));
  }
}

// ---------------------------------------------------------------------------
// Services
// ---------------------------------------------------------------------------

/// Singleton backend server instance.  Created once, shared across the app.
final backendServerProvider = Provider<BackendServerService>((ref) {
  final server = BackendServerService();
  ref.onDispose(() {
    server.stop(); // fire-and-forget on provider teardown
  });
  return server;
});

/// Android-native llama.cpp service (binaries + server lifecycle).
final llamaCppAndroidProvider = Provider<LlamaCppAndroidService>((ref) {
  final service = LlamaCppAndroidService();
  ref.onDispose(() {
    service.stopServer(); // fire-and-forget
  });
  return service;
});

/// Model download service for Android.
final modelDownloadProvider = Provider<ModelDownloadService>((ref) {
  return ModelDownloadService();
});

/// Model registry service (loads assets/models.json).
final modelRegistryProvider = Provider<ModelRegistryService>((ref) {
  return ModelRegistryService();
});

/// Tracks whether the backend server is ready.
///
/// The [ServerStartupNotifier] is initialised eagerly by the startup gate
/// widget.  It starts the backend and exposes the current status so the UI
/// can show a loading / error screen.
enum ServerStatus {
  starting,
  ready,
  error,
  downloadingPython,
  downloadingLlama,
  downloading,
}

class ServerStartupState {
  const ServerStartupState({
    this.status = ServerStatus.starting,
    this.message = 'Starting backend server…',
    this.downloadFile = '',
    this.downloadedBytes = 0,
    this.totalBytes = 0,
    this.technicalDetail,
  });
  final ServerStatus status;
  final String message;

  /// Currently-downloading file name.
  final String downloadFile;

  /// Bytes downloaded so far.
  final int downloadedBytes;

  /// Total bytes expected (0 = unknown).
  final int totalBytes;

  /// Raw exception text + stderr/log buffer for the error screen's
  /// "technical details" panel and "Copy diagnostics" button.
  final String? technicalDetail;

  /// Convenience: download progress as 0..1 (or 0 when unknown).
  double get downloadProgress =>
      totalBytes > 0 ? downloadedBytes / totalBytes : 0;
}

final serverStartupProvider =
    NotifierProvider<ServerStartupNotifier, ServerStartupState>(
        ServerStartupNotifier.new);

class ServerStartupNotifier extends Notifier<ServerStartupState> {
  bool _disposed = false;

  @override
  ServerStartupState build() {
    _disposed = false;
    ref.onDispose(() => _disposed = true);
    _boot();
    return const ServerStartupState();
  }

  Future<void> _boot() async {
    if (Platform.isAndroid) {
      await _bootAndroid();
      return;
    }
    await _bootDesktop();
  }

  /// Android boot sequence: extract native binaries, download models, start llama-server.
  Future<void> _bootAndroid() async {
    try {
      final llama = ref.read(llamaCppAndroidProvider);
      final settings = ref.read(settingsProvider);
      final registry = ref.read(modelRegistryProvider);

      // Resolve the active model from settings (falls back to default).
      final model = await registry.getModel(settings.activeModelId) ??
          await registry.getDefaultModel();
      llama.setActiveModel(model);

      // Pass GPU preference so ensureBinaries can verify the right variant.
      llama.setGpuConfig(
        gpuMode: settings.gpuMode,
        nGpuLayers: settings.nGpuLayers,
      );

      // Step 1: Extract native binaries from assets.
      state = const ServerStartupState(
        status: ServerStatus.starting,
        message: 'Extracting native binaries…',
      );
      await llama.ensureBinaries();
      if (_disposed) return;

      // Step 2: Check / download models.
      var modelsExist = await llama.modelsExist;
      if (_disposed) return;

      // Verify checksums if files exist — corrupted downloads cause cryptic
      // crashes inside llama-server, so we catch them early.
      if (modelsExist) {
        state = const ServerStartupState(
          status: ServerStatus.starting,
          message: 'Verifying model integrity…',
        );
        final ok = await llama.verifyModels();
        if (_disposed) return;
        if (!ok) {
          // Delete corrupt files so the download path below re-fetches them.
          try {
            File(llama.modelPath).deleteSync();
          } catch (_) {}
          try {
            File(llama.mmprojPath).deleteSync();
          } catch (_) {}
          modelsExist = false;
        }
      }

      if (!modelsExist) {
        final settings = ref.read(settingsProvider);

        final totalGb = (model.totalSizeBytes / (1024 * 1024 * 1024))
            .toStringAsFixed(1);

        // Storage guard — refuse to start a download when the partition
        // does not have enough headroom (models + ~1 GB for .part files,
        // OS overhead, and post-download mv into place).
        final freeBytes = await SystemChannel.getAvailableStorageBytes();
        final requiredBytes = model.totalSizeBytes + 1024 * 1024 * 1024;
        if (freeBytes >= 0 && freeBytes < requiredBytes) {
          final freeGb = (freeBytes / 1e9).toStringAsFixed(1);
          state = ServerStartupState(
            status: ServerStatus.error,
            message:
                'Not enough storage. The AI model needs ~$totalGb GB free '
                '(only $freeGb GB available). Free some space and tap retry.',
          );
          return;
        }

        // WiFi-only download guard.
        if (settings.wifiOnlyDownloads) {
          final connectivity = await Connectivity().checkConnectivity();
          final hasWifi = connectivity.contains(ConnectivityResult.wifi);
          if (!hasWifi) {
            state = ServerStartupState(
              status: ServerStatus.error,
              message:
                  'WiFi required for model download (~$totalGb GB). '
                'Connect to WiFi and tap retry, or disable "WiFi-only downloads" in Settings.',
            );
            return;
          }
        }

        final downloader = ref.read(modelDownloadProvider);
        state = ServerStartupState(
          status: ServerStatus.downloading,
          message: 'Downloading ${model.name} (~$totalGb GB, one-time setup)…',
        );
        // Start the foreground service so Android does not kill the
        // download when the user backgrounds the app.  The service stays
        // up through the subsequent server-load phase via the existing
        // [LlamaCppAndroidService.startServer] path.
        await ForegroundTaskService.start(
          detail: 'Downloading ${model.name} — ~$totalGb GB',
        );
        try {
          await downloader.downloadModel(
            model,
            onProgress: (downloaded, total, file) {
              if (_disposed) return;
              state = ServerStartupState(
                status: ServerStatus.downloading,
                message: 'Downloading $file…',
                downloadFile: file,
                downloadedBytes: downloaded,
                totalBytes: total,
              );
            },
          );
          if (_disposed) return;
        } catch (dlErr, dlStack) {
          if (_disposed) return;
          final dlMsg = dlErr.toString().toLowerCase();
          String dlUserMessage;
          if (dlMsg.contains('cancelled')) {
            dlUserMessage = 'Download cancelled by user.';
          } else if (dlMsg.contains('no space') || dlMsg.contains('nospace')) {
            dlUserMessage = 'Download failed: not enough storage space. Free up ~$totalGb GB and retry.';
          } else if (dlMsg.contains('socket') || dlMsg.contains('connection') || dlMsg.contains('http')) {
            dlUserMessage = 'Download failed: network error. Check your internet connection and retry.';
          } else {
            dlUserMessage = 'Model download failed. See technical details below.';
          }
          state = ServerStartupState(
            status: ServerStatus.error,
            message: dlUserMessage,
            technicalDetail: '$dlErr\n\nStack trace:\n$dlStack',
          );
          return;
        }
      }

      // Step 3: Start llama-server.
      state = const ServerStartupState(
        status: ServerStatus.starting,
        message: 'Starting language model server…',
      );

      // The server will start its own foreground service to stay alive.
      await llama.startServer();
      if (_disposed) return;

      state = const ServerStartupState(
        status: ServerStatus.ready,
        message: 'Backend ready.',
      );

      // Opportunistic update check (fire-and-forget).
      ref.read(updateProvider.notifier).check();
    } catch (e, stack) {
      if (_disposed) return;
      final errStr = e.toString();
      final msg = errStr.toLowerCase();
      String userMessage;
      // Order matters: check "exited prematurely / exec failed" before
      // "permission" because the noexec EACCES case ends up matching both
      // and the exec story is more accurate.
      if (msg.contains('exited prematurely') ||
          msg.contains('exit code') ||
          msg.contains('failed to launch') ||
          msg.contains('did not become healthy')) {
        userMessage = 'AI server failed to start. The native binary may not be executable on this Android device. See technical details below.';
      } else if (msg.contains('no space') || msg.contains('nospace')) {
        userMessage = 'Storage full: not enough space to extract binaries or download models. Free up space and retry.';
      } else if (msg.contains('model not found') || msg.contains('vision projector')) {
        userMessage = 'Model files missing or corrupted. The app will try re-downloading on next launch.';
      } else if (msg.contains('socket') || msg.contains('connection refused')) {
        userMessage = 'Cannot connect to the local AI server. It may have been killed by the system. Tap retry to restart it.';
      } else if (msg.contains('permission') || msg.contains('denied') || msg.contains('eacces')) {
        userMessage = 'Permission denied. Most likely Android is blocking the binary from being executed from app storage. See technical details below.';
      } else {
        userMessage = 'Failed to start Android backend. See technical details below.';
      }
      state = ServerStartupState(
        status: ServerStatus.error,
        message: userMessage,
        technicalDetail: '$errStr\n\nStack trace:\n$stack',
      );
    }
  }

  /// Desktop boot sequence: Python backend + auto-downloads.
  Future<void> _bootDesktop() async {
    try {
      final server = ref.read(backendServerProvider);
      state = const ServerStartupState(
        status: ServerStatus.starting,
        message: 'Starting backend server…',
      );
      await server.start(timeout: const Duration(seconds: 120));
      if (_disposed) return;

      // Send the user's GPU mode preference to the backend so it can
      // initialise with the right n_gpu_layers before we check binaries
      // or download models.
      await _syncGpuMode(server.url);
      if (_disposed) return;

      // Ensure llama.cpp vision binary is available (auto-download if missing).
      final llamaOk = await _checkLlamaBinary(server.url);
      if (_disposed) return;

      if (!llamaOk) {
        try {
          await _downloadLlamaAuto(server.url);
          if (_disposed) return;
        } catch (dlErr, dlStack) {
          if (_disposed) return;
          state = ServerStartupState(
            status: ServerStatus.error,
            message: 'Vision engine download failed. See technical details below.',
            technicalDetail: '$dlErr\n\nStack trace:\n$dlStack',
          );
          return;
        }
      }

      // Server is up — check whether model files are present.
      final modelsOk = await _checkModels(server.url);
      if (_disposed) return;

      if (!modelsOk) {
        // Auto-download models, keeping the user informed with progress.
        try {
          await _downloadModelsAuto(server.url);
          if (_disposed) return;
        } catch (dlErr, dlStack) {
          if (_disposed) return;
          state = ServerStartupState(
            status: ServerStatus.error,
            message: 'Model download failed. See technical details below.',
            technicalDetail: '$dlErr\n\nStack trace:\n$dlStack',
          );
          return;
        }
      }

      state = const ServerStartupState(
        status: ServerStatus.ready,
        message: 'Backend ready.',
      );

      // Opportunistic update check (fire-and-forget).
      ref.read(updateProvider.notifier).check();
    } catch (e) {
      if (_disposed) return;

      // Python not found — silently download a portable runtime and retry.
      final msg = e.toString();
      if (e is StateError &&
          (msg.contains('Cannot find python') ||
              msg.contains('Portable Python not found'))) {
        try {
          await _downloadPython();
          if (_disposed) return;
          // Python is now available — restart the full boot sequence.
          await _boot();
          return;
        } catch (dlErr) {
          if (_disposed) return;
          state = ServerStartupState(
            status: ServerStatus.error,
            message: 'Failed to set up Python: $dlErr',
          );
          return;
        }
      }

      state = ServerStartupState(
        status: ServerStatus.error,
        message: 'Failed to start backend: $e',
      );
    }
  }

  /// Download portable Python silently, showing progress.
  Future<void> _downloadPython() async {
    state = const ServerStartupState(
      status: ServerStatus.downloadingPython,
      message: 'Setting up — downloading Python runtime…',
    );

    await BackendServerService.downloadPortablePython(
      onProgress: (downloaded, total) {
        if (_disposed) return;
        state = ServerStartupState(
          status: ServerStatus.downloadingPython,
          message: 'Setting up — downloading Python runtime…',
          downloadFile: 'Python',
          downloadedBytes: downloaded,
          totalBytes: total,
        );
      },
    );
  }

  /// Returns true if the llama.cpp vision binary is already available.
  Future<bool> _checkLlamaBinary(String baseUrl) async {
    try {
      final resp = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      if (resp.statusCode == 200) {
        final body = jsonDecode(resp.body) as Map<String, dynamic>;
        return body['llama_binary_available'] == true;
      }
    } catch (_) {}
    return true; // assume ok if we can't tell
  }

  /// Auto-download llama.cpp binaries, showing progress.
  Future<void> _downloadLlamaAuto(String baseUrl) async {
    state = const ServerStartupState(
      status: ServerStatus.downloadingLlama,
      message: 'Downloading vision engine (~50 MB, one-time setup)…',
    );

    final request =
        http.Request('POST', Uri.parse('$baseUrl/llama/download'));
    final client = http.Client();
    try {
      final streamed = await client.send(request);
      final lines = streamed.stream
          .transform(const Utf8Decoder())
          .transform(const LineSplitter());

      await for (final line in lines) {
        if (_disposed) return;
        if (!line.startsWith('data: ')) continue;
        final payload = line.substring(6);
        try {
          final j = jsonDecode(payload) as Map<String, dynamic>;
          if (j['done'] == true) {
            if (j['error'] != null) {
              throw Exception(j['error']);
            }
            break;
          }
          state = ServerStartupState(
            status: ServerStatus.downloadingLlama,
            message: 'Downloading vision engine…',
            downloadFile: j['file'] as String? ?? '',
            downloadedBytes: (j['downloaded'] as num?)?.toInt() ?? 0,
            totalBytes: (j['total'] as num?)?.toInt() ?? 0,
          );
        } catch (e) {
          if (e is Exception && e.toString().contains('Exception:')) {
            rethrow;
          }
          // parse error — skip line
        }
      }
    } finally {
      client.close();
    }

    // Reinitialise backends now that binaries are available.
    await _reinitBackends(baseUrl);
  }

  /// Auto-download models and reinitialise backends, showing progress.
  Future<void> _downloadModelsAuto(String baseUrl) async {
    state = const ServerStartupState(
      status: ServerStatus.downloading,
      message: 'Downloading AI model (~3.2 GB, one-time setup)…',
    );

    await _downloadModels(baseUrl);
    if (_disposed) return;

    // Reinitialise backends now that files are on disk.
    await _reinitBackends(baseUrl);
  }

  /// Send the user's GPU mode preference to the backend.
  Future<void> _syncGpuMode(String baseUrl) async {
    try {
      final settings = ref.read(settingsProvider);
      await http
          .post(
            Uri.parse('$baseUrl/config/gpu'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'mode': settings.gpuMode}),
          )
          .timeout(const Duration(seconds: 30));
    } catch (e) {
      // Non-fatal — backend will use its platform defaults.
      debugPrint('Failed to sync GPU mode: $e');
    }
  }

  /// Allow retry from the error screen.
  Future<void> retry() async => _boot();

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  /// Returns true if all model files are present.
  Future<bool> _checkModels(String baseUrl) async {
    try {
      final resp = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      if (resp.statusCode == 200) {
        final body = jsonDecode(resp.body) as Map<String, dynamic>;
        return body['models_downloaded'] == true;
      }
    } catch (_) {}
    return true; // assume ok if we can't tell
  }

  /// Stream-download missing models, updating state with progress.
  Future<void> _downloadModels(String baseUrl) async {
    final request = http.Request('POST', Uri.parse('$baseUrl/models/download'));
    final client = http.Client();
    try {
      final streamed = await client.send(request);
      final lines = streamed.stream
          .transform(const Utf8Decoder())
          .transform(const LineSplitter());

      await for (final line in lines) {
        if (_disposed) return;
        if (!line.startsWith('data: ')) continue;
        final payload = line.substring(6);
        try {
          final j = jsonDecode(payload) as Map<String, dynamic>;
          if (j['done'] == true) {
            if (j['error'] != null) {
              throw Exception(j['error']);
            }
            break;
          }
          state = ServerStartupState(
            status: ServerStatus.downloading,
            message: 'Downloading ${j['file']}…',
            downloadFile: j['file'] as String? ?? '',
            downloadedBytes: (j['downloaded'] as num?)?.toInt() ?? 0,
            totalBytes: (j['total'] as num?)?.toInt() ?? 0,
          );
        } catch (e) {
          if (e is Exception &&
              e.toString().contains('Exception:')) {
            rethrow;
          }
          // parse error — skip line
        }
      }
    } finally {
      client.close();
    }
  }

  /// Ask the backend to reload vision + text after model download.
  Future<void> _reinitBackends(String baseUrl) async {
    state = const ServerStartupState(
      status: ServerStatus.downloading,
      message: 'Initialising models…',
    );
    await http
        .post(Uri.parse('$baseUrl/models/reinit'))
        .timeout(const Duration(seconds: 60));
  }
}

final inferenceServiceProvider = Provider<InferenceService>((ref) {
  final settings = ref.watch(settingsProvider);
  final android = Platform.isAndroid
      ? ref.read(llamaCppAndroidProvider)
      : null;
  return InferenceService(settings: settings, androidService: android);
});

final ankiExportServiceProvider = Provider<AnkiExportService>((ref) {
  final settings = ref.watch(settingsProvider);
  return AnkiExportService(settings: settings);
});

final highlightDetectorProvider = Provider<HighlightDetector>((ref) {
  final settings = ref.watch(settingsProvider);
  return HighlightDetector(
    colorTolerance: settings.colorTolerance,
    minArea: settings.minArea,
    padding: settings.padding,
    mergeNearby: settings.mergeNearby,
    mergeDistance: settings.mergeDistance,
    adaptiveMode: settings.adaptiveMode,
  );
});

// ---------------------------------------------------------------------------
// Update checking
// ---------------------------------------------------------------------------

enum UpdateStatus { idle, checking, available, downloading, error }

class UpdateState {
  const UpdateState({
    this.status = UpdateStatus.idle,
    this.info,
    this.downloadProgress = 0,
    this.error,
  });

  final UpdateStatus status;
  final UpdateInfo? info;
  final double downloadProgress;
  final String? error;

  UpdateState copyWith({
    UpdateStatus? status,
    UpdateInfo? info,
    double? downloadProgress,
    String? error,
  }) =>
      UpdateState(
        status: status ?? this.status,
        info: info ?? this.info,
        downloadProgress: downloadProgress ?? this.downloadProgress,
        error: error,
      );
}

final updateProvider =
    NotifierProvider<UpdateNotifier, UpdateState>(UpdateNotifier.new);

class UpdateNotifier extends Notifier<UpdateState> {
  UpdateService? _service;

  @override
  UpdateState build() => const UpdateState();

  Future<UpdateService> _ensureService() async {
    if (_service != null) return _service!;

    final packageInfo = await PackageInfo.fromPlatform();
    final version = packageInfo.version;

    _service = UpdateService(
      serverUrl: ref.read(settingsProvider).serverUrl,
      currentVersion: version,
      isAndroid: Platform.isAndroid,
    );
    return _service!;
  }

  /// Check for updates if auto-check is enabled and not already checked.
  Future<void> check({bool force = false}) async {
    final settings = ref.read(settingsProvider);
    if (!force && !settings.autoCheckUpdates) return;
    if (state.status == UpdateStatus.checking) return;

    state = state.copyWith(status: UpdateStatus.checking);

    try {
      final service = await _ensureService();
      final info = await service.checkForUpdate();

      if (!info.hasUpdate) {
        state = state.copyWith(status: UpdateStatus.idle);
        return;
      }

      // Respect "skip this version" preference.
      if (settings.skipVersion == info.latestVersion) {
        state = state.copyWith(status: UpdateStatus.idle);
        return;
      }

      state = state.copyWith(
        status: UpdateStatus.available,
        info: info,
      );
    } catch (e) {
      state = state.copyWith(
        status: UpdateStatus.error,
        error: e.toString(),
      );
    }
  }

  /// Download and apply the update.
  Future<void> downloadAndApply() async {
    final info = state.info;
    if (info == null || info.downloadUrl.isEmpty) return;

    state = state.copyWith(
      status: UpdateStatus.downloading,
      downloadProgress: 0,
    );

    final service = await _ensureService();
    try {
      final archivePath = await service.downloadUpdate(
        info.downloadUrl,
        onProgress: (p) {
          state = state.copyWith(
            downloadProgress: p.fraction,
          );
        },
      );

      await service.applyUpdate(archivePath);
      // applyUpdate calls exit(0) — we never return.
    } catch (e) {
      state = state.copyWith(
        status: UpdateStatus.error,
        error: e.toString(),
      );
    }
  }

  /// Mark the current available version as skipped.
  void skipVersion() {
    final info = state.info;
    if (info == null) return;

    ref.read(settingsProvider.notifier).update(
      (s) => s..skipVersion = info.latestVersion,
    );
    state = state.copyWith(status: UpdateStatus.idle);
  }

  /// Dismiss the update notification without skipping.
  void dismiss() {
    state = state.copyWith(status: UpdateStatus.idle);
  }

}

// ---------------------------------------------------------------------------
// Processing state
// ---------------------------------------------------------------------------

class ProcessingState {
  const ProcessingState({
    this.phase = ProcessingPhase.idle,
    this.progress = 0,
    this.statusMessage = '',
    this.words = const [],
    this.enrichedWords = const [],
    this.ocrText = '',
    this.error,
    this.activityLog = const [],
    this.startTime,
    this.enrichmentSkipped = false,
  });

  final ProcessingPhase phase;
  final double progress;
  final String statusMessage;
  final List<String> words;
  final List<EnrichWordResult> enrichedWords;
  final String ocrText;
  final String? error;

  /// Timestamped log of each step for the user to follow along.
  final List<String> activityLog;

  /// When processing started (for elapsed timer).
  final DateTime? startTime;

  /// Whether the user chose to skip enrichment.
  final bool enrichmentSkipped;

  ProcessingState copyWith({
    ProcessingPhase? phase,
    double? progress,
    String? statusMessage,
    List<String>? words,
    List<EnrichWordResult>? enrichedWords,
    String? ocrText,
    String? error,
    List<String>? activityLog,
    DateTime? startTime,
    bool? enrichmentSkipped,
  }) =>
      ProcessingState(
        phase: phase ?? this.phase,
        progress: progress ?? this.progress,
        statusMessage: statusMessage ?? this.statusMessage,
        words: words ?? this.words,
        enrichedWords: enrichedWords ?? this.enrichedWords,
        ocrText: ocrText ?? this.ocrText,
        error: error,
        activityLog: activityLog ?? this.activityLog,
        startTime: startTime ?? this.startTime,
        enrichmentSkipped: enrichmentSkipped ?? this.enrichmentSkipped,
      );
}

final processingProvider =
    NotifierProvider<ProcessingNotifier, ProcessingState>(
        ProcessingNotifier.new);

/// Result from the word-review gate: confirmed words + optional per-word
/// language overrides (keyed by lowercase word).
typedef WordReviewResult = ({List<String> words, Map<String, String> wordLanguages});

class ProcessingNotifier extends Notifier<ProcessingState> {
  bool _cancelled = false;
  Timer? _heartbeat;

  /// Completer that the pipeline awaits during word review.
  Completer<WordReviewResult?>? _wordReviewCompleter;

  @override
  ProcessingState build() => const ProcessingState();

  void reset() {
    _cancelled = false;
    _heartbeat?.cancel();
    _heartbeat = null;
    _wordReviewCompleter = null;
    state = const ProcessingState();
  }

  /// User confirmed the edited word list — resume enrichment.
  ///
  /// [wordLanguages] maps lowercase word → language code for words
  /// whose source language was explicitly tagged by the user.
  void confirmWords(List<String> words, {Map<String, String> wordLanguages = const {}}) {
    _wordReviewCompleter?.complete((words: words, wordLanguages: wordLanguages));
  }

  /// User chose to skip enrichment entirely.
  void skipEnrichment() {
    _wordReviewCompleter?.complete(null);
  }

  /// Request cancellation of the current pipeline.
  void cancel() {
    if (_cancelled) return; // already requested
    _cancelled = true;
    // Stop the heartbeat timer immediately so no more log noise.
    _heartbeat?.cancel();
    _heartbeat = null;
    _log('Cancellation requested...', progress: state.progress);
    // Unblock the word-review gate so the pipeline can reach
    // _checkCancelled() and throw the cancellation exception.
    if (_wordReviewCompleter != null && !_wordReviewCompleter!.isCompleted) {
      _wordReviewCompleter!.complete(null);
    }
    // Kill the server-side subprocess and abort the HTTP request.
    final inference = ref.read(inferenceServiceProvider);
    inference.cancelOcr();
    inference.cancelEnrichment();
  }

  /// Throws if cancel() has been called.
  void _checkCancelled() {
    if (_cancelled) throw Exception('Processing cancelled by user.');
  }

  /// Append a line to the activity log and update the status message.
  void _log(String message, {ProcessingPhase? phase, double? progress}) {
    var log = [...state.activityLog, message];
    if (log.length > 100) {
      log = log.sublist(log.length - 100);
    }
    state = state.copyWith(
      activityLog: log,
      statusMessage: message,
      phase: phase,
      progress: progress,
    );
  }

  String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(0)} KB';
    return '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';
  }

  /// Fire a desktop notification (Linux: notify-send). Best-effort, never
  /// throws even if the tool is missing.
  void _notify(String title, String body) {
    if (Platform.isLinux) {
      // Use .then + .catchError (returning a ProcessResult) or just ignore
      // the future. We use an async IIFE so exceptions never propagate.
      () async {
        try {
          await Process.run('notify-send', [
            '-a', 'OCR to Anki',
            '-i', 'dialog-information',
            title,
            body,
          ]);
        } catch (_) {
          // notify-send may not be installed — ignore silently.
        }
      }();
    }
    // macOS / Windows could be added later.
  }

  // -------------------------------------------------------------------------
  // Session persistence (in-flight recovery)
  // -------------------------------------------------------------------------

  static const _pendingBatchKey = 'pending_batch';

  /// Directory where pending-batch image temp files are stored.
  Future<Directory> _pendingBatchTempDir() async {
    final tmp = await getTemporaryDirectory();
    final dir = Directory('${tmp.path}/pending_batch_images');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  /// Save the current processing state so it can survive app restarts.
  Future<void> _savePendingBatch({
    required List<ImageEntry> images,
    List<String>? ocrResults,
    List<String>? words,
    Map<String, String>? wordLanguages,
    List<EnrichWordResult>? enrichedWords,
  }) async {
    final db = ref.read(databaseProvider);
    final tmpDir = await _pendingBatchTempDir();

    // Write image bytes to temp files.
    final pendingImages = <PendingImageEntry>[];
    for (var i = 0; i < images.length; i++) {
      final entry = images[i];
      final path = '${tmpDir.path}/img_$i.jpg';
      await File(path).writeAsBytes(entry.bytes);
      pendingImages.add(PendingImageEntry(
        path: path,
        name: entry.name,
        cropRegion: entry.cropRegion,
        hsvOverride: entry.hsvOverride,
        termLanguage: entry.termLanguage,
      ));
    }

    final batch = PendingBatch(
      phase: state.phase,
      imageEntries: pendingImages,
      ocrResults: ocrResults ?? [],
      words: words ?? state.words,
      wordLanguages: wordLanguages ?? {},
      enrichedWords: enrichedWords ?? state.enrichedWords,
      ocrText: state.ocrText,
      progress: state.progress,
      statusMessage: state.statusMessage,
      activityLog: state.activityLog,
    );

    await db.setSetting(_pendingBatchKey, batch.toJsonString());
  }

  /// Update the persisted batch after a phase transition, preserving the
  /// existing image entries and OCR results.
  Future<void> _updatePendingBatch({
    ProcessingPhase? phase,
    List<String>? words,
    Map<String, String>? wordLanguages,
    List<EnrichWordResult>? enrichedWords,
    String? ocrText,
    double? progress,
    String? statusMessage,
    List<String>? activityLog,
  }) async {
    final db = ref.read(databaseProvider);
    final existingJson = await db.getSetting(_pendingBatchKey);
    if (existingJson == null || existingJson.isEmpty) return;

    try {
      final existing = PendingBatch.fromJsonString(existingJson);
      final updated = PendingBatch(
        phase: phase ?? state.phase,
        imageEntries: existing.imageEntries,
        ocrResults: existing.ocrResults,
        words: words ?? state.words,
        wordLanguages: wordLanguages ?? existing.wordLanguages,
        enrichedWords: enrichedWords ?? state.enrichedWords,
        ocrText: ocrText ?? state.ocrText,
        progress: progress ?? state.progress,
        statusMessage: statusMessage ?? state.statusMessage,
        activityLog: activityLog ?? state.activityLog,
      );
      await db.setSetting(_pendingBatchKey, updated.toJsonString());
    } catch (_) {
      // Ignore parse errors.
    }
  }

  /// Clear the persisted batch and delete temp files.
  Future<void> clearPendingBatch() async {
    final db = ref.read(databaseProvider);
    await db.setSetting(_pendingBatchKey, '');

    try {
      final tmpDir = await _pendingBatchTempDir();
      if (await tmpDir.exists()) {
        await tmpDir.delete(recursive: true);
      }
    } catch (_) {}
  }

  /// Check for an existing pending batch.
  static Future<PendingBatch?> loadPendingBatch(AppDatabase db) async {
    final json = await db.getSetting(_pendingBatchKey);
    if (json == null || json.isEmpty) return null;
    try {
      final batch = PendingBatch.fromJsonString(json);
      if (batch.isTerminal) return null;
      return batch;
    } catch (_) {
      return null;
    }
  }

  /// Resume an interrupted batch from the given [PendingBatch].
  ///
  /// Reconstructs the state and continues processing from the saved phase.
  Future<void> resumeBatch(PendingBatch batch) async {
    state = ProcessingState(
      phase: batch.phase,
      progress: batch.progress,
      statusMessage: batch.statusMessage,
      words: batch.words,
      enrichedWords: batch.enrichedWords,
      ocrText: batch.ocrText,
      activityLog: batch.activityLog,
      startTime: DateTime.now(),
    );

    try {
      if (batch.phase == ProcessingPhase.cropping ||
          batch.phase == ProcessingPhase.ocr) {
        // Re-run the full pipeline from the saved images.
        final images = <ImageEntry>[];
        for (final p in batch.imageEntries) {
          final bytes = await p.loadBytes();
          images.add(ImageEntry(
            bytes: bytes,
            name: p.name,
            cropRegion: p.cropRegion,
            hsvOverride: p.hsvOverride,
            termLanguage: p.termLanguage,
          ));
        }
        // Reconstruct global HSV range from first override or null.
        HsvRange? hsvRange;
        if (batch.imageEntries.isNotEmpty) {
          hsvRange = batch.imageEntries.first.hsvOverride;
        }
        await processImages(images: images, hsvRange: hsvRange);
        return;
      }

      if (batch.phase == ProcessingPhase.wordReview) {
        // Re-establish the word-review gate.
        _wordReviewCompleter = Completer<WordReviewResult?>();
        final reviewResult = await _wordReviewCompleter!.future;
        _wordReviewCompleter = null;

        _checkCancelled();

        final confirmedWords = reviewResult?.words;
        final wordLanguages = reviewResult?.wordLanguages ?? {};

        if (confirmedWords == null || confirmedWords.isEmpty) {
          _log('Enrichment skipped by user.', progress: 0.95);
          final wordsToKeep = confirmedWords ?? state.words;
          final stubCards = wordsToKeep
              .map((w) => EnrichWordResult(
                    word: w, definition: '', examples: ''))
              .toList();
          state = state.copyWith(
            words: wordsToKeep,
            enrichedWords: stubCards,
            enrichmentSkipped: true,
          );
        } else {
          state = state.copyWith(words: confirmedWords);
          await _doEnrichment(
            allWords: confirmedWords,
            wordLanguages: wordLanguages,
          );
        }

        await _persistSession(
          imageNames: batch.imageEntries.map((e) => e.name).toList(),
          hsvRange: batch.imageEntries.isNotEmpty
              ? batch.imageEntries.first.hsvOverride
              : null,
        );
        await clearPendingBatch();
        return;
      }

      if (batch.phase == ProcessingPhase.enriching) {
        // Continue enrichment for words that don't have results yet.
        final missingWords = batch.words
            .where((w) => !batch.enrichedWords
                .any((e) => e.word.toLowerCase() == w.toLowerCase()))
            .toList();

        if (missingWords.isNotEmpty) {
          await _doEnrichment(
            allWords: batch.words,
            wordsToEnrich: missingWords,
            wordLanguages: batch.wordLanguages,
          );
        } else {
          state = state.copyWith(
            phase: ProcessingPhase.enriching,
            enrichedWords: batch.enrichedWords,
          );
        }

        await _persistSession(
          imageNames: batch.imageEntries.map((e) => e.name).toList(),
          hsvRange: batch.imageEntries.isNotEmpty
              ? batch.imageEntries.first.hsvOverride
              : null,
        );
        await clearPendingBatch();
        return;
      }

      // Any other phase — clear and done.
      await clearPendingBatch();
    } catch (e) {
      _heartbeat?.cancel();
      _heartbeat = null;
      if (_cancelled) {
        _log('Processing cancelled.', progress: 0);
        state = state.copyWith(
          phase: ProcessingPhase.done,
          statusMessage: 'Cancelled',
        );
      } else {
        _log('Error: $e');
        state = state.copyWith(
          phase: ProcessingPhase.error,
          error: e.toString(),
          statusMessage: 'Error: $e',
        );
      }
    } finally {
      if (Platform.isAndroid) {
        await ForegroundTaskService.update(detail: 'AI model ready');
      }
      _heartbeat?.cancel();
      _heartbeat = null;
    }
  }

  /// Shared enrichment pipeline used by [processImages], [processWordsOnly],
  /// and [resumeBatch].
  ///
  /// [allWords] is the full ordered word list.  [wordsToEnrich] is the subset
  /// that actually needs LLM calls (defaults to [allWords]).
  Future<void> _doEnrichment({
    required List<String> allWords,
    Map<String, String> wordLanguages = const {},
    List<String>? wordsToEnrich,
  }) async {
    final settings = ref.read(settingsProvider);
    final db = ref.read(databaseProvider);
    final inference = ref.read(inferenceServiceProvider);
    final targetWords = wordsToEnrich ?? allWords;

    _checkCancelled();
    if (targetWords.isEmpty) return;

    // Cache lookup
    final cachedMap = await db.getCachedEnrichments(
      words: targetWords,
      definitionLanguage: settings.definitionLanguage,
      examplesLanguage: settings.examplesLanguage,
    );

    final cachedResults = <EnrichWordResult>[];
    final uncachedWords = <String>[];
    for (final w in targetWords) {
      final hit = cachedMap[w.toLowerCase()];
      if (hit != null && hit.warning != 'not_found') {
        cachedResults.add(EnrichWordResult(
          word: w,
          definition: hit.definition.replaceAll('*', ''),
          examples: hit.examples.replaceAll('*', ''),
          warning: hit.warning,
          fromCache: true,
        ));
      } else {
        uncachedWords.add(w);
      }
    }

    if (cachedResults.isNotEmpty) {
      _log(
        '${cachedResults.length} word(s) found in cache, '
        '${uncachedWords.length} word(s) need enrichment',
        progress: 0.64,
      );
      state = state.copyWith(
        enrichedWords: [...state.enrichedWords, ...cachedResults],
      );
    }

    if (uncachedWords.isNotEmpty) {
      final wordCount = uncachedWords.length;
      _log(
        'Enriching $wordCount word(s) with definitions '
        '(${settings.definitionLanguage})...',
        phase: ProcessingPhase.enriching,
        progress: 0.65,
      );
      _log(
        'Generating definitions and example sentences via LLM '
        '(1 word per call, 10 min timeout each)...',
        progress: 0.68,
      );

      final enrichStopwatch = Stopwatch()..start();

      _heartbeat?.cancel();
      _heartbeat = Timer.periodic(
        const Duration(seconds: 15),
        (timer) {
          final secs = enrichStopwatch.elapsed.inSeconds;
          _log(
            'Enrichment in progress... ${secs}s elapsed '
            '(chunked LLM calls for $wordCount word(s))',
            progress: 0.68 + 0.20 * (secs / 120).clamp(0.0, 1.0),
          );
        },
      );

      try {
        final enriched = await inference.enrichWords(
          words: uncachedWords,
          definitionLanguage: settings.definitionLanguage,
          examplesLanguage: settings.examplesLanguage,
          termLanguage: settings.termLanguage,
          wordLanguages: wordLanguages,
          chunkSize: 1,
          chunkTimeout: const Duration(minutes: 10),
          onChunkDone: (completed, total, chunkResults) {
            state = state.copyWith(
              enrichedWords: [...state.enrichedWords, ...chunkResults],
            );
            _log(
              'Enrichment: $completed/$total word(s) done',
              progress: 0.68 + 0.20 * (completed / total),
            );
            // Persist partial progress so a kill mid-enrichment resumes
            // from the last completed chunk instead of starting over.
            _updatePendingBatch(enrichedWords: state.enrichedWords);
            // Stop immediately if the user cancelled during this chunk.
            _checkCancelled();
          },
        );
        enrichStopwatch.stop();
        _heartbeat?.cancel();
        _heartbeat = null;

        final enrichElapsed =
            (enrichStopwatch.elapsedMilliseconds / 1000).toStringAsFixed(1);

        _log(
          'Enrichment complete: ${enriched.length} card(s) in ${enrichElapsed}s',
          progress: 0.90,
        );

        // Cache new results
        final toCache = enriched
            .where((e) => e.warning != 'not_found')
            .map((e) => EnrichmentCacheEntriesCompanion.insert(
                  word: e.word.toLowerCase(),
                  definitionLanguage: settings.definitionLanguage,
                  examplesLanguage: settings.examplesLanguage,
                  definition: e.definition,
                  examples: e.examples,
                  warning: Value(e.warning),
                ))
            .toList();
        if (toCache.isNotEmpty) {
          await db.cacheEnrichments(toCache);
          _log('Cached ${toCache.length} enrichment(s) for future re-use');
        }

        // Merge cached + fresh in original order
        final allResultsMap = <String, EnrichWordResult>{};
        for (final r in cachedResults) {
          allResultsMap[r.word.toLowerCase()] = r;
        }
        for (final r in enriched) {
          allResultsMap[r.word.toLowerCase()] = r;
        }
        final orderedResults = allWords
            .map((w) => allResultsMap[w.toLowerCase()])
            .whereType<EnrichWordResult>()
            .toList();

        state = state.copyWith(enrichedWords: orderedResults);
      } catch (e) {
        _heartbeat?.cancel();
        _heartbeat = null;
        rethrow;
      }
    } else {
      _log('All words served from cache!', progress: 0.90);
      state = state.copyWith(
        phase: ProcessingPhase.enriching,
        enrichedWords: cachedResults,
      );
    }
  }

  /// Persist the completed session to the database.
  ///
  /// [imageNames] and [hsvRange] are used to populate the session record.
  Future<void> _persistSession({
    required List<String> imageNames,
    HsvRange? hsvRange,
  }) async {
    final db = ref.read(databaseProvider);
    _log('Saving session to local database...', progress: 0.92);

    final sessionId = await db.insertSession(
      ProcessingSessionsCompanion.insert(
        imagePath: imageNames.join(', '),
        context: hsvRange != null ? 'highlighted' : 'handwrittenOrPrinted',
        highlightColor: Value(hsvRange?.label),
        ocrText: Value(state.ocrText),
      ),
    );

    final wordCompanions = state.enrichedWords
        .map((e) => WordEntriesCompanion.insert(
              sessionId: sessionId,
              word: e.word,
              definition: Value(e.definition),
              examples: Value(e.examples),
            ))
        .toList();

    if (wordCompanions.isNotEmpty) {
      await db.insertWords(wordCompanions);
      _log('Saved ${wordCompanions.length} word(s) to database',
          progress: 0.96);
    }

    if (state.enrichedWords.isNotEmpty) {
      _log(
        'Done! ${state.enrichedWords.length} card(s) ready for export',
        phase: ProcessingPhase.done,
        progress: 1.0,
      );
    } else {
      _log(
        'Done. No words to process',
        phase: ProcessingPhase.done,
        progress: 1.0,
      );
    }
  }

  /// Skip OCR and go straight to word review → enrichment.
  ///
  /// Used when the user wants to add words manually without any images.
  Future<void> processWordsOnly(List<String> words) async {
    final stopwatch = Stopwatch()..start();
    final bench = BenchmarkData(
      timestamp: DateTime.now(),
      imageSizeBytes: 0,
    );

    try {
      if (Platform.isAndroid) {
        await ForegroundTaskService.update(detail: 'Enriching words…');
      }

      state = const ProcessingState().copyWith(
        phase: ProcessingPhase.wordReview,
        progress: 0.62,
        statusMessage: 'Review your word list before enrichment.',
        startTime: DateTime.now(),
        activityLog: ['Manual word entry (no images).'],
        words: words,
        ocrText: words.join('\n'),
      );

      final settings = ref.read(settingsProvider);
      bench.definitionLanguage = settings.definitionLanguage;
      bench.examplesLanguage = settings.examplesLanguage;

      _log(
        'Word extraction complete – review the word list before enrichment.',
        phase: ProcessingPhase.wordReview,
        progress: 0.62,
      );

      // Persist so a kill during word review can be resumed.
      await _savePendingBatch(images: [], words: words);

      // Reuse the same word-review gate and enrichment pipeline as
      // processImages (everything from the Completer onwards).
      _wordReviewCompleter = Completer<WordReviewResult?>();
      final reviewResult = await _wordReviewCompleter!.future;
      _wordReviewCompleter = null;

      _checkCancelled();

      final confirmedWords = reviewResult?.words;
      final wordLanguages = reviewResult?.wordLanguages ?? {};

      if (confirmedWords == null || confirmedWords.isEmpty) {
        _log('Enrichment skipped by user.', progress: 0.95);
        final wordsToKeep = confirmedWords ?? state.words;
        final stubCards = wordsToKeep
            .map((w) => EnrichWordResult(
                  word: w, definition: '', examples: ''))
            .toList();
        state = state.copyWith(
          words: wordsToKeep,
          enrichedWords: stubCards,
          enrichmentSkipped: true,
        );
      } else {
        state = state.copyWith(words: confirmedWords);

        // Persist so a kill during enrichment can be resumed.
        await _updatePendingBatch(
          phase: ProcessingPhase.enriching,
          words: confirmedWords,
          wordLanguages: wordLanguages,
        );

        _checkCancelled();
        if (confirmedWords.isNotEmpty) {
          final inference = ref.read(inferenceServiceProvider);
          final db = ref.read(databaseProvider);

          // Server check
          _log('Checking server connection...', progress: 0.63);
          final serverOk = await inference.isAvailable();
          if (!serverOk) {
            final detail = inference.debugMessage ?? 'no details';
            final hint = Platform.isAndroid
                ? 'Make sure the app finished its initial setup.'
                : 'Make sure the FastAPI backend is running.';
            throw Exception(
              'Cannot reach inference server ($detail). $hint',
            );
          }

          // Cache lookup
          final cachedMap = await db.getCachedEnrichments(
            words: confirmedWords,
            definitionLanguage: settings.definitionLanguage,
            examplesLanguage: settings.examplesLanguage,
          );

          final cachedResults = <EnrichWordResult>[];
          final uncachedWords = <String>[];
          for (final w in confirmedWords) {
            final hit = cachedMap[w.toLowerCase()];
            if (hit != null && hit.warning != 'not_found') {
              cachedResults.add(EnrichWordResult(
                word: w,
                definition: hit.definition.replaceAll('*', ''),
                examples: hit.examples.replaceAll('*', ''),
                warning: hit.warning,
                fromCache: true,
              ));
            } else {
              uncachedWords.add(w);
            }
          }

          if (cachedResults.isNotEmpty) {
            _log(
              '${cachedResults.length} word(s) found in cache, '
              '${uncachedWords.length} word(s) need enrichment',
              progress: 0.64,
            );
            state = state.copyWith(
              enrichedWords: [...state.enrichedWords, ...cachedResults],
            );
          }

          List<EnrichWordResult> freshResults = [];
          if (uncachedWords.isNotEmpty) {
            _log(
              'Enriching ${uncachedWords.length} word(s)...',
              phase: ProcessingPhase.enriching,
              progress: 0.70,
            );

            final enrichWatch = Stopwatch()..start();
            freshResults = await inference.enrichWords(
              words: uncachedWords,
              definitionLanguage: settings.definitionLanguage,
              examplesLanguage: settings.examplesLanguage,
              termLanguage: settings.termLanguage,
              wordLanguages: wordLanguages,
              chunkSize: 1,
              chunkTimeout: const Duration(minutes: 10),
              onChunkDone: (completed, total, chunkResults) {
                state = state.copyWith(
                  enrichedWords: [...state.enrichedWords, ...chunkResults],
                );
                _log(
                  'Enrichment: $completed/$total word(s) done',
                  progress: 0.68 + 0.20 * (completed / total),
                );
                _updatePendingBatch(enrichedWords: state.enrichedWords);
                _checkCancelled();
              },
            );
            enrichWatch.stop();
            bench.enrichElapsedS = enrichWatch.elapsedMilliseconds / 1000;

            // Cache new results
            final toCache = freshResults
                .where((e) => e.warning != 'not_found')
                .map((e) => EnrichmentCacheEntriesCompanion.insert(
                      word: e.word.toLowerCase(),
                      definitionLanguage: settings.definitionLanguage,
                      examplesLanguage: settings.examplesLanguage,
                      definition: e.definition,
                      examples: e.examples,
                      warning: Value(e.warning),
                    ))
                .toList();
            if (toCache.isNotEmpty) {
              await db.cacheEnrichments(toCache);
              _log('Cached ${toCache.length} enrichment(s) for future re-use');
            }
          }

          bench.enrichedWordCount = cachedResults.length + freshResults.length;

          // Merge in original order
          final allResultsMap = <String, EnrichWordResult>{};
          for (final r in cachedResults) {
            allResultsMap[r.word.toLowerCase()] = r;
          }
          for (final r in freshResults) {
            allResultsMap[r.word.toLowerCase()] = r;
          }
          final orderedResults = confirmedWords
              .map((w) => allResultsMap[w.toLowerCase()])
              .whereType<EnrichWordResult>()
              .toList();

          state = state.copyWith(enrichedWords: orderedResults);
        }
      }

      // Done
      _checkCancelled();
      _log('Saving session to local database...', progress: 0.92);
      final totalElapsed =
          (stopwatch.elapsedMilliseconds / 1000).toStringAsFixed(1);

      if (state.enrichedWords.isNotEmpty) {
        _log(
          'Done! ${state.enrichedWords.length} card(s) ready for export '
          '(${totalElapsed}s total)',
          phase: ProcessingPhase.done,
          progress: 1.0,
        );
      } else {
        _log(
          'Done. No words to process (${totalElapsed}s total)',
          phase: ProcessingPhase.done,
          progress: 1.0,
        );
      }
    } catch (e) {
      if (_cancelled) {
        _log('Processing cancelled.', progress: 0);
        state = state.copyWith(
          phase: ProcessingPhase.done,
          statusMessage: 'Cancelled',
        );
      } else {
        _log('Error: $e');
        state = state.copyWith(
          phase: ProcessingPhase.error,
          error: e.toString(),
          statusMessage: 'Error: $e',
        );
      }
    } finally {
      await clearPendingBatch();
      if (Platform.isAndroid) {
        await ForegroundTaskService.update(detail: 'AI model ready');
      }
      stopwatch.stop();
      _heartbeat?.cancel();
      _heartbeat = null;
    }
  }

  /// Run the full pipeline on a single image.
  Future<void> processImage({
    required Uint8List imageBytes,
    required String filename,
    HighlightColor? highlightColor,
  }) =>
      processImages(
        images: [
          ImageEntry(
            bytes: imageBytes,
            name: filename,
          ),
        ],
        hsvRange: highlightColor != null
            ? HsvRange.fromPreset(highlightColor)
            : null,
      );

  /// Run the full pipeline on a batch of images.
  ///
  /// Each image is optionally cropped and OCR'd; the resulting words from
  /// ALL images are de-duplicated then sent through a single word-review
  /// gate and a single enrichment pass.
  ///
  /// [hsvRange] is the global highlight colour; individual images may
  /// override it via [ImageEntry.hsvOverride].  Similarly, per-image crop
  /// regions in [ImageEntry.cropRegion] override the global crop that was
  /// already applied before calling this method.
  ///
  /// [confirmedBoxes] maps image indices to user-confirmed bounding boxes
  /// from the preview dialog.  When present for an image index, those boxes
  /// are used instead of auto-detecting.
  Future<void> processImages({
    required List<ImageEntry> images,
    HsvRange? hsvRange,
    Map<int, List<HighlightBBox>>? confirmedBoxes,
  }) async {
    const maxImages = 10;
    if (images.length > maxImages) {
      images = images.sublist(0, maxImages);
    }

    final stopwatch = Stopwatch()..start();

    // Benchmark data collector.
    final totalBytes = images.fold<int>(0, (s, i) => s + i.bytes.length);
    final bench = BenchmarkData(
      timestamp: DateTime.now(),
      imageSizeBytes: totalBytes,
    );

    try {
      if (Platform.isAndroid) {
        await ForegroundTaskService.update(
          detail: 'Processing ${images.length} image(s)...',
        );
      }

      final names = images.map((i) => i.name).join(', ');
      state = const ProcessingState().copyWith(
        phase: ProcessingPhase.ocr,
        progress: 0,
        statusMessage: 'Preparing ${images.length} image(s)...',
        startTime: DateTime.now(),
        activityLog: ['Processing started for: $names'],
      );

      final inference = ref.read(inferenceServiceProvider);
      final settings = ref.read(settingsProvider);

      bench.definitionLanguage = settings.definitionLanguage;
      bench.examplesLanguage = settings.examplesLanguage;

      // Resolve the effective term language: if every image agrees on a
      // per-image override, use that; otherwise fall back to global default.
      final perImageLangs = images
          .map((e) => e.termLanguage)
          .whereType<String>()
          .toSet();
      final effectiveTermLang = perImageLangs.length == 1
          ? perImageLangs.first
          : settings.termLanguage;

      _log(
        '${images.length} image(s) loaded (${_formatBytes(totalBytes)})',
        progress: 0.05,
      );

      _checkCancelled();

      // Step 1: Verify server connectivity.
      _log('Checking server connection...', progress: 0.07);
      final serverOk = await inference.isAvailable();
      if (!serverOk) {
        final detail = inference.debugMessage ?? 'no details';
        final hint = Platform.isAndroid
            ? 'Make sure the app finished its initial setup.'
            : 'Make sure the FastAPI backend is running.';
        throw Exception(
          'Cannot reach inference server ($detail). $hint',
        );
      }
      _log('Server is online', progress: 0.08);

      // Steps 2-3: For each image → crop (if highlighted) → OCR.
      final allWords = <String>[];
      final ocrTexts = <String>[];
      var totalCropCount = 0;

      for (var imgIdx = 0; imgIdx < images.length; imgIdx++) {
        _checkCancelled();
        final entry = images[imgIdx];
        final imgBytes = entry.bytes;
        final imgName = entry.name;
        final imgProgress = 0.10 + 0.45 * imgIdx / images.length;

        // Per-image colour override takes precedence over global.
        final effectiveHsv = entry.hsvOverride ?? hsvRange;

        if (images.length > 1) {
          _log(
            '── Image ${imgIdx + 1}/${images.length}: $imgName'
            '${entry.hasColorOverride ? " [custom colour]" : ""}'
            '${entry.hasCrop ? " [custom crop]" : ""}'
            ' ──',
            progress: imgProgress,
          );
        }

        // Step 2: If highlighted context, crop first.
        List<Uint8List> imagesToProcess = [imgBytes];
        if (effectiveHsv != null) {
          _log(
            'Scanning for ${effectiveHsv.label} highlights...',
            phase: ProcessingPhase.cropping,
            progress: imgProgress,
          );
          final cropWatch = Stopwatch()..start();
          final detector = ref.read(highlightDetectorProvider);

          // Use user-confirmed boxes from the preview dialog when
          // available; otherwise auto-detect.
          List<Uint8List> crops;
          final previewBoxes = confirmedBoxes?[imgIdx];
          if (previewBoxes != null) {
            if (previewBoxes.isNotEmpty) {
              crops = detector.cropBoxes(
                imageBytes: imgBytes,
                boxes: previewBoxes,
              );
              _log('Using ${crops.length} user-confirmed region(s)');
            } else {
              crops = [];
              _log('All regions removed by user, using full image');
            }
          } else {
            crops = detector.detectAndCrop(
              imageBytes: imgBytes,
              color: effectiveHsv,
            );
          }

          cropWatch.stop();
          bench.cropElapsedS += cropWatch.elapsedMilliseconds / 1000;
          if (crops.isNotEmpty) {
            imagesToProcess = crops;
            totalCropCount += crops.length;
            if (previewBoxes == null) {
              _log('Found ${crops.length} highlighted region(s)');
            }
          } else {
            _log('No highlighted regions detected, using full image');
          }
        } else if (imgIdx == 0) {
          _log(
            'Mode: full-image OCR (no highlight colour set)',
            progress: 0.10,
          );
        }

        // Step 3: Vision OCR on each image/crop.
        //
        // Three modes:
        //   a) Parallel crops   – send each crop individually, concurrently.
        //   b) Montage          – stitch crops into one image, single OCR call.
        //   c) Single image     – one crop or full image, one OCR call.
        //
        // Parallel mode is faster when a discrete GPU is available and the
        // model is kept loaded between calls (llama-server).  Montage is
        // better on iGPUs where each subprocess launch reloads the model.
        final useParallel = settings.parallelCrops && imagesToProcess.length > 1;
        final useMontage = !useParallel && imagesToProcess.length > 1;

        if (useParallel) {
          // ── Parallel crop OCR ──────────────────────────────────────
          _checkCancelled();
          final n = imagesToProcess.length;
          _log(
            'Processing $n crops sequentially...',
            phase: ProcessingPhase.ocr,
            progress: imgProgress + 0.02,
          );

          // Downscale each crop before sending if montageMaxWidth > 0.
          final mw = settings.montageMaxWidth;
          final cropsToSend = mw > 0
              ? imagesToProcess
                    .map((c) => HighlightDetector.buildMontage([c], maxWidth: mw))
                    .toList()
              : imagesToProcess;

          final parallelProgress = 0.10 + 0.45 * (imgIdx + 0.5) / images.length;
          final ocrStopwatch = Stopwatch()..start();

          _heartbeat?.cancel();
          var completed = 0;
          _heartbeat = Timer.periodic(
            const Duration(seconds: 15),
            (timer) {
              final secs = ocrStopwatch.elapsed.inSeconds;
              final mins = (secs / 60).toStringAsFixed(1);
              _log(
                'OCR in progress... $mins min elapsed - $completed/$n crops done',
                progress: parallelProgress +
                    0.30 * (completed / n).clamp(0.0, 1.0),
              );
            },
          );

          try {
            // Process crops sequentially — llama-mtmd-cli loads the full
            // model into GPU VRAM each call, so running multiple in
            // parallel crashes on Windows/iGPUs (0xC0000409).
            final results = <VisionOcrResult>[];
            for (final crop in cropsToSend) {
              _checkCancelled();
              final result = await inference.visionOcr(imageBytes: crop);
              completed++;
              results.add(result);
            }
            ocrStopwatch.stop();
            _heartbeat?.cancel();
            _heartbeat = null;

            final totalS = ocrStopwatch.elapsedMilliseconds / 1000;
            bench.perCropOcrS = [totalS];

            for (final result in results) {
              if (result.backend.isNotEmpty) bench.backend = result.backend;
              ocrTexts.add(result.text);

              final words = result.text
                  .split(RegExp(r'[\n,]+'))
                  .map((w) => w
                      .trim()
                      .replaceAll(RegExp(r'^[\s*\-\u2022\u00b7]+'), '')
                      .replaceAll(RegExp(r'^\d+\.\s*'), '')
                      .trim())
                  .where(
                      (w) => w.length > 1 && w.split(RegExp(r'\s+')).length <= 4)
                  .toList();
              allWords.addAll(words);
            }

            _log(
              'Parallel OCR complete: ${allWords.length} word(s) from $n crops in ${totalS.toStringAsFixed(1)}s'
              '${bench.backend.isNotEmpty ? " [${bench.backend}]" : ""}',
            );
          } catch (e) {
            _heartbeat?.cancel();
            _heartbeat = null;
            rethrow;
          }
        } else if (useMontage) {
          // ── Montage OCR ────────────────────────────────────────────
          _checkCancelled();
          _log(
            'Stitching ${imagesToProcess.length} crops into montage for single OCR pass...',
            phase: ProcessingPhase.ocr,
            progress: imgProgress + 0.02,
          );

          final montage = HighlightDetector.buildMontage(
            imagesToProcess,
            maxWidth: settings.montageMaxWidth,
          );
          final mLabel = '${imagesToProcess.length}-crop montage';
          final montageProgress = 0.10 +
              0.45 * (imgIdx + 0.5) / images.length;
          _log(
            'Running OCR on $mLabel (single inference call)...',
            progress: montageProgress,
          );

          final ocrStopwatch = Stopwatch()..start();

          _heartbeat?.cancel();
          _heartbeat = Timer.periodic(
            const Duration(seconds: 15),
            (timer) {
              final secs = ocrStopwatch.elapsed.inSeconds;
              final mins = (secs / 60).toStringAsFixed(1);
              String hint;
              if (secs < 15) {
                hint = 'preparing montage for vision encoder...';
              } else if (secs < 30) {
                hint = 'vision encode on GPU...';
              } else if (secs < 60) {
                hint = 'prompt eval on iGPU...';
              } else if (secs < 300) {
                hint = 'generating text from visual features...';
              } else {
                hint = 'still working (montage may be complex)...';
              }
              _log(
                'OCR in progress... $mins min elapsed - $hint',
                progress: montageProgress + 0.30 * (secs / 300).clamp(0.0, 1.0),
              );
            },
          );

          try {
            final result = await inference.visionOcr(imageBytes: montage);
            ocrStopwatch.stop();
            _heartbeat?.cancel();
            _heartbeat = null;

            final cropOcrS = ocrStopwatch.elapsedMilliseconds / 1000;
            bench.perCropOcrS = [cropOcrS];
            if (result.backend.isNotEmpty) bench.backend = result.backend;

            ocrTexts.add(result.text);

            final words = result.text
                .split(RegExp(r'[\n,]+'))
                .map((w) => w
                    .trim()
                    .replaceAll(RegExp(r'^[\s*\-\u2022\u00b7]+'), '')
                    .replaceAll(RegExp(r'^\d+\.\s*'), '')
                    .trim())
                .where((w) => w.length > 1 && w.split(RegExp(r'\s+')).length <= 4)
                .toList();
            allWords.addAll(words);

            final elapsed = (cropOcrS).toStringAsFixed(1);
            _log(
              'OCR on $mLabel complete: ${words.length} word(s) found in ${elapsed}s'
              '${result.backend.isNotEmpty ? " [${result.backend}]" : ""}',
            );
          } catch (e) {
            _heartbeat?.cancel();
            _heartbeat = null;
            rethrow;
          }
        } else {
        // ── Single image/crop ────────────────────────────────────────
        for (var i = 0; i < imagesToProcess.length; i++) {
          _checkCancelled();
          final label = imagesToProcess.length > 1
              ? 'crop ${i + 1}/${imagesToProcess.length}'
              : (images.length > 1 ? imgName : 'image');

          final cropProgress = 0.10 +
              0.45 *
                  (imgIdx + (i / imagesToProcess.length)) /
                  images.length;
          _log(
            'Running OCR on $label (this may take a minute)...',
            phase: ProcessingPhase.ocr,
            progress: cropProgress,
          );

        final ocrStopwatch = Stopwatch()..start();

        _heartbeat?.cancel();
        _heartbeat = Timer.periodic(
          const Duration(seconds: 15),
          (timer) {
            final secs = ocrStopwatch.elapsed.inSeconds;
            final mins = (secs / 60).toStringAsFixed(1);
            String hint;
            if (secs < 15) {
              hint = 'preparing image for vision encoder...';
            } else if (secs < 30) {
              hint = 'vision encode on GPU...';
            } else if (secs < 60) {
              hint = 'prompt eval on iGPU with flash attention...';
            } else if (secs < 300) {
              hint = 'generating text from visual features...';
            } else {
              hint = 'still working (this image may be complex)...';
            }
            _log(
              'OCR in progress... $mins min elapsed - $hint',
              progress: cropProgress + 0.30 * (secs / 300).clamp(0.0, 1.0),
            );
          },
        );

        try {
          final result = await inference.visionOcr(
            imageBytes: imagesToProcess[i],
          );
          ocrStopwatch.stop();
          _heartbeat?.cancel();
          _heartbeat = null;

          // Record per-crop OCR timing.
          final cropOcrS = ocrStopwatch.elapsedMilliseconds / 1000;
          bench.perCropOcrS = [...bench.perCropOcrS, cropOcrS];
          if (result.backend.isNotEmpty) bench.backend = result.backend;

          ocrTexts.add(result.text);

          final words = result.text
              .split(RegExp(r'[\n,]+'))
              .map((w) => w
                  .trim()
                  // Strip markdown bullets (*, -, •, ·), numbering (1., 2.)
                  .replaceAll(RegExp(r'^[\s*\-\u2022\u00b7]+'), '')
                  .replaceAll(RegExp(r'^\d+\.\s*'), '')
                  .trim())
              // Filter: must be >1 char, ≤4 words (skip LLM commentary sentences)
              .where((w) => w.length > 1 && w.split(RegExp(r'\s+')).length <= 4)
              .toList();
          allWords.addAll(words);

          final elapsed = (ocrStopwatch.elapsedMilliseconds / 1000).toStringAsFixed(1);
          _log(
            'OCR on $label complete: ${words.length} word(s) found in ${elapsed}s'
            '${result.backend.isNotEmpty ? " [${result.backend}]" : ""}',
          );
        } catch (e) {
          _heartbeat?.cancel();
          _heartbeat = null;
          rethrow;
        }
        }
        } // end parallel / montage / single branch
      } // end image loop

      bench.cropCount = totalCropCount;

      // Deduplicate (case-insensitive).
      final seen = <String>{};
      final uniqueWords = <String>[];
      for (final w in allWords) {
        if (seen.add(w.toLowerCase())) uniqueWords.add(w);
      }

      bench.rawWordCount = allWords.length;
      bench.uniqueWordCount = uniqueWords.length;
      // Total OCR time is the sum of per-crop durations.
      bench.ocrElapsedS =
          bench.perCropOcrS.fold(0.0, (sum, v) => sum + v);

      final dupes = allWords.length - uniqueWords.length;
      _log(
        '${uniqueWords.length} unique word(s) extracted'
        '${dupes > 0 ? " ($dupes duplicate(s) removed)" : ""}',
        progress: 0.60,
      );

      state = state.copyWith(
        words: uniqueWords,
        ocrText: ocrTexts.join('\n---\n'),
      );

      // Persist so a kill during word review can be resumed.
      await _savePendingBatch(
        images: images,
        words: uniqueWords,
        ocrResults: ocrTexts,
      );

      // Release image bytes after saving to reduce memory pressure.
      for (var i = 0; i < images.length; i++) {
        images[i].bytes = Uint8List(0);
      }

      // ── Word Review gate ────────────────────────────────────────────
      // Pause the pipeline so the user can edit / remove words before
      // spending GPU time on enrichment.
      _log(
        'Word extraction complete – review the word list before enrichment.',
        phase: ProcessingPhase.wordReview,
        progress: 0.62,
      );

      _wordReviewCompleter = Completer<WordReviewResult?>();
      final reviewResult = await _wordReviewCompleter!.future;
      _wordReviewCompleter = null;

      _checkCancelled();

      final confirmedWords = reviewResult?.words;
      final wordLanguages = reviewResult?.wordLanguages ?? {};

      // null = user chose to skip enrichment entirely.
      if (confirmedWords == null || confirmedWords.isEmpty) {
        _log('Enrichment skipped by user.', progress: 0.95);
        // Keep the existing word list and create stub cards so the
        // user can still review / export the raw words.
        final wordsToKeep = confirmedWords ?? state.words;
        final stubCards = wordsToKeep
            .map((w) => EnrichWordResult(
                  word: w,
                  definition: '',
                  examples: '',
                ))
            .toList();
        state = state.copyWith(
          words: wordsToKeep,
          enrichedWords: stubCards,
          enrichmentSkipped: true,
        );
      } else {
        // Update the word list with whatever the user confirmed.
        state = state.copyWith(words: confirmedWords);

        // Persist so a kill during enrichment can be resumed.
        await _updatePendingBatch(
          phase: ProcessingPhase.enriching,
          words: confirmedWords,
          wordLanguages: wordLanguages,
        );

      // Step 4: Enrich words (with cache integration).
      _checkCancelled();
      if (confirmedWords.isNotEmpty) {
        final db = ref.read(databaseProvider);

        // ── Cache lookup ─────────────────────────────────────────
        final cachedMap = await db.getCachedEnrichments(
          words: confirmedWords,
          definitionLanguage: settings.definitionLanguage,
          examplesLanguage: settings.examplesLanguage,
        );

        final cachedResults = <EnrichWordResult>[];
        final uncachedWords = <String>[];
        for (final w in confirmedWords) {
          final hit = cachedMap[w.toLowerCase()];
          if (hit != null && hit.warning != 'not_found') {
            cachedResults.add(EnrichWordResult(
              word: w,
              definition: hit.definition,
              examples: hit.examples,
              warning: hit.warning,
            ));
          } else {
            uncachedWords.add(w);
          }
        }

        if (cachedResults.isNotEmpty) {
          _log(
            '${cachedResults.length} word(s) found in cache, '
            '${uncachedWords.length} word(s) need enrichment',
            progress: 0.64,
          );
          // Push cached results immediately so the UI shows them.
          state = state.copyWith(
            enrichedWords: [...state.enrichedWords, ...cachedResults],
          );
        }

        // ── Send OCR-complete notification ────────────────────────
        _notify('OCR Complete',
            '${confirmedWords.length} word(s) extracted – enriching…');

        if (uncachedWords.isNotEmpty) {
        final wordCount = uncachedWords.length;
        _log(
          'Enriching $wordCount word(s) with definitions '
          '(${settings.definitionLanguage})...',
          phase: ProcessingPhase.enriching,
          progress: 0.65,
        );
        _log(
          'Generating definitions and example sentences via LLM '
          '(1 word per call, 10 min timeout each)...',
          progress: 0.68,
        );

        final enrichStopwatch = Stopwatch()..start();

        // Heartbeat for enrichment phase.
        _heartbeat?.cancel();
        _heartbeat = Timer.periodic(
          const Duration(seconds: 15),
          (timer) {
            final secs = enrichStopwatch.elapsed.inSeconds;
            _log(
              'Enrichment in progress... ${secs}s elapsed '
              '(chunked LLM calls for $wordCount word(s))',
              progress: 0.68 + 0.20 * (secs / 120).clamp(0.0, 1.0),
            );
          },
        );

        final enriched = await inference.enrichWords(
          words: uncachedWords,
          definitionLanguage: settings.definitionLanguage,
          examplesLanguage: settings.examplesLanguage,
          termLanguage: effectiveTermLang,
          wordLanguages: wordLanguages,
          chunkSize: 1,
          chunkTimeout: const Duration(minutes: 10),
          onChunkDone: (completed, total, chunkResults) {
            // Push partial results to state so the UI shows cards appearing.
            state = state.copyWith(
              enrichedWords: [...state.enrichedWords, ...chunkResults],
            );
            _log(
              'Enrichment: $completed/$total word(s) done',
              progress: 0.68 + 0.20 * (completed / total),
            );
            _updatePendingBatch(enrichedWords: state.enrichedWords);
            _checkCancelled();
          },
        );
        enrichStopwatch.stop();
        _heartbeat?.cancel();
        _heartbeat = null;

        final enrichElapsed =
            (enrichStopwatch.elapsedMilliseconds / 1000).toStringAsFixed(1);

        // Record enrichment benchmark data.
        bench.enrichElapsedS = enrichStopwatch.elapsedMilliseconds / 1000;
        bench.enrichedWordCount =
            cachedResults.length + enriched.length;
        bench.warningNotFoundCount =
            enriched.where((e) => e.warning == 'not_found').length;
        bench.warningTruncatedCount =
            enriched.where((e) => e.warning == 'truncated').length;

        _log(
          'Enrichment complete: ${enriched.length} card(s) in ${enrichElapsed}s'
          '${bench.totalWarnings > 0 ? " (${bench.totalWarnings} warning(s))" : ""}',
          progress: 0.90,
        );

        // ── Store new results in cache ───────────────────────────
        final toCache = enriched
            .where((e) => e.warning != 'not_found')
            .map((e) => EnrichmentCacheEntriesCompanion.insert(
                  word: e.word.toLowerCase(),
                  definitionLanguage: settings.definitionLanguage,
                  examplesLanguage: settings.examplesLanguage,
                  definition: e.definition,
                  examples: e.examples,
                  warning: Value(e.warning),
                ))
            .toList();
        if (toCache.isNotEmpty) {
          await db.cacheEnrichments(toCache);
          _log('Cached ${toCache.length} enrichment(s) for future re-use');
        }

        // Combine cached + freshly enriched in original word order.
        final allResultsMap = <String, EnrichWordResult>{};
        for (final r in cachedResults) {
          allResultsMap[r.word.toLowerCase()] = r;
        }
        for (final r in enriched) {
          allResultsMap[r.word.toLowerCase()] = r;
        }
        final orderedResults = confirmedWords
            .map((w) => allResultsMap[w.toLowerCase()])
            .whereType<EnrichWordResult>()
            .toList();

        state = state.copyWith(enrichedWords: orderedResults);

        } else {
          // All words were cached — no LLM call needed.
          _log('All words served from cache!', progress: 0.90);
          bench.enrichedWordCount = cachedResults.length;
          state = state.copyWith(
            phase: ProcessingPhase.enriching,
            enrichedWords: cachedResults,
          );
        }

        // ── Enrichment-done notification ─────────────────────────
        _notify('Enrichment Done',
            '${state.enrichedWords.length} card(s) ready for review');

      } else {
        _log('No words to enrich', progress: 0.90);
      }
      } // end of confirmedWords else block

      // Step 5: Persist to local database + benchmark log.
      _checkCancelled();
      _log('Saving session to local database...', progress: 0.92);

      stopwatch.stop();
      bench.totalElapsedS = stopwatch.elapsedMilliseconds / 1000;

      final dbPersist = ref.read(databaseProvider);
      final sessionId = await dbPersist.insertSession(
        ProcessingSessionsCompanion.insert(
          imagePath: images.map((i) => i.name).join(', '),
          context: hsvRange != null ? 'highlighted' : 'handwrittenOrPrinted',
          highlightColor: Value(hsvRange?.label),
          ocrText: Value(state.ocrText),
          ocrElapsedS: Value(bench.ocrElapsedS),
          enrichElapsedS: Value(bench.enrichElapsedS),
          backend: Value(bench.backend),
          benchmarkJson: Value(bench.toJsonString()),
        ),
      );

      final wordCompanions = state.enrichedWords
          .map((e) => WordEntriesCompanion.insert(
                sessionId: sessionId,
                word: e.word,
                definition: Value(e.definition),
                examples: Value(e.examples),
              ))
          .toList();

      if (wordCompanions.isNotEmpty) {
        await dbPersist.insertWords(wordCompanions);
        _log('Saved ${wordCompanions.length} word(s) to database',
            progress: 0.96);
      }

      final totalElapsed = bench.totalElapsedS.toStringAsFixed(1);

      // Log benchmark summary.
      _log(
        '── Benchmark ──'
        '\n  Image: ${bench.imageSizeFormatted}'
        '${bench.cropCount > 0 ? "\n  Crops: ${bench.cropCount} (${bench.cropElapsedS.toStringAsFixed(1)}s)" : ""}'
        '\n  OCR: ${bench.ocrElapsedS.toStringAsFixed(1)}s'
        '${bench.perCropOcrS.length > 1 ? " (${bench.perCropOcrS.map((s) => "${s.toStringAsFixed(1)}s").join(", ")})" : ""}'
        '\n  Enrichment: ${bench.enrichElapsedS.toStringAsFixed(1)}s'
        ' (${bench.enrichedWordCount} words, '
        '${bench.avgEnrichPerWordS.toStringAsFixed(1)}s/word)'
        '${bench.totalWarnings > 0 ? "\n  Warnings: ${bench.warningNotFoundCount} not_found, ${bench.warningTruncatedCount} truncated" : ""}'
        '\n  Total: ${totalElapsed}s'
        '${bench.backend.isNotEmpty ? "\n  Backend: ${bench.backend}" : ""}',
      );

      if (state.enrichedWords.isNotEmpty) {
        _log(
          'Done! ${state.enrichedWords.length} card(s) ready for export '
          '(${totalElapsed}s total)',
          phase: ProcessingPhase.done,
          progress: 1.0,
        );
      } else {
        _log(
          'Done. No words found in image (${totalElapsed}s total)',
          phase: ProcessingPhase.done,
          progress: 1.0,
        );
      }
    } catch (e) {
      _heartbeat?.cancel();
      _heartbeat = null;
      stopwatch.stop();
      if (_cancelled) {
        final log = [...state.activityLog, 'Processing cancelled.'];
        state = state.copyWith(
          phase: ProcessingPhase.done,
          statusMessage: 'Cancelled',
          activityLog: log,
        );
      } else {
        final log = [...state.activityLog, 'ERROR: $e'];
        state = state.copyWith(
          phase: ProcessingPhase.error,
          error: e.toString(),
          statusMessage: 'Error: $e',
          activityLog: log,
        );
      }
    } finally {
      await clearPendingBatch();
      if (Platform.isAndroid) {
        await ForegroundTaskService.update(detail: 'AI model ready');
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Android system permission state
// ---------------------------------------------------------------------------

/// Live observation of POST_NOTIFICATIONS grant.  Returns true on
/// non-Android platforms and on Android API < 33.  Invalidate this
/// provider after the user returns from app-details settings to refresh
/// the home-screen banner.
final notificationsGrantedProvider = FutureProvider<bool>((ref) {
  return SystemChannel.isPostNotificationsGranted();
});

/// Per-session dismissal flag for the "notifications denied" banner on
/// the home screen.  Resets to false on every app launch.
final notificationsBannerDismissedProvider =
    NotifierProvider<_NotificationsBannerDismissedNotifier, bool>(
        _NotificationsBannerDismissedNotifier.new);

class _NotificationsBannerDismissedNotifier extends Notifier<bool> {
  @override
  bool build() => false;
  void dismiss() => state = true;
}

/// Live observation of battery-optimisation whitelist state.  Used by the
/// settings screen tile and refreshed on app resume.
final batteryOptimizationDisabledProvider = FutureProvider<bool>((ref) {
  return SystemChannel.isBatteryOptimizationDisabled();
});

// ---------------------------------------------------------------------------
// Inbound share intents (Android only)
// ---------------------------------------------------------------------------

/// Which screen is shown in the detail pane of a two-pane layout.
/// Driven by pipeline state rather than navigation routes.
enum DetailScreen { none, processing, review }

final detailScreenProvider =
    NotifierProvider<_DetailScreenNotifier, DetailScreen>(
        _DetailScreenNotifier.new);

class _DetailScreenNotifier extends Notifier<DetailScreen> {
  @override
  DetailScreen build() => DetailScreen.none;
  void show(DetailScreen screen) => state = screen;
  void clear() => state = DetailScreen.none;
}

/// Singleton handler for `ACTION_SEND` / `ACTION_SEND_MULTIPLE` image
/// intents.  Constructed eagerly during app boot via [OcrToAnkiApp.initState]
/// so neither the cold-launch share intent nor warm-launch onNewIntent
/// events are dropped.  No-op on non-Android platforms.
final shareIntentHandlerProvider = Provider<ShareIntentHandler>((ref) {
  final handler = ShareIntentHandler();
  ref.onDispose(handler.dispose);
  return handler;
});