import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:drift/drift.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../database/database.dart';
import '../models/models.dart';
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
        // 2) Embedded mode is not yet implemented -- force remote.
        if (loaded.inferenceMode == InferenceMode.embedded) {
          loaded.inferenceMode = InferenceMode.remote;
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

/// Tracks whether the backend server is ready.
///
/// The [ServerStartupNotifier] is initialised eagerly by the startup gate
/// widget.  It starts the backend and exposes the current status so the UI
/// can show a loading / error screen.
enum ServerStatus { starting, ready, error }

class ServerStartupState {
  const ServerStartupState({
    this.status = ServerStatus.starting,
    this.message = 'Starting backend server…',
  });
  final ServerStatus status;
  final String message;
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
    try {
      final server = ref.read(backendServerProvider);
      state = const ServerStartupState(
        status: ServerStatus.starting,
        message: 'Starting backend server…',
      );
      await server.start(timeout: const Duration(seconds: 60));
      if (_disposed) return;
      state = const ServerStartupState(
        status: ServerStatus.ready,
        message: 'Backend ready.',
      );
    } catch (e) {
      if (_disposed) return;
      state = ServerStartupState(
        status: ServerStatus.error,
        message: 'Failed to start backend: $e',
      );
    }
  }

  /// Allow retry from the error screen.
  Future<void> retry() async => _boot();
}

final inferenceServiceProvider = Provider<InferenceService>((ref) {
  final settings = ref.watch(settingsProvider);
  return InferenceService(settings: settings);
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
// Processing state
// ---------------------------------------------------------------------------

enum ProcessingPhase { idle, cropping, ocr, wordReview, enriching, done, error }

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
      );
}

final processingProvider =
    NotifierProvider<ProcessingNotifier, ProcessingState>(
        ProcessingNotifier.new);

class ProcessingNotifier extends Notifier<ProcessingState> {
  bool _cancelled = false;
  Timer? _heartbeat;

  /// Completer that the pipeline awaits during word review.
  Completer<List<String>?>? _wordReviewCompleter;

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
  void confirmWords(List<String> words) {
    _wordReviewCompleter?.complete(words);
  }

  /// User chose to skip enrichment entirely.
  void skipEnrichment() {
    _wordReviewCompleter?.complete(null);
  }

  /// Request cancellation of the current pipeline.
  void cancel() {
    _cancelled = true;
    // Stop the heartbeat timer immediately so no more log noise.
    _heartbeat?.cancel();
    _heartbeat = null;
    _log('Cancellation requested...', progress: state.progress);
    // Kill the server-side subprocess and abort the HTTP request.
    final inference = ref.read(inferenceServiceProvider);
    inference.cancelOcr();
  }

  /// Throws if cancel() has been called.
  void _checkCancelled() {
    if (_cancelled) throw Exception('Processing cancelled by user.');
  }

  /// Append a line to the activity log and update the status message.
  void _log(String message, {ProcessingPhase? phase, double? progress}) {
    final log = [...state.activityLog, message];
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

  /// Run the full pipeline: optionally crop highlights -> vision OCR -> enrich.
  Future<void> processImage({
    required Uint8List imageBytes,
    required String filename,
    required OcrContext context,
    HighlightColor? highlightColor,
  }) async {
    final imgBytes = imageBytes;
    final stopwatch = Stopwatch()..start();

    // Benchmark data collector.
    final bench = BenchmarkData(
      timestamp: DateTime.now(),
      imageSizeBytes: imgBytes.length,
    );

    try {
      state = const ProcessingState().copyWith(
        phase: ProcessingPhase.ocr,
        progress: 0,
        statusMessage: 'Preparing image...',
        startTime: DateTime.now(),
        activityLog: ['Processing started for: $filename'],
      );

      final inference = ref.read(inferenceServiceProvider);
      final settings = ref.read(settingsProvider);

      bench.definitionLanguage = settings.definitionLanguage;
      bench.examplesLanguage = settings.examplesLanguage;

      _log('Image loaded (${_formatBytes(imgBytes.length)})', progress: 0.05);

      _checkCancelled();

      // Step 1: Verify server connectivity.
      _log('Checking server connection...', progress: 0.07);
      final serverOk = await inference.isAvailable();
      if (!serverOk) {
        final detail = inference.debugMessage ?? 'no details';
        throw Exception(
          'Cannot reach inference server at ${settings.serverUrl} ($detail). '
          'Make sure the FastAPI backend is running.',
        );
      }
      _log('Server is online (${settings.serverUrl})', progress: 0.08);

      // Step 2: If highlighted context, crop first.
      List<Uint8List> imagesToProcess = [imgBytes];
      if (context == OcrContext.highlighted && highlightColor != null) {
        _log(
          'Scanning for ${highlightColor.label} highlights...',
          phase: ProcessingPhase.cropping,
          progress: 0.10,
        );
        final cropWatch = Stopwatch()..start();
        final detector = ref.read(highlightDetectorProvider);
        final crops = detector.detectAndCrop(
          imageBytes: imgBytes,
          color: highlightColor,
        );
        cropWatch.stop();
        bench.cropElapsedS = cropWatch.elapsedMilliseconds / 1000;
        if (crops.isNotEmpty) {
          imagesToProcess = crops;
          bench.cropCount = crops.length;
          _log('Found ${crops.length} highlighted region(s)', progress: 0.15);
        } else {
          _log('No highlighted regions detected, using full image',
              progress: 0.15);
        }
      } else {
        _log(
          'Mode: ${context == OcrContext.handwrittenOrPrinted ? "handwritten/printed text" : "highlighted text"}',
          progress: 0.10,
        );
      }

      // Step 3: Vision OCR on each image/crop.
      _log(
        'Sending ${imagesToProcess.length} image(s) to vision model...',
        phase: ProcessingPhase.ocr,
        progress: 0.18,
      );

      final allWords = <String>[];
      final ocrTexts = <String>[];

      for (var i = 0; i < imagesToProcess.length; i++) {
        _checkCancelled();
        final label = imagesToProcess.length > 1
            ? 'crop ${i + 1}/${imagesToProcess.length}'
            : 'image';
        _log(
          'Running OCR on $label (this may take a minute)...',
          progress: 0.20 + 0.35 * i / imagesToProcess.length,
        );

        final ocrStopwatch = Stopwatch()..start();

        // Heartbeat: log progress every 15s so the user isn't staring
        // at a frozen screen.  Images are auto-downscaled to 768px max
        // on the server side to reduce SigLIP tile count.
        //
        // GPU Vulkan timing profile (Intel iGPU, patched ggml-vulkan.cpp
        // with serialize_graph + force_f32_matmul):
        //   0-15s   : image prep / downscale / server start
        //   15-30s  : SigLIP vision encode on GPU (~14s)
        //   30-60s  : prompt eval with flash attention (~40s)
        //   60-240s : token generation (~27s per ~100 tokens)
        //   ~4 min  : done (total ~228s typical)
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
              progress: 0.20 + 0.30 * (secs / 300).clamp(0.0, 1.0),
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
            progress: 0.20 + 0.35 * (i + 1) / imagesToProcess.length,
          );
        } catch (e) {
          _heartbeat?.cancel();
          _heartbeat = null;
          rethrow;
        }
      }

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

      // ── Word Review gate ────────────────────────────────────────────
      // Pause the pipeline so the user can edit / remove words before
      // spending GPU time on enrichment.
      _log(
        'Word extraction complete – review the word list before enrichment.',
        phase: ProcessingPhase.wordReview,
        progress: 0.62,
      );

      _wordReviewCompleter = Completer<List<String>?>();
      final confirmedWords = await _wordReviewCompleter!.future;
      _wordReviewCompleter = null;

      _checkCancelled();

      // null = user chose to skip enrichment entirely.
      if (confirmedWords == null || confirmedWords.isEmpty) {
        _log('Enrichment skipped by user.', progress: 0.95);
        state = state.copyWith(words: confirmedWords ?? []);
      } else {
        // Update the word list with whatever the user confirmed.
        state = state.copyWith(words: confirmedWords);

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
          imagePath: filename,
          context: context.name,
          highlightColor: Value(highlightColor?.name),
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
      final log = [...state.activityLog, 'ERROR: $e'];
      state = state.copyWith(
        phase: ProcessingPhase.error,
        error: e.toString(),
        statusMessage: 'Error: $e',
        activityLog: log,
      );
    }
  }
}
