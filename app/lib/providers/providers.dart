import 'dart:async';
import 'dart:convert';

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
    StateNotifierProvider<SettingsNotifier, AppSettings>((ref) {
  return SettingsNotifier(ref.watch(databaseProvider));
});

class SettingsNotifier extends StateNotifier<AppSettings> {
  SettingsNotifier(this._db) : super(AppSettings()) {
    _load();
  }

  final AppDatabase _db;

  Future<void> _load() async {
    final json = await _db.getSetting('app_settings');
    if (json != null) {
      try {
        state = AppSettings.fromJson(
          jsonDecode(json) as Map<String, dynamic>,
        );
        // Migrations:
        var dirty = false;
        // 1) "localhost" resolves to IPv6 ::1 on some systems while the
        //    server only listens on IPv4, causing connection failures.
        if (state.serverUrl.contains('://localhost:')) {
          state.serverUrl =
              state.serverUrl.replaceFirst('://localhost:', '://127.0.0.1:');
          dirty = true;
        }
        // 2) Embedded mode is not yet implemented -- force remote.
        if (state.inferenceMode == InferenceMode.embedded) {
          state.inferenceMode = InferenceMode.remote;
          dirty = true;
        }
        if (dirty) {
          state = state; // notify listeners
          await _db.setSetting('app_settings', jsonEncode(state.toJson()));
        }
      } catch (_) {
        // Corrupted settings -- keep defaults.
      }
    }
  }

  Future<void> update(AppSettings Function(AppSettings) updater) async {
    state = updater(state);
    await _db.setSetting('app_settings', jsonEncode(state.toJson()));
  }
}

// ---------------------------------------------------------------------------
// Services
// ---------------------------------------------------------------------------

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

enum ProcessingPhase { idle, cropping, ocr, enriching, done, error }

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
    StateNotifierProvider<ProcessingNotifier, ProcessingState>((ref) {
  return ProcessingNotifier(ref);
});

class ProcessingNotifier extends StateNotifier<ProcessingState> {
  ProcessingNotifier(this._ref) : super(const ProcessingState());

  final Ref _ref;
  bool _cancelled = false;
  Timer? _heartbeat;

  void reset() {
    _cancelled = false;
    _heartbeat?.cancel();
    _heartbeat = null;
    state = const ProcessingState();
  }

  /// Request cancellation of the current pipeline.
  void cancel() {
    _cancelled = true;
    // Stop the heartbeat timer immediately so no more log noise.
    _heartbeat?.cancel();
    _heartbeat = null;
    _log('Cancellation requested...', progress: state.progress);
    // Kill the server-side subprocess and abort the HTTP request.
    final inference = _ref.read(inferenceServiceProvider);
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

  /// Run the full pipeline: optionally crop highlights -> vision OCR -> enrich.
  Future<void> processImage({
    required Uint8List imageBytes,
    required String filename,
    required OcrContext context,
    HighlightColor? highlightColor,
  }) async {
    final imgBytes = imageBytes;
    final stopwatch = Stopwatch()..start();
    try {
      state = const ProcessingState().copyWith(
        phase: ProcessingPhase.ocr,
        progress: 0,
        statusMessage: 'Preparing image...',
        startTime: DateTime.now(),
        activityLog: ['Processing started for: $filename'],
      );

      final inference = _ref.read(inferenceServiceProvider);
      final settings = _ref.read(settingsProvider);

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
        final detector = _ref.read(highlightDetectorProvider);
        final crops = detector.detectAndCrop(
          imageBytes: imgBytes,
          color: highlightColor,
        );
        if (crops.isNotEmpty) {
          imagesToProcess = crops;
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
        // Two timing profiles depending on GPU mmproj offloading:
        //
        // FAST (GPU mmproj, preempt_timeout_ms=0):
        //   0-30s  : image prep / model loading
        //   ~22s   : SigLIP vision encode on GPU
        //   ~40s   : prompt eval with FA on GPU
        //   ~27s   : token generation
        //   ~90s   : done
        //
        // SLOW (CPU mmproj, default preemption timeout):
        //   0-30s  : image prep / model loading
        //   ~20min : SigLIP vision encode on CPU (bottleneck)
        //   ~40s   : prompt eval with FA on GPU
        //   ~27s   : token generation
        //   ~21min : done
        //
        // We can't know which mode the server uses, so we show adaptive
        // hints that work for both cases.
        _heartbeat?.cancel();
        _heartbeat = Timer.periodic(
          const Duration(seconds: 15),
          (timer) {
            final secs = ocrStopwatch.elapsed.inSeconds;
            final mins = (secs / 60).toStringAsFixed(1);
            String hint;
            if (secs < 30) {
              hint = 'preparing image for vision encoder...';
            } else if (secs < 90) {
              hint = 'vision encoding (GPU: ~22s, CPU: ~20 min)...';
            } else if (secs < 120) {
              hint = 'prompt eval + generating text...';
            } else if (secs < 1200) {
              hint = 'vision encode on CPU (slowest phase, ~20 min)...';
            } else if (secs < 1320) {
              hint = 'prompt eval on iGPU with flash attention (~40s)...';
            } else if (secs < 1400) {
              hint = 'generating text from visual features (~27s)...';
            } else {
              hint = 'still working (this image may be complex)...';
            }
            _log(
              'OCR in progress... $mins min elapsed - $hint',
              progress: 0.20 + 0.30 * (secs / 1400).clamp(0.0, 1.0),
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

          ocrTexts.add(result.text);

          final words = result.text
              .split(RegExp(r'[\n,]+'))
              .map((w) => w.trim().replaceAll(RegExp(r'^[-\u2022\u00b7]+'), '').trim())
              .where((w) => w.length > 1)
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

      // Step 4: Enrich words.
      _checkCancelled();
      if (uniqueWords.isNotEmpty) {
        final wordCount = uniqueWords.length.clamp(0, 20);
        _log(
          'Enriching $wordCount word(s) with definitions '
          '(${settings.definitionLanguage})...',
          phase: ProcessingPhase.enriching,
          progress: 0.65,
        );
        _log(
          'Generating definitions and example sentences via LLM...',
          progress: 0.68,
        );

        final enrichStopwatch = Stopwatch()..start();
        final enriched = await inference.enrichWords(
          words: uniqueWords.take(20).toList(),
          definitionLanguage: settings.definitionLanguage,
          examplesLanguage: settings.examplesLanguage,
        );
        enrichStopwatch.stop();

        final enrichElapsed =
            (enrichStopwatch.elapsedMilliseconds / 1000).toStringAsFixed(1);
        _log(
          'Enrichment complete: ${enriched.length} card(s) in ${enrichElapsed}s',
          progress: 0.90,
        );

        state = state.copyWith(enrichedWords: enriched);
      } else {
        _log('No words to enrich', progress: 0.90);
      }

      // Step 5: Persist to local database.
      _checkCancelled();
      _log('Saving session to local database...', progress: 0.92);
      final db = _ref.read(databaseProvider);
      final ocrElapsed = stopwatch.elapsedMilliseconds / 1000;
      final sessionId = await db.insertSession(
        ProcessingSessionsCompanion.insert(
          imagePath: filename,
          context: context.name,
          highlightColor: Value(highlightColor?.name),
          ocrText: Value(state.ocrText),
          ocrElapsedS: Value(ocrElapsed),
          enrichElapsedS: const Value(0),
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

      stopwatch.stop();
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
          'Done. No words found in image (${totalElapsed}s total)',
          phase: ProcessingPhase.done,
          progress: 1.0,
        );
      }
    } catch (e) {
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
