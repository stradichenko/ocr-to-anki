import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

import '../models/app_settings.dart';
import 'llama_cpp_android_service.dart';

/// Result of a vision OCR call.
class VisionOcrResult {
  const VisionOcrResult({
    required this.text,
    required this.elapsedSeconds,
    required this.backend,
  });

  final String text;
  final double elapsedSeconds;
  final String backend;
}

/// Result of a text-generation / enrichment call.
class EnrichWordResult {
  const EnrichWordResult({
    required this.word,
    required this.definition,
    required this.examples,
    this.warning = '',
    this.correctedWord = '',
    this.fromCache = false,
  });

  final String word;
  final String definition;
  final String examples;

  /// Quality warning: 'not_found', 'truncated', or '' (ok).
  final String warning;

  /// LLM-suggested correct spelling when OCR produced a misspelling.
  /// Empty if the word is already correct.
  final String correctedWord;

  /// Whether this result was served from the local enrichment cache.
  final bool fromCache;

  Map<String, dynamic> toJson() => {
        'word': word,
        'definition': definition,
        'examples': examples,
        'warning': warning,
        'correctedWord': correctedWord,
        'fromCache': fromCache,
      };

  factory EnrichWordResult.fromJson(Map<String, dynamic> json) =>
      EnrichWordResult(
        word: json['word'] as String,
        definition: json['definition'] as String,
        examples: json['examples'] as String,
        warning: json['warning'] as String? ?? '',
        correctedWord: json['correctedWord'] as String? ?? '',
        fromCache: json['fromCache'] as bool? ?? false,
      );
}

/// Unified inference service that forwards requests to the Python FastAPI
/// backend defined in `src/api/app.py`.
///
/// On Android, uses [LlamaCppAndroidService] to talk directly to the
/// bundled llama.cpp binaries instead of going through the Python backend.
class InferenceService {
  InferenceService({
    required AppSettings settings,
    LlamaCppAndroidService? androidService,
  })  : _settings = settings,
        _androidService = androidService;

  final AppSettings _settings;
  final LlamaCppAndroidService? _androidService;

  bool get _isAndroid => _androidService != null;

  /// HTTP client for the current OCR request; can be closed to abort.
  http.Client? _ocrClient;

  /// HTTP client for the current enrichment request; can be closed to abort.
  http.Client? _enrichClient;

  /// Cancel any in-flight OCR request.
  ///
  /// This (1) closes the HTTP client to abort the request and
  /// (2) tells the server to kill the running subprocess.
  Future<void> cancelOcr() async {
    // Abort the in-flight HTTP request.
    _ocrClient?.close();
    _ocrClient = null;

    // Best-effort: ask the server to kill the subprocess.
    try {
      await http
          .post(Uri.parse('${_settings.serverUrl}/ocr/cancel'))
          .timeout(const Duration(seconds: 5));
    } catch (_) {
      // Server may already be done or unreachable -- that is fine.
    }
  }

  /// Cancel any in-flight enrichment request.
  ///
  /// Closes the HTTP client to abort the current chunk request.
  /// Also signals the Android service to cancel generation if running.
  Future<void> cancelEnrichment() async {
    _enrichClient?.close();
    _enrichClient = null;

    if (_isAndroid) {
      _androidService?.cancelGeneration();
    }
  }

  // ---------------------------------------------------------------------------
  // Health
  // ---------------------------------------------------------------------------

  /// Check whether the backend is reachable.
  Future<bool> isAvailable() async {
    if (_isAndroid) {
      return _androidService!.checkHealth();
    }
    return _remoteHealthCheck();
  }

  Future<bool> _remoteHealthCheck() async {
    debugMessage = null;
    final url = '${_settings.serverUrl}/health';
    // Retry up to 3 times with a short delay to handle transient failures
    // (e.g. server still starting up after GPU coordination pause).
    for (var attempt = 1; attempt <= 3; attempt++) {
      try {
        final uri = Uri.parse(url);
        final response = await http.get(uri).timeout(
              Duration(seconds: _settings.ankiConnectTimeout),
            );
        if (response.statusCode == 200) {
          final body = jsonDecode(response.body) as Map<String, dynamic>;
          final ok = body['status'] == 'ok' || body['status'] == 'degraded';
          if (!ok) {
            debugMessage = 'Server returned status: ${body['status']}';
          } else {
            debugMessage = null; // clear any prior attempt error
          }
          return ok;
        }
        debugMessage =
            'GET $url returned HTTP ${response.statusCode} (attempt $attempt)';
      } catch (e) {
        debugMessage = 'GET $url -- $e (attempt $attempt)';
      }
      if (attempt < 3) {
        await Future<void>.delayed(const Duration(seconds: 2));
      }
    }
    return false;
  }

  /// Diagnostic message from the last failed health check (if any).
  String? debugMessage;

  // ---------------------------------------------------------------------------
  // Vision OCR
  // ---------------------------------------------------------------------------

  /// Extract text from an image.
  ///
  /// [imageBytes] -- raw JPEG / PNG bytes.
  /// [prompt] -- vision prompt for the model.
  Future<VisionOcrResult> visionOcr({
    required Uint8List imageBytes,
    String prompt =
        'List every word visible in this image. Output ONLY the words, one per line. No bullet points, no numbering, no descriptions, no commentary.',
    int timeoutSeconds = 2700,
  }) async {
    if (_isAndroid) {
      return _androidVisionOcr(imageBytes: imageBytes, prompt: prompt);
    }
    return _remoteVisionOcr(
      imageBytes: imageBytes,
      prompt: prompt,
      timeoutSeconds: timeoutSeconds,
    );
  }

  Future<VisionOcrResult> _androidVisionOcr({
    required Uint8List imageBytes,
    required String prompt,
  }) async {
    await _androidService!.ensureServerRunning();
    final stopwatch = Stopwatch()..start();
    Uint8List bytesToOcr = imageBytes;
    // On Android, always downscale images to save memory regardless of the
    // user's compressLargeImages setting.  Vision models don't need full
    // resolution; 1024 px on the long edge is more than enough.
    if (Platform.isAndroid) {
      bytesToOcr = _maybeCompressImage(imageBytes);
    } else if (_settings.compressLargeImages &&
        imageBytes.length > 1024 * 1024) {
      bytesToOcr = _maybeCompressImage(imageBytes);
    }
    final text = await _androidService.runVisionOcr(
      imageBytes: bytesToOcr,
      prompt: prompt,
    );
    stopwatch.stop();
    return VisionOcrResult(
      text: text,
      elapsedSeconds: stopwatch.elapsedMilliseconds / 1000.0,
      backend: 'llama-mtmd-cli (Android)',
    );
  }

  Future<VisionOcrResult> _remoteVisionOcr({
    required Uint8List imageBytes,
    required String prompt,
    required int timeoutSeconds,
  }) async {
    final uri = Uri.parse('${_settings.serverUrl}/ocr/vision');
    final b64 = base64Encode(imageBytes);

    _ocrClient = http.Client();
    try {
      final response = await _ocrClient!
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'image_base64': b64,
              'prompt': prompt,
              'timeout': timeoutSeconds,
            }),
          )
          .timeout(Duration(seconds: timeoutSeconds + 30));

      if (response.statusCode != 200) {
        throw Exception(
          'Vision OCR failed (${response.statusCode}): ${response.body}',
        );
      }

      final body = jsonDecode(response.body) as Map<String, dynamic>;
      return VisionOcrResult(
        text: body['text'] as String? ?? '',
        elapsedSeconds: (body['elapsed_s'] as num?)?.toDouble() ?? 0,
        backend: body['backend'] as String? ?? 'remote',
      );
    } finally {
      _ocrClient?.close();
      _ocrClient = null;
    }
  }

  /// Compress an image if it exceeds 1 MB by resizing so the long edge is
  /// at most 1024 px and re-encoding as JPEG at quality 85.
  Uint8List _maybeCompressImage(Uint8List bytes) {
    try {
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return bytes;

      const maxDimension = 1024;
      if (decoded.width <= maxDimension && decoded.height <= maxDimension) {
        return bytes;
      }

      final resized = img.copyResize(
        decoded,
        width: decoded.width > decoded.height ? maxDimension : null,
        height: decoded.height >= decoded.width ? maxDimension : null,
      );
      return Uint8List.fromList(img.encodeJpg(resized, quality: 85));
    } catch (_) {
      // If anything goes wrong, fall back to the original bytes.
      return bytes;
    }
  }

  // ---------------------------------------------------------------------------
  // Vocabulary Enrichment
  // ---------------------------------------------------------------------------

  /// Enrich a list of words with definitions and example sentences.
  ///
  /// Words are sent in chunks of [chunkSize] with a per-chunk [chunkTimeout].
  /// This makes the timeout scale naturally regardless of total word count
  /// (100s or 1000s of words all work the same).
  ///
  /// [onChunkDone] is called after each chunk finishes so the UI can show
  /// live progress.  It receives the completed count, total, and the results
  /// from the just-finished chunk.
  Future<List<EnrichWordResult>> enrichWords({
    required List<String> words,
    String? definitionLanguage,
    String? examplesLanguage,
    String? termLanguage,
    Map<String, String> wordLanguages = const {},
    int? maxTokens,
    double? temperature,
    int chunkSize = 6,
    Duration chunkTimeout = const Duration(minutes: 5),
    void Function(int completedWords, int totalWords,
        List<EnrichWordResult> chunkResults)? onChunkDone,
  }) async {
    final defLang = definitionLanguage ?? _settings.definitionLanguage;
    final exLang = examplesLanguage ?? _settings.examplesLanguage;
    final termLang = termLanguage ?? _settings.termLanguage;
    final temp = temperature ?? _settings.temperature;
    final tokens = maxTokens ?? 256;

    if (_isAndroid) {
      return _androidEnrichChunked(
        words: words,
        definitionLanguage: defLang,
        examplesLanguage: exLang,
        termLanguage: termLang,
        wordLanguages: wordLanguages,
        maxTokens: tokens,
        temperature: temp,
        chunkSize: chunkSize,
        onChunkDone: onChunkDone,
      );
    }

    return _remoteEnrichChunked(
      words: words,
      definitionLanguage: defLang,
      examplesLanguage: exLang,
      termLanguage: termLang,
      wordLanguages: wordLanguages,
      maxTokens: tokens,
      temperature: temp,
      chunkSize: chunkSize,
      chunkTimeout: chunkTimeout,
      onChunkDone: onChunkDone,
    );
  }

  /// Send words in small chunks, each with its own timeout.
  Future<List<EnrichWordResult>> _remoteEnrichChunked({
    required List<String> words,
    required String definitionLanguage,
    required String examplesLanguage,
    required String termLanguage,
    required Map<String, String> wordLanguages,
    required int maxTokens,
    required double temperature,
    required int chunkSize,
    required Duration chunkTimeout,
    void Function(int completedWords, int totalWords,
        List<EnrichWordResult> chunkResults)? onChunkDone,
  }) async {
    final allResults = <EnrichWordResult>[];
    final chunks = <List<String>>[];

    for (var i = 0; i < words.length; i += chunkSize) {
      chunks.add(words.sublist(i, (i + chunkSize).clamp(0, words.length)));
    }

    _enrichClient = http.Client();
    try {
      for (var ci = 0; ci < chunks.length; ci++) {
        final chunk = chunks[ci];
        final uri = Uri.parse('${_settings.serverUrl}/enrich');
        final prevCount = allResults.length;

        // Resolve per-chunk term_language: if every word in the chunk
        // has the same explicit language override, use it; otherwise
        // fall back to the global termLanguage.
        final chunkLangs = chunk
            .map((w) => wordLanguages[w.toLowerCase()])
            .whereType<String>()
            .toSet();
        final effectiveLang = chunkLangs.length == 1
            ? chunkLangs.first
            : termLanguage;

        try {
          final response = await _enrichClient!
              .post(
                uri,
                headers: {'Content-Type': 'application/json'},
                body: jsonEncode({
                  'words': chunk,
                  'definition_language': definitionLanguage,
                  'examples_language': examplesLanguage,
                  'term_language': effectiveLang,
                  'max_tokens': maxTokens,
                  'temperature': temperature,
                }),
              )
              .timeout(chunkTimeout);

        if (response.statusCode != 200) {
          throw Exception(
            'Enrich chunk ${ci + 1}/${chunks.length} failed '
            '(${response.statusCode}): ${response.body}',
          );
        }

        final body = jsonDecode(response.body) as Map<String, dynamic>;
        final results = body['results'] as List<dynamic>? ?? [];

        allResults.addAll(results.map((r) {
          final m = r as Map<String, dynamic>;
          return EnrichWordResult(
            word: m['word'] as String? ?? '',
            definition: (m['definition'] as String? ?? '').replaceAll('*', ''),
            examples: (m['examples'] as String? ?? '').replaceAll('*', ''),
            warning: m['warning'] as String? ?? '',
            correctedWord: m['corrected_word'] as String? ?? '',
          );
        }));
      } catch (e) {
        // On timeout / error, fill this chunk with not_found entries
        // so processing continues with the remaining chunks.
        for (final w in chunk) {
          allResults.add(EnrichWordResult(
            word: w,
            definition: '',
            examples: '',
            warning: 'not_found',
          ));
        }
      }

      onChunkDone?.call(
        allResults.length.clamp(0, words.length),
        words.length,
        allResults.sublist(prevCount),
      );
    }

    } finally {
      _enrichClient = null;
    }

    return allResults;
  }

  // ---------------------------------------------------------------------------
  // Android-native enrichment (direct llama-server)
  // ---------------------------------------------------------------------------

  Future<List<EnrichWordResult>> _androidEnrichChunked({
    required List<String> words,
    required String definitionLanguage,
    required String examplesLanguage,
    required String termLanguage,
    required Map<String, String> wordLanguages,
    required int maxTokens,
    required double temperature,
    required int chunkSize,
    void Function(int completedWords, int totalWords,
        List<EnrichWordResult> chunkResults)? onChunkDone,
  }) async {
    final allResults = <EnrichWordResult>[];
    final chunks = <List<String>>[];

    for (var i = 0; i < words.length; i += chunkSize) {
      chunks.add(words.sublist(i, (i + chunkSize).clamp(0, words.length)));
    }

    // Ensure the server is healthy once before starting.  If it can't start
    // (missing binary, model, OOM, etc.) we fail fast so the user sees the
    // real error instead of every word silently marked not_found.
    await _androidService!.ensureServerRunning();

    for (var ci = 0; ci < chunks.length; ci++) {
      final chunk = chunks[ci];
      final prevCount = allResults.length;

      // Resolve per-chunk term_language.
      final chunkLangs = chunk
          .map((w) => wordLanguages[w.toLowerCase()])
          .whereType<String>()
          .toSet();
      final effectiveLang =
          chunkLangs.length == 1 ? chunkLangs.first : termLanguage;

      try {
        final prompt = _buildEnrichPrompt(
          chunk,
          definitionLanguage,
          examplesLanguage,
          effectiveLang,
        );
        final response = await _androidService.generate(
          prompt: prompt,
          maxTokens: maxTokens,
          temperature: temperature,
        );
        final parsed = _parseEnrichResponse(response, chunk);
        // If the model returned content but parsing produced no results,
        // the response format was unexpected. Log it so we can diagnose.
        if (parsed.isEmpty && response.trim().isNotEmpty) {
          debugPrint('[enrich] parsing produced 0 results for chunk '
              '${ci + 1}/${chunks.length}. Raw response:\n$response');
        }
        allResults.addAll(parsed);
      } catch (e) {
        // On Android the server is local — if one chunk fails (timeout,
        // HTTP error, etc.) the rest will too.  Fail fast so the user
        // sees the real error instead of every word silently marked
        // not_found.
        debugPrint('[enrich] chunk ${ci + 1}/${chunks.length} failed: $e');
        rethrow;
      }

      onChunkDone?.call(
        allResults.length.clamp(0, words.length),
        words.length,
        allResults.sublist(prevCount),
      );
    }

    return allResults;
  }

  String _buildEnrichPrompt(
    List<String> words,
    String defLang,
    String exLang,
    String termLang,
  ) {
    final wordList = words.map((w) => '- $w').join('\n');
    final langHint = termLang.toLowerCase() != 'auto'
        ? 'The word is in $termLang. '
        : '';
    return (
        '$langHint'
        'For each word, give a definition in $defLang and 2 example sentences in $exLang.\n'
        'WORD: must contain the EXACT original word, unchanged. Do NOT translate, correct, or modify it.\n'
        'No asterisks, no bold, no markdown. Plain text only.\n\n'
        'If unrecognizable, write DEF: UNKNOWN and skip examples.\n\n'
        'Format (labels must be in English):\n'
        'WORD: <original word, unchanged>\n'
        'DEF: <$defLang definition>\n'
        'EX1: <$exLang sentence>\n'
        'EX2: <$exLang sentence>\n\n'
        'Example:\n'
        'WORD: serendipity\n'
        'DEF: The occurrence of events by chance in a happy or beneficial way.\n'
        'EX1: Meeting my best friend was a serendipity.\n'
        'EX2: The discovery of penicillin was pure serendipity.\n\n'
        'Words:\n$wordList\n');
  }

  List<EnrichWordResult> _parseEnrichResponse(String text, List<String> words) {
    final results = <EnrichWordResult>[];
    final lines = text.split('\n');
    String? currentWord;
    String? currentDef;
    String? currentEx1;
    String? currentEx2;
    // Track which expected word we're on when WORD: is missing.
    var fallbackWordIndex = 0;

    void flush() {
      if (currentWord != null) {
        final examples = [currentEx1, currentEx2]
            .whereType<String>()
            .join('\n');
        results.add(EnrichWordResult(
          word: currentWord!,
          definition: currentDef ?? '',
          examples: examples,
          warning: (currentDef == null || currentDef!.isEmpty) ? 'not_found' : '',
        ));
      }
      currentWord = null;
      currentDef = null;
      currentEx1 = null;
      currentEx2 = null;
    }

    for (final rawLine in lines) {
      final line = rawLine.trim();
      if (line.isEmpty) continue;

      final upper = line.toUpperCase();
      if (upper.startsWith('WORD:')) {
        flush();
        currentWord = line.substring(5).trim();
      } else if (upper.startsWith('DEF:')) {
        // If no WORD: was seen before DEF:, try to infer the word.
        if (currentWord == null) {
          flush();
          // Try matching the definition text against expected words.
          final defText = line.substring(4).trim().toLowerCase();
          for (var i = fallbackWordIndex; i < words.length; i++) {
            final w = words[i].toLowerCase();
            if (defText.contains(w)) {
              currentWord = words[i];
              fallbackWordIndex = i + 1;
              break;
            }
          }
          // Fallback: use the next expected word if no match.
          if (currentWord == null && fallbackWordIndex < words.length) {
            currentWord = words[fallbackWordIndex];
            fallbackWordIndex++;
          }
        }
        currentDef = line.substring(4).trim();
      } else if (upper.startsWith('EX1:')) {
        currentEx1 = line.substring(4).trim();
      } else if (upper.startsWith('EX2:')) {
        currentEx2 = line.substring(4).trim();
      }
    }
    flush();

    // If parsing produced fewer results than words, fill gaps with not_found.
    if (results.length < words.length) {
      final parsedWords = results.map((r) => r.word.toLowerCase()).toSet();
      for (final w in words) {
        if (!parsedWords.contains(w.toLowerCase())) {
          results.add(EnrichWordResult(
            word: w,
            definition: '',
            examples: '',
            warning: 'not_found',
          ));
        }
      }
    }

    return results;
  }
}
