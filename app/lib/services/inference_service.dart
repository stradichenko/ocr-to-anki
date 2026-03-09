import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../models/app_settings.dart';

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
}

/// Unified inference service that forwards requests to the Python FastAPI
/// backend defined in `src/api/app.py`.
class InferenceService {
  InferenceService({required AppSettings settings}) : _settings = settings;

  final AppSettings _settings;

  /// HTTP client for the current OCR request; can be closed to abort.
  http.Client? _ocrClient;

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

  // ---------------------------------------------------------------------------
  // Health
  // ---------------------------------------------------------------------------

  /// Check whether the backend is reachable.
  Future<bool> isAvailable() async {
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
    return _remoteVisionOcr(
      imageBytes: imageBytes,
      prompt: prompt,
      timeoutSeconds: timeoutSeconds,
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
    int? maxTokens,
    double? temperature,
    int chunkSize = 6,
    Duration chunkTimeout = const Duration(minutes: 5),
    void Function(int completedWords, int totalWords,
        List<EnrichWordResult> chunkResults)? onChunkDone,
  }) async {
    return _remoteEnrichChunked(
      words: words,
      definitionLanguage:
          definitionLanguage ?? _settings.definitionLanguage,
      examplesLanguage: examplesLanguage ?? _settings.examplesLanguage,
      termLanguage: termLanguage ?? _settings.termLanguage,
      maxTokens: maxTokens ?? 256,
      temperature: temperature ?? _settings.temperature,
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

    for (var ci = 0; ci < chunks.length; ci++) {
      final chunk = chunks[ci];
      final uri = Uri.parse('${_settings.serverUrl}/enrich');
      final prevCount = allResults.length;

      try {
        final response = await http
            .post(
              uri,
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({
                'words': chunk,
                'definition_language': definitionLanguage,
                'examples_language': examplesLanguage,
                'term_language': termLanguage,
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

    return allResults;
  }

}
