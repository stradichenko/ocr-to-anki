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
    this.imageHash = '',
  });

  final String text;
  final double elapsedSeconds;
  final String backend;
  final String imageHash;
}

/// Result of a text-generation / enrichment call.
class EnrichWordResult {
  const EnrichWordResult({
    required this.word,
    required this.definition,
    required this.examples,
    this.warning = '',
    this.correctedWord = '',
  });

  final String word;
  final String definition;
  final String examples;

  /// Quality warning: 'not_found', 'truncated', or '' (ok).
  final String warning;

  /// LLM-suggested correct spelling when OCR produced a misspelling.
  /// Empty if the word is already correct.
  final String correctedWord;
}

/// Unified inference service that can run in two modes:
///
///  * **remote** -- forward requests to the existing Python FastAPI backend
///  * **embedded** -- (future) on-device inference via llama.cpp FFI
///
/// The remote mode is fully functional and talks to the API defined in
/// `src/api/app.py`.
class InferenceService {
  InferenceService({required AppSettings settings}) : _settings = settings;

  AppSettings _settings;

  /// HTTP client for the current OCR request; can be closed to abort.
  http.Client? _ocrClient;

  void updateSettings(AppSettings settings) => _settings = settings;

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

  /// Check whether the backend is reachable (remote) or model is loaded
  /// (embedded).
  Future<bool> isAvailable() async {
    switch (_settings.inferenceMode) {
      case InferenceMode.embedded:
        debugMessage = 'Embedded mode not yet implemented';
        return false;
      case InferenceMode.remote:
        return _remoteHealthCheck();
    }
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
    switch (_settings.inferenceMode) {
      case InferenceMode.embedded:
        throw UnimplementedError(
          'Embedded inference not yet available -- use remote mode.',
        );
      case InferenceMode.remote:
        return _remoteVisionOcr(
          imageBytes: imageBytes,
          prompt: prompt,
          timeoutSeconds: timeoutSeconds,
        );
    }
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
        imageHash: body['image_hash'] as String? ?? '',
      );
    } finally {
      _ocrClient?.close();
      _ocrClient = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Text Generation
  // ---------------------------------------------------------------------------

  /// Raw text generation.
  Future<String> generate({
    required String prompt,
    String? system,
    int? maxTokens,
    double? temperature,
  }) async {
    switch (_settings.inferenceMode) {
      case InferenceMode.embedded:
        throw UnimplementedError(
          'Embedded inference not yet available -- use remote mode.',
        );
      case InferenceMode.remote:
        return _remoteGenerate(
          prompt: prompt,
          system: system,
          maxTokens: maxTokens ?? _settings.maxTokens,
          temperature: temperature ?? _settings.temperature,
        );
    }
  }

  Future<String> _remoteGenerate({
    required String prompt,
    String? system,
    required int maxTokens,
    required double temperature,
  }) async {
    final uri = Uri.parse('${_settings.serverUrl}/generate');

    final body = <String, dynamic>{
      'prompt': prompt,
      'max_tokens': maxTokens,
      'temperature': temperature,
    };
    if (system != null) body['system'] = system;

    final response = await http
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(body),
        )
        .timeout(const Duration(seconds: 120));

    if (response.statusCode != 200) {
      throw Exception(
        'Generate failed (${response.statusCode}): ${response.body}',
      );
    }

    final respBody = jsonDecode(response.body) as Map<String, dynamic>;
    return respBody['content'] as String? ?? '';
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
    int? maxTokens,
    double? temperature,
    int chunkSize = 6,
    Duration chunkTimeout = const Duration(minutes: 5),
    void Function(int completedWords, int totalWords,
        List<EnrichWordResult> chunkResults)? onChunkDone,
  }) async {
    switch (_settings.inferenceMode) {
      case InferenceMode.embedded:
        throw UnimplementedError(
          'Embedded inference not yet available -- use remote mode.',
        );
      case InferenceMode.remote:
        return _remoteEnrichChunked(
          words: words,
          definitionLanguage:
              definitionLanguage ?? _settings.definitionLanguage,
          examplesLanguage: examplesLanguage ?? _settings.examplesLanguage,
          maxTokens: maxTokens ?? 256,
          temperature: temperature ?? _settings.temperature,
          chunkSize: chunkSize,
          chunkTimeout: chunkTimeout,
          onChunkDone: onChunkDone,
        );
    }
  }

  /// Send words in small chunks, each with its own timeout.
  Future<List<EnrichWordResult>> _remoteEnrichChunked({
    required List<String> words,
    required String definitionLanguage,
    required String examplesLanguage,
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
            definition: m['definition'] as String? ?? '',
            examples: m['examples'] as String? ?? '',
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

  // ---------------------------------------------------------------------------
  // Full pipeline: image → OCR → enrich → cards
  // ---------------------------------------------------------------------------

  /// Runs the full pipeline via the remote FastAPI `/pipeline/image-to-cards`
  /// endpoint.
  Future<PipelineResult> runPipeline({
    required Uint8List imageBytes,
    required String filename,
    String? definitionLanguage,
    String? examplesLanguage,
    String? ocrPrompt,
  }) async {
    if (_settings.inferenceMode != InferenceMode.remote) {
      throw UnimplementedError(
        'Pipeline is only available in remote mode for now.',
      );
    }

    final uri = Uri.parse('${_settings.serverUrl}/pipeline/image-to-cards');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes(
        'file',
        imageBytes,
        filename: filename,
      ))
      ..fields['definition_language'] =
          definitionLanguage ?? _settings.definitionLanguage
      ..fields['examples_language'] =
          examplesLanguage ?? _settings.examplesLanguage;

    if (ocrPrompt != null) {
      request.fields['ocr_prompt'] = ocrPrompt;
    }

    final streamed = await request.send().timeout(
          const Duration(minutes: 15),
        );

    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw Exception(
        'Pipeline failed (${response.statusCode}): ${response.body}',
      );
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    final cards = (body['cards'] as List<dynamic>? ?? []).map((c) {
      final m = c as Map<String, dynamic>;
      return EnrichWordResult(
        word: m['word'] as String? ?? '',
        definition: m['definition'] as String? ?? '',
        examples: m['examples'] as String? ?? '',
      );
    }).toList();

    return PipelineResult(
      ocrText: body['ocr_text'] as String? ?? '',
      ocrBackend: body['ocr_backend'] as String? ?? '',
      ocrElapsedSeconds: (body['ocr_elapsed_s'] as num?)?.toDouble() ?? 0,
      cards: cards,
      totalElapsedSeconds:
          (body['total_elapsed_s'] as num?)?.toDouble() ?? 0,
    );
  }
}

/// Full pipeline result from the FastAPI backend.
class PipelineResult {
  const PipelineResult({
    required this.ocrText,
    required this.ocrBackend,
    required this.ocrElapsedSeconds,
    required this.cards,
    required this.totalElapsedSeconds,
  });

  final String ocrText;
  final String ocrBackend;
  final double ocrElapsedSeconds;
  final List<EnrichWordResult> cards;
  final double totalElapsedSeconds;
}
