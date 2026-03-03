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
  });

  final String word;
  final String definition;
  final String examples;
}

/// Unified inference service that can run in two modes:
///
///  * **remote** — forward requests to the existing Python FastAPI backend
///  * **embedded** — (future) on-device inference via llama.cpp FFI
///
/// The remote mode is fully functional and talks to the API defined in
/// `src/api/app.py`.
class InferenceService {
  InferenceService({required AppSettings settings}) : _settings = settings;

  AppSettings _settings;

  void updateSettings(AppSettings settings) => _settings = settings;

  // ---------------------------------------------------------------------------
  // Health
  // ---------------------------------------------------------------------------

  /// Check whether the backend is reachable (remote) or model is loaded
  /// (embedded).
  Future<bool> isAvailable() async {
    switch (_settings.inferenceMode) {
      case InferenceMode.embedded:
        // TODO: implement when llamadart FFI is integrated
        return false;
      case InferenceMode.remote:
        return _remoteHealthCheck();
    }
  }

  Future<bool> _remoteHealthCheck() async {
    try {
      final uri = Uri.parse('${_settings.serverUrl}/health');
      final response = await http.get(uri).timeout(
            Duration(seconds: _settings.ankiConnectTimeout),
          );
      if (response.statusCode == 200) {
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        return body['status'] == 'ok' || body['status'] == 'degraded';
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Vision OCR
  // ---------------------------------------------------------------------------

  /// Extract text from an image.
  ///
  /// [imageBytes] — raw JPEG / PNG bytes.
  /// [prompt] — vision prompt for the model.
  Future<VisionOcrResult> visionOcr({
    required Uint8List imageBytes,
    String prompt =
        'Extract all visible text from this image. List each word or phrase you can read.',
    int timeoutSeconds = 600,
  }) async {
    switch (_settings.inferenceMode) {
      case InferenceMode.embedded:
        throw UnimplementedError(
          'Embedded inference not yet available — use remote mode.',
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

    final response = await http
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
          'Embedded inference not yet available — use remote mode.',
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
  Future<List<EnrichWordResult>> enrichWords({
    required List<String> words,
    String? definitionLanguage,
    String? examplesLanguage,
    int? maxTokens,
    double? temperature,
  }) async {
    switch (_settings.inferenceMode) {
      case InferenceMode.embedded:
        throw UnimplementedError(
          'Embedded inference not yet available — use remote mode.',
        );
      case InferenceMode.remote:
        return _remoteEnrich(
          words: words,
          definitionLanguage:
              definitionLanguage ?? _settings.definitionLanguage,
          examplesLanguage: examplesLanguage ?? _settings.examplesLanguage,
          maxTokens: maxTokens ?? 256,
          temperature: temperature ?? _settings.temperature,
        );
    }
  }

  Future<List<EnrichWordResult>> _remoteEnrich({
    required List<String> words,
    required String definitionLanguage,
    required String examplesLanguage,
    required int maxTokens,
    required double temperature,
  }) async {
    final uri = Uri.parse('${_settings.serverUrl}/enrich');

    final response = await http
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'words': words,
            'definition_language': definitionLanguage,
            'examples_language': examplesLanguage,
            'max_tokens': maxTokens,
            'temperature': temperature,
          }),
        )
        .timeout(Duration(seconds: words.length * 60 + 30));

    if (response.statusCode != 200) {
      throw Exception(
        'Enrich failed (${response.statusCode}): ${response.body}',
      );
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    final results = body['results'] as List<dynamic>? ?? [];

    return results.map((r) {
      final m = r as Map<String, dynamic>;
      return EnrichWordResult(
        word: m['word'] as String? ?? '',
        definition: m['definition'] as String? ?? '',
        examples: m['examples'] as String? ?? '',
      );
    }).toList();
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
