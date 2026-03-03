import 'package:flutter/material.dart';

import 'highlight_color.dart';

/// Application-wide settings, persisted in the local database.
class AppSettings {
  AppSettings({
    this.themeMode = ThemeMode.dark,
    this.ankiConnectUrl = 'http://localhost:8765',
    this.ankiConnectVersion = 6,
    this.ankiConnectTimeout = 10,
    this.defaultDeck = 'Vocabulary::Japanese',
    this.defaultModel = 'Basic',
    this.batchSize = 10,
    this.allowDuplicates = false,
    this.duplicateScope = 'deck',
    this.definitionLanguage = 'english',
    this.examplesLanguage = 'english',
    this.ocrLanguage = 'eng',
    this.enabledColors = const {HighlightColor.orange},
    this.adaptiveMode = false,
    this.colorTolerance = 25,
    this.minArea = 200,
    this.padding = 10,
    this.mergeNearby = true,
    this.mergeDistance = 25,
    this.inferenceMode = InferenceMode.remote,
    this.serverUrl = 'http://127.0.0.1:8000',
    this.modelPath = '',
    this.temperature = 0.1,
    this.maxTokens = 512,
    this.contextSize = 4096,
  });

  // -- Appearance --
  ThemeMode themeMode;

  // -- AnkiConnect settings --
  String ankiConnectUrl;
  int ankiConnectVersion;
  int ankiConnectTimeout;

  // -- Import defaults --
  String defaultDeck;
  String defaultModel;
  int batchSize;
  bool allowDuplicates;
  String duplicateScope;

  // -- Language preferences --
  String definitionLanguage;
  String examplesLanguage;
  String ocrLanguage;

  // -- Highlight detection --
  Set<HighlightColor> enabledColors;
  bool adaptiveMode;
  int colorTolerance;
  int minArea;
  int padding;
  bool mergeNearby;
  int mergeDistance;

  // -- Inference --
  InferenceMode inferenceMode;

  /// URL of the FastAPI backend (when [inferenceMode] is [InferenceMode.remote]).
  String serverUrl;

  /// Local path to the GGUF model (when [inferenceMode] is [InferenceMode.embedded]).
  String modelPath;

  double temperature;
  int maxTokens;
  int contextSize;

  Map<String, dynamic> toJson() => {
        'themeMode': themeMode.name,
        'ankiConnectUrl': ankiConnectUrl,
        'ankiConnectVersion': ankiConnectVersion,
        'ankiConnectTimeout': ankiConnectTimeout,
        'defaultDeck': defaultDeck,
        'defaultModel': defaultModel,
        'batchSize': batchSize,
        'allowDuplicates': allowDuplicates,
        'duplicateScope': duplicateScope,
        'definitionLanguage': definitionLanguage,
        'examplesLanguage': examplesLanguage,
        'ocrLanguage': ocrLanguage,
        'enabledColors':
            enabledColors.map((c) => c.name).toList(),
        'adaptiveMode': adaptiveMode,
        'colorTolerance': colorTolerance,
        'minArea': minArea,
        'padding': padding,
        'mergeNearby': mergeNearby,
        'mergeDistance': mergeDistance,
        'inferenceMode': inferenceMode.name,
        'serverUrl': serverUrl,
        'modelPath': modelPath,
        'temperature': temperature,
        'maxTokens': maxTokens,
        'contextSize': contextSize,
      };

  factory AppSettings.fromJson(Map<String, dynamic> json) {
    return AppSettings(
      themeMode: ThemeMode.values.byName(
          json['themeMode'] as String? ?? 'dark'),
      ankiConnectUrl:
          json['ankiConnectUrl'] as String? ?? 'http://localhost:8765',
      ankiConnectVersion: json['ankiConnectVersion'] as int? ?? 6,
      ankiConnectTimeout: json['ankiConnectTimeout'] as int? ?? 10,
      defaultDeck:
          json['defaultDeck'] as String? ?? 'Vocabulary::Japanese',
      defaultModel: json['defaultModel'] as String? ?? 'Basic',
      batchSize: json['batchSize'] as int? ?? 10,
      allowDuplicates: json['allowDuplicates'] as bool? ?? false,
      duplicateScope: json['duplicateScope'] as String? ?? 'deck',
      definitionLanguage:
          json['definitionLanguage'] as String? ?? 'english',
      examplesLanguage:
          json['examplesLanguage'] as String? ?? 'english',
      ocrLanguage: json['ocrLanguage'] as String? ?? 'eng',
      enabledColors: (json['enabledColors'] as List<dynamic>?)
              ?.map((e) => HighlightColor.values.byName(e as String))
              .toSet() ??
          {HighlightColor.orange},
      adaptiveMode: json['adaptiveMode'] as bool? ?? false,
      colorTolerance: json['colorTolerance'] as int? ?? 25,
      minArea: json['minArea'] as int? ?? 200,
      padding: json['padding'] as int? ?? 10,
      mergeNearby: json['mergeNearby'] as bool? ?? true,
      mergeDistance: json['mergeDistance'] as int? ?? 25,
      inferenceMode: InferenceMode.values.byName(
          json['inferenceMode'] as String? ?? 'remote'),
      serverUrl:
          json['serverUrl'] as String? ?? 'http://127.0.0.1:8000',
      modelPath: json['modelPath'] as String? ?? '',
      temperature: (json['temperature'] as num?)?.toDouble() ?? 0.1,
      maxTokens: json['maxTokens'] as int? ?? 512,
      contextSize: json['contextSize'] as int? ?? 4096,
    );
  }
}

/// Whether the app runs the LLM locally or connects to the Python FastAPI server.
enum InferenceMode {
  /// On-device inference via llama.cpp FFI (llamadart).
  embedded,

  /// Forward requests to the existing Python FastAPI backend.
  remote,
}
