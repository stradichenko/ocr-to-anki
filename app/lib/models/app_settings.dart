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
    this.termLanguage = 'auto',
    this.ocrLanguage = 'eng',
    this.enabledColors = const {HighlightColor.orange},
    this.adaptiveMode = false,
    this.colorTolerance = 25,
    this.minArea = 200,
    this.padding = 0,
    this.mergeNearby = true,
    this.mergeDistance = 10,
    this.serverUrl = 'http://127.0.0.1:8000',
    this.temperature = 0.1,
    this.maxTokens = 512,
    this.contextSize = 4096,
    this.montageMaxWidth = 768,
    this.parallelCrops = true,
    this.preferDiscreteGpu = true,
    this.colorSchemeSeed = 'deepOrange',
    this.customColorHex = '',
  });

  // -- Appearance --
  ThemeMode themeMode;
  String colorSchemeSeed;
  String customColorHex;

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
  String termLanguage;
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

  /// URL of the FastAPI backend.
  String serverUrl;

  double temperature;
  int maxTokens;
  int contextSize;

  // -- OCR Performance --

  /// Max width (px) for each crop before stitching into the montage.
  /// Lower values speed up vision encoding.  0 = no downscale.
  int montageMaxWidth;

  /// When true, process crops as parallel individual OCR calls instead of
  /// a single montage.  Faster when a discrete GPU is available.
  bool parallelCrops;

  /// Ask the backend to prefer a discrete GPU over an integrated one.
  bool preferDiscreteGpu;

  Map<String, dynamic> toJson() => {
        'themeMode': themeMode.name,
        'colorSchemeSeed': colorSchemeSeed,
        'customColorHex': customColorHex,
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
        'termLanguage': termLanguage,
        'ocrLanguage': ocrLanguage,
        'enabledColors':
            enabledColors.map((c) => c.name).toList(),
        'adaptiveMode': adaptiveMode,
        'colorTolerance': colorTolerance,
        'minArea': minArea,
        'padding': padding,
        'mergeNearby': mergeNearby,
        'mergeDistance': mergeDistance,
        'serverUrl': serverUrl,
        'temperature': temperature,
        'maxTokens': maxTokens,
        'contextSize': contextSize,
        'montageMaxWidth': montageMaxWidth,
        'parallelCrops': parallelCrops,
        'preferDiscreteGpu': preferDiscreteGpu,
      };

  factory AppSettings.fromJson(Map<String, dynamic> json) {
    return AppSettings(
      themeMode: ThemeMode.values.byName(
          json['themeMode'] as String? ?? 'dark'),
      colorSchemeSeed:
          json['colorSchemeSeed'] as String? ?? 'deepOrange',
      customColorHex:
          json['customColorHex'] as String? ?? '',
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
      termLanguage:
          json['termLanguage'] as String? ?? 'auto',
      ocrLanguage: json['ocrLanguage'] as String? ?? 'eng',
      enabledColors: (json['enabledColors'] as List<dynamic>?)
              ?.map((e) => HighlightColor.values.byName(e as String))
              .toSet() ??
          {HighlightColor.orange},
      adaptiveMode: json['adaptiveMode'] as bool? ?? false,
      colorTolerance: json['colorTolerance'] as int? ?? 25,
      minArea: json['minArea'] as int? ?? 200,
      padding: json['padding'] as int? ?? 0,
      mergeNearby: json['mergeNearby'] as bool? ?? true,
      mergeDistance: json['mergeDistance'] as int? ?? 10,
      serverUrl:
          json['serverUrl'] as String? ?? 'http://127.0.0.1:8000',
      temperature: (json['temperature'] as num?)?.toDouble() ?? 0.1,
      maxTokens: json['maxTokens'] as int? ?? 512,
      contextSize: json['contextSize'] as int? ?? 4096,
      montageMaxWidth: json['montageMaxWidth'] as int? ?? 768,
      parallelCrops: json['parallelCrops'] as bool? ?? true,
      preferDiscreteGpu: json['preferDiscreteGpu'] as bool? ?? true,
    );
  }
}

/// Languages the user can select for definitions / examples.
const kSupportedLanguages = [
  'english',
  'spanish',
  'french',
  'german',
  'portuguese',
  'italian',
  'japanese',
  'chinese',
  'korean',
  'russian',
  'arabic',
  'dutch',
  'swedish',
  'polish',
  'turkish',
  'hindi',
];
