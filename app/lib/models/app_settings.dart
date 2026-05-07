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
    this.gpuMode = 'auto',
    this.nGpuLayers = 999,
    this.colorSchemeSeed = 'deepOrange',
    this.customColorHex = '',
    this.autoCheckUpdates = true,
    this.skipVersion = '',
    this.wifiOnlyDownloads = true,
    this.compressLargeImages = true,
    this.notificationsPermissionAsked = false,
    this.batteryOptimizationPromptShown = false,
    this.lastAnkiDroidDeckId = 0,
    this.activeModelId = 'gemma-3-4b-q4',
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

  /// GPU acceleration mode: 'auto' (platform default), 'gpu' (force all
  /// layers on GPU), or 'cpu' (force CPU-only).  On Windows 'auto'
  /// means CPU because the Vulkan drivers are often unstable.
  /// On Android: 'auto', 'vulkan', 'opencl', or 'cpu'.
  String gpuMode;

  /// Number of model layers to offload to GPU. 999 means all layers.
  int nGpuLayers;

  // -- Updates --

  /// Whether to automatically check for updates on startup.
  bool autoCheckUpdates;

  /// Version string the user chose to skip (e.g. "0.2.0"). Empty = none.
  String skipVersion;

  /// On Android, only download models when connected to WiFi.
  bool wifiOnlyDownloads;

  /// Compress images larger than 1 MB before vision OCR on Android.
  bool compressLargeImages;

  /// True once the app has shown the runtime POST_NOTIFICATIONS prompt
  /// (Android 13+).  Stays true regardless of the user's grant choice so
  /// we don't re-prompt on every launch.
  bool notificationsPermissionAsked;

  /// True once the app has shown the one-time battery-optimisation
  /// whitelist prompt.  Stays true regardless of the user's choice.
  bool batteryOptimizationPromptShown;

  /// Last-selected AnkiDroid deck ID for one-tap export.
  int lastAnkiDroidDeckId;

  /// Active model ID from the registry. Defaults to the bundled model.
  String activeModelId;

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
        'gpuMode': gpuMode,
        'nGpuLayers': nGpuLayers,
        'autoCheckUpdates': autoCheckUpdates,
        'skipVersion': skipVersion,
        'wifiOnlyDownloads': wifiOnlyDownloads,
        'compressLargeImages': compressLargeImages,
        'notificationsPermissionAsked': notificationsPermissionAsked,
        'batteryOptimizationPromptShown': batteryOptimizationPromptShown,
        'lastAnkiDroidDeckId': lastAnkiDroidDeckId,
        'activeModelId': activeModelId,
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
      gpuMode: json['gpuMode'] as String? ?? 'auto',
      nGpuLayers: json['nGpuLayers'] as int? ?? 999,
      autoCheckUpdates: json['autoCheckUpdates'] as bool? ?? true,
      skipVersion: json['skipVersion'] as String? ?? '',
      wifiOnlyDownloads: json['wifiOnlyDownloads'] as bool? ?? true,
      compressLargeImages: json['compressLargeImages'] as bool? ?? true,
      notificationsPermissionAsked:
          json['notificationsPermissionAsked'] as bool? ?? false,
      batteryOptimizationPromptShown:
          json['batteryOptimizationPromptShown'] as bool? ?? false,
      lastAnkiDroidDeckId:
          json['lastAnkiDroidDeckId'] as int? ?? 0,
      activeModelId:
          json['activeModelId'] as String? ?? 'gemma-3-4b-q4',
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
