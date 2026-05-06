import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../services/highlight_detector.dart';
import '../services/inference_service.dart';
import 'highlight_color.dart';
import 'processing_phase.dart';

/// Transient processing state persisted to the settings table so that
/// an in-flight batch can survive app restarts.
class PendingBatch {
  PendingBatch({
    required this.phase,
    required this.imageEntries,
    this.ocrResults = const [],
    this.words = const [],
    this.wordLanguages = const {},
    this.enrichedWords = const [],
    this.ocrText = '',
    this.progress = 0,
    this.statusMessage = '',
    this.activityLog = const [],
  });

  final ProcessingPhase phase;
  final List<PendingImageEntry> imageEntries;
  final List<String> ocrResults;
  final List<String> words;
  final Map<String, String> wordLanguages;
  final List<EnrichWordResult> enrichedWords;
  final String ocrText;
  final double progress;
  final String statusMessage;
  final List<String> activityLog;

  bool get isTerminal =>
      phase == ProcessingPhase.done || phase == ProcessingPhase.error;

  /// Whether this batch has image data (vs words-only).
  bool get hasImages => imageEntries.isNotEmpty;

  Map<String, dynamic> toJson() => {
        'phase': phase.name,
        'imageEntries': imageEntries.map((e) => e.toJson()).toList(),
        'ocrResults': ocrResults,
        'words': words,
        'wordLanguages': wordLanguages,
        'enrichedWords': enrichedWords.map((e) => e.toJson()).toList(),
        'ocrText': ocrText,
        'progress': progress,
        'statusMessage': statusMessage,
        'activityLog': activityLog,
      };

  factory PendingBatch.fromJson(Map<String, dynamic> json) {
    return PendingBatch(
      phase: ProcessingPhase.values.byName(json['phase'] as String),
      imageEntries: (json['imageEntries'] as List<dynamic>?)
              ?.map((e) =>
                  PendingImageEntry.fromJson(e as Map<String, dynamic>))
              .toList() ??
          const [],
      ocrResults: (json['ocrResults'] as List<dynamic>?)?.cast<String>() ??
          const [],
      words: (json['words'] as List<dynamic>?)?.cast<String>() ??
          const [],
      wordLanguages:
          (json['wordLanguages'] as Map<String, dynamic>?)?.cast<String, String>() ??
              const {},
      enrichedWords: (json['enrichedWords'] as List<dynamic>?)
              ?.map((e) =>
                  EnrichWordResult.fromJson(e as Map<String, dynamic>))
              .toList() ??
          const [],
      ocrText: json['ocrText'] as String? ?? '',
      progress: (json['progress'] as num?)?.toDouble() ?? 0,
      statusMessage: json['statusMessage'] as String? ?? '',
      activityLog: (json['activityLog'] as List<dynamic>?)?.cast<String>() ??
          const [],
    );
  }

  String toJsonString() => jsonEncode(toJson());

  factory PendingBatch.fromJsonString(String s) =>
      PendingBatch.fromJson(jsonDecode(s) as Map<String, dynamic>);
}

/// Serializable metadata for a single image in a pending batch.
///
/// The raw bytes are stored separately on disk; [path] points to the temp file.
class PendingImageEntry {
  const PendingImageEntry({
    required this.path,
    required this.name,
    this.cropRegion,
    this.hsvOverride,
    this.termLanguage,
  });

  /// Path to the temp file containing the image bytes.
  final String path;

  /// Original filename for display.
  final String name;

  /// Optional per-image crop region.
  final HighlightBBox? cropRegion;

  /// Optional per-image HSV colour override.
  final HsvRange? hsvOverride;

  /// Optional per-image term language.
  final String? termLanguage;

  Map<String, dynamic> toJson() => {
        'path': path,
        'name': name,
        if (cropRegion != null) 'cropRegion': cropRegion!.toJson(),
        if (hsvOverride != null) 'hsvOverride': hsvOverride!.toJson(),
        if (termLanguage != null) 'termLanguage': termLanguage,
      };

  factory PendingImageEntry.fromJson(Map<String, dynamic> json) =>
      PendingImageEntry(
        path: json['path'] as String,
        name: json['name'] as String,
        cropRegion: json['cropRegion'] != null
            ? HighlightBBox.fromJson(
                json['cropRegion'] as Map<String, dynamic>)
            : null,
        hsvOverride: json['hsvOverride'] != null
            ? HsvRange.fromJson(
                json['hsvOverride'] as Map<String, dynamic>)
            : null,
        termLanguage: json['termLanguage'] as String?,
      );

  /// Load the image bytes from [path].
  Future<Uint8List> loadBytes() async {
    final file = File(path);
    return file.readAsBytes();
  }
}
