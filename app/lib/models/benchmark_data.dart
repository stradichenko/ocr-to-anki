import 'dart:convert';

/// Structured benchmark data collected during a single processing session.
///
/// Persisted as JSON in the `benchmark_json` column of `ProcessingSessions`.
class BenchmarkData {
  BenchmarkData({
    this.totalElapsedS = 0,
    this.cropElapsedS = 0,
    this.ocrElapsedS = 0,
    this.enrichElapsedS = 0,
    this.perCropOcrS = const [],
    this.cropCount = 0,
    this.rawWordCount = 0,
    this.uniqueWordCount = 0,
    this.enrichedWordCount = 0,
    this.warningNotFoundCount = 0,
    this.warningTruncatedCount = 0,
    this.imageSizeBytes = 0,
    this.backend = '',
    this.modelName = '',
    this.definitionLanguage = '',
    this.examplesLanguage = '',
    this.timestamp,
  });

  // -- Timing (seconds) --
  double totalElapsedS;
  double cropElapsedS;
  double ocrElapsedS;
  double enrichElapsedS;

  /// Per-crop OCR durations in seconds, in order.
  List<double> perCropOcrS;

  // -- Counts --
  int cropCount;
  int rawWordCount;
  int uniqueWordCount;
  int enrichedWordCount;
  int warningNotFoundCount;
  int warningTruncatedCount;

  // -- Context --
  int imageSizeBytes;
  String backend;
  String modelName;
  String definitionLanguage;
  String examplesLanguage;
  DateTime? timestamp;

  // -- Derived --

  double get avgOcrPerCropS =>
      perCropOcrS.isEmpty ? 0 : ocrElapsedS / perCropOcrS.length;

  double get avgEnrichPerWordS =>
      enrichedWordCount == 0 ? 0 : enrichElapsedS / enrichedWordCount;

  int get totalWarnings => warningNotFoundCount + warningTruncatedCount;

  String get imageSizeFormatted {
    if (imageSizeBytes < 1024) return '$imageSizeBytes B';
    if (imageSizeBytes < 1024 * 1024) {
      return '${(imageSizeBytes / 1024).toStringAsFixed(0)} KB';
    }
    return '${(imageSizeBytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  Map<String, dynamic> toJson() => {
        'total_elapsed_s': totalElapsedS,
        'crop_elapsed_s': cropElapsedS,
        'ocr_elapsed_s': ocrElapsedS,
        'enrich_elapsed_s': enrichElapsedS,
        'per_crop_ocr_s': perCropOcrS,
        'crop_count': cropCount,
        'raw_word_count': rawWordCount,
        'unique_word_count': uniqueWordCount,
        'enriched_word_count': enrichedWordCount,
        'warning_not_found_count': warningNotFoundCount,
        'warning_truncated_count': warningTruncatedCount,
        'image_size_bytes': imageSizeBytes,
        'backend': backend,
        'model_name': modelName,
        'definition_language': definitionLanguage,
        'examples_language': examplesLanguage,
        'timestamp': timestamp?.toIso8601String(),
      };

  String toJsonString() =>
      const JsonEncoder.withIndent('  ').convert(toJson());

  factory BenchmarkData.fromJson(Map<String, dynamic> json) {
    return BenchmarkData(
      totalElapsedS: (json['total_elapsed_s'] as num?)?.toDouble() ?? 0,
      cropElapsedS: (json['crop_elapsed_s'] as num?)?.toDouble() ?? 0,
      ocrElapsedS: (json['ocr_elapsed_s'] as num?)?.toDouble() ?? 0,
      enrichElapsedS: (json['enrich_elapsed_s'] as num?)?.toDouble() ?? 0,
      perCropOcrS: (json['per_crop_ocr_s'] as List<dynamic>?)
              ?.map((e) => (e as num).toDouble())
              .toList() ??
          [],
      cropCount: json['crop_count'] as int? ?? 0,
      rawWordCount: json['raw_word_count'] as int? ?? 0,
      uniqueWordCount: json['unique_word_count'] as int? ?? 0,
      enrichedWordCount: json['enriched_word_count'] as int? ?? 0,
      warningNotFoundCount: json['warning_not_found_count'] as int? ?? 0,
      warningTruncatedCount: json['warning_truncated_count'] as int? ?? 0,
      imageSizeBytes: json['image_size_bytes'] as int? ?? 0,
      backend: json['backend'] as String? ?? '',
      modelName: json['model_name'] as String? ?? '',
      definitionLanguage: json['definition_language'] as String? ?? '',
      examplesLanguage: json['examples_language'] as String? ?? '',
      timestamp: json['timestamp'] != null
          ? DateTime.tryParse(json['timestamp'] as String)
          : null,
    );
  }

  factory BenchmarkData.fromJsonString(String jsonStr) {
    try {
      return BenchmarkData.fromJson(
        jsonDecode(jsonStr) as Map<String, dynamic>,
      );
    } catch (_) {
      return BenchmarkData();
    }
  }
}
