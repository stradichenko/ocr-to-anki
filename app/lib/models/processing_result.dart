import 'highlight_color.dart';

/// The two main OCR contexts the app supports.
enum OcrContext {
  /// Handwritten or printed words in an image (no highlight detection).
  handwrittenOrPrinted,

  /// Highlighted words — user picks the marker colour, app crops regions first.
  highlighted,
}

/// Result of processing a single image through the pipeline.
class ProcessingResult {
  ProcessingResult({
    required this.imagePath,
    required this.context,
    this.highlightColor,
    this.ocrText = '',
    this.words = const [],
    this.ocrElapsedSeconds = 0,
    this.enrichElapsedSeconds = 0,
    this.backend = '',
    this.croppedImagePaths = const [],
    this.error,
  });

  /// Path to the original image.
  final String imagePath;

  /// Which OCR context was used.
  final OcrContext context;

  /// If [context] is [OcrContext.highlighted], the colour that was selected.
  final HighlightColor? highlightColor;

  /// Raw text returned by the vision model.
  String ocrText;

  /// Individual words parsed from [ocrText].
  List<String> words;

  /// Time taken for vision OCR in seconds.
  double ocrElapsedSeconds;

  /// Time taken for enrichment in seconds.
  double enrichElapsedSeconds;

  /// Backend that was used (e.g. "opencl", "vulkan", "cpu").
  String backend;

  /// Paths to cropped highlight sub-images (for highlighted context).
  List<String> croppedImagePaths;

  /// Non-null when something went wrong.
  String? error;

  bool get hasError => error != null;

  double get totalElapsedSeconds => ocrElapsedSeconds + enrichElapsedSeconds;
}
