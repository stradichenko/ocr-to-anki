/// The two main OCR contexts the app supports.
enum OcrContext {
  /// Handwritten or printed words in an image (no highlight detection).
  handwrittenOrPrinted,

  /// Highlighted words -- user picks the marker colour, app crops regions first.
  highlighted,
}

