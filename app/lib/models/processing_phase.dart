/// Phase of the image-processing pipeline.
enum ProcessingPhase {
  idle,
  cropping,
  ocr,
  wordReview,
  enriching,
  done,
  error,
}
