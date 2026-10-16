/// Shared tuning for the vocabulary enrichment step.
///
/// The backend derives its own batch size from the model context, so these
/// values control only how many words the client puts into a single request.
/// Keeping the client chunk at or below the server's batch means the UI gets
/// progress between batches instead of one long silent stretch.
///
/// Every call site previously passed `chunkSize: 1`, which sent one word per
/// HTTP request. Because prompt caching is disabled for vision and text alike,
/// each of those requests re-evaluated the whole system prompt and worked
/// example before producing anything. A twenty-word page cost twenty round
/// trips; it now costs four.
library;

/// Words per enrichment request. Must not exceed the backend's
/// `MAX_ENRICH_BATCH_WORDS`, or the server silently splits the request into
/// several generations the client cannot report progress for.
const int kEnrichChunkSize = 6;

/// Timeout allowance. The integrated GPU is slow at prompt evaluation, so a
/// fixed window was either too tight for a wide batch or wastefully long for a
/// narrow one. Scale it with the work, with a floor matching the old value.
const int _kEnrichTimeoutBaseSeconds = 240;
const int _kEnrichTimeoutPerWordSeconds = 90;
const int _kEnrichTimeoutFloorSeconds = 600;

Duration enrichChunkTimeout(int words) {
  final int seconds =
      _kEnrichTimeoutBaseSeconds + _kEnrichTimeoutPerWordSeconds * words;
  return Duration(
    seconds: seconds < _kEnrichTimeoutFloorSeconds
        ? _kEnrichTimeoutFloorSeconds
        : seconds,
  );
}