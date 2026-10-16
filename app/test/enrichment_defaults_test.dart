import 'package:flutter_test/flutter_test.dart';
import 'package:ocr_to_anki/models/enrichment_defaults.dart';

void main() {
  group('kEnrichChunkSize', () {
    test('stays within the backend batch cap', () {
      // Mirrors MAX_ENRICH_BATCH_WORDS in src/api/enrichment.py. If the client
      // asks for more words than the server batches, the server splits the
      // request and the UI loses its per-batch progress.
      expect(kEnrichChunkSize, lessThanOrEqualTo(8));
    });

    test('sends more than one word per request', () {
      // The regression this guards: every call site used to pass 1, so a
      // twenty-word page cost twenty round trips.
      expect(kEnrichChunkSize, greaterThan(1));
    });
  });

  group('enrichChunkTimeout', () {
    test('keeps the floor for a single word', () {
      expect(
        enrichChunkTimeout(1),
        greaterThanOrEqualTo(const Duration(minutes: 10)),
      );
    });

    test('grows with the number of words', () {
      expect(
        enrichChunkTimeout(kEnrichChunkSize),
        greaterThan(enrichChunkTimeout(1)),
      );
    });

    test('allows a full chunk a longer window than the old fixed value', () {
      expect(
        enrichChunkTimeout(kEnrichChunkSize),
        greaterThan(const Duration(minutes: 10)),
      );
    });
  });
}