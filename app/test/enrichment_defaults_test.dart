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

  group('kAndroidEnrichChunkSize', () {
    test('stays within the desktop chunk', () {
      expect(kAndroidEnrichChunkSize, lessThanOrEqualTo(kEnrichChunkSize));
    });

    test('is small enough for UI progress on a slow device', () {
      // Two words at ~1.9 tok/s foreground CPU is roughly 90 seconds per
      // request; six would hold the screen silent for 4+ minutes.
      expect(kAndroidEnrichChunkSize, lessThanOrEqualTo(2));
    });
  });

  group('androidChunkBudget', () {
    test('keeps the old flat cap as the single-word floor', () {
      expect(androidChunkBudget(1), 128);
    });

    test('scales with the words in the chunk', () {
      expect(androidChunkBudget(2), greaterThan(androidChunkBudget(1)));
      expect(androidChunkBudget(6), greaterThan(androidChunkBudget(2)));
    });

    test('leaves room for the fixed prompt overhead', () {
      // 64 tokens per word plus 32 of slack — the regression this guards is
      // the flat 128 cap that truncated a 6-word chunk halfway through its
      // second entry.
      expect(androidChunkBudget(6), 32 + 64 * 6);
    });

    test('never exceeds a sane upper bound', () {
      expect(androidChunkBudget(100), lessThanOrEqualTo(512));
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