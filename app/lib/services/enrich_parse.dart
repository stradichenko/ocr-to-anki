import 'inference_service.dart';

/// Parsing for the enrichment responses produced on device.
///
/// Lives apart from InferenceService so it can be tested directly. This is the
/// component that decides which definition belongs to which word, and getting
/// that wrong produces flashcards that look correct and are not. Android parses
/// responses here rather than through the Python backend, so this is a second
/// implementation of the same logic and it had drifted from the first.
///
/// Line-oriented rather than marker-oriented, because the model does not
/// reliably repeat the WORD: label. Asked for several words it often labels
/// only the first:
///
///     WORD: 探す
///     DEF: ...
///     - 静か
///     DEF: ...
///
/// Two earlier attempts to cope with that were both unsafe. Matching the
/// definition text against the expected words let any definition that happened
/// to mention another word claim that word's slot, and a positional cursor that
/// only advanced drifted as soon as the model skipped or merged an entry, so one
/// word was handed another word's definition.
///
/// Entries now start either at a WORD: line or at a bare line that turns out to
/// precede a DEF:, and a definition always belongs to the nearest entry above
/// it. A word the model never mentioned comes back empty for the retry pass.
/// The result is always one entry per requested word, in the requested order.

List<EnrichWordResult> parseEnrichResponse(String text, List<String> words) {
  final entries = <_ParsedEntry>[];
  _ParsedEntry? current;
  String? pendingLabel;

  for (final rawLine in text.split('\n')) {
    final line = rawLine.trim();
    if (line.isEmpty) continue;
    final upper = line.toUpperCase();

    if (upper.startsWith('WORD:')) {
      current = _ParsedEntry(line.substring(5).trim());
      entries.add(current);
      pendingLabel = null;
    } else if (upper.startsWith('CORRECTED:')) {
      continue;
    } else if (upper.startsWith('DEF:')) {
      // A DEF: with no entry open means the line above it was the word.
      if (current == null || pendingLabel != null) {
        current = _ParsedEntry(pendingLabel ?? '');
        entries.add(current);
        pendingLabel = null;
      }
      current.definition = line.substring(4).trim();
    } else if (upper.startsWith('EX1:')) {
      current?.ex1 = line.substring(4).trim();
    } else if (upper.startsWith('EX2:')) {
      current?.ex2 = line.substring(4).trim();
    } else if (current == null || current.definition.isNotEmpty) {
      // A bare line once the open entry is complete is the next word,
      // missing its label.
      pendingLabel = line.replaceFirst(RegExp(r'^[\s*\-.•·#>]+'), '').trim();
    }
  }

  final byWord = <String, EnrichWordResult>{};
  for (final entry in entries) {
    if (entry.word.isEmpty) continue;
    // The marker is the whole definition, per the prompt's instruction to
    // "write DEF: UNKNOWN". A substring test would discard correct definitions
    // that merely contain the word.
    final marker = entry.definition.replaceAll(RegExp(r'[.!?\s]+$'), '');
    final isUnknown = marker.toUpperCase() == 'UNKNOWN';
    final definition = isUnknown ? '' : entry.definition;
    final warning = definition.isEmpty ? 'not_found' : '';
    final examples = warning.isEmpty
        ? [entry.ex1, entry.ex2].whereType<String>().join('\n')
        : '';
    // First mention wins: a repeated label is the model drifting, and
    // overwriting would decide which one is right with no evidence.
    byWord.putIfAbsent(
      entry.word.toLowerCase(),
      () => EnrichWordResult(
        word: entry.word,
        definition: definition,
        examples: examples,
        warning: warning,
      ),
    );
  }

  return [
    for (final w in words)
      byWord[w.toLowerCase()] ??
          EnrichWordResult(
            word: w,
            definition: '',
            examples: '',
            warning: 'not_found',
          ),
  ];
}

/// One word's block, while a response is being read.
class _ParsedEntry {
  _ParsedEntry(this.word);

  final String word;
  String definition = '';
  String? ex1;
  String? ex2;
}
