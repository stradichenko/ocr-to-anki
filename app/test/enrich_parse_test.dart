import 'package:flutter_test/flutter_test.dart';
import 'package:ocr_to_anki/services/enrich_parse.dart';

/// Tests for the on-device enrichment parser.
///
/// Android parses model responses here rather than through the Python backend,
/// so this is a second implementation of the same job and the two had drifted:
/// the Python side was fixed for responses where the model labels only the
/// first word, and this one still carried the two unsafe guesses that fix
/// replaced. Both produce the same visible failure, a flashcard whose
/// definition belongs to a different word.
void main() {
  String block(String word, String def, {String? ex1, String? ex2}) =>
      'WORD: $word\nDEF: $def\n'
      '${ex1 == null ? '' : 'EX1: $ex1\n'}'
      '${ex2 == null ? '' : 'EX2: $ex2\n'}';

  group('well-formed responses', () {
    test('each word keeps its own definition', () {
      final text = block('uno', 'The number one.', ex1: 'One plus one is two.')
          + block('dos', 'The number two.', ex1: 'Two plus two is four.');
      final r = parseEnrichResponse(text, ['uno', 'dos']);

      expect(r.map((e) => e.word), ['uno', 'dos']);
      expect(r[0].definition, 'The number one.');
      expect(r[1].definition, 'The number two.');
      expect(r.every((e) => e.warning == ''), isTrue);
    });

    test('always returns one entry per requested word, in order', () {
      final text = block('uno', 'One.');
      final r = parseEnrichResponse(text, ['uno', 'dos', 'tres']);
      expect(r.map((e) => e.word), ['uno', 'dos', 'tres']);
      expect(r[0].warning, '');
      expect(r[1].warning, 'not_found');
      expect(r[2].warning, 'not_found');
    });
  });

  group('missing WORD: labels', () {
    // The model labels only the first entry and emits the rest bare. This is
    // the shape that produced the misaligned export.
    final text = 'WORD: anal\n'
        'DEF: Relating to the anus.\n'
        'EX1: A medical term.\n'
        '- venezolana\n'
        'DEF: A woman from Venezuela.\n'
        'EX1: She is Venezuelan.\n'
        'amor\n'
        'DEF: A deep affection.\n'
        'EX1: Love is patient.\n';

    test('every word gets its own definition', () {
      final r = parseEnrichResponse(text, ['anal', 'venezolana', 'amor']);
      expect(r[0].definition, 'Relating to the anus.');
      expect(r[1].definition, 'A woman from Venezuela.');
      expect(r[2].definition, 'A deep affection.');
    });

    test('no two words share a definition', () {
      final r = parseEnrichResponse(text, ['anal', 'venezolana', 'amor']);
      final defs = r.map((e) => e.definition).toList();
      expect(defs.toSet().length, defs.length);
    });

    test('a bulleted word is read without the bullet', () {
      final r = parseEnrichResponse(text, ['anal', 'venezolana', 'amor']);
      expect(r[1].word, 'venezolana');
    });
  });

  group('the guesses that were removed', () {
    test('a definition mentioning another word does not steal its slot', () {
      // The old parser matched the definition text against the expected words,
      // so a definition containing "amor" claimed the slot for "amor".
      final text = 'WORD: amor\n'
          'DEF: A deep affection, as in the phrase amor fati.\n'
          'WORD: casa\n'
          'DEF: A house.\n';
      final r = parseEnrichResponse(text, ['amor', 'casa']);
      expect(r[0].word, 'amor');
      expect(r[0].definition, contains('deep affection'));
      expect(r[1].word, 'casa');
      expect(r[1].definition, 'A house.');
    });

    test('a skipped word does not shift the ones after it', () {
      // The old parser walked a cursor over the expected words, so one missing
      // entry gave every later word the wrong definition.
      final text = block('alpha', 'The first letter.')
          + block('gamma', 'The third letter.');
      final r = parseEnrichResponse(text, ['alpha', 'beta', 'gamma']);

      expect(r[0].definition, 'The first letter.');
      expect(r[1].warning, 'not_found');
      expect(r[1].definition, '');
      expect(r[2].definition, 'The third letter.');
    });
  });

  group('the UNKNOWN marker', () {
    test('a bare marker is a miss', () {
      final r = parseEnrichResponse('WORD: xyzzy\nDEF: UNKNOWN\n', ['xyzzy']);
      expect(r[0].warning, 'not_found');
      expect(r[0].definition, '');
    });

    test('a definition merely containing it is not', () {
      final r = parseEnrichResponse(
        'WORD: casa\nDEF: Of an unknown origin.\nEX1: A house.\n',
        ['casa'],
      );
      expect(r[0].warning, '');
      expect(r[0].definition, 'Of an unknown origin.');
    });
  });

  group('robustness', () {
    test('empty input yields a miss per word', () {
      final r = parseEnrichResponse('', ['uno', 'dos']);
      expect(r.length, 2);
      expect(r.every((e) => e.warning == 'not_found'), isTrue);
    });

    test('a repeated word keeps the first block', () {
      final text = block('uno', 'The first answer.')
          + block('uno', 'A different answer.');
      final r = parseEnrichResponse(text, ['uno']);
      expect(r[0].definition, 'The first answer.');
    });

    test('examples accompany their own definition', () {
      final text = block('uno', 'One.', ex1: 'Uno mas.', ex2: 'Solo uno.')
          + block('dos', 'Two.', ex1: 'Dos mas.', ex2: 'Solo dos.');
      final r = parseEnrichResponse(text, ['uno', 'dos']);
      expect(r[0].examples, 'Uno mas.\nSolo uno.');
      expect(r[1].examples, 'Dos mas.\nSolo dos.');
    });
  });


  group('reasoning channel', () {
    // Gemma 4 narrates before answering. The closing marker is glued to the
    // first label line, so without stripping it the label is never recognised
    // and the run reports nothing enriched.
    const narration = '<|channel>thought\n'
        'Thinking process:\n'
        '1. The user wants definitions.\n'
        '2. I will write them out.';

    test('a closed channel is stripped before parsing', () {
      final text = '$narration'
          '<channel|>WORD: casa\n'
          'DEF: A house.\n'
          'EX1: I live in a big house.\n';
      final r = parseEnrichResponse(text, ['casa']);
      expect(r[0].definition, 'A house.');
      expect(r[0].warning, '');
    });

    test('entries after the channel are not lost', () {
      final text = '$narration'
          '<channel|>WORD: casa\nDEF: A house.\n'
          'WORD: perro\nDEF: A dog.\n';
      final r = parseEnrichResponse(text, ['casa', 'perro']);
      expect(r[0].definition, 'A house.');
      expect(r[1].definition, 'A dog.');
    });

    test('an unclosed channel yields misses, not the narration', () {
      final r = parseEnrichResponse(narration, ['casa']);
      expect(r[0].warning, 'not_found');
      expect(r[0].definition, '');
    });

    test('narration text never becomes a definition', () {
      final text = '$narration<channel|>WORD: casa\nDEF: A house.\n';
      final r = parseEnrichResponse(text, ['casa']);
      expect(r[0].definition, isNot(contains('Thinking')));
      expect(r[0].definition, isNot(contains('user wants')));
    });

    test('a response with no channel is untouched', () {
      final r = parseEnrichResponse('WORD: casa\nDEF: A house.\n', ['casa']);
      expect(r[0].definition, 'A house.');
    });
  });
}
