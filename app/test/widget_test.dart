import 'package:flutter_test/flutter_test.dart';
import 'package:ocr_to_anki/models/models.dart';

void main() {
  group('AnkiNote', () {
    test('toJson round-trip', () {
      final note = AnkiNote(
        front: 'hello',
        definition: 'a greeting',
        examples: 'Hello, world!',
      );
      final json = note.toJson();
      final restored = AnkiNote.fromJson(json);
      expect(restored.front, 'hello');
      expect(restored.definition, 'a greeting');
    });

    test('composeBack merges definition and examples', () {
      final note = AnkiNote(
        front: 'test',
        definition: 'a trial',
        examples: 'This is a test.',
      )..composeBack();
      expect(note.back, contains('a trial'));
      expect(note.back, contains('This is a test.'));
    });
  });

  group('HighlightColor', () {
    test('has 6 colours', () {
      expect(HighlightColor.values.length, 6);
    });
  });

  group('AppSettings', () {
    test('toJson round-trip', () {
      final s = AppSettings();
      final json = s.toJson();
      final restored = AppSettings.fromJson(json);
      expect(restored.defaultDeck, s.defaultDeck);
      expect(restored.inferenceMode, s.inferenceMode);
    });
  });
}
