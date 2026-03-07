/// Data model for a single Anki flashcard note.
///
/// Mirrors the JSON structure used by AnkiConnect and the Python backend's
/// `data/test_notes.json` format.
class AnkiNote {
  AnkiNote({
    required this.front,
    this.back = '',
    this.deckName,
    this.modelName,
    this.tags = const ['ocr'],
    this.allowDuplicate = false,
    this.definition = '',
    this.examples = '',
  });

  /// The front field (word / term).
  String front;

  /// The back field (definition + examples).
  String back;

  /// Override deck name (uses default from settings if null).
  String? deckName;

  /// Override note model name (uses default from settings if null).
  String? modelName;

  /// Tags attached to this note.
  List<String> tags;

  /// Whether to allow duplicate notes in Anki.
  bool allowDuplicate;

  /// LLM-generated definition (stored separately before merging into [back]).
  String definition;

  /// LLM-generated example sentences.
  String examples;

  /// Merge [definition] and [examples] into the [back] field.
  void composeBack() {
    final parts = <String>[];
    if (definition.isNotEmpty) parts.add(definition);
    if (examples.isNotEmpty) parts.add(examples);
    back = parts.join('\n\n');
  }

  /// Serialise to AnkiConnect-compatible JSON.
  Map<String, dynamic> toAnkiJson({
    required String defaultDeck,
    required String defaultModel,
  }) {
    return {
      'deckName': deckName ?? defaultDeck,
      'modelName': modelName ?? defaultModel,
      'fields': {
        'Front': front,
        'Back': back,
      },
      'tags': tags,
      'options': {
        'allowDuplicate': allowDuplicate,
        'duplicateScope': 'deck',
      },
    };
  }

  /// Serialise for local JSON storage.
  Map<String, dynamic> toJson() => {
        'front': front,
        'back': back,
        'deckName': deckName,
        'modelName': modelName,
        'tags': tags,
        'allowDuplicate': allowDuplicate,
        'definition': definition,
        'examples': examples,
      };

  factory AnkiNote.fromJson(Map<String, dynamic> json) => AnkiNote(
        front: json['front'] as String? ?? '',
        back: json['back'] as String? ?? '',
        deckName: json['deckName'] as String?,
        modelName: json['modelName'] as String?,
        tags: (json['tags'] as List<dynamic>?)
                ?.map((e) => e as String)
                .toList() ??
            const ['ocr'],
        allowDuplicate: json['allowDuplicate'] as bool? ?? false,
        definition: json['definition'] as String? ?? '',
        examples: json['examples'] as String? ?? '',
      );

  @override
  String toString() => 'AnkiNote(front: $front)';
}
