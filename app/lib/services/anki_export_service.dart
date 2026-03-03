import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/anki_note.dart';
import '../models/app_settings.dart';

/// Service for exporting flashcards to Anki.
///
/// Supports:
///  * **AnkiConnect** (desktop) -- HTTP JSON-RPC to localhost:8765
///  * **JSON file export** -- write an Anki-importable JSON file
///
/// Future:
///  * AnkiDroid Instant-Add API (Android)
///  * .apkg binary generation (cross-platform offline fallback)
class AnkiExportService {
  AnkiExportService({required AppSettings settings}) : _settings = settings;

  AppSettings _settings;

  void updateSettings(AppSettings settings) => _settings = settings;

  // ---------------------------------------------------------------------------
  // AnkiConnect helpers
  // ---------------------------------------------------------------------------

  /// Send a request to the AnkiConnect API.
  Future<dynamic> _invoke(String action,
      [Map<String, dynamic> params = const {}]) async {
    final uri = Uri.parse(_settings.ankiConnectUrl);
    final payload = {
      'action': action,
      'version': _settings.ankiConnectVersion,
      'params': params,
    };

    final response = await http
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload),
        )
        .timeout(Duration(seconds: _settings.ankiConnectTimeout));

    if (response.statusCode != 200) {
      throw AnkiConnectException(
        'AnkiConnect HTTP ${response.statusCode}: ${response.body}',
      );
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    if (body['error'] != null) {
      throw AnkiConnectException('AnkiConnect error: ${body['error']}');
    }

    return body['result'];
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Check whether AnkiConnect is reachable.
  Future<bool> checkConnection() async {
    try {
      final version = await _invoke('version');
      return version != null;
    } catch (_) {
      return false;
    }
  }

  /// Get the AnkiConnect version number.
  Future<int?> getVersion() async {
    try {
      return await _invoke('version') as int?;
    } catch (_) {
      return null;
    }
  }

  /// List all available deck names.
  Future<List<String>> getDecks() async {
    final result = await _invoke('deckNames');
    return (result as List<dynamic>).cast<String>();
  }

  /// List all available model (note type) names.
  Future<List<String>> getModels() async {
    final result = await _invoke('modelNames');
    return (result as List<dynamic>).cast<String>();
  }

  /// Create a deck if it does not exist.
  Future<void> createDeck(String deckName) async {
    await _invoke('createDeck', {'deck': deckName});
  }

  /// Add a single note to Anki. Returns the note ID or `null` on failure.
  Future<int?> addNote(AnkiNote note) async {
    try {
      final result = await _invoke('addNote', {
        'note': note.toAnkiJson(
          defaultDeck: _settings.defaultDeck,
          defaultModel: _settings.defaultModel,
        ),
      });
      return result as int?;
    } catch (_) {
      return null;
    }
  }

  /// Add multiple notes in a single batch. Returns a list of note IDs (null
  /// entries indicate failures).
  Future<List<int?>> addNotes(List<AnkiNote> notes) async {
    final ankiNotes = notes
        .map((n) => n.toAnkiJson(
              defaultDeck: _settings.defaultDeck,
              defaultModel: _settings.defaultModel,
            ))
        .toList();

    try {
      final result = await _invoke('addNotes', {'notes': ankiNotes});
      return (result as List<dynamic>).map((e) => e as int?).toList();
    } catch (_) {
      return List<int?>.filled(notes.length, null);
    }
  }

  /// Export notes as a JSON string in the same format as `data/test_notes.json`.
  String exportToJson(
    List<AnkiNote> notes, {
    AnkiImportSettings? importSettings,
  }) {
    final settings = importSettings ??
        AnkiImportSettings(
          defaultDeck: _settings.defaultDeck,
          defaultModel: _settings.defaultModel,
          batchSize: _settings.batchSize,
        );

    final output = {
      'settings': settings.toJson(),
      'notes': notes
          .map((n) => {
                'fields': {
                  'Front': n.front,
                  'Back': n.back,
                },
                'tags': n.tags,
                'allowDuplicate': n.allowDuplicate,
              })
          .toList(),
    };

    return const JsonEncoder.withIndent('  ').convert(output);
  }

  /// Full import flow: create deck → batch-add notes → return summary.
  Future<ImportResult> importNotes(List<AnkiNote> notes) async {
    if (notes.isEmpty) {
      return const ImportResult(total: 0, success: 0, failed: 0);
    }

    // Ensure deck exists.
    await createDeck(_settings.defaultDeck);

    // Batch import.
    final ids = await addNotes(notes);

    var success = 0;
    var failed = 0;
    for (final id in ids) {
      if (id != null) {
        success++;
      } else {
        failed++;
      }
    }

    return ImportResult(total: notes.length, success: success, failed: failed);
  }
}

/// Summary of an import operation.
class ImportResult {
  const ImportResult({
    required this.total,
    required this.success,
    required this.failed,
  });

  final int total;
  final int success;
  final int failed;

  @override
  String toString() => 'ImportResult(total=$total, ok=$success, fail=$failed)';
}

class AnkiConnectException implements Exception {
  AnkiConnectException(this.message);
  final String message;

  @override
  String toString() => message;
}
