import 'dart:convert';
import 'dart:io';

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

  final AppSettings _settings;

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

    http.Response response;
    try {
      response = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(Duration(seconds: _settings.ankiConnectTimeout));
    } on SocketException catch (e) {
      if (e.osError?.errorCode == 111 || e.message.contains('Connection refused')) {
        throw AnkiConnectException(
          'Connection refused – is Anki running with AnkiConnect installed?\n\n'
          'AnkiConnect URL: ${_settings.ankiConnectUrl}\n'
          'Error: $e',
        );
      }
      throw AnkiConnectException(
        'Cannot reach Anki at ${_settings.ankiConnectUrl}\n'
        'Error: $e',
      );
    }

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

  /// Export notes as a tab-separated text file importable by Anki.
  ///
  /// Format: `Front\tBack\ttag1 tag2`
  /// Anki File → Import understands this format.
  String exportToTsv(List<AnkiNote> notes) {
    final buffer = StringBuffer();
    // Header comment so the user knows what this file is.
    buffer.writeln('#separator:tab');
    buffer.writeln('#html:false');
    buffer.writeln('#deck:${_settings.defaultDeck}');
    buffer.writeln('#notetype:${_settings.defaultModel}');
    buffer.writeln('#tags column:3');

    for (final note in notes) {
      final front = _escapeTsvField(note.front);
      final back = _escapeTsvField(
        note.back.isNotEmpty ? note.back : _composeBack(note),
      );
      final tags = note.tags.join(' ');
      buffer.writeln('$front\t$back\t$tags');
    }

    return buffer.toString();
  }

  String _composeBack(AnkiNote note) {
    final parts = <String>[];
    if (note.definition.isNotEmpty) parts.add(note.definition);
    if (note.examples.isNotEmpty) parts.add(note.examples);
    return parts.join('\n\n');
  }

  /// Escape a TSV field: replace tabs with spaces, newlines with <br>.
  String _escapeTsvField(String value) {
    return value
        .replaceAll('\t', '    ')
        .replaceAll('\r\n', '<br>')
        .replaceAll('\n', '<br>');
  }

  /// Export notes as a JSON string in the same format as `data/test_notes.json`.
  String exportToJson(List<AnkiNote> notes) {
    final output = {
      'settings': {
        'defaultDeck': _settings.defaultDeck,
        'defaultModel': _settings.defaultModel,
        'batchSize': _settings.batchSize,
        'allowDuplicates': _settings.allowDuplicates,
        'duplicateScope': 'deck',
      },
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
