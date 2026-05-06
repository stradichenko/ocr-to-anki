import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../models/anki_note.dart';
import '../models/app_settings.dart';
import 'system_channel.dart';

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

  /// Optional callback fired with status messages during auto-launch.
  /// Set this before calling methods if you want progress updates.
  void Function(String message)? onStatusUpdate;

  /// Prevents multiple simultaneous Anki launch attempts across instances.
  static bool _launching = false;

  // ---------------------------------------------------------------------------
  // AnkiConnect helpers
  // ---------------------------------------------------------------------------

  /// Try to auto-launch Anki and wait for AnkiConnect to become reachable.
  /// Returns `true` if AnkiConnect became reachable, `false` otherwise.
  Future<bool> _tryLaunchAnki() async {
    // Only one launch attempt at a time.
    if (_launching) return false;
    _launching = true;
    try {
      onStatusUpdate?.call('Launching Anki…');
      await Process.start('anki', [], mode: ProcessStartMode.detached);
      for (var i = 0; i < 10; i++) {
        await Future.delayed(const Duration(seconds: 2));
        if (await checkConnection()) {
          return true;
        }
        onStatusUpdate?.call('Waiting for AnkiConnect… (${i + 1}/10)');
      }
      return false;
    } catch (_) {
      // `anki` binary not found – nothing we can do.
      return false;
    } finally {
      _launching = false;
    }
  }

  /// Send a request to the AnkiConnect API.
  ///
  /// On connection refused, attempts to auto-launch Anki and retries once.
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
        // Try launching Anki automatically.
        if (await _tryLaunchAnki()) {
          // Retry the request now that Anki is running.
          return _invoke(action, params);
        }
        throw AnkiConnectException(
          'Connection refused – could not reach or launch Anki.\n\n'
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

  // ---------------------------------------------------------------------------
  // AnkiDroid direct add (Android)
  // ---------------------------------------------------------------------------

  /// Whether AnkiDroid is installed and we have permission to add notes.
  Future<bool> canUseAnkiDroid() async {
    if (!Platform.isAndroid) return false;
    if (!await SystemChannel.isAnkiDroidInstalled()) return false;
    return await SystemChannel.requestAnkiDroidPermission();
  }

  /// List available AnkiDroid decks.
  Future<List<({int id, String name})>> getAnkiDroidDecks() async {
    final raw = await SystemChannel.getAnkiDroidDecks();
    return raw
        .map((d) => (id: (d['id'] as num).toInt(), name: d['name'] as String))
        .toList();
  }

  /// Add notes directly to AnkiDroid.
  Future<int> addNotesToAnkiDroid(
    List<AnkiNote> notes, {
    required int deckId,
  }) async {
    final payload = notes.map((n) {
      final back = n.back.isNotEmpty ? n.back : _composeBack(n);
      return <String, dynamic>{
        'fields': [n.front, back],
        'tags': n.tags,
      };
    }).toList();
    return SystemChannel.addNotesToAnkiDroid(payload, deckId);
  }

  /// Share notes as a TSV file with AnkiDroid (Android only).
  ///
  /// Writes a temporary file and opens the Android share sheet so the user
  /// can select AnkiDroid to import the cards.
  Future<void> shareToAnkiDroid(List<AnkiNote> notes) async {
    final tsv = exportToTsv(notes);
    final tmpDir = await getTemporaryDirectory();
    final timestamp = DateTime.now()
        .toIso8601String()
        .replaceAll(':', '-')
        .split('.')
        .first;
    final filePath = '${tmpDir.path}/anki_cards_$timestamp.txt';
    final file = File(filePath);
    await file.writeAsString(tsv);

    await Share.shareXFiles(
      [XFile(filePath, mimeType: 'text/tab-separated-values')],
      subject: 'Anki cards',
      text: '${notes.length} Anki card(s) to import',
    );
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
