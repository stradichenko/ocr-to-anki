import 'package:sqlite3/sqlite3.dart';

import '../models/dictionary_info.dart';
import 'dictionary_registry_service.dart';

/// One word's dictionary entry.
class DictionaryLookup {
  const DictionaryLookup({
    required this.word,
    required this.definition,
    this.examples = const [],
  });

  final String word;
  final String definition;
  final List<String> examples;

  bool get found => definition.trim().isNotEmpty;
}

/// Reads an installed dictionary index.
///
/// The index is a small SQLite file built ahead of time from a Wiktextract
/// extract (see scripts/build-dictionary-index.py). Lookups are index reads, so
/// they cost microseconds where a generated definition costs a model call.
///
/// Opened lazily and read-only. A missing or unreadable index is not an error:
/// the caller falls back to the model, which is the normal path for any
/// language pair with no dictionary.
class DictionaryService {
  DictionaryService(this._registry);

  final DictionaryRegistryService _registry;

  Database? _db;
  String? _openFor;

  /// Look up [words] in the index for this language pair.
  ///
  /// Returns an empty map when no index is installed for the pair, so the
  /// caller can treat that identically to a set of misses.
  Future<Map<String, DictionaryLookup>> lookup(
    List<String> words, {
    required String termLanguage,
    required String definitionLanguage,
  }) async {
    final info = await _registry.forLanguagePair(termLanguage, definitionLanguage);
    if (info == null) return const {};

    final db = await _open(info);
    if (db == null) return const {};

    final out = <String, DictionaryLookup>{};
    final selectGloss = db.prepare(
      'SELECT gloss FROM entries WHERE word_norm = ? LIMIT 1',
    );
    final selectExamples = db.prepare(
      'SELECT sentence FROM examples WHERE word_norm = ? LIMIT 2',
    );
    try {
      for (final word in words) {
        final key = _normalise(word);
        if (key.isEmpty) continue;

        final rows = selectGloss.select([key]);
        if (rows.isEmpty) continue;

        final gloss = (rows.first['gloss'] as String?)?.trim() ?? '';
        if (gloss.isEmpty) continue;

        final examples = selectExamples
            .select([key])
            .map((r) => (r['sentence'] as String?)?.trim() ?? '')
            .where((s) => s.isNotEmpty)
            .toList();

        out[key] = DictionaryLookup(
          word: word,
          definition: gloss,
          examples: examples,
        );
      }
    } finally {
      selectGloss.dispose();
      selectExamples.dispose();
    }
    return out;
  }

  Future<Database?> _open(DictionaryInfo info) async {
    if (_db != null && _openFor == info.id) return _db;

    // A bundled index is copied into app storage on first use, since SQLite
    // needs a real file and on Android assets live inside the APK.
    final path = await _registry.ensureInstalled(info);
    if (path == null) return null;

    close();
    try {
      _db = sqlite3.open(path, mode: OpenMode.readOnly);
      _openFor = info.id;
      return _db;
    } catch (_) {
      // A corrupt index falls back to the model rather than failing the run.
      _db = null;
      _openFor = null;
      return null;
    }
  }

  /// Whether an index is installed and can be opened for this pair.
  Future<bool> isAvailable(String termLanguage, String definitionLanguage) async {
    final info = await _registry.forLanguagePair(termLanguage, definitionLanguage);
    if (info == null) return false;
    return _registry.isInstalled(info);
  }

  void close() {
    _db?.dispose();
    _db = null;
    _openFor = null;
  }

  /// Must match the normalisation in scripts/build-dictionary-index.py.
  String _normalise(String word) => word.trim().toLowerCase();
}