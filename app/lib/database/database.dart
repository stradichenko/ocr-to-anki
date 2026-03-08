import 'dart:io';

import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

part 'database.g.dart';

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// A processing session -- one image processed through the pipeline.
class ProcessingSessions extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get imagePath => text()();
  TextColumn get context => text()(); // 'handwrittenOrPrinted' | 'highlighted'
  TextColumn get highlightColor => text().nullable()();
  TextColumn get ocrText => text().withDefault(const Constant(''))();
  RealColumn get ocrElapsedS => real().withDefault(const Constant(0))();
  RealColumn get enrichElapsedS => real().withDefault(const Constant(0))();
  TextColumn get backend => text().withDefault(const Constant(''))();
  TextColumn get error => text().nullable()();

  /// Structured benchmark data as JSON (see [BenchmarkData]).
  TextColumn get benchmarkJson =>
      text().withDefault(const Constant(''))();

  DateTimeColumn get createdAt =>
      dateTime().withDefault(currentDateAndTime)();
}

/// Individual words extracted during a session.
class WordEntries extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get sessionId =>
      integer().references(ProcessingSessions, #id)();
  TextColumn get word => text()();
  TextColumn get definition => text().withDefault(const Constant(''))();
  TextColumn get examples => text().withDefault(const Constant(''))();
  BoolColumn get exported =>
      boolean().withDefault(const Constant(false))();
  IntColumn get ankiNoteId => integer().nullable()();
  DateTimeColumn get createdAt =>
      dateTime().withDefault(currentDateAndTime)();
}

/// Tracks Anki export operations.
class ExportLogs extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get sessionId =>
      integer().nullable().references(ProcessingSessions, #id)();
  TextColumn get method =>
      text()(); // 'ankiconnect' | 'json' | 'apkg'
  IntColumn get totalNotes => integer()();
  IntColumn get successCount => integer()();
  IntColumn get failedCount => integer()();
  TextColumn get targetDeck => text()();
  DateTimeColumn get createdAt =>
      dateTime().withDefault(currentDateAndTime)();
}

/// User settings stored as key-value pairs.
class SettingsEntries extends Table {
  TextColumn get key => text()();
  TextColumn get value => text()();

  @override
  Set<Column> get primaryKey => {key};
}

/// Cache for enriched word definitions so repeated words skip the LLM.
class EnrichmentCacheEntries extends Table {
  /// Lowercased word.
  TextColumn get word => text()();

  /// Language the definition was generated in.
  TextColumn get definitionLanguage => text()();

  /// Language the examples were generated in.
  TextColumn get examplesLanguage => text()();

  TextColumn get definition => text()();
  TextColumn get examples => text()();
  TextColumn get warning => text().withDefault(const Constant(''))();

  DateTimeColumn get createdAt =>
      dateTime().withDefault(currentDateAndTime)();

  @override
  Set<Column> get primaryKey => {word, definitionLanguage, examplesLanguage};
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

@DriftDatabase(
  tables: [ProcessingSessions, WordEntries, ExportLogs, SettingsEntries, EnrichmentCacheEntries],
)
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_openConnection());

  /// For testing only.
  AppDatabase.forTesting(super.e);

  @override
  int get schemaVersion => 3;

  @override
  MigrationStrategy get migration => MigrationStrategy(
        onCreate: (m) => m.createAll(),
        onUpgrade: (m, from, to) async {
          if (from < 2) {
            // v2: add benchmark_json column to processing_sessions.
            await m.addColumn(
                processingSessions, processingSessions.benchmarkJson);
          }
          if (from < 3) {
            // v3: add enrichment_cache_entries table.
            await m.createTable(enrichmentCacheEntries);
          }
        },
      );

  // -------------------------------------------------------------------------
  // Processing sessions
  // -------------------------------------------------------------------------

  Future<int> insertSession(ProcessingSessionsCompanion entry) =>
      into(processingSessions).insert(entry);

  Future<List<ProcessingSession>> getAllSessions() =>
      (select(processingSessions)
            ..orderBy([
              (t) => OrderingTerm.desc(t.createdAt),
            ]))
          .get();

  Future<ProcessingSession?> getSession(int id) =>
      (select(processingSessions)..where((t) => t.id.equals(id)))
          .getSingleOrNull();

  // -------------------------------------------------------------------------
  // Word entries
  // -------------------------------------------------------------------------

  Future<int> insertWord(WordEntriesCompanion entry) =>
      into(wordEntries).insert(entry);

  Future<void> insertWords(List<WordEntriesCompanion> entries) async {
    await batch((b) {
      b.insertAll(wordEntries, entries);
    });
  }

  Future<List<WordEntry>> getWordsForSession(int sessionId) =>
      (select(wordEntries)..where((t) => t.sessionId.equals(sessionId)))
          .get();

  Future<List<WordEntry>> getUnexportedWords() =>
      (select(wordEntries)..where((t) => t.exported.equals(false))).get();

  Future<void> markExported(List<int> wordIds, {int? ankiNoteId}) async {
    await (update(wordEntries)..where((t) => t.id.isIn(wordIds))).write(
      WordEntriesCompanion(
        exported: const Value(true),
        ankiNoteId: ankiNoteId != null ? Value(ankiNoteId) : const Value.absent(),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Export logs
  // -------------------------------------------------------------------------

  Future<int> insertExportLog(ExportLogsCompanion entry) =>
      into(exportLogs).insert(entry);

  Future<List<ExportLog>> getExportHistory() =>
      (select(exportLogs)
            ..orderBy([(t) => OrderingTerm.desc(t.createdAt)]))
          .get();

  // -------------------------------------------------------------------------
  // Settings
  // -------------------------------------------------------------------------

  Future<String?> getSetting(String key) async {
    final row = await (select(settingsEntries)
          ..where((t) => t.key.equals(key)))
        .getSingleOrNull();
    return row?.value;
  }

  Future<void> setSetting(String key, String value) =>
      into(settingsEntries).insertOnConflictUpdate(
        SettingsEntriesCompanion.insert(key: key, value: value),
      );

  // -------------------------------------------------------------------------
  // Stats
  // -------------------------------------------------------------------------

  Future<int> totalSessionCount() async {
    final count = processingSessions.id.count();
    final query = selectOnly(processingSessions)..addColumns([count]);
    final row = await query.getSingle();
    return row.read(count) ?? 0;
  }

  Future<int> totalWordCount() async {
    final count = wordEntries.id.count();
    final query = selectOnly(wordEntries)..addColumns([count]);
    final row = await query.getSingle();
    return row.read(count) ?? 0;
  }

  Future<int> totalExportedCount() async {
    final count = wordEntries.id.count();
    final query = selectOnly(wordEntries)
      ..addColumns([count])
      ..where(wordEntries.exported.equals(true));
    final row = await query.getSingle();
    return row.read(count) ?? 0;
  }

  // -------------------------------------------------------------------------
  // Enrichment cache
  // -------------------------------------------------------------------------

  /// Look up a cached enrichment result for the given word + language pair.
  Future<EnrichmentCacheEntry?> getCachedEnrichment({
    required String word,
    required String definitionLanguage,
    required String examplesLanguage,
  }) {
    final lw = word.toLowerCase();
    return (select(enrichmentCacheEntries)
          ..where((t) =>
              t.word.equals(lw) &
              t.definitionLanguage.equals(definitionLanguage) &
              t.examplesLanguage.equals(examplesLanguage)))
        .getSingleOrNull();
  }

  /// Batch lookup: returns a map from lowercase word → cached entry.
  Future<Map<String, EnrichmentCacheEntry>> getCachedEnrichments({
    required List<String> words,
    required String definitionLanguage,
    required String examplesLanguage,
  }) async {
    if (words.isEmpty) return {};
    final lowerWords = words.map((w) => w.toLowerCase()).toList();
    final rows = await (select(enrichmentCacheEntries)
          ..where((t) =>
              t.word.isIn(lowerWords) &
              t.definitionLanguage.equals(definitionLanguage) &
              t.examplesLanguage.equals(examplesLanguage)))
        .get();
    return {for (final r in rows) r.word: r};
  }

  /// Insert or update a cache entry.
  Future<void> cacheEnrichment({
    required String word,
    required String definitionLanguage,
    required String examplesLanguage,
    required String definition,
    required String examples,
    String warning = '',
  }) {
    return into(enrichmentCacheEntries).insertOnConflictUpdate(
      EnrichmentCacheEntriesCompanion.insert(
        word: word.toLowerCase(),
        definitionLanguage: definitionLanguage,
        examplesLanguage: examplesLanguage,
        definition: definition,
        examples: examples,
        warning: Value(warning),
      ),
    );
  }

  /// Batch-insert cache entries.
  Future<void> cacheEnrichments(
      List<EnrichmentCacheEntriesCompanion> entries) async {
    await batch((b) {
      for (final entry in entries) {
        b.insert(enrichmentCacheEntries, entry,
            onConflict: DoUpdate((_) => entry));
      }
    });
  }

  /// Clear the entire enrichment cache.
  Future<int> clearEnrichmentCache() =>
      delete(enrichmentCacheEntries).go();

  /// Delete a single cache entry by its composite key.
  Future<int> deleteCacheEntry({
    required String word,
    required String definitionLanguage,
    required String examplesLanguage,
  }) {
    final lw = word.toLowerCase();
    return (delete(enrichmentCacheEntries)
          ..where((t) =>
              t.word.equals(lw) &
              t.definitionLanguage.equals(definitionLanguage) &
              t.examplesLanguage.equals(examplesLanguage)))
        .go();
  }

  /// Delete all cache entries for a specific language pair.
  Future<int> clearCacheByLanguagePair({
    required String definitionLanguage,
    required String examplesLanguage,
  }) {
    return (delete(enrichmentCacheEntries)
          ..where((t) =>
              t.definitionLanguage.equals(definitionLanguage) &
              t.examplesLanguage.equals(examplesLanguage)))
        .go();
  }

  /// List cached entries with optional search and language filter.
  /// Results are ordered alphabetically by word.
  Future<List<EnrichmentCacheEntry>> listCachedEntries({
    String? search,
    String? definitionLanguage,
    String? examplesLanguage,
    int? limit,
    int? offset,
  }) {
    final q = select(enrichmentCacheEntries)
      ..orderBy([(t) => OrderingTerm.asc(t.word)]);

    q.where((t) {
      Expression<bool> expr = const Constant(true);
      if (search != null && search.isNotEmpty) {
        expr = expr & t.word.like('%${search.toLowerCase()}%');
      }
      if (definitionLanguage != null) {
        expr = expr & t.definitionLanguage.equals(definitionLanguage);
      }
      if (examplesLanguage != null) {
        expr = expr & t.examplesLanguage.equals(examplesLanguage);
      }
      return expr;
    });

    if (limit != null) q.limit(limit, offset: offset);
    return q.get();
  }

  /// Get a breakdown of cache entry counts grouped by language pair.
  Future<List<CacheLanguagePairStat>> getCacheLanguagePairStats() async {
    final defLang = enrichmentCacheEntries.definitionLanguage;
    final exLang = enrichmentCacheEntries.examplesLanguage;
    final cnt = enrichmentCacheEntries.word.count();

    final query = selectOnly(enrichmentCacheEntries)
      ..addColumns([defLang, exLang, cnt])
      ..groupBy([defLang, exLang])
      ..orderBy([OrderingTerm.desc(cnt)]);

    final rows = await query.get();
    return rows.map((row) {
      return CacheLanguagePairStat(
        definitionLanguage: row.read(defLang)!,
        examplesLanguage: row.read(exLang)!,
        count: row.read(cnt)!,
      );
    }).toList();
  }

  /// Count of cached entries.
  Future<int> enrichmentCacheCount() async {
    final count = enrichmentCacheEntries.word.count();
    final query = selectOnly(enrichmentCacheEntries)..addColumns([count]);
    final row = await query.getSingle();
    return row.read(count) ?? 0;
  }
}

LazyDatabase _openConnection() {
  return LazyDatabase(() async {
    // Use applicationSupport (~/.local/share/<app> on Linux) instead of
    // documents, which may not exist on some Linux desktop environments.
    final dir = await getApplicationSupportDirectory();
    final file = File(p.join(dir.path, 'ocr_to_anki.sqlite'));
    return NativeDatabase.createInBackground(file);
  });
}

// ---------------------------------------------------------------------------
// Helper classes
// ---------------------------------------------------------------------------

class CacheLanguagePairStat {
  const CacheLanguagePairStat({
    required this.definitionLanguage,
    required this.examplesLanguage,
    required this.count,
  });

  final String definitionLanguage;
  final String examplesLanguage;
  final int count;
}
