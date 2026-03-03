import 'dart:io';

import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

part 'database.g.dart';

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// A processing session — one image processed through the pipeline.
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

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

@DriftDatabase(
  tables: [ProcessingSessions, WordEntries, ExportLogs, SettingsEntries],
)
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_openConnection());

  /// For testing only.
  AppDatabase.forTesting(super.e);

  @override
  int get schemaVersion => 1;

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
