import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import '../models/dictionary_info.dart';

/// Loads and queries the bundled dictionary registry (`assets/dictionaries.json`).
class DictionaryRegistryService {
  DictionaryRegistryService();

  DictionaryRegistry? _registry;

  Future<DictionaryRegistry> loadRegistry() async {
    if (_registry != null) return _registry!;
    final jsonString = await rootBundle.loadString('assets/dictionaries.json');
    final json = jsonDecode(jsonString) as Map<String, dynamic>;
    _registry = DictionaryRegistry.fromJson(json);
    return _registry!;
  }

  Future<List<DictionaryInfo>> listDictionaries() async {
    final r = await loadRegistry();
    return r.dictionaries;
  }

  Future<DictionaryInfo?> getDictionary(String id) async {
    final r = await loadRegistry();
    return r.byId(id);
  }

  /// The dictionary for this pair, or null when there is none.
  ///
  /// Most pairs have none. The caller falls back to the model, which is the
  /// normal path rather than an error case.
  Future<DictionaryInfo?> forLanguagePair(String term, String definition) async {
    final r = await loadRegistry();
    return r.forLanguagePair(term, definition);
  }

  /// Directory where installed indices live.
  Future<Directory> dictionaryDir() async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}/dictionaries');
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  Future<String> dictionaryPath(DictionaryInfo info) async {
    final dir = await dictionaryDir();
    return '${dir.path}/${info.fileName}';
  }

  Future<bool> isInstalled(DictionaryInfo info) async {
    // A bundled index counts as installed: it ships inside the app, so there
    // is nothing to download and the feature works on first run.
    if (await isBundled(info)) return true;

    final path = await dictionaryPath(info);
    final file = File(path);
    if (!await file.exists()) return false;
    // A truncated index is worse than none: it would silently answer some
    // lookups and miss others. Size is the cheap check; the hash is verified
    // once at download time.
    if (info.sizeBytes > 0 && await file.length() != info.sizeBytes) return false;
    return true;
  }

  /// Whether this index ships inside the app bundle.
  Future<bool> isBundled(DictionaryInfo info) async {
    try {
      await rootBundle.load('assets/dictionaries/${info.fileName}');
      return true;
    } catch (_) {
      return false;
    }
  }

  /// Path to an index that can be opened, copying a bundled one out first.
  ///
  /// SQLite needs a real file, and on Android assets live inside the APK, so a
  /// bundled index is copied into app storage once. Returns null when the index
  /// is neither bundled nor already downloaded.
  Future<String?> ensureInstalled(DictionaryInfo info) async {
    final path = await dictionaryPath(info);
    final file = File(path);
    if (await file.exists()) {
      if (info.sizeBytes == 0 || await file.length() == info.sizeBytes) {
        return path;
      }
      // A partial or stale copy: fall through and rewrite it.
      await file.delete();
    }

    if (!await isBundled(info)) return null;

    final data = await rootBundle.load('assets/dictionaries/${info.fileName}');
    await file.writeAsBytes(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      flush: true,
    );
    return path;
  }

  Future<int> installedBytes() async {
    final dir = await dictionaryDir();
    if (!await dir.exists()) return 0;
    var total = 0;
    await for (final entity in dir.list()) {
      if (entity is File) total += await entity.length();
    }
    return total;
  }

  Future<void> delete(DictionaryInfo info) async {
    final path = await dictionaryPath(info);
    final file = File(path);
    if (await file.exists()) await file.delete();
  }
}