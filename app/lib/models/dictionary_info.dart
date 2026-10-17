/// Metadata for a downloadable offline dictionary index.
///
/// One entry covers one term language looked up in one definition language,
/// because a Wiktextract extract carries glosses in the language of the
/// Wiktionary edition it came from. Japanese terms with English glosses come
/// from the English edition; French definitions would need the French one.
class DictionaryInfo {
  const DictionaryInfo({
    required this.id,
    required this.name,
    required this.description,
    required this.termLanguage,
    required this.definitionLanguage,
    required this.fileName,
    required this.sizeBytes,
    required this.sha256,
    required this.entryCount,
    required this.license,
    required this.licenseUrl,
    required this.attribution,
    required this.sourceUrl,
  });

  final String id;
  final String name;
  final String description;

  /// Language of the words being looked up, lowercased.
  final String termLanguage;

  /// Language the definitions are written in, lowercased.
  final String definitionLanguage;

  /// Local filename once installed.
  final String fileName;

  final int sizeBytes;

  /// SHA-256 of the downloaded index.
  final String sha256;

  /// Roughly how many headwords it holds, for the settings screen.
  final int entryCount;

  // Licence fields. The data is derived from Wiktionary, which is share-alike,
  // so these are shown in the app rather than only recorded here.
  final String license;
  final String licenseUrl;
  final String attribution;
  final String sourceUrl;

  /// Whether this dictionary serves a given language pair.
  bool serves(String term, String definition) =>
      termLanguage == term.trim().toLowerCase() &&
      definitionLanguage == definition.trim().toLowerCase();

  String get sizeLabel => '${(sizeBytes / (1024 * 1024)).toStringAsFixed(1)} MB';

  factory DictionaryInfo.fromJson(Map<String, dynamic> json) {
    return DictionaryInfo(
      id: json['id'] as String,
      name: json['name'] as String,
      description: json['description'] as String? ?? '',
      termLanguage: json['termLanguage'] as String,
      definitionLanguage: json['definitionLanguage'] as String,
      fileName: json['fileName'] as String,
      sizeBytes: json['sizeBytes'] as int? ?? 0,
      sha256: json['sha256'] as String? ?? '',
      entryCount: json['entryCount'] as int? ?? 0,
      license: json['license'] as String? ?? '',
      licenseUrl: json['licenseUrl'] as String? ?? '',
      attribution: json['attribution'] as String? ?? '',
      sourceUrl: json['sourceUrl'] as String? ?? '',
    );
  }

  /// Download URL, built from the registry's base.
  String urlFor(String baseUrl) {
    final base = baseUrl.endsWith('/') ? baseUrl : '$baseUrl/';
    return '$base$fileName';
  }
}

/// Root object in the dictionary registry JSON asset.
class DictionaryRegistry {
  const DictionaryRegistry({required this.baseUrl, required this.dictionaries});

  /// Where the indices are published. Kept in the registry rather than in code
  /// so moving the hosting does not need a release.
  final String baseUrl;

  final List<DictionaryInfo> dictionaries;

  DictionaryInfo? byId(String id) {
    for (final d in dictionaries) {
      if (d.id == id) return d;
    }
    return null;
  }

  /// The dictionary serving this pair, or null when there is none.
  ///
  /// Coverage is uneven by nature: only so many Wiktionary editions publish an
  /// extract, so a pair with no dictionary simply falls back to the model.
  DictionaryInfo? forLanguagePair(String term, String definition) {
    for (final d in dictionaries) {
      if (d.serves(term, definition)) return d;
    }
    return null;
  }

  factory DictionaryRegistry.fromJson(Map<String, dynamic> json) {
    final raw = (json['dictionaries'] as List<dynamic>?) ?? const [];
    return DictionaryRegistry(
      baseUrl: json['baseUrl'] as String? ?? '',
      dictionaries: raw
          .map((e) => DictionaryInfo.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}