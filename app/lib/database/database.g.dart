// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'database.dart';

// ignore_for_file: type=lint
class $ProcessingSessionsTable extends ProcessingSessions
    with TableInfo<$ProcessingSessionsTable, ProcessingSession> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $ProcessingSessionsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _imagePathMeta = const VerificationMeta(
    'imagePath',
  );
  @override
  late final GeneratedColumn<String> imagePath = GeneratedColumn<String>(
    'image_path',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _contextMeta = const VerificationMeta(
    'context',
  );
  @override
  late final GeneratedColumn<String> context = GeneratedColumn<String>(
    'context',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _highlightColorMeta = const VerificationMeta(
    'highlightColor',
  );
  @override
  late final GeneratedColumn<String> highlightColor = GeneratedColumn<String>(
    'highlight_color',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _ocrTextMeta = const VerificationMeta(
    'ocrText',
  );
  @override
  late final GeneratedColumn<String> ocrText = GeneratedColumn<String>(
    'ocr_text',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant(''),
  );
  static const VerificationMeta _ocrElapsedSMeta = const VerificationMeta(
    'ocrElapsedS',
  );
  @override
  late final GeneratedColumn<double> ocrElapsedS = GeneratedColumn<double>(
    'ocr_elapsed_s',
    aliasedName,
    false,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
    defaultValue: const Constant(0),
  );
  static const VerificationMeta _enrichElapsedSMeta = const VerificationMeta(
    'enrichElapsedS',
  );
  @override
  late final GeneratedColumn<double> enrichElapsedS = GeneratedColumn<double>(
    'enrich_elapsed_s',
    aliasedName,
    false,
    type: DriftSqlType.double,
    requiredDuringInsert: false,
    defaultValue: const Constant(0),
  );
  static const VerificationMeta _backendMeta = const VerificationMeta(
    'backend',
  );
  @override
  late final GeneratedColumn<String> backend = GeneratedColumn<String>(
    'backend',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant(''),
  );
  static const VerificationMeta _errorMeta = const VerificationMeta('error');
  @override
  late final GeneratedColumn<String> error = GeneratedColumn<String>(
    'error',
    aliasedName,
    true,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _createdAtMeta = const VerificationMeta(
    'createdAt',
  );
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
    'created_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    imagePath,
    context,
    highlightColor,
    ocrText,
    ocrElapsedS,
    enrichElapsedS,
    backend,
    error,
    createdAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'processing_sessions';
  @override
  VerificationContext validateIntegrity(
    Insertable<ProcessingSession> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('image_path')) {
      context.handle(
        _imagePathMeta,
        imagePath.isAcceptableOrUnknown(data['image_path']!, _imagePathMeta),
      );
    } else if (isInserting) {
      context.missing(_imagePathMeta);
    }
    if (data.containsKey('context')) {
      context.handle(
        _contextMeta,
        this.context.isAcceptableOrUnknown(data['context']!, _contextMeta),
      );
    } else if (isInserting) {
      context.missing(_contextMeta);
    }
    if (data.containsKey('highlight_color')) {
      context.handle(
        _highlightColorMeta,
        highlightColor.isAcceptableOrUnknown(
          data['highlight_color']!,
          _highlightColorMeta,
        ),
      );
    }
    if (data.containsKey('ocr_text')) {
      context.handle(
        _ocrTextMeta,
        ocrText.isAcceptableOrUnknown(data['ocr_text']!, _ocrTextMeta),
      );
    }
    if (data.containsKey('ocr_elapsed_s')) {
      context.handle(
        _ocrElapsedSMeta,
        ocrElapsedS.isAcceptableOrUnknown(
          data['ocr_elapsed_s']!,
          _ocrElapsedSMeta,
        ),
      );
    }
    if (data.containsKey('enrich_elapsed_s')) {
      context.handle(
        _enrichElapsedSMeta,
        enrichElapsedS.isAcceptableOrUnknown(
          data['enrich_elapsed_s']!,
          _enrichElapsedSMeta,
        ),
      );
    }
    if (data.containsKey('backend')) {
      context.handle(
        _backendMeta,
        backend.isAcceptableOrUnknown(data['backend']!, _backendMeta),
      );
    }
    if (data.containsKey('error')) {
      context.handle(
        _errorMeta,
        error.isAcceptableOrUnknown(data['error']!, _errorMeta),
      );
    }
    if (data.containsKey('created_at')) {
      context.handle(
        _createdAtMeta,
        createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  ProcessingSession map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return ProcessingSession(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      imagePath: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}image_path'],
      )!,
      context: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}context'],
      )!,
      highlightColor: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}highlight_color'],
      ),
      ocrText: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}ocr_text'],
      )!,
      ocrElapsedS: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}ocr_elapsed_s'],
      )!,
      enrichElapsedS: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}enrich_elapsed_s'],
      )!,
      backend: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}backend'],
      )!,
      error: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}error'],
      ),
      createdAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}created_at'],
      )!,
    );
  }

  @override
  $ProcessingSessionsTable createAlias(String alias) {
    return $ProcessingSessionsTable(attachedDatabase, alias);
  }
}

class ProcessingSession extends DataClass
    implements Insertable<ProcessingSession> {
  final int id;
  final String imagePath;
  final String context;
  final String? highlightColor;
  final String ocrText;
  final double ocrElapsedS;
  final double enrichElapsedS;
  final String backend;
  final String? error;
  final DateTime createdAt;
  const ProcessingSession({
    required this.id,
    required this.imagePath,
    required this.context,
    this.highlightColor,
    required this.ocrText,
    required this.ocrElapsedS,
    required this.enrichElapsedS,
    required this.backend,
    this.error,
    required this.createdAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['image_path'] = Variable<String>(imagePath);
    map['context'] = Variable<String>(context);
    if (!nullToAbsent || highlightColor != null) {
      map['highlight_color'] = Variable<String>(highlightColor);
    }
    map['ocr_text'] = Variable<String>(ocrText);
    map['ocr_elapsed_s'] = Variable<double>(ocrElapsedS);
    map['enrich_elapsed_s'] = Variable<double>(enrichElapsedS);
    map['backend'] = Variable<String>(backend);
    if (!nullToAbsent || error != null) {
      map['error'] = Variable<String>(error);
    }
    map['created_at'] = Variable<DateTime>(createdAt);
    return map;
  }

  ProcessingSessionsCompanion toCompanion(bool nullToAbsent) {
    return ProcessingSessionsCompanion(
      id: Value(id),
      imagePath: Value(imagePath),
      context: Value(context),
      highlightColor: highlightColor == null && nullToAbsent
          ? const Value.absent()
          : Value(highlightColor),
      ocrText: Value(ocrText),
      ocrElapsedS: Value(ocrElapsedS),
      enrichElapsedS: Value(enrichElapsedS),
      backend: Value(backend),
      error: error == null && nullToAbsent
          ? const Value.absent()
          : Value(error),
      createdAt: Value(createdAt),
    );
  }

  factory ProcessingSession.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return ProcessingSession(
      id: serializer.fromJson<int>(json['id']),
      imagePath: serializer.fromJson<String>(json['imagePath']),
      context: serializer.fromJson<String>(json['context']),
      highlightColor: serializer.fromJson<String?>(json['highlightColor']),
      ocrText: serializer.fromJson<String>(json['ocrText']),
      ocrElapsedS: serializer.fromJson<double>(json['ocrElapsedS']),
      enrichElapsedS: serializer.fromJson<double>(json['enrichElapsedS']),
      backend: serializer.fromJson<String>(json['backend']),
      error: serializer.fromJson<String?>(json['error']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'imagePath': serializer.toJson<String>(imagePath),
      'context': serializer.toJson<String>(context),
      'highlightColor': serializer.toJson<String?>(highlightColor),
      'ocrText': serializer.toJson<String>(ocrText),
      'ocrElapsedS': serializer.toJson<double>(ocrElapsedS),
      'enrichElapsedS': serializer.toJson<double>(enrichElapsedS),
      'backend': serializer.toJson<String>(backend),
      'error': serializer.toJson<String?>(error),
      'createdAt': serializer.toJson<DateTime>(createdAt),
    };
  }

  ProcessingSession copyWith({
    int? id,
    String? imagePath,
    String? context,
    Value<String?> highlightColor = const Value.absent(),
    String? ocrText,
    double? ocrElapsedS,
    double? enrichElapsedS,
    String? backend,
    Value<String?> error = const Value.absent(),
    DateTime? createdAt,
  }) => ProcessingSession(
    id: id ?? this.id,
    imagePath: imagePath ?? this.imagePath,
    context: context ?? this.context,
    highlightColor: highlightColor.present
        ? highlightColor.value
        : this.highlightColor,
    ocrText: ocrText ?? this.ocrText,
    ocrElapsedS: ocrElapsedS ?? this.ocrElapsedS,
    enrichElapsedS: enrichElapsedS ?? this.enrichElapsedS,
    backend: backend ?? this.backend,
    error: error.present ? error.value : this.error,
    createdAt: createdAt ?? this.createdAt,
  );
  ProcessingSession copyWithCompanion(ProcessingSessionsCompanion data) {
    return ProcessingSession(
      id: data.id.present ? data.id.value : this.id,
      imagePath: data.imagePath.present ? data.imagePath.value : this.imagePath,
      context: data.context.present ? data.context.value : this.context,
      highlightColor: data.highlightColor.present
          ? data.highlightColor.value
          : this.highlightColor,
      ocrText: data.ocrText.present ? data.ocrText.value : this.ocrText,
      ocrElapsedS: data.ocrElapsedS.present
          ? data.ocrElapsedS.value
          : this.ocrElapsedS,
      enrichElapsedS: data.enrichElapsedS.present
          ? data.enrichElapsedS.value
          : this.enrichElapsedS,
      backend: data.backend.present ? data.backend.value : this.backend,
      error: data.error.present ? data.error.value : this.error,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('ProcessingSession(')
          ..write('id: $id, ')
          ..write('imagePath: $imagePath, ')
          ..write('context: $context, ')
          ..write('highlightColor: $highlightColor, ')
          ..write('ocrText: $ocrText, ')
          ..write('ocrElapsedS: $ocrElapsedS, ')
          ..write('enrichElapsedS: $enrichElapsedS, ')
          ..write('backend: $backend, ')
          ..write('error: $error, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    imagePath,
    context,
    highlightColor,
    ocrText,
    ocrElapsedS,
    enrichElapsedS,
    backend,
    error,
    createdAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is ProcessingSession &&
          other.id == this.id &&
          other.imagePath == this.imagePath &&
          other.context == this.context &&
          other.highlightColor == this.highlightColor &&
          other.ocrText == this.ocrText &&
          other.ocrElapsedS == this.ocrElapsedS &&
          other.enrichElapsedS == this.enrichElapsedS &&
          other.backend == this.backend &&
          other.error == this.error &&
          other.createdAt == this.createdAt);
}

class ProcessingSessionsCompanion extends UpdateCompanion<ProcessingSession> {
  final Value<int> id;
  final Value<String> imagePath;
  final Value<String> context;
  final Value<String?> highlightColor;
  final Value<String> ocrText;
  final Value<double> ocrElapsedS;
  final Value<double> enrichElapsedS;
  final Value<String> backend;
  final Value<String?> error;
  final Value<DateTime> createdAt;
  const ProcessingSessionsCompanion({
    this.id = const Value.absent(),
    this.imagePath = const Value.absent(),
    this.context = const Value.absent(),
    this.highlightColor = const Value.absent(),
    this.ocrText = const Value.absent(),
    this.ocrElapsedS = const Value.absent(),
    this.enrichElapsedS = const Value.absent(),
    this.backend = const Value.absent(),
    this.error = const Value.absent(),
    this.createdAt = const Value.absent(),
  });
  ProcessingSessionsCompanion.insert({
    this.id = const Value.absent(),
    required String imagePath,
    required String context,
    this.highlightColor = const Value.absent(),
    this.ocrText = const Value.absent(),
    this.ocrElapsedS = const Value.absent(),
    this.enrichElapsedS = const Value.absent(),
    this.backend = const Value.absent(),
    this.error = const Value.absent(),
    this.createdAt = const Value.absent(),
  }) : imagePath = Value(imagePath),
       context = Value(context);
  static Insertable<ProcessingSession> custom({
    Expression<int>? id,
    Expression<String>? imagePath,
    Expression<String>? context,
    Expression<String>? highlightColor,
    Expression<String>? ocrText,
    Expression<double>? ocrElapsedS,
    Expression<double>? enrichElapsedS,
    Expression<String>? backend,
    Expression<String>? error,
    Expression<DateTime>? createdAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (imagePath != null) 'image_path': imagePath,
      if (context != null) 'context': context,
      if (highlightColor != null) 'highlight_color': highlightColor,
      if (ocrText != null) 'ocr_text': ocrText,
      if (ocrElapsedS != null) 'ocr_elapsed_s': ocrElapsedS,
      if (enrichElapsedS != null) 'enrich_elapsed_s': enrichElapsedS,
      if (backend != null) 'backend': backend,
      if (error != null) 'error': error,
      if (createdAt != null) 'created_at': createdAt,
    });
  }

  ProcessingSessionsCompanion copyWith({
    Value<int>? id,
    Value<String>? imagePath,
    Value<String>? context,
    Value<String?>? highlightColor,
    Value<String>? ocrText,
    Value<double>? ocrElapsedS,
    Value<double>? enrichElapsedS,
    Value<String>? backend,
    Value<String?>? error,
    Value<DateTime>? createdAt,
  }) {
    return ProcessingSessionsCompanion(
      id: id ?? this.id,
      imagePath: imagePath ?? this.imagePath,
      context: context ?? this.context,
      highlightColor: highlightColor ?? this.highlightColor,
      ocrText: ocrText ?? this.ocrText,
      ocrElapsedS: ocrElapsedS ?? this.ocrElapsedS,
      enrichElapsedS: enrichElapsedS ?? this.enrichElapsedS,
      backend: backend ?? this.backend,
      error: error ?? this.error,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (imagePath.present) {
      map['image_path'] = Variable<String>(imagePath.value);
    }
    if (context.present) {
      map['context'] = Variable<String>(context.value);
    }
    if (highlightColor.present) {
      map['highlight_color'] = Variable<String>(highlightColor.value);
    }
    if (ocrText.present) {
      map['ocr_text'] = Variable<String>(ocrText.value);
    }
    if (ocrElapsedS.present) {
      map['ocr_elapsed_s'] = Variable<double>(ocrElapsedS.value);
    }
    if (enrichElapsedS.present) {
      map['enrich_elapsed_s'] = Variable<double>(enrichElapsedS.value);
    }
    if (backend.present) {
      map['backend'] = Variable<String>(backend.value);
    }
    if (error.present) {
      map['error'] = Variable<String>(error.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('ProcessingSessionsCompanion(')
          ..write('id: $id, ')
          ..write('imagePath: $imagePath, ')
          ..write('context: $context, ')
          ..write('highlightColor: $highlightColor, ')
          ..write('ocrText: $ocrText, ')
          ..write('ocrElapsedS: $ocrElapsedS, ')
          ..write('enrichElapsedS: $enrichElapsedS, ')
          ..write('backend: $backend, ')
          ..write('error: $error, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }
}

class $WordEntriesTable extends WordEntries
    with TableInfo<$WordEntriesTable, WordEntry> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $WordEntriesTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _sessionIdMeta = const VerificationMeta(
    'sessionId',
  );
  @override
  late final GeneratedColumn<int> sessionId = GeneratedColumn<int>(
    'session_id',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES processing_sessions (id)',
    ),
  );
  static const VerificationMeta _wordMeta = const VerificationMeta('word');
  @override
  late final GeneratedColumn<String> word = GeneratedColumn<String>(
    'word',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _definitionMeta = const VerificationMeta(
    'definition',
  );
  @override
  late final GeneratedColumn<String> definition = GeneratedColumn<String>(
    'definition',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant(''),
  );
  static const VerificationMeta _examplesMeta = const VerificationMeta(
    'examples',
  );
  @override
  late final GeneratedColumn<String> examples = GeneratedColumn<String>(
    'examples',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: false,
    defaultValue: const Constant(''),
  );
  static const VerificationMeta _exportedMeta = const VerificationMeta(
    'exported',
  );
  @override
  late final GeneratedColumn<bool> exported = GeneratedColumn<bool>(
    'exported',
    aliasedName,
    false,
    type: DriftSqlType.bool,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'CHECK ("exported" IN (0, 1))',
    ),
    defaultValue: const Constant(false),
  );
  static const VerificationMeta _ankiNoteIdMeta = const VerificationMeta(
    'ankiNoteId',
  );
  @override
  late final GeneratedColumn<int> ankiNoteId = GeneratedColumn<int>(
    'anki_note_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
  );
  static const VerificationMeta _createdAtMeta = const VerificationMeta(
    'createdAt',
  );
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
    'created_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    sessionId,
    word,
    definition,
    examples,
    exported,
    ankiNoteId,
    createdAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'word_entries';
  @override
  VerificationContext validateIntegrity(
    Insertable<WordEntry> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('session_id')) {
      context.handle(
        _sessionIdMeta,
        sessionId.isAcceptableOrUnknown(data['session_id']!, _sessionIdMeta),
      );
    } else if (isInserting) {
      context.missing(_sessionIdMeta);
    }
    if (data.containsKey('word')) {
      context.handle(
        _wordMeta,
        word.isAcceptableOrUnknown(data['word']!, _wordMeta),
      );
    } else if (isInserting) {
      context.missing(_wordMeta);
    }
    if (data.containsKey('definition')) {
      context.handle(
        _definitionMeta,
        definition.isAcceptableOrUnknown(data['definition']!, _definitionMeta),
      );
    }
    if (data.containsKey('examples')) {
      context.handle(
        _examplesMeta,
        examples.isAcceptableOrUnknown(data['examples']!, _examplesMeta),
      );
    }
    if (data.containsKey('exported')) {
      context.handle(
        _exportedMeta,
        exported.isAcceptableOrUnknown(data['exported']!, _exportedMeta),
      );
    }
    if (data.containsKey('anki_note_id')) {
      context.handle(
        _ankiNoteIdMeta,
        ankiNoteId.isAcceptableOrUnknown(
          data['anki_note_id']!,
          _ankiNoteIdMeta,
        ),
      );
    }
    if (data.containsKey('created_at')) {
      context.handle(
        _createdAtMeta,
        createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  WordEntry map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return WordEntry(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      sessionId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}session_id'],
      )!,
      word: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}word'],
      )!,
      definition: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}definition'],
      )!,
      examples: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}examples'],
      )!,
      exported: attachedDatabase.typeMapping.read(
        DriftSqlType.bool,
        data['${effectivePrefix}exported'],
      )!,
      ankiNoteId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}anki_note_id'],
      ),
      createdAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}created_at'],
      )!,
    );
  }

  @override
  $WordEntriesTable createAlias(String alias) {
    return $WordEntriesTable(attachedDatabase, alias);
  }
}

class WordEntry extends DataClass implements Insertable<WordEntry> {
  final int id;
  final int sessionId;
  final String word;
  final String definition;
  final String examples;
  final bool exported;
  final int? ankiNoteId;
  final DateTime createdAt;
  const WordEntry({
    required this.id,
    required this.sessionId,
    required this.word,
    required this.definition,
    required this.examples,
    required this.exported,
    this.ankiNoteId,
    required this.createdAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['session_id'] = Variable<int>(sessionId);
    map['word'] = Variable<String>(word);
    map['definition'] = Variable<String>(definition);
    map['examples'] = Variable<String>(examples);
    map['exported'] = Variable<bool>(exported);
    if (!nullToAbsent || ankiNoteId != null) {
      map['anki_note_id'] = Variable<int>(ankiNoteId);
    }
    map['created_at'] = Variable<DateTime>(createdAt);
    return map;
  }

  WordEntriesCompanion toCompanion(bool nullToAbsent) {
    return WordEntriesCompanion(
      id: Value(id),
      sessionId: Value(sessionId),
      word: Value(word),
      definition: Value(definition),
      examples: Value(examples),
      exported: Value(exported),
      ankiNoteId: ankiNoteId == null && nullToAbsent
          ? const Value.absent()
          : Value(ankiNoteId),
      createdAt: Value(createdAt),
    );
  }

  factory WordEntry.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return WordEntry(
      id: serializer.fromJson<int>(json['id']),
      sessionId: serializer.fromJson<int>(json['sessionId']),
      word: serializer.fromJson<String>(json['word']),
      definition: serializer.fromJson<String>(json['definition']),
      examples: serializer.fromJson<String>(json['examples']),
      exported: serializer.fromJson<bool>(json['exported']),
      ankiNoteId: serializer.fromJson<int?>(json['ankiNoteId']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'sessionId': serializer.toJson<int>(sessionId),
      'word': serializer.toJson<String>(word),
      'definition': serializer.toJson<String>(definition),
      'examples': serializer.toJson<String>(examples),
      'exported': serializer.toJson<bool>(exported),
      'ankiNoteId': serializer.toJson<int?>(ankiNoteId),
      'createdAt': serializer.toJson<DateTime>(createdAt),
    };
  }

  WordEntry copyWith({
    int? id,
    int? sessionId,
    String? word,
    String? definition,
    String? examples,
    bool? exported,
    Value<int?> ankiNoteId = const Value.absent(),
    DateTime? createdAt,
  }) => WordEntry(
    id: id ?? this.id,
    sessionId: sessionId ?? this.sessionId,
    word: word ?? this.word,
    definition: definition ?? this.definition,
    examples: examples ?? this.examples,
    exported: exported ?? this.exported,
    ankiNoteId: ankiNoteId.present ? ankiNoteId.value : this.ankiNoteId,
    createdAt: createdAt ?? this.createdAt,
  );
  WordEntry copyWithCompanion(WordEntriesCompanion data) {
    return WordEntry(
      id: data.id.present ? data.id.value : this.id,
      sessionId: data.sessionId.present ? data.sessionId.value : this.sessionId,
      word: data.word.present ? data.word.value : this.word,
      definition: data.definition.present
          ? data.definition.value
          : this.definition,
      examples: data.examples.present ? data.examples.value : this.examples,
      exported: data.exported.present ? data.exported.value : this.exported,
      ankiNoteId: data.ankiNoteId.present
          ? data.ankiNoteId.value
          : this.ankiNoteId,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('WordEntry(')
          ..write('id: $id, ')
          ..write('sessionId: $sessionId, ')
          ..write('word: $word, ')
          ..write('definition: $definition, ')
          ..write('examples: $examples, ')
          ..write('exported: $exported, ')
          ..write('ankiNoteId: $ankiNoteId, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    sessionId,
    word,
    definition,
    examples,
    exported,
    ankiNoteId,
    createdAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is WordEntry &&
          other.id == this.id &&
          other.sessionId == this.sessionId &&
          other.word == this.word &&
          other.definition == this.definition &&
          other.examples == this.examples &&
          other.exported == this.exported &&
          other.ankiNoteId == this.ankiNoteId &&
          other.createdAt == this.createdAt);
}

class WordEntriesCompanion extends UpdateCompanion<WordEntry> {
  final Value<int> id;
  final Value<int> sessionId;
  final Value<String> word;
  final Value<String> definition;
  final Value<String> examples;
  final Value<bool> exported;
  final Value<int?> ankiNoteId;
  final Value<DateTime> createdAt;
  const WordEntriesCompanion({
    this.id = const Value.absent(),
    this.sessionId = const Value.absent(),
    this.word = const Value.absent(),
    this.definition = const Value.absent(),
    this.examples = const Value.absent(),
    this.exported = const Value.absent(),
    this.ankiNoteId = const Value.absent(),
    this.createdAt = const Value.absent(),
  });
  WordEntriesCompanion.insert({
    this.id = const Value.absent(),
    required int sessionId,
    required String word,
    this.definition = const Value.absent(),
    this.examples = const Value.absent(),
    this.exported = const Value.absent(),
    this.ankiNoteId = const Value.absent(),
    this.createdAt = const Value.absent(),
  }) : sessionId = Value(sessionId),
       word = Value(word);
  static Insertable<WordEntry> custom({
    Expression<int>? id,
    Expression<int>? sessionId,
    Expression<String>? word,
    Expression<String>? definition,
    Expression<String>? examples,
    Expression<bool>? exported,
    Expression<int>? ankiNoteId,
    Expression<DateTime>? createdAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (sessionId != null) 'session_id': sessionId,
      if (word != null) 'word': word,
      if (definition != null) 'definition': definition,
      if (examples != null) 'examples': examples,
      if (exported != null) 'exported': exported,
      if (ankiNoteId != null) 'anki_note_id': ankiNoteId,
      if (createdAt != null) 'created_at': createdAt,
    });
  }

  WordEntriesCompanion copyWith({
    Value<int>? id,
    Value<int>? sessionId,
    Value<String>? word,
    Value<String>? definition,
    Value<String>? examples,
    Value<bool>? exported,
    Value<int?>? ankiNoteId,
    Value<DateTime>? createdAt,
  }) {
    return WordEntriesCompanion(
      id: id ?? this.id,
      sessionId: sessionId ?? this.sessionId,
      word: word ?? this.word,
      definition: definition ?? this.definition,
      examples: examples ?? this.examples,
      exported: exported ?? this.exported,
      ankiNoteId: ankiNoteId ?? this.ankiNoteId,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (sessionId.present) {
      map['session_id'] = Variable<int>(sessionId.value);
    }
    if (word.present) {
      map['word'] = Variable<String>(word.value);
    }
    if (definition.present) {
      map['definition'] = Variable<String>(definition.value);
    }
    if (examples.present) {
      map['examples'] = Variable<String>(examples.value);
    }
    if (exported.present) {
      map['exported'] = Variable<bool>(exported.value);
    }
    if (ankiNoteId.present) {
      map['anki_note_id'] = Variable<int>(ankiNoteId.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('WordEntriesCompanion(')
          ..write('id: $id, ')
          ..write('sessionId: $sessionId, ')
          ..write('word: $word, ')
          ..write('definition: $definition, ')
          ..write('examples: $examples, ')
          ..write('exported: $exported, ')
          ..write('ankiNoteId: $ankiNoteId, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }
}

class $ExportLogsTable extends ExportLogs
    with TableInfo<$ExportLogsTable, ExportLog> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $ExportLogsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _sessionIdMeta = const VerificationMeta(
    'sessionId',
  );
  @override
  late final GeneratedColumn<int> sessionId = GeneratedColumn<int>(
    'session_id',
    aliasedName,
    true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'REFERENCES processing_sessions (id)',
    ),
  );
  static const VerificationMeta _methodMeta = const VerificationMeta('method');
  @override
  late final GeneratedColumn<String> method = GeneratedColumn<String>(
    'method',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _totalNotesMeta = const VerificationMeta(
    'totalNotes',
  );
  @override
  late final GeneratedColumn<int> totalNotes = GeneratedColumn<int>(
    'total_notes',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _successCountMeta = const VerificationMeta(
    'successCount',
  );
  @override
  late final GeneratedColumn<int> successCount = GeneratedColumn<int>(
    'success_count',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _failedCountMeta = const VerificationMeta(
    'failedCount',
  );
  @override
  late final GeneratedColumn<int> failedCount = GeneratedColumn<int>(
    'failed_count',
    aliasedName,
    false,
    type: DriftSqlType.int,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _targetDeckMeta = const VerificationMeta(
    'targetDeck',
  );
  @override
  late final GeneratedColumn<String> targetDeck = GeneratedColumn<String>(
    'target_deck',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _createdAtMeta = const VerificationMeta(
    'createdAt',
  );
  @override
  late final GeneratedColumn<DateTime> createdAt = GeneratedColumn<DateTime>(
    'created_at',
    aliasedName,
    false,
    type: DriftSqlType.dateTime,
    requiredDuringInsert: false,
    defaultValue: currentDateAndTime,
  );
  @override
  List<GeneratedColumn> get $columns => [
    id,
    sessionId,
    method,
    totalNotes,
    successCount,
    failedCount,
    targetDeck,
    createdAt,
  ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'export_logs';
  @override
  VerificationContext validateIntegrity(
    Insertable<ExportLog> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('session_id')) {
      context.handle(
        _sessionIdMeta,
        sessionId.isAcceptableOrUnknown(data['session_id']!, _sessionIdMeta),
      );
    }
    if (data.containsKey('method')) {
      context.handle(
        _methodMeta,
        method.isAcceptableOrUnknown(data['method']!, _methodMeta),
      );
    } else if (isInserting) {
      context.missing(_methodMeta);
    }
    if (data.containsKey('total_notes')) {
      context.handle(
        _totalNotesMeta,
        totalNotes.isAcceptableOrUnknown(data['total_notes']!, _totalNotesMeta),
      );
    } else if (isInserting) {
      context.missing(_totalNotesMeta);
    }
    if (data.containsKey('success_count')) {
      context.handle(
        _successCountMeta,
        successCount.isAcceptableOrUnknown(
          data['success_count']!,
          _successCountMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_successCountMeta);
    }
    if (data.containsKey('failed_count')) {
      context.handle(
        _failedCountMeta,
        failedCount.isAcceptableOrUnknown(
          data['failed_count']!,
          _failedCountMeta,
        ),
      );
    } else if (isInserting) {
      context.missing(_failedCountMeta);
    }
    if (data.containsKey('target_deck')) {
      context.handle(
        _targetDeckMeta,
        targetDeck.isAcceptableOrUnknown(data['target_deck']!, _targetDeckMeta),
      );
    } else if (isInserting) {
      context.missing(_targetDeckMeta);
    }
    if (data.containsKey('created_at')) {
      context.handle(
        _createdAtMeta,
        createdAt.isAcceptableOrUnknown(data['created_at']!, _createdAtMeta),
      );
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  ExportLog map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return ExportLog(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      sessionId: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}session_id'],
      ),
      method: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}method'],
      )!,
      totalNotes: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}total_notes'],
      )!,
      successCount: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}success_count'],
      )!,
      failedCount: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}failed_count'],
      )!,
      targetDeck: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}target_deck'],
      )!,
      createdAt: attachedDatabase.typeMapping.read(
        DriftSqlType.dateTime,
        data['${effectivePrefix}created_at'],
      )!,
    );
  }

  @override
  $ExportLogsTable createAlias(String alias) {
    return $ExportLogsTable(attachedDatabase, alias);
  }
}

class ExportLog extends DataClass implements Insertable<ExportLog> {
  final int id;
  final int? sessionId;
  final String method;
  final int totalNotes;
  final int successCount;
  final int failedCount;
  final String targetDeck;
  final DateTime createdAt;
  const ExportLog({
    required this.id,
    this.sessionId,
    required this.method,
    required this.totalNotes,
    required this.successCount,
    required this.failedCount,
    required this.targetDeck,
    required this.createdAt,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    if (!nullToAbsent || sessionId != null) {
      map['session_id'] = Variable<int>(sessionId);
    }
    map['method'] = Variable<String>(method);
    map['total_notes'] = Variable<int>(totalNotes);
    map['success_count'] = Variable<int>(successCount);
    map['failed_count'] = Variable<int>(failedCount);
    map['target_deck'] = Variable<String>(targetDeck);
    map['created_at'] = Variable<DateTime>(createdAt);
    return map;
  }

  ExportLogsCompanion toCompanion(bool nullToAbsent) {
    return ExportLogsCompanion(
      id: Value(id),
      sessionId: sessionId == null && nullToAbsent
          ? const Value.absent()
          : Value(sessionId),
      method: Value(method),
      totalNotes: Value(totalNotes),
      successCount: Value(successCount),
      failedCount: Value(failedCount),
      targetDeck: Value(targetDeck),
      createdAt: Value(createdAt),
    );
  }

  factory ExportLog.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return ExportLog(
      id: serializer.fromJson<int>(json['id']),
      sessionId: serializer.fromJson<int?>(json['sessionId']),
      method: serializer.fromJson<String>(json['method']),
      totalNotes: serializer.fromJson<int>(json['totalNotes']),
      successCount: serializer.fromJson<int>(json['successCount']),
      failedCount: serializer.fromJson<int>(json['failedCount']),
      targetDeck: serializer.fromJson<String>(json['targetDeck']),
      createdAt: serializer.fromJson<DateTime>(json['createdAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'sessionId': serializer.toJson<int?>(sessionId),
      'method': serializer.toJson<String>(method),
      'totalNotes': serializer.toJson<int>(totalNotes),
      'successCount': serializer.toJson<int>(successCount),
      'failedCount': serializer.toJson<int>(failedCount),
      'targetDeck': serializer.toJson<String>(targetDeck),
      'createdAt': serializer.toJson<DateTime>(createdAt),
    };
  }

  ExportLog copyWith({
    int? id,
    Value<int?> sessionId = const Value.absent(),
    String? method,
    int? totalNotes,
    int? successCount,
    int? failedCount,
    String? targetDeck,
    DateTime? createdAt,
  }) => ExportLog(
    id: id ?? this.id,
    sessionId: sessionId.present ? sessionId.value : this.sessionId,
    method: method ?? this.method,
    totalNotes: totalNotes ?? this.totalNotes,
    successCount: successCount ?? this.successCount,
    failedCount: failedCount ?? this.failedCount,
    targetDeck: targetDeck ?? this.targetDeck,
    createdAt: createdAt ?? this.createdAt,
  );
  ExportLog copyWithCompanion(ExportLogsCompanion data) {
    return ExportLog(
      id: data.id.present ? data.id.value : this.id,
      sessionId: data.sessionId.present ? data.sessionId.value : this.sessionId,
      method: data.method.present ? data.method.value : this.method,
      totalNotes: data.totalNotes.present
          ? data.totalNotes.value
          : this.totalNotes,
      successCount: data.successCount.present
          ? data.successCount.value
          : this.successCount,
      failedCount: data.failedCount.present
          ? data.failedCount.value
          : this.failedCount,
      targetDeck: data.targetDeck.present
          ? data.targetDeck.value
          : this.targetDeck,
      createdAt: data.createdAt.present ? data.createdAt.value : this.createdAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('ExportLog(')
          ..write('id: $id, ')
          ..write('sessionId: $sessionId, ')
          ..write('method: $method, ')
          ..write('totalNotes: $totalNotes, ')
          ..write('successCount: $successCount, ')
          ..write('failedCount: $failedCount, ')
          ..write('targetDeck: $targetDeck, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(
    id,
    sessionId,
    method,
    totalNotes,
    successCount,
    failedCount,
    targetDeck,
    createdAt,
  );
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is ExportLog &&
          other.id == this.id &&
          other.sessionId == this.sessionId &&
          other.method == this.method &&
          other.totalNotes == this.totalNotes &&
          other.successCount == this.successCount &&
          other.failedCount == this.failedCount &&
          other.targetDeck == this.targetDeck &&
          other.createdAt == this.createdAt);
}

class ExportLogsCompanion extends UpdateCompanion<ExportLog> {
  final Value<int> id;
  final Value<int?> sessionId;
  final Value<String> method;
  final Value<int> totalNotes;
  final Value<int> successCount;
  final Value<int> failedCount;
  final Value<String> targetDeck;
  final Value<DateTime> createdAt;
  const ExportLogsCompanion({
    this.id = const Value.absent(),
    this.sessionId = const Value.absent(),
    this.method = const Value.absent(),
    this.totalNotes = const Value.absent(),
    this.successCount = const Value.absent(),
    this.failedCount = const Value.absent(),
    this.targetDeck = const Value.absent(),
    this.createdAt = const Value.absent(),
  });
  ExportLogsCompanion.insert({
    this.id = const Value.absent(),
    this.sessionId = const Value.absent(),
    required String method,
    required int totalNotes,
    required int successCount,
    required int failedCount,
    required String targetDeck,
    this.createdAt = const Value.absent(),
  }) : method = Value(method),
       totalNotes = Value(totalNotes),
       successCount = Value(successCount),
       failedCount = Value(failedCount),
       targetDeck = Value(targetDeck);
  static Insertable<ExportLog> custom({
    Expression<int>? id,
    Expression<int>? sessionId,
    Expression<String>? method,
    Expression<int>? totalNotes,
    Expression<int>? successCount,
    Expression<int>? failedCount,
    Expression<String>? targetDeck,
    Expression<DateTime>? createdAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (sessionId != null) 'session_id': sessionId,
      if (method != null) 'method': method,
      if (totalNotes != null) 'total_notes': totalNotes,
      if (successCount != null) 'success_count': successCount,
      if (failedCount != null) 'failed_count': failedCount,
      if (targetDeck != null) 'target_deck': targetDeck,
      if (createdAt != null) 'created_at': createdAt,
    });
  }

  ExportLogsCompanion copyWith({
    Value<int>? id,
    Value<int?>? sessionId,
    Value<String>? method,
    Value<int>? totalNotes,
    Value<int>? successCount,
    Value<int>? failedCount,
    Value<String>? targetDeck,
    Value<DateTime>? createdAt,
  }) {
    return ExportLogsCompanion(
      id: id ?? this.id,
      sessionId: sessionId ?? this.sessionId,
      method: method ?? this.method,
      totalNotes: totalNotes ?? this.totalNotes,
      successCount: successCount ?? this.successCount,
      failedCount: failedCount ?? this.failedCount,
      targetDeck: targetDeck ?? this.targetDeck,
      createdAt: createdAt ?? this.createdAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (sessionId.present) {
      map['session_id'] = Variable<int>(sessionId.value);
    }
    if (method.present) {
      map['method'] = Variable<String>(method.value);
    }
    if (totalNotes.present) {
      map['total_notes'] = Variable<int>(totalNotes.value);
    }
    if (successCount.present) {
      map['success_count'] = Variable<int>(successCount.value);
    }
    if (failedCount.present) {
      map['failed_count'] = Variable<int>(failedCount.value);
    }
    if (targetDeck.present) {
      map['target_deck'] = Variable<String>(targetDeck.value);
    }
    if (createdAt.present) {
      map['created_at'] = Variable<DateTime>(createdAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('ExportLogsCompanion(')
          ..write('id: $id, ')
          ..write('sessionId: $sessionId, ')
          ..write('method: $method, ')
          ..write('totalNotes: $totalNotes, ')
          ..write('successCount: $successCount, ')
          ..write('failedCount: $failedCount, ')
          ..write('targetDeck: $targetDeck, ')
          ..write('createdAt: $createdAt')
          ..write(')'))
        .toString();
  }
}

class $SettingsEntriesTable extends SettingsEntries
    with TableInfo<$SettingsEntriesTable, SettingsEntry> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $SettingsEntriesTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _keyMeta = const VerificationMeta('key');
  @override
  late final GeneratedColumn<String> key = GeneratedColumn<String>(
    'key',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _valueMeta = const VerificationMeta('value');
  @override
  late final GeneratedColumn<String> value = GeneratedColumn<String>(
    'value',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  @override
  List<GeneratedColumn> get $columns => [key, value];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'settings_entries';
  @override
  VerificationContext validateIntegrity(
    Insertable<SettingsEntry> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('key')) {
      context.handle(
        _keyMeta,
        key.isAcceptableOrUnknown(data['key']!, _keyMeta),
      );
    } else if (isInserting) {
      context.missing(_keyMeta);
    }
    if (data.containsKey('value')) {
      context.handle(
        _valueMeta,
        value.isAcceptableOrUnknown(data['value']!, _valueMeta),
      );
    } else if (isInserting) {
      context.missing(_valueMeta);
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {key};
  @override
  SettingsEntry map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return SettingsEntry(
      key: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}key'],
      )!,
      value: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}value'],
      )!,
    );
  }

  @override
  $SettingsEntriesTable createAlias(String alias) {
    return $SettingsEntriesTable(attachedDatabase, alias);
  }
}

class SettingsEntry extends DataClass implements Insertable<SettingsEntry> {
  final String key;
  final String value;
  const SettingsEntry({required this.key, required this.value});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['key'] = Variable<String>(key);
    map['value'] = Variable<String>(value);
    return map;
  }

  SettingsEntriesCompanion toCompanion(bool nullToAbsent) {
    return SettingsEntriesCompanion(key: Value(key), value: Value(value));
  }

  factory SettingsEntry.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return SettingsEntry(
      key: serializer.fromJson<String>(json['key']),
      value: serializer.fromJson<String>(json['value']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'key': serializer.toJson<String>(key),
      'value': serializer.toJson<String>(value),
    };
  }

  SettingsEntry copyWith({String? key, String? value}) =>
      SettingsEntry(key: key ?? this.key, value: value ?? this.value);
  SettingsEntry copyWithCompanion(SettingsEntriesCompanion data) {
    return SettingsEntry(
      key: data.key.present ? data.key.value : this.key,
      value: data.value.present ? data.value.value : this.value,
    );
  }

  @override
  String toString() {
    return (StringBuffer('SettingsEntry(')
          ..write('key: $key, ')
          ..write('value: $value')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(key, value);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is SettingsEntry &&
          other.key == this.key &&
          other.value == this.value);
}

class SettingsEntriesCompanion extends UpdateCompanion<SettingsEntry> {
  final Value<String> key;
  final Value<String> value;
  final Value<int> rowid;
  const SettingsEntriesCompanion({
    this.key = const Value.absent(),
    this.value = const Value.absent(),
    this.rowid = const Value.absent(),
  });
  SettingsEntriesCompanion.insert({
    required String key,
    required String value,
    this.rowid = const Value.absent(),
  }) : key = Value(key),
       value = Value(value);
  static Insertable<SettingsEntry> custom({
    Expression<String>? key,
    Expression<String>? value,
    Expression<int>? rowid,
  }) {
    return RawValuesInsertable({
      if (key != null) 'key': key,
      if (value != null) 'value': value,
      if (rowid != null) 'rowid': rowid,
    });
  }

  SettingsEntriesCompanion copyWith({
    Value<String>? key,
    Value<String>? value,
    Value<int>? rowid,
  }) {
    return SettingsEntriesCompanion(
      key: key ?? this.key,
      value: value ?? this.value,
      rowid: rowid ?? this.rowid,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (key.present) {
      map['key'] = Variable<String>(key.value);
    }
    if (value.present) {
      map['value'] = Variable<String>(value.value);
    }
    if (rowid.present) {
      map['rowid'] = Variable<int>(rowid.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('SettingsEntriesCompanion(')
          ..write('key: $key, ')
          ..write('value: $value, ')
          ..write('rowid: $rowid')
          ..write(')'))
        .toString();
  }
}

abstract class _$AppDatabase extends GeneratedDatabase {
  _$AppDatabase(QueryExecutor e) : super(e);
  $AppDatabaseManager get managers => $AppDatabaseManager(this);
  late final $ProcessingSessionsTable processingSessions =
      $ProcessingSessionsTable(this);
  late final $WordEntriesTable wordEntries = $WordEntriesTable(this);
  late final $ExportLogsTable exportLogs = $ExportLogsTable(this);
  late final $SettingsEntriesTable settingsEntries = $SettingsEntriesTable(
    this,
  );
  @override
  Iterable<TableInfo<Table, Object?>> get allTables =>
      allSchemaEntities.whereType<TableInfo<Table, Object?>>();
  @override
  List<DatabaseSchemaEntity> get allSchemaEntities => [
    processingSessions,
    wordEntries,
    exportLogs,
    settingsEntries,
  ];
}

typedef $$ProcessingSessionsTableCreateCompanionBuilder =
    ProcessingSessionsCompanion Function({
      Value<int> id,
      required String imagePath,
      required String context,
      Value<String?> highlightColor,
      Value<String> ocrText,
      Value<double> ocrElapsedS,
      Value<double> enrichElapsedS,
      Value<String> backend,
      Value<String?> error,
      Value<DateTime> createdAt,
    });
typedef $$ProcessingSessionsTableUpdateCompanionBuilder =
    ProcessingSessionsCompanion Function({
      Value<int> id,
      Value<String> imagePath,
      Value<String> context,
      Value<String?> highlightColor,
      Value<String> ocrText,
      Value<double> ocrElapsedS,
      Value<double> enrichElapsedS,
      Value<String> backend,
      Value<String?> error,
      Value<DateTime> createdAt,
    });

final class $$ProcessingSessionsTableReferences
    extends
        BaseReferences<
          _$AppDatabase,
          $ProcessingSessionsTable,
          ProcessingSession
        > {
  $$ProcessingSessionsTableReferences(
    super.$_db,
    super.$_table,
    super.$_typedResult,
  );

  static MultiTypedResultKey<$WordEntriesTable, List<WordEntry>>
  _wordEntriesRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.wordEntries,
    aliasName: $_aliasNameGenerator(
      db.processingSessions.id,
      db.wordEntries.sessionId,
    ),
  );

  $$WordEntriesTableProcessedTableManager get wordEntriesRefs {
    final manager = $$WordEntriesTableTableManager(
      $_db,
      $_db.wordEntries,
    ).filter((f) => f.sessionId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_wordEntriesRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }

  static MultiTypedResultKey<$ExportLogsTable, List<ExportLog>>
  _exportLogsRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
    db.exportLogs,
    aliasName: $_aliasNameGenerator(
      db.processingSessions.id,
      db.exportLogs.sessionId,
    ),
  );

  $$ExportLogsTableProcessedTableManager get exportLogsRefs {
    final manager = $$ExportLogsTableTableManager(
      $_db,
      $_db.exportLogs,
    ).filter((f) => f.sessionId.id.sqlEquals($_itemColumn<int>('id')!));

    final cache = $_typedResult.readTableOrNull(_exportLogsRefsTable($_db));
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: cache),
    );
  }
}

class $$ProcessingSessionsTableFilterComposer
    extends Composer<_$AppDatabase, $ProcessingSessionsTable> {
  $$ProcessingSessionsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get imagePath => $composableBuilder(
    column: $table.imagePath,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get context => $composableBuilder(
    column: $table.context,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get highlightColor => $composableBuilder(
    column: $table.highlightColor,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get ocrText => $composableBuilder(
    column: $table.ocrText,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get ocrElapsedS => $composableBuilder(
    column: $table.ocrElapsedS,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get enrichElapsedS => $composableBuilder(
    column: $table.enrichElapsedS,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get backend => $composableBuilder(
    column: $table.backend,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get error => $composableBuilder(
    column: $table.error,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnFilters(column),
  );

  Expression<bool> wordEntriesRefs(
    Expression<bool> Function($$WordEntriesTableFilterComposer f) f,
  ) {
    final $$WordEntriesTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.wordEntries,
      getReferencedColumn: (t) => t.sessionId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$WordEntriesTableFilterComposer(
            $db: $db,
            $table: $db.wordEntries,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<bool> exportLogsRefs(
    Expression<bool> Function($$ExportLogsTableFilterComposer f) f,
  ) {
    final $$ExportLogsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.exportLogs,
      getReferencedColumn: (t) => t.sessionId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExportLogsTableFilterComposer(
            $db: $db,
            $table: $db.exportLogs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$ProcessingSessionsTableOrderingComposer
    extends Composer<_$AppDatabase, $ProcessingSessionsTable> {
  $$ProcessingSessionsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get imagePath => $composableBuilder(
    column: $table.imagePath,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get context => $composableBuilder(
    column: $table.context,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get highlightColor => $composableBuilder(
    column: $table.highlightColor,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get ocrText => $composableBuilder(
    column: $table.ocrText,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get ocrElapsedS => $composableBuilder(
    column: $table.ocrElapsedS,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get enrichElapsedS => $composableBuilder(
    column: $table.enrichElapsedS,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get backend => $composableBuilder(
    column: $table.backend,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get error => $composableBuilder(
    column: $table.error,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnOrderings(column),
  );
}

class $$ProcessingSessionsTableAnnotationComposer
    extends Composer<_$AppDatabase, $ProcessingSessionsTable> {
  $$ProcessingSessionsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get imagePath =>
      $composableBuilder(column: $table.imagePath, builder: (column) => column);

  GeneratedColumn<String> get context =>
      $composableBuilder(column: $table.context, builder: (column) => column);

  GeneratedColumn<String> get highlightColor => $composableBuilder(
    column: $table.highlightColor,
    builder: (column) => column,
  );

  GeneratedColumn<String> get ocrText =>
      $composableBuilder(column: $table.ocrText, builder: (column) => column);

  GeneratedColumn<double> get ocrElapsedS => $composableBuilder(
    column: $table.ocrElapsedS,
    builder: (column) => column,
  );

  GeneratedColumn<double> get enrichElapsedS => $composableBuilder(
    column: $table.enrichElapsedS,
    builder: (column) => column,
  );

  GeneratedColumn<String> get backend =>
      $composableBuilder(column: $table.backend, builder: (column) => column);

  GeneratedColumn<String> get error =>
      $composableBuilder(column: $table.error, builder: (column) => column);

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  Expression<T> wordEntriesRefs<T extends Object>(
    Expression<T> Function($$WordEntriesTableAnnotationComposer a) f,
  ) {
    final $$WordEntriesTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.wordEntries,
      getReferencedColumn: (t) => t.sessionId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$WordEntriesTableAnnotationComposer(
            $db: $db,
            $table: $db.wordEntries,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }

  Expression<T> exportLogsRefs<T extends Object>(
    Expression<T> Function($$ExportLogsTableAnnotationComposer a) f,
  ) {
    final $$ExportLogsTableAnnotationComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.id,
      referencedTable: $db.exportLogs,
      getReferencedColumn: (t) => t.sessionId,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ExportLogsTableAnnotationComposer(
            $db: $db,
            $table: $db.exportLogs,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return f(composer);
  }
}

class $$ProcessingSessionsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $ProcessingSessionsTable,
          ProcessingSession,
          $$ProcessingSessionsTableFilterComposer,
          $$ProcessingSessionsTableOrderingComposer,
          $$ProcessingSessionsTableAnnotationComposer,
          $$ProcessingSessionsTableCreateCompanionBuilder,
          $$ProcessingSessionsTableUpdateCompanionBuilder,
          (ProcessingSession, $$ProcessingSessionsTableReferences),
          ProcessingSession,
          PrefetchHooks Function({bool wordEntriesRefs, bool exportLogsRefs})
        > {
  $$ProcessingSessionsTableTableManager(
    _$AppDatabase db,
    $ProcessingSessionsTable table,
  ) : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$ProcessingSessionsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$ProcessingSessionsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$ProcessingSessionsTableAnnotationComposer(
                $db: db,
                $table: table,
              ),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<String> imagePath = const Value.absent(),
                Value<String> context = const Value.absent(),
                Value<String?> highlightColor = const Value.absent(),
                Value<String> ocrText = const Value.absent(),
                Value<double> ocrElapsedS = const Value.absent(),
                Value<double> enrichElapsedS = const Value.absent(),
                Value<String> backend = const Value.absent(),
                Value<String?> error = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => ProcessingSessionsCompanion(
                id: id,
                imagePath: imagePath,
                context: context,
                highlightColor: highlightColor,
                ocrText: ocrText,
                ocrElapsedS: ocrElapsedS,
                enrichElapsedS: enrichElapsedS,
                backend: backend,
                error: error,
                createdAt: createdAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required String imagePath,
                required String context,
                Value<String?> highlightColor = const Value.absent(),
                Value<String> ocrText = const Value.absent(),
                Value<double> ocrElapsedS = const Value.absent(),
                Value<double> enrichElapsedS = const Value.absent(),
                Value<String> backend = const Value.absent(),
                Value<String?> error = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => ProcessingSessionsCompanion.insert(
                id: id,
                imagePath: imagePath,
                context: context,
                highlightColor: highlightColor,
                ocrText: ocrText,
                ocrElapsedS: ocrElapsedS,
                enrichElapsedS: enrichElapsedS,
                backend: backend,
                error: error,
                createdAt: createdAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable(table),
                  $$ProcessingSessionsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback:
              ({wordEntriesRefs = false, exportLogsRefs = false}) {
                return PrefetchHooks(
                  db: db,
                  explicitlyWatchedTables: [
                    if (wordEntriesRefs) db.wordEntries,
                    if (exportLogsRefs) db.exportLogs,
                  ],
                  addJoins: null,
                  getPrefetchedDataCallback: (items) async {
                    return [
                      if (wordEntriesRefs)
                        await $_getPrefetchedData<
                          ProcessingSession,
                          $ProcessingSessionsTable,
                          WordEntry
                        >(
                          currentTable: table,
                          referencedTable: $$ProcessingSessionsTableReferences
                              ._wordEntriesRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$ProcessingSessionsTableReferences(
                                db,
                                table,
                                p0,
                              ).wordEntriesRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.sessionId == item.id,
                              ),
                          typedResults: items,
                        ),
                      if (exportLogsRefs)
                        await $_getPrefetchedData<
                          ProcessingSession,
                          $ProcessingSessionsTable,
                          ExportLog
                        >(
                          currentTable: table,
                          referencedTable: $$ProcessingSessionsTableReferences
                              ._exportLogsRefsTable(db),
                          managerFromTypedResult: (p0) =>
                              $$ProcessingSessionsTableReferences(
                                db,
                                table,
                                p0,
                              ).exportLogsRefs,
                          referencedItemsForCurrentItem:
                              (item, referencedItems) => referencedItems.where(
                                (e) => e.sessionId == item.id,
                              ),
                          typedResults: items,
                        ),
                    ];
                  },
                );
              },
        ),
      );
}

typedef $$ProcessingSessionsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $ProcessingSessionsTable,
      ProcessingSession,
      $$ProcessingSessionsTableFilterComposer,
      $$ProcessingSessionsTableOrderingComposer,
      $$ProcessingSessionsTableAnnotationComposer,
      $$ProcessingSessionsTableCreateCompanionBuilder,
      $$ProcessingSessionsTableUpdateCompanionBuilder,
      (ProcessingSession, $$ProcessingSessionsTableReferences),
      ProcessingSession,
      PrefetchHooks Function({bool wordEntriesRefs, bool exportLogsRefs})
    >;
typedef $$WordEntriesTableCreateCompanionBuilder =
    WordEntriesCompanion Function({
      Value<int> id,
      required int sessionId,
      required String word,
      Value<String> definition,
      Value<String> examples,
      Value<bool> exported,
      Value<int?> ankiNoteId,
      Value<DateTime> createdAt,
    });
typedef $$WordEntriesTableUpdateCompanionBuilder =
    WordEntriesCompanion Function({
      Value<int> id,
      Value<int> sessionId,
      Value<String> word,
      Value<String> definition,
      Value<String> examples,
      Value<bool> exported,
      Value<int?> ankiNoteId,
      Value<DateTime> createdAt,
    });

final class $$WordEntriesTableReferences
    extends BaseReferences<_$AppDatabase, $WordEntriesTable, WordEntry> {
  $$WordEntriesTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $ProcessingSessionsTable _sessionIdTable(_$AppDatabase db) =>
      db.processingSessions.createAlias(
        $_aliasNameGenerator(
          db.wordEntries.sessionId,
          db.processingSessions.id,
        ),
      );

  $$ProcessingSessionsTableProcessedTableManager get sessionId {
    final $_column = $_itemColumn<int>('session_id')!;

    final manager = $$ProcessingSessionsTableTableManager(
      $_db,
      $_db.processingSessions,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_sessionIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$WordEntriesTableFilterComposer
    extends Composer<_$AppDatabase, $WordEntriesTable> {
  $$WordEntriesTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get word => $composableBuilder(
    column: $table.word,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get definition => $composableBuilder(
    column: $table.definition,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get examples => $composableBuilder(
    column: $table.examples,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<bool> get exported => $composableBuilder(
    column: $table.exported,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get ankiNoteId => $composableBuilder(
    column: $table.ankiNoteId,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnFilters(column),
  );

  $$ProcessingSessionsTableFilterComposer get sessionId {
    final $$ProcessingSessionsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.sessionId,
      referencedTable: $db.processingSessions,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ProcessingSessionsTableFilterComposer(
            $db: $db,
            $table: $db.processingSessions,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$WordEntriesTableOrderingComposer
    extends Composer<_$AppDatabase, $WordEntriesTable> {
  $$WordEntriesTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get word => $composableBuilder(
    column: $table.word,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get definition => $composableBuilder(
    column: $table.definition,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get examples => $composableBuilder(
    column: $table.examples,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<bool> get exported => $composableBuilder(
    column: $table.exported,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get ankiNoteId => $composableBuilder(
    column: $table.ankiNoteId,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$ProcessingSessionsTableOrderingComposer get sessionId {
    final $$ProcessingSessionsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.sessionId,
      referencedTable: $db.processingSessions,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ProcessingSessionsTableOrderingComposer(
            $db: $db,
            $table: $db.processingSessions,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$WordEntriesTableAnnotationComposer
    extends Composer<_$AppDatabase, $WordEntriesTable> {
  $$WordEntriesTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get word =>
      $composableBuilder(column: $table.word, builder: (column) => column);

  GeneratedColumn<String> get definition => $composableBuilder(
    column: $table.definition,
    builder: (column) => column,
  );

  GeneratedColumn<String> get examples =>
      $composableBuilder(column: $table.examples, builder: (column) => column);

  GeneratedColumn<bool> get exported =>
      $composableBuilder(column: $table.exported, builder: (column) => column);

  GeneratedColumn<int> get ankiNoteId => $composableBuilder(
    column: $table.ankiNoteId,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  $$ProcessingSessionsTableAnnotationComposer get sessionId {
    final $$ProcessingSessionsTableAnnotationComposer composer =
        $composerBuilder(
          composer: this,
          getCurrentColumn: (t) => t.sessionId,
          referencedTable: $db.processingSessions,
          getReferencedColumn: (t) => t.id,
          builder:
              (
                joinBuilder, {
                $addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer,
              }) => $$ProcessingSessionsTableAnnotationComposer(
                $db: $db,
                $table: $db.processingSessions,
                $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
                joinBuilder: joinBuilder,
                $removeJoinBuilderFromRootComposer:
                    $removeJoinBuilderFromRootComposer,
              ),
        );
    return composer;
  }
}

class $$WordEntriesTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $WordEntriesTable,
          WordEntry,
          $$WordEntriesTableFilterComposer,
          $$WordEntriesTableOrderingComposer,
          $$WordEntriesTableAnnotationComposer,
          $$WordEntriesTableCreateCompanionBuilder,
          $$WordEntriesTableUpdateCompanionBuilder,
          (WordEntry, $$WordEntriesTableReferences),
          WordEntry,
          PrefetchHooks Function({bool sessionId})
        > {
  $$WordEntriesTableTableManager(_$AppDatabase db, $WordEntriesTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$WordEntriesTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$WordEntriesTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$WordEntriesTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int> sessionId = const Value.absent(),
                Value<String> word = const Value.absent(),
                Value<String> definition = const Value.absent(),
                Value<String> examples = const Value.absent(),
                Value<bool> exported = const Value.absent(),
                Value<int?> ankiNoteId = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => WordEntriesCompanion(
                id: id,
                sessionId: sessionId,
                word: word,
                definition: definition,
                examples: examples,
                exported: exported,
                ankiNoteId: ankiNoteId,
                createdAt: createdAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required int sessionId,
                required String word,
                Value<String> definition = const Value.absent(),
                Value<String> examples = const Value.absent(),
                Value<bool> exported = const Value.absent(),
                Value<int?> ankiNoteId = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => WordEntriesCompanion.insert(
                id: id,
                sessionId: sessionId,
                word: word,
                definition: definition,
                examples: examples,
                exported: exported,
                ankiNoteId: ankiNoteId,
                createdAt: createdAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable(table),
                  $$WordEntriesTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({sessionId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (sessionId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.sessionId,
                                referencedTable: $$WordEntriesTableReferences
                                    ._sessionIdTable(db),
                                referencedColumn: $$WordEntriesTableReferences
                                    ._sessionIdTable(db)
                                    .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ),
      );
}

typedef $$WordEntriesTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $WordEntriesTable,
      WordEntry,
      $$WordEntriesTableFilterComposer,
      $$WordEntriesTableOrderingComposer,
      $$WordEntriesTableAnnotationComposer,
      $$WordEntriesTableCreateCompanionBuilder,
      $$WordEntriesTableUpdateCompanionBuilder,
      (WordEntry, $$WordEntriesTableReferences),
      WordEntry,
      PrefetchHooks Function({bool sessionId})
    >;
typedef $$ExportLogsTableCreateCompanionBuilder =
    ExportLogsCompanion Function({
      Value<int> id,
      Value<int?> sessionId,
      required String method,
      required int totalNotes,
      required int successCount,
      required int failedCount,
      required String targetDeck,
      Value<DateTime> createdAt,
    });
typedef $$ExportLogsTableUpdateCompanionBuilder =
    ExportLogsCompanion Function({
      Value<int> id,
      Value<int?> sessionId,
      Value<String> method,
      Value<int> totalNotes,
      Value<int> successCount,
      Value<int> failedCount,
      Value<String> targetDeck,
      Value<DateTime> createdAt,
    });

final class $$ExportLogsTableReferences
    extends BaseReferences<_$AppDatabase, $ExportLogsTable, ExportLog> {
  $$ExportLogsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $ProcessingSessionsTable _sessionIdTable(_$AppDatabase db) =>
      db.processingSessions.createAlias(
        $_aliasNameGenerator(db.exportLogs.sessionId, db.processingSessions.id),
      );

  $$ProcessingSessionsTableProcessedTableManager? get sessionId {
    final $_column = $_itemColumn<int>('session_id');
    if ($_column == null) return null;
    final manager = $$ProcessingSessionsTableTableManager(
      $_db,
      $_db.processingSessions,
    ).filter((f) => f.id.sqlEquals($_column));
    final item = $_typedResult.readTableOrNull(_sessionIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
      manager.$state.copyWith(prefetchedData: [item]),
    );
  }
}

class $$ExportLogsTableFilterComposer
    extends Composer<_$AppDatabase, $ExportLogsTable> {
  $$ExportLogsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get method => $composableBuilder(
    column: $table.method,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get totalNotes => $composableBuilder(
    column: $table.totalNotes,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get successCount => $composableBuilder(
    column: $table.successCount,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<int> get failedCount => $composableBuilder(
    column: $table.failedCount,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get targetDeck => $composableBuilder(
    column: $table.targetDeck,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnFilters(column),
  );

  $$ProcessingSessionsTableFilterComposer get sessionId {
    final $$ProcessingSessionsTableFilterComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.sessionId,
      referencedTable: $db.processingSessions,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ProcessingSessionsTableFilterComposer(
            $db: $db,
            $table: $db.processingSessions,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ExportLogsTableOrderingComposer
    extends Composer<_$AppDatabase, $ExportLogsTable> {
  $$ExportLogsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get method => $composableBuilder(
    column: $table.method,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get totalNotes => $composableBuilder(
    column: $table.totalNotes,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get successCount => $composableBuilder(
    column: $table.successCount,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<int> get failedCount => $composableBuilder(
    column: $table.failedCount,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get targetDeck => $composableBuilder(
    column: $table.targetDeck,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<DateTime> get createdAt => $composableBuilder(
    column: $table.createdAt,
    builder: (column) => ColumnOrderings(column),
  );

  $$ProcessingSessionsTableOrderingComposer get sessionId {
    final $$ProcessingSessionsTableOrderingComposer composer = $composerBuilder(
      composer: this,
      getCurrentColumn: (t) => t.sessionId,
      referencedTable: $db.processingSessions,
      getReferencedColumn: (t) => t.id,
      builder:
          (
            joinBuilder, {
            $addJoinBuilderToRootComposer,
            $removeJoinBuilderFromRootComposer,
          }) => $$ProcessingSessionsTableOrderingComposer(
            $db: $db,
            $table: $db.processingSessions,
            $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
            joinBuilder: joinBuilder,
            $removeJoinBuilderFromRootComposer:
                $removeJoinBuilderFromRootComposer,
          ),
    );
    return composer;
  }
}

class $$ExportLogsTableAnnotationComposer
    extends Composer<_$AppDatabase, $ExportLogsTable> {
  $$ExportLogsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<String> get method =>
      $composableBuilder(column: $table.method, builder: (column) => column);

  GeneratedColumn<int> get totalNotes => $composableBuilder(
    column: $table.totalNotes,
    builder: (column) => column,
  );

  GeneratedColumn<int> get successCount => $composableBuilder(
    column: $table.successCount,
    builder: (column) => column,
  );

  GeneratedColumn<int> get failedCount => $composableBuilder(
    column: $table.failedCount,
    builder: (column) => column,
  );

  GeneratedColumn<String> get targetDeck => $composableBuilder(
    column: $table.targetDeck,
    builder: (column) => column,
  );

  GeneratedColumn<DateTime> get createdAt =>
      $composableBuilder(column: $table.createdAt, builder: (column) => column);

  $$ProcessingSessionsTableAnnotationComposer get sessionId {
    final $$ProcessingSessionsTableAnnotationComposer composer =
        $composerBuilder(
          composer: this,
          getCurrentColumn: (t) => t.sessionId,
          referencedTable: $db.processingSessions,
          getReferencedColumn: (t) => t.id,
          builder:
              (
                joinBuilder, {
                $addJoinBuilderToRootComposer,
                $removeJoinBuilderFromRootComposer,
              }) => $$ProcessingSessionsTableAnnotationComposer(
                $db: $db,
                $table: $db.processingSessions,
                $addJoinBuilderToRootComposer: $addJoinBuilderToRootComposer,
                joinBuilder: joinBuilder,
                $removeJoinBuilderFromRootComposer:
                    $removeJoinBuilderFromRootComposer,
              ),
        );
    return composer;
  }
}

class $$ExportLogsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $ExportLogsTable,
          ExportLog,
          $$ExportLogsTableFilterComposer,
          $$ExportLogsTableOrderingComposer,
          $$ExportLogsTableAnnotationComposer,
          $$ExportLogsTableCreateCompanionBuilder,
          $$ExportLogsTableUpdateCompanionBuilder,
          (ExportLog, $$ExportLogsTableReferences),
          ExportLog,
          PrefetchHooks Function({bool sessionId})
        > {
  $$ExportLogsTableTableManager(_$AppDatabase db, $ExportLogsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$ExportLogsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$ExportLogsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$ExportLogsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int?> sessionId = const Value.absent(),
                Value<String> method = const Value.absent(),
                Value<int> totalNotes = const Value.absent(),
                Value<int> successCount = const Value.absent(),
                Value<int> failedCount = const Value.absent(),
                Value<String> targetDeck = const Value.absent(),
                Value<DateTime> createdAt = const Value.absent(),
              }) => ExportLogsCompanion(
                id: id,
                sessionId: sessionId,
                method: method,
                totalNotes: totalNotes,
                successCount: successCount,
                failedCount: failedCount,
                targetDeck: targetDeck,
                createdAt: createdAt,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<int?> sessionId = const Value.absent(),
                required String method,
                required int totalNotes,
                required int successCount,
                required int failedCount,
                required String targetDeck,
                Value<DateTime> createdAt = const Value.absent(),
              }) => ExportLogsCompanion.insert(
                id: id,
                sessionId: sessionId,
                method: method,
                totalNotes: totalNotes,
                successCount: successCount,
                failedCount: failedCount,
                targetDeck: targetDeck,
                createdAt: createdAt,
              ),
          withReferenceMapper: (p0) => p0
              .map(
                (e) => (
                  e.readTable(table),
                  $$ExportLogsTableReferences(db, table, e),
                ),
              )
              .toList(),
          prefetchHooksCallback: ({sessionId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins:
                  <
                    T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic
                    >
                  >(state) {
                    if (sessionId) {
                      state =
                          state.withJoin(
                                currentTable: table,
                                currentColumn: table.sessionId,
                                referencedTable: $$ExportLogsTableReferences
                                    ._sessionIdTable(db),
                                referencedColumn: $$ExportLogsTableReferences
                                    ._sessionIdTable(db)
                                    .id,
                              )
                              as T;
                    }

                    return state;
                  },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ),
      );
}

typedef $$ExportLogsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $ExportLogsTable,
      ExportLog,
      $$ExportLogsTableFilterComposer,
      $$ExportLogsTableOrderingComposer,
      $$ExportLogsTableAnnotationComposer,
      $$ExportLogsTableCreateCompanionBuilder,
      $$ExportLogsTableUpdateCompanionBuilder,
      (ExportLog, $$ExportLogsTableReferences),
      ExportLog,
      PrefetchHooks Function({bool sessionId})
    >;
typedef $$SettingsEntriesTableCreateCompanionBuilder =
    SettingsEntriesCompanion Function({
      required String key,
      required String value,
      Value<int> rowid,
    });
typedef $$SettingsEntriesTableUpdateCompanionBuilder =
    SettingsEntriesCompanion Function({
      Value<String> key,
      Value<String> value,
      Value<int> rowid,
    });

class $$SettingsEntriesTableFilterComposer
    extends Composer<_$AppDatabase, $SettingsEntriesTable> {
  $$SettingsEntriesTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<String> get key => $composableBuilder(
    column: $table.key,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get value => $composableBuilder(
    column: $table.value,
    builder: (column) => ColumnFilters(column),
  );
}

class $$SettingsEntriesTableOrderingComposer
    extends Composer<_$AppDatabase, $SettingsEntriesTable> {
  $$SettingsEntriesTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<String> get key => $composableBuilder(
    column: $table.key,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get value => $composableBuilder(
    column: $table.value,
    builder: (column) => ColumnOrderings(column),
  );
}

class $$SettingsEntriesTableAnnotationComposer
    extends Composer<_$AppDatabase, $SettingsEntriesTable> {
  $$SettingsEntriesTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<String> get key =>
      $composableBuilder(column: $table.key, builder: (column) => column);

  GeneratedColumn<String> get value =>
      $composableBuilder(column: $table.value, builder: (column) => column);
}

class $$SettingsEntriesTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $SettingsEntriesTable,
          SettingsEntry,
          $$SettingsEntriesTableFilterComposer,
          $$SettingsEntriesTableOrderingComposer,
          $$SettingsEntriesTableAnnotationComposer,
          $$SettingsEntriesTableCreateCompanionBuilder,
          $$SettingsEntriesTableUpdateCompanionBuilder,
          (
            SettingsEntry,
            BaseReferences<_$AppDatabase, $SettingsEntriesTable, SettingsEntry>,
          ),
          SettingsEntry,
          PrefetchHooks Function()
        > {
  $$SettingsEntriesTableTableManager(
    _$AppDatabase db,
    $SettingsEntriesTable table,
  ) : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$SettingsEntriesTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$SettingsEntriesTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$SettingsEntriesTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<String> key = const Value.absent(),
                Value<String> value = const Value.absent(),
                Value<int> rowid = const Value.absent(),
              }) => SettingsEntriesCompanion(
                key: key,
                value: value,
                rowid: rowid,
              ),
          createCompanionCallback:
              ({
                required String key,
                required String value,
                Value<int> rowid = const Value.absent(),
              }) => SettingsEntriesCompanion.insert(
                key: key,
                value: value,
                rowid: rowid,
              ),
          withReferenceMapper: (p0) => p0
              .map((e) => (e.readTable(table), BaseReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: null,
        ),
      );
}

typedef $$SettingsEntriesTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $SettingsEntriesTable,
      SettingsEntry,
      $$SettingsEntriesTableFilterComposer,
      $$SettingsEntriesTableOrderingComposer,
      $$SettingsEntriesTableAnnotationComposer,
      $$SettingsEntriesTableCreateCompanionBuilder,
      $$SettingsEntriesTableUpdateCompanionBuilder,
      (
        SettingsEntry,
        BaseReferences<_$AppDatabase, $SettingsEntriesTable, SettingsEntry>,
      ),
      SettingsEntry,
      PrefetchHooks Function()
    >;

class $AppDatabaseManager {
  final _$AppDatabase _db;
  $AppDatabaseManager(this._db);
  $$ProcessingSessionsTableTableManager get processingSessions =>
      $$ProcessingSessionsTableTableManager(_db, _db.processingSessions);
  $$WordEntriesTableTableManager get wordEntries =>
      $$WordEntriesTableTableManager(_db, _db.wordEntries);
  $$ExportLogsTableTableManager get exportLogs =>
      $$ExportLogsTableTableManager(_db, _db.exportLogs);
  $$SettingsEntriesTableTableManager get settingsEntries =>
      $$SettingsEntriesTableTableManager(_db, _db.settingsEntries);
}
