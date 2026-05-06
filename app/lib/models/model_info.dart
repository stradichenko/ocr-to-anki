/// Metadata for a downloadable GGUF model.
class ModelInfo {
  const ModelInfo({
    required this.id,
    required this.name,
    required this.description,
    required this.modelUrl,
    required this.mmprojUrl,
    required this.modelFilename,
    required this.mmprojFilename,
    required this.modelSizeBytes,
    required this.mmprojSizeBytes,
    required this.sha256Model,
    required this.sha256Mmproj,
    required this.supportsVision,
    required this.contextSize,
    this.tags = const [],
  });

  final String id;
  final String name;
  final String description;

  /// Download URL for the main GGUF model file.
  final String modelUrl;

  /// Download URL for the mmproj (vision projector) file.
  final String mmprojUrl;

  /// Local filename for the model (stored in models dir).
  final String modelFilename;

  /// Local filename for the mmproj (stored in models dir).
  final String mmprojFilename;

  /// Expected size of the model file in bytes.
  final int modelSizeBytes;

  /// Expected size of the mmproj file in bytes.
  final int mmprojSizeBytes;

  /// SHA-256 of the model file.
  final String sha256Model;

  /// SHA-256 of the mmproj file.
  final String sha256Mmproj;

  /// Whether this model supports vision (has a mmproj file).
  final bool supportsVision;

  /// Context size (max tokens) this model supports.
  final int contextSize;

  /// Optional tags like 'recommended', 'vision', 'fast'.
  final List<String> tags;

  /// Total size of both files.
  int get totalSizeBytes => modelSizeBytes + mmprojSizeBytes;

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      id: json['id'] as String,
      name: json['name'] as String,
      description: json['description'] as String,
      modelUrl: json['modelUrl'] as String,
      mmprojUrl: json['mmprojUrl'] as String,
      modelFilename: json['modelFilename'] as String,
      mmprojFilename: json['mmprojFilename'] as String,
      modelSizeBytes: json['modelSizeBytes'] as int,
      mmprojSizeBytes: json['mmprojSizeBytes'] as int,
      sha256Model: json['sha256Model'] as String,
      sha256Mmproj: json['sha256Mmproj'] as String,
      supportsVision: json['supportsVision'] as bool,
      contextSize: json['contextSize'] as int,
      tags: (json['tags'] as List<dynamic>?)?.cast<String>() ?? const [],
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'description': description,
        'modelUrl': modelUrl,
        'mmprojUrl': mmprojUrl,
        'modelFilename': modelFilename,
        'mmprojFilename': mmprojFilename,
        'modelSizeBytes': modelSizeBytes,
        'mmprojSizeBytes': mmprojSizeBytes,
        'sha256Model': sha256Model,
        'sha256Mmproj': sha256Mmproj,
        'supportsVision': supportsVision,
        'contextSize': contextSize,
        'tags': tags,
      };
}

/// Root object in the model registry JSON asset.
class ModelRegistry {
  const ModelRegistry({required this.models});

  final List<ModelInfo> models;

  /// Look up a model by its unique ID.
  ModelInfo? byId(String id) {
    for (final m in models) {
      if (m.id == id) return m;
    }
    return null;
  }

  /// Default model to use when none is selected.
  ModelInfo get defaultModel => models.first;

  factory ModelRegistry.fromJson(Map<String, dynamic> json) {
    final list = (json['models'] as List<dynamic>)
        .map((e) => ModelInfo.fromJson(e as Map<String, dynamic>))
        .toList();
    return ModelRegistry(models: list);
  }
}
