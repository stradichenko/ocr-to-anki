import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import '../models/model_info.dart';

/// Loads and queries the bundled model registry (`assets/models.json`).
class ModelRegistryService {
  ModelRegistryService();

  ModelRegistry? _registry;

  /// Load the model registry from the bundled JSON asset.
  Future<ModelRegistry> loadRegistry() async {
    if (_registry != null) return _registry!;
    final jsonString = await rootBundle.loadString('assets/models.json');
    final json = jsonDecode(jsonString) as Map<String, dynamic>;
    _registry = ModelRegistry.fromJson(json);
    return _registry!;
  }

  /// All models defined in the registry.
  Future<List<ModelInfo>> listModels() async {
    final r = await loadRegistry();
    return r.models;
  }

  /// Look up a model by ID.
  Future<ModelInfo?> getModel(String id) async {
    final r = await loadRegistry();
    return r.byId(id);
  }

  /// Default model (first in registry).
  Future<ModelInfo> getDefaultModel() async {
    final r = await loadRegistry();
    return r.defaultModel;
  }

  /// Directory where models are stored.
  Future<Directory> modelDir() async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}/llama_cpp/models');
    await dir.create(recursive: true);
    return dir;
  }

  /// Full path to a model file on disk.
  Future<String> modelPath(ModelInfo info) async {
    final dir = await modelDir();
    return '${dir.path}/${info.modelFilename}';
  }

  /// Full path to an mmproj file on disk.
  Future<String> mmprojPath(ModelInfo info) async {
    final dir = await modelDir();
    return '${dir.path}/${info.mmprojFilename}';
  }

  /// Whether both files for [info] exist on disk.
  Future<bool> isDownloaded(ModelInfo info) async {
    final model = File(await modelPath(info));
    final mmproj = File(await mmprojPath(info));
    return model.existsSync() && mmproj.existsSync();
  }

  /// Total bytes used by all downloaded model files.
  Future<int> totalDiskUsageBytes() async {
    final dir = await modelDir();
    if (!dir.existsSync()) return 0;
    var total = 0;
    for (final entity in dir.listSync()) {
      if (entity is File) {
        total += entity.lengthSync();
      }
    }
    return total;
  }

  /// Delete the files for [info] from disk.
  Future<void> deleteModel(ModelInfo info) async {
    final model = File(await modelPath(info));
    final mmproj = File(await mmprojPath(info));
    if (model.existsSync()) await model.delete();
    if (mmproj.existsSync()) await mmproj.delete();
  }
}
