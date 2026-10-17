import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/dictionary_info.dart';
import '../providers/providers.dart';
import '../services/dictionary_registry_service.dart';

/// Licences and attribution.
///
/// Two parts. The static notices cover the software this app is built on and
/// do not change. The dictionary notices are per installed index and are read
/// from the registry, because the data is derived from Wiktionary, which is
/// share-alike: every dictionary the user downloads carries its own attribution
/// requirement, so a list fixed at build time would be wrong.
class CreditsScreen extends ConsumerWidget {
  const CreditsScreen({super.key});

  static const List<(String, String, String)> _staticNotices = [
    (
      'llama.cpp',
      'MIT License',
      'Local inference engine. Bundled and auto-downloaded builds come from '
          'ggml-org/llama.cpp.',
    ),
    (
      'Gemma 4',
      'Apache License 2.0',
      'Language and vision model. Downloaded on first run; not redistributed '
          'with this app.',
    ),
    (
      'Flutter',
      'BSD 3-Clause License',
      'User interface framework, with the package licences listed in the '
          'generated NOTICES file.',
    ),
    (
      'OCR to Anki',
      'MIT License',
      'This application.',
    ),
  ];

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final registry = ref.watch(dictionaryRegistryProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Licences')),
      body: FutureBuilder<List<DictionaryInfo>>(
        future: _installed(registry),
        builder: (context, snapshot) {
          final installed = snapshot.data ?? const <DictionaryInfo>[];

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Text('Dictionaries', style: theme.textTheme.titleMedium),
              const SizedBox(height: 4),
              Text(
                installed.isEmpty
                    ? 'No offline dictionaries are installed. Definitions for '
                        'every word come from the local model.'
                    : 'Dictionary data is derived from Wiktionary and is '
                        'share-alike. Each installed dictionary is listed '
                        'below with its own attribution.',
                style: theme.textTheme.bodySmall,
              ),
              const SizedBox(height: 12),
              for (final d in installed) _dictionaryTile(theme, d),
              const Divider(height: 32),
              Text('Software', style: theme.textTheme.titleMedium),
              const SizedBox(height: 12),
              for (final (name, license, detail) in _staticNotices)
                _noticeTile(theme, name, license, detail),
            ],
          );
        },
      ),
    );
  }

  Widget _dictionaryTile(ThemeData theme, DictionaryInfo d) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(d.name, style: theme.textTheme.titleSmall),
            const SizedBox(height: 4),
            Text(d.license, style: theme.textTheme.labelMedium),
            const SizedBox(height: 6),
            Text(d.attribution, style: theme.textTheme.bodySmall),
            const SizedBox(height: 6),
            Text(
              '${d.entryCount} entries, ${d.sizeLabel}',
              style: theme.textTheme.bodySmall,
            ),
            if (d.licenseUrl.isNotEmpty) ...[
              const SizedBox(height: 4),
              SelectableText(
                d.licenseUrl,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.primary,
                ),
              ),
            ],
            if (d.sourceUrl.isNotEmpty)
              SelectableText(
                d.sourceUrl,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.primary,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _noticeTile(ThemeData theme, String name, String license, String detail) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(name, style: theme.textTheme.titleSmall),
            const SizedBox(height: 4),
            Text(license, style: theme.textTheme.labelMedium),
            const SizedBox(height: 6),
            Text(detail, style: theme.textTheme.bodySmall),
          ],
        ),
      ),
    );
  }

  /// Registry entries whose index is present on this device.
  ///
  /// Installation is derived from the filesystem rather than recorded in the
  /// database, so there is no state that can disagree with what is on disk.
  Future<List<DictionaryInfo>> _installed(
    DictionaryRegistryService registry,
  ) async {
    final all = await registry.listDictionaries();
    final out = <DictionaryInfo>[];
    for (final d in all) {
      if (await registry.isInstalled(d)) out.add(d);
    }
    return out;
  }
}