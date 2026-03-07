import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/models.dart';
import '../providers/providers.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final settings = ref.watch(settingsProvider);
    final notifier = ref.read(settingsProvider.notifier);

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.symmetric(vertical: 8),
        children: [
          // ---------------------------------------------------------------
          // Appearance
          // ---------------------------------------------------------------
          _SectionHeader('Appearance'),
          ListTile(
            title: const Text('Theme'),
            trailing: SegmentedButton<ThemeMode>(
              segments: const [
                ButtonSegment(
                  value: ThemeMode.light,
                  icon: Icon(Icons.light_mode, size: 18),
                  label: Text('Light'),
                ),
                ButtonSegment(
                  value: ThemeMode.dark,
                  icon: Icon(Icons.dark_mode, size: 18),
                  label: Text('Dark'),
                ),
                ButtonSegment(
                  value: ThemeMode.system,
                  icon: Icon(Icons.settings_brightness, size: 18),
                  label: Text('System'),
                ),
              ],
              selected: {settings.themeMode},
              onSelectionChanged: (v) => notifier.update(
                (s) => s..themeMode = v.first,
              ),
            ),
          ),

          // ---------------------------------------------------------------
          // Inference
          // ---------------------------------------------------------------
          _SectionHeader('Inference'),
          ListTile(
            title: const Text('Mode'),
            subtitle: Text(settings.inferenceMode == InferenceMode.remote
                ? 'Remote (FastAPI server)'
                : 'Embedded (on-device)'),
            trailing: SegmentedButton<InferenceMode>(
              segments: const [
                ButtonSegment(
                  value: InferenceMode.remote,
                  label: Text('Remote'),
                ),
                ButtonSegment(
                  value: InferenceMode.embedded,
                  label: Text('Local'),
                ),
              ],
              selected: {settings.inferenceMode},
              onSelectionChanged: (v) => notifier.update(
                (s) => s..inferenceMode = v.first,
              ),
            ),
          ),
          if (settings.inferenceMode == InferenceMode.remote)
            _TextFieldTile(
              label: 'Server URL',
              value: settings.serverUrl,
              onChanged: (v) => notifier.update((s) => s..serverUrl = v),
            ),

          // ---------------------------------------------------------------
          // Language
          // ---------------------------------------------------------------
          _SectionHeader('Languages'),
          _LanguagePicker(
            label: 'Definition language',
            value: settings.definitionLanguage,
            onChanged: (v) =>
                notifier.update((s) => s..definitionLanguage = v),
          ),
          _LanguagePicker(
            label: 'Examples language',
            value: settings.examplesLanguage,
            onChanged: (v) =>
                notifier.update((s) => s..examplesLanguage = v),
          ),

          // ---------------------------------------------------------------
          // Anki
          // ---------------------------------------------------------------
          _SectionHeader('Anki Export'),
          _TextFieldTile(
            label: 'AnkiConnect URL',
            value: settings.ankiConnectUrl,
            onChanged: (v) =>
                notifier.update((s) => s..ankiConnectUrl = v),
          ),
          _AnkiDropdownTile(
            label: 'Default deck',
            value: settings.defaultDeck,
            fetcher: () => ref.read(ankiExportServiceProvider).getDecks(),
            onChanged: (v) => notifier.update((s) => s..defaultDeck = v),
          ),
          _AnkiDropdownTile(
            label: 'Default note model',
            value: settings.defaultModel,
            fetcher: () => ref.read(ankiExportServiceProvider).getModels(),
            onChanged: (v) =>
                notifier.update((s) => s..defaultModel = v),
          ),
          SwitchListTile(
            title: const Text('Allow duplicates'),
            value: settings.allowDuplicates,
            onChanged: (v) =>
                notifier.update((s) => s..allowDuplicates = v),
          ),

          // ---------------------------------------------------------------
          // Highlight detection
          // ---------------------------------------------------------------
          _SectionHeader('Highlight Detection'),
          SwitchListTile(
            title: const Text('Adaptive mode'),
            subtitle: const Text('Auto-adjust for image lighting'),
            value: settings.adaptiveMode,
            onChanged: (v) =>
                notifier.update((s) => s..adaptiveMode = v),
          ),
          ListTile(
            title: const Text('Color tolerance'),
            subtitle: Slider(
              value: settings.colorTolerance.toDouble(),
              min: 10,
              max: 60,
              divisions: 50,
              label: settings.colorTolerance.toString(),
              onChanged: (v) =>
                  notifier.update((s) => s..colorTolerance = v.round()),
            ),
          ),
          ListTile(
            title: const Text('Minimum area (px)'),
            subtitle: Slider(
              value: settings.minArea.toDouble(),
              min: 50,
              max: 1000,
              divisions: 19,
              label: settings.minArea.toString(),
              onChanged: (v) =>
                  notifier.update((s) => s..minArea = v.round()),
            ),
          ),

          // ---------------------------------------------------------------
          // LLM parameters
          // ---------------------------------------------------------------
          _SectionHeader('LLM Parameters'),
          ListTile(
            title: const Text('Temperature'),
            subtitle: Slider(
              value: settings.temperature,
              min: 0.0,
              max: 2.0,
              divisions: 20,
              label: settings.temperature.toStringAsFixed(1),
              onChanged: (v) =>
                  notifier.update((s) => s..temperature = v),
            ),
          ),
          ListTile(
            title: const Text('Max tokens'),
            subtitle: Slider(
              value: settings.maxTokens.toDouble(),
              min: 32,
              max: 4096,
              divisions: 31,
              label: settings.maxTokens.toString(),
              onChanged: (v) =>
                  notifier.update((s) => s..maxTokens = v.round()),
            ),
          ),

          const SizedBox(height: 32),

          // ---------------------------------------------------------------
          // Enrichment Cache
          // ---------------------------------------------------------------
          _SectionHeader('Enrichment Cache'),
          _CacheTile(),

          // ---------------------------------------------------------------
          // Connection test
          // ---------------------------------------------------------------
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: _ConnectionTestButton(),
          ),

          const SizedBox(height: 32),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Reusable widgets
// ---------------------------------------------------------------------------

class _SectionHeader extends StatelessWidget {
  const _SectionHeader(this.title);
  final String title;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 24, 16, 4),
      child: Text(
        title,
        style: Theme.of(context).textTheme.titleSmall?.copyWith(
              color: Theme.of(context).colorScheme.primary,
            ),
      ),
    );
  }
}

class _TextFieldTile extends StatelessWidget {
  const _TextFieldTile({
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final String value;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: TextFormField(
        initialValue: value,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        onChanged: onChanged,
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Anki live-query dropdown (deck / model picker)
// ---------------------------------------------------------------------------

class _AnkiDropdownTile extends StatefulWidget {
  const _AnkiDropdownTile({
    required this.label,
    required this.value,
    required this.fetcher,
    required this.onChanged,
  });

  final String label;
  final String value;
  final Future<List<String>> Function() fetcher;
  final ValueChanged<String> onChanged;

  @override
  State<_AnkiDropdownTile> createState() => _AnkiDropdownTileState();
}

class _AnkiDropdownTileState extends State<_AnkiDropdownTile> {
  List<String>? _options;
  bool _loading = true;
  bool _fallback = false;

  @override
  void initState() {
    super.initState();
    _fetch();
  }

  Future<void> _fetch() async {
    try {
      final result = await widget.fetcher();
      if (mounted) {
        setState(() {
          _options = result;
          _loading = false;
          _fallback = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _loading = false;
          _fallback = true;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return ListTile(
        title: TextFormField(
          initialValue: widget.value,
          readOnly: true,
          decoration: InputDecoration(
            labelText: widget.label,
            border: const OutlineInputBorder(),
            isDense: true,
            suffixIcon: const SizedBox(
              width: 20,
              height: 20,
              child: Padding(
                padding: EdgeInsets.all(12),
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            ),
          ),
        ),
      );
    }

    if (_fallback || _options == null || _options!.isEmpty) {
      return ListTile(
        title: TextFormField(
          initialValue: widget.value,
          decoration: InputDecoration(
            labelText: widget.label,
            border: const OutlineInputBorder(),
            isDense: true,
            suffixIcon: IconButton(
              icon: const Icon(Icons.refresh, size: 18),
              tooltip: 'Retry fetching from AnkiConnect',
              onPressed: () {
                setState(() => _loading = true);
                _fetch();
              },
            ),
          ),
          onChanged: widget.onChanged,
        ),
      );
    }

    // Ensure current value is in the list so the dropdown doesn't break.
    final options = [..._options!];
    if (!options.contains(widget.value) && widget.value.isNotEmpty) {
      options.insert(0, widget.value);
    }

    return ListTile(
      title: DropdownButtonFormField<String>(
        initialValue: options.contains(widget.value) ? widget.value : null,
        decoration: InputDecoration(
          labelText: widget.label,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        items: options.map((o) {
          return DropdownMenuItem(value: o, child: Text(o));
        }).toList(),
        onChanged: (v) {
          if (v != null) widget.onChanged(v);
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Language picker (dropdown menu)
// ---------------------------------------------------------------------------

class _LanguagePicker extends StatelessWidget {
  const _LanguagePicker({
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final String value;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: DropdownButtonFormField<String>(
        initialValue: kSupportedLanguages.contains(value) ? value : kSupportedLanguages.first,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        items: kSupportedLanguages.map((lang) {
          return DropdownMenuItem(
            value: lang,
            child: Text(lang[0].toUpperCase() + lang.substring(1)),
          );
        }).toList(),
        onChanged: (v) {
          if (v != null) onChanged(v);
        },
      ),
    );
  }
}

class _ConnectionTestButton extends ConsumerStatefulWidget {
  @override
  ConsumerState<_ConnectionTestButton> createState() =>
      _ConnectionTestButtonState();
}

class _ConnectionTestButtonState
    extends ConsumerState<_ConnectionTestButton> {
  String? _result;
  bool _testing = false;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _testing ? null : _testInference,
                icon: const Icon(Icons.link),
                label: const Text('Test Inference'),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _testing ? null : _testAnki,
                icon: const Icon(Icons.link),
                label: const Text('Test AnkiConnect'),
              ),
            ),
          ],
        ),
        if (_result != null)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              _result!,
              style: Theme.of(context).textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ),
      ],
    );
  }

  Future<void> _testInference() async {
    setState(() {
      _testing = true;
      _result = null;
    });
    final ok = await ref.read(inferenceServiceProvider).isAvailable();
    setState(() {
      _testing = false;
      _result = ok ? '[OK] Inference backend reachable' : '[FAIL] Cannot reach inference backend';
    });
  }

  Future<void> _testAnki() async {
    setState(() {
      _testing = true;
      _result = null;
    });
    final ok = await ref.read(ankiExportServiceProvider).checkConnection();
    setState(() {
      _testing = false;
      _result = ok ? '[OK] AnkiConnect reachable' : '[FAIL] Cannot reach AnkiConnect';
    });
  }
}

// ---------------------------------------------------------------------------
// Enrichment cache management tile
// ---------------------------------------------------------------------------

class _CacheTile extends ConsumerStatefulWidget {
  @override
  ConsumerState<_CacheTile> createState() => _CacheTileState();
}

class _CacheTileState extends ConsumerState<_CacheTile> {
  int? _count;

  @override
  void initState() {
    super.initState();
    _loadCount();
  }

  Future<void> _loadCount() async {
    final db = ref.read(databaseProvider);
    final count = await db.enrichmentCacheCount();
    if (mounted) setState(() => _count = count);
  }

  Future<void> _clearCache() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear enrichment cache?'),
        content: const Text(
          'All cached word definitions will be deleted. '
          'Future enrichments will need to call the LLM again.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    final db = ref.read(databaseProvider);
    await db.clearEnrichmentCache();
    await _loadCount();

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Enrichment cache cleared')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(
        _count != null
            ? '$_count cached word definition(s)'
            : 'Loading cache info…',
      ),
      subtitle: const Text(
        'Cached definitions are reused instantly, skipping the LLM',
      ),
      trailing: TextButton.icon(
        onPressed: (_count ?? 0) > 0 ? _clearCache : null,
        icon: const Icon(Icons.delete_sweep, size: 18),
        label: const Text('Clear'),
      ),
    );
  }
}
