import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;

import '../database/database.dart';
import '../models/models.dart';
import '../providers/providers.dart';
import '../services/model_registry_service.dart';
import '../services/system_channel.dart';
import '../utils/responsive.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final settings = ref.watch(settingsProvider);
    final notifier = ref.read(settingsProvider.notifier);

    final compact = isCompact(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: SafeArea(
        child: ListView(
          padding: EdgeInsets.symmetric(
            vertical: 8,
            horizontal: compact ? 4 : 0,
          ),
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

          // -- Color scheme --
          ListTile(
            title: const Text('Color Scheme'),
            subtitle: Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: kColorSchemes.entries.map((entry) {
                  final isSelected = settings.colorSchemeSeed == entry.key;
                  final color = resolveColorSeed(entry.key,
                      entry.key == 'custom' ? settings.customColorHex : '');
                  return _ColorSchemeChip(
                    label: entry.value,
                    color: entry.key == 'custom' && settings.customColorHex.isEmpty
                        ? null
                        : color,
                    selected: isSelected,
                    onTap: () {
                      notifier.update((s) => s..colorSchemeSeed = entry.key);
                    },
                  );
                }).toList(),
              ),
            ),
          ),

          // -- Custom color hex field (visible when 'custom' is selected) --
          if (settings.colorSchemeSeed == 'custom')
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: resolveColorSeed('custom', settings.customColorHex),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: Theme.of(context).colorScheme.outline,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: TextFormField(
                      initialValue: settings.customColorHex,
                      decoration: const InputDecoration(
                        labelText: 'Hex color (e.g. #FF5722)',
                        border: OutlineInputBorder(),
                        isDense: true,
                        prefixText: '#',
                      ),
                      onChanged: (v) {
                        final hex = v.replaceFirst('#', '');
                        notifier.update((s) => s..customColorHex = hex);
                      },
                    ),
                  ),
                ],
              ),
            ),

          // ---------------------------------------------------------------
          // Inference
          // ---------------------------------------------------------------
          if (!Platform.isAndroid) ...[
            _SectionHeader('Inference'),
            _TextFieldTile(
              label: 'Server URL',
              value: settings.serverUrl,
              onChanged: (v) => notifier.update((s) => s..serverUrl = v),
            ),
          ],

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
          _LanguagePicker(
            label: 'Term language',
            value: settings.termLanguage,
            includeAuto: true,
            onChanged: (v) =>
                notifier.update((s) => s..termLanguage = v),
          ),

          // ---------------------------------------------------------------
          // Anki
          // ---------------------------------------------------------------
          if (!Platform.isAndroid) ...[
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
          ],

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
          ListTile(
            title: const Text('Padding (px)'),
            subtitle: Slider(
              value: settings.padding.toDouble(),
              min: 0,
              max: 20,
              divisions: 20,
              label: settings.padding.toString(),
              onChanged: (v) =>
                  notifier.update((s) => s..padding = v.round()),
            ),
          ),
          ListTile(
            title: const Text('Merge distance (px)'),
            subtitle: Slider(
              value: settings.mergeDistance.toDouble(),
              min: 0,
              max: 50,
              divisions: 50,
              label: settings.mergeDistance.toString(),
              onChanged: (v) =>
                  notifier.update((s) => s..mergeDistance = v.round()),
            ),
          ),

          const SizedBox(height: 32),

          // ---------------------------------------------------------------
          // OCR Performance
          // ---------------------------------------------------------------
          _SectionHeader('OCR Performance'),
          if (!Platform.isAndroid) ...[
            ListTile(
              title: const Text('GPU acceleration'),
              subtitle: Text(
                switch (settings.gpuMode) {
                  'gpu' => 'Force GPU — use for discrete NVIDIA / AMD GPUs.',
                  'cpu' => 'Force CPU — slower but stable on all hardware.',
                  _ => 'Auto — GPU on Linux/macOS, CPU on Windows.',
                },
              ),
              trailing: DropdownButton<String>(
                value: settings.gpuMode,
                items: const [
                  DropdownMenuItem(value: 'auto', child: Text('Auto')),
                  DropdownMenuItem(value: 'gpu', child: Text('GPU')),
                  DropdownMenuItem(value: 'cpu', child: Text('CPU')),
                ],
                onChanged: (v) {
                  if (v == null) return;
                  notifier.update((s) => s..gpuMode = v);
                  // Notify backend to reinit with the new GPU mode.
                  http.post(
                    Uri.parse('${settings.serverUrl}/config/gpu'),
                    headers: {'Content-Type': 'application/json'},
                    body: jsonEncode({'mode': v}),
                  ).catchError((_) => http.Response('', 500));
                },
              ),
            ),
            SwitchListTile(
              title: const Text('Prefer discrete GPU'),
              subtitle: const Text(
                'When available, use a discrete GPU (e.g. NVIDIA, AMD, Arc) '
                'instead of the integrated one.',
              ),
              value: settings.preferDiscreteGpu,
              onChanged: (v) =>
                  notifier.update((s) => s..preferDiscreteGpu = v),
            ),
          ],
          if (Platform.isAndroid) ...[
            ListTile(
              title: const Text('GPU backend'),
              subtitle: Text(
                switch (settings.gpuMode) {
                  'vulkan' => 'Vulkan — GPU acceleration via Vulkan.',
                  'opencl' => 'OpenCL — GPU acceleration via OpenCL.',
                  'cpu' => 'CPU only — slower but most compatible.',
                  _ => 'Auto — use best available GPU backend.',
                },
              ),
              trailing: DropdownButton<String>(
                value: settings.gpuMode,
                items: const [
                  DropdownMenuItem(value: 'auto', child: Text('Auto')),
                  DropdownMenuItem(value: 'vulkan', child: Text('Vulkan')),
                  DropdownMenuItem(value: 'opencl', child: Text('OpenCL')),
                  DropdownMenuItem(value: 'cpu', child: Text('CPU')),
                ],
                onChanged: (v) {
                  if (v == null) return;
                  notifier.update((s) => s..gpuMode = v);
                },
              ),
            ),
            ListTile(
              title: const Text('GPU layers'),
              subtitle: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    settings.nGpuLayers >= 999
                        ? 'All layers on GPU'
                        : '${settings.nGpuLayers} layers',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                  Slider(
                    value: settings.nGpuLayers.toDouble(),
                    min: 0,
                    max: 999,
                    divisions: 999,
                    label: settings.nGpuLayers >= 999
                        ? 'All'
                        : '${settings.nGpuLayers}',
                    onChanged: (v) =>
                        notifier.update((s) => s..nGpuLayers = v.round()),
                  ),
                ],
              ),
            ),
          ],
          SwitchListTile(
            title: const Text('Parallel crop processing'),
            subtitle: const Text(
              'OCR each crop concurrently instead of stitching a montage. '
              'Faster with a discrete GPU; disable on iGPU-only systems.',
            ),
            value: settings.parallelCrops,
            onChanged: (v) =>
                notifier.update((s) => s..parallelCrops = v),
          ),
          ListTile(
            title: const Text('Montage max width (px)'),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  settings.montageMaxWidth == 0
                      ? 'No limit (original resolution)'
                      : '${settings.montageMaxWidth} px',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                Slider(
                  value: settings.montageMaxWidth.toDouble(),
                  min: 0,
                  max: 2048,
                  divisions: 16,
                  label: settings.montageMaxWidth == 0
                      ? 'Off'
                      : '${settings.montageMaxWidth}',
                  onChanged: (v) =>
                      notifier.update((s) => s..montageMaxWidth = v.round()),
                ),
              ],
            ),
          ),
          if (Platform.isAndroid)
            SwitchListTile(
              title: const Text('Compress large images'),
              subtitle: const Text(
                'Downscale images larger than 1 MB before OCR to reduce '
                'memory usage and speed up vision encoding.',
              ),
              value: settings.compressLargeImages,
              onChanged: (v) =>
                  notifier.update((s) => s..compressLargeImages = v),
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
          const _CacheStatsPanel(),
          const SizedBox(height: 8),
          const _CacheBrowseButton(),

          const SizedBox(height: 32),

          // ---------------------------------------------------------------
          // Updates
          // ---------------------------------------------------------------
          _SectionHeader('Updates'),
          SwitchListTile(
            title: const Text('Auto-check for updates'),
            subtitle: const Text('Check on startup if a new release is available'),
            value: settings.autoCheckUpdates,
            onChanged: (v) =>
                notifier.update((s) => s..autoCheckUpdates = v),
          ),
          const _UpdateCheckTile(),
          const SizedBox(height: 32),

          if (Platform.isAndroid) ...[
            _SectionHeader('Downloads'),
            SwitchListTile(
              title: const Text('WiFi-only downloads'),
              subtitle: const Text(
                'Only download the AI model when connected to WiFi '
                '(~3.2 GB, recommended).',
              ),
              value: settings.wifiOnlyDownloads,
              onChanged: (v) =>
                  notifier.update((s) => s..wifiOnlyDownloads = v),
            ),
            const SizedBox(height: 32),
            _SectionHeader('AI Model'),
            const _AiModelTile(),
            const SizedBox(height: 32),
            _SectionHeader('Battery & Background'),
            _BatteryOptimizationTile(),
            ListTile(
              leading: const Icon(Icons.help_outline),
              title: const Text('Device-specific settings'),
              subtitle: const Text(
                'Some Xiaomi/Samsung/OnePlus devices need extra steps to '
                'allow background OCR. Open the OS app info page.',
              ),
              trailing: const Icon(Icons.open_in_new),
              onTap: () => SystemChannel.openAppDetailsSettings(),
            ),
            const SizedBox(height: 32),
          ],

          const SizedBox(height: 32),

          // ---------------------------------------------------------------
          // Connection test
          // ---------------------------------------------------------------
          if (!Platform.isAndroid) ...[
            _SectionHeader('Connection'),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: _ConnectionTestButton(),
            ),
            const SizedBox(height: 32),
          ],
        ],
        ),
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

class _ColorSchemeChip extends StatelessWidget {
  const _ColorSchemeChip({
    required this.label,
    required this.selected,
    required this.onTap,
    this.color,
  });

  final String label;
  final Color? color;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final chipColor = color ?? theme.colorScheme.outline;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: selected
              ? chipColor.withValues(alpha: 0.2)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: selected ? chipColor : theme.colorScheme.outlineVariant,
            width: selected ? 2 : 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 14,
              height: 14,
              decoration: BoxDecoration(
                color: chipColor,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: theme.textTheme.labelMedium?.copyWith(
                fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
              ),
            ),
          ],
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
    this.includeAuto = false,
  });

  final String label;
  final String value;
  final ValueChanged<String> onChanged;

  /// When true, an "Auto-detect" entry is prepended to the language list.
  final bool includeAuto;

  @override
  Widget build(BuildContext context) {
    final langs = [
      if (includeAuto) 'auto',
      ...kSupportedLanguages,
    ];
    return ListTile(
      title: DropdownButtonFormField<String>(
        initialValue: langs.contains(value) ? value : langs.first,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        items: langs.map((lang) {
          return DropdownMenuItem(
            value: lang,
            child: Text(
              lang == 'auto'
                  ? 'Auto-detect'
                  : lang[0].toUpperCase() + lang.substring(1),
            ),
          );
        }).toList(),
        onChanged: (v) {
          if (v != null) onChanged(v);
        },
      ),
    );
  }
}

class _UpdateCheckTile extends ConsumerWidget {
  const _UpdateCheckTile();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final update = ref.watch(updateProvider);
    final notifier = ref.read(updateProvider.notifier);

    return ListTile(
      leading: const Icon(Icons.system_update),
      title: const Text('Check for updates now'),
      subtitle: switch (update.status) {
        UpdateStatus.checking => const Text('Checking...'),
        UpdateStatus.available => Text(
            'Version ${update.info!.latestVersion} is available',
            style: TextStyle(
              color: Theme.of(context).colorScheme.primary,
              fontWeight: FontWeight.w600,
            ),
          ),
        UpdateStatus.upToDate => Text(
            'You are on the latest version',
            style: TextStyle(
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
        UpdateStatus.downloading => Text(
            'Downloading... ${(update.downloadProgress * 100).toStringAsFixed(0)}%',
          ),
        UpdateStatus.error => Text(
            'Error: ${update.error ?? "Unknown error"}',
            style: TextStyle(
              color: Theme.of(context).colorScheme.error,
            ),
          ),
        _ => const Text('Tap to check for a newer release'),
      },
      trailing: switch (update.status) {
        UpdateStatus.checking => const SizedBox(
            width: 20,
            height: 20,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
        UpdateStatus.available => FilledButton(
            onPressed: () => _showUpdateDialog(context, ref),
            child: const Text('Update'),
          ),
        UpdateStatus.upToDate => Icon(
            Icons.check_circle,
            color: Theme.of(context).colorScheme.primary,
          ),
        _ => const Icon(Icons.chevron_right),
      },
      onTap: update.status == UpdateStatus.checking ||
              update.status == UpdateStatus.downloading
          ? null
          : () async {
              await notifier.check(force: true);
              if (!context.mounted) return;
              final status = ref.read(updateProvider).status;
              if (status == UpdateStatus.upToDate) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('You are on the latest version'),
                    duration: Duration(seconds: 2),
                  ),
                );
              } else if (status == UpdateStatus.error) {
                final error = ref.read(updateProvider).error ?? 'Unknown error';
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('Update check failed: $error'),
                    backgroundColor: Theme.of(context).colorScheme.error,
                    duration: const Duration(seconds: 4),
                  ),
                );
              }
            },
    );
  }

  void _showUpdateDialog(BuildContext context, WidgetRef ref) {
    final info = ref.read(updateProvider).info;
    if (info == null) return;

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Update Available'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('A new version of OCR to Anki is available.'),
            const SizedBox(height: 12),
            Text(
              'Current: ${info.currentVersion}',
              style: Theme.of(ctx).textTheme.bodySmall,
            ),
            Text(
              'Latest: ${info.latestVersion}',
              style: Theme.of(ctx).textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
            ),
            if (info.publishedAt.isNotEmpty)
              Text(
                'Released: ${info.publishedAt.substring(0, 10)}',
                style: Theme.of(ctx).textTheme.bodySmall,
              ),
            if (info.releaseNotes.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text(
                'Release Notes:',
                style: Theme.of(ctx).textTheme.titleSmall,
              ),
              const SizedBox(height: 4),
              Container(
                constraints: const BoxConstraints(maxHeight: 200),
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Theme.of(ctx).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: SingleChildScrollView(
                  child: Text(
                    info.releaseNotes,
                    style: Theme.of(ctx).textTheme.bodySmall,
                  ),
                ),
              ),
            ],
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              ref.read(updateProvider.notifier).skipVersion();
              Navigator.pop(ctx);
            },
            child: const Text('Skip this version'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Later'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.pop(ctx);
              ref.read(updateProvider.notifier).downloadAndApply();
            },
            child: const Text('Download & Install'),
          ),
        ],
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

    final service = ref.read(ankiExportServiceProvider);
    // Hook into status updates so the user sees launch progress.
    service.onStatusUpdate = (msg) {
      if (mounted) setState(() => _result = msg);
    };

    try {
      // checkConnection calls _invoke which auto-launches Anki on failure.
      await service.getDecks(); // validates full round-trip, not just ping
      if (mounted) {
        setState(() {
          _testing = false;
          _result = '[OK] AnkiConnect reachable';
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _testing = false;
          _result = '[FAIL] Cannot reach AnkiConnect – is Anki installed?';
        });
      }
    } finally {
      service.onStatusUpdate = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Enrichment cache management widgets
// ---------------------------------------------------------------------------

/// Shows total count + per-language-pair breakdown with clear buttons.
class _CacheStatsPanel extends ConsumerStatefulWidget {
  const _CacheStatsPanel();

  @override
  ConsumerState<_CacheStatsPanel> createState() => _CacheStatsPanelState();
}

class _CacheStatsPanelState extends ConsumerState<_CacheStatsPanel> {
  int? _total;
  List<CacheLanguagePairStat>? _langStats;

  @override
  void initState() {
    super.initState();
    _reload();
  }

  Future<void> _reload() async {
    final db = ref.read(databaseProvider);
    final total = await db.enrichmentCacheCount();
    final stats = await db.getCacheLanguagePairStats();
    if (mounted) setState(() { _total = total; _langStats = stats; });
  }

  Future<void> _clearAll() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear entire cache?'),
        content: const Text(
          'All cached word definitions will be deleted. '
          'Future enrichments will call the LLM again.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(ctx).colorScheme.error,
            ),
            child: const Text('Clear All'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    final db = ref.read(databaseProvider);
    final deleted = await db.clearEnrichmentCache();
    await _reload();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Cleared $deleted cached entries')),
      );
    }
  }

  Future<void> _clearLanguagePair(CacheLanguagePairStat stat) async {
    final label = '${_cap(stat.definitionLanguage)} / ${_cap(stat.examplesLanguage)}';
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Clear "$label" cache?'),
        content: Text(
          '${stat.count} cached entries for this language pair will be deleted.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(ctx).colorScheme.error,
            ),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    final db = ref.read(databaseProvider);
    final deleted = await db.clearCacheByLanguagePair(
      definitionLanguage: stat.definitionLanguage,
      examplesLanguage: stat.examplesLanguage,
    );
    await _reload();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Cleared $deleted entries for $label')),
      );
    }
  }

  String _cap(String s) => s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // -- Total + Clear All --
        ListTile(
          leading: const Icon(Icons.storage),
          title: Text(
            _total != null
                ? '$_total cached definition${_total == 1 ? '' : 's'}'
                : 'Loading…',
          ),
          subtitle: const Text(
            'Cached definitions are reused instantly, skipping the LLM',
          ),
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              IconButton(
                icon: const Icon(Icons.refresh, size: 18),
                tooltip: 'Refresh',
                onPressed: _reload,
              ),
              TextButton.icon(
                onPressed: (_total ?? 0) > 0 ? _clearAll : null,
                icon: const Icon(Icons.delete_sweep, size: 18),
                label: const Text('Clear All'),
              ),
            ],
          ),
        ),

        // -- Per-language breakdown --
        if (_langStats != null && _langStats!.isNotEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 4),
                Text('By language pair',
                    style: theme.textTheme.labelMedium?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    )),
                const SizedBox(height: 4),
                ..._langStats!.map((stat) {
                  final label =
                      '${_cap(stat.definitionLanguage)} → ${_cap(stat.examplesLanguage)}';
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2),
                    child: Row(
                      children: [
                        const Icon(Icons.translate, size: 16),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text('$label  (${stat.count})',
                              style: theme.textTheme.bodySmall),
                        ),
                        SizedBox(
                          height: 28,
                          child: TextButton(
                            onPressed: () => _clearLanguagePair(stat),
                            style: TextButton.styleFrom(
                              padding: const EdgeInsets.symmetric(horizontal: 8),
                              textStyle: theme.textTheme.labelSmall,
                            ),
                            child: const Text('Clear'),
                          ),
                        ),
                      ],
                    ),
                  );
                }),
                const SizedBox(height: 8),
              ],
            ),
          ),
      ],
    );
  }
}

/// Button that opens a full-screen dialog to browse / search / delete cached entries.
class _CacheBrowseButton extends ConsumerWidget {
  const _CacheBrowseButton();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: OutlinedButton.icon(
        onPressed: () {
          Navigator.of(context).push(
            MaterialPageRoute(builder: (_) => const _CacheBrowseScreen()),
          );
        },
        icon: const Icon(Icons.search, size: 18),
        label: const Text('Browse & manage cached entries'),
      ),
    );
  }
}

/// Full-screen page to browse, search, and delete individual cache entries.
class _CacheBrowseScreen extends ConsumerStatefulWidget {
  const _CacheBrowseScreen();

  @override
  ConsumerState<_CacheBrowseScreen> createState() =>
      _CacheBrowseScreenState();
}

class _CacheBrowseScreenState extends ConsumerState<_CacheBrowseScreen> {
  final _searchController = TextEditingController();
  List<EnrichmentCacheEntry> _entries = [];
  List<CacheLanguagePairStat>? _langPairs;
  String? _filterDefLang;
  String? _filterExLang;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadLangPairs();
    _search();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _loadLangPairs() async {
    final db = ref.read(databaseProvider);
    final pairs = await db.getCacheLanguagePairStats();
    if (mounted) setState(() => _langPairs = pairs);
  }

  Future<void> _search() async {
    setState(() => _loading = true);
    final db = ref.read(databaseProvider);
    final results = await db.listCachedEntries(
      search: _searchController.text,
      definitionLanguage: _filterDefLang,
      examplesLanguage: _filterExLang,
    );
    if (mounted) setState(() { _entries = results; _loading = false; });
  }

  Future<void> _deleteEntry(EnrichmentCacheEntry entry) async {
    final db = ref.read(databaseProvider);
    await db.deleteCacheEntry(
      word: entry.word,
      definitionLanguage: entry.definitionLanguage,
      examplesLanguage: entry.examplesLanguage,
    );
    _search(); // refresh
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Deleted "${entry.word}"'),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  String _cap(String s) => s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    // Build language pair filter chips
    final filterChips = <Widget>[];
    if (_langPairs != null) {
      filterChips.add(
        FilterChip(
          label: const Text('All'),
          selected: _filterDefLang == null && _filterExLang == null,
          onSelected: (_) {
            setState(() { _filterDefLang = null; _filterExLang = null; });
            _search();
          },
        ),
      );
      for (final pair in _langPairs!) {
        final selected = _filterDefLang == pair.definitionLanguage &&
            _filterExLang == pair.examplesLanguage;
        filterChips.add(
          FilterChip(
            label: Text(
              '${_cap(pair.definitionLanguage)}/${_cap(pair.examplesLanguage)} (${pair.count})',
            ),
            selected: selected,
            onSelected: (_) {
              setState(() {
                if (selected) {
                  _filterDefLang = null;
                  _filterExLang = null;
                } else {
                  _filterDefLang = pair.definitionLanguage;
                  _filterExLang = pair.examplesLanguage;
                }
              });
              _search();
            },
          ),
        );
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Enrichment Cache'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Center(
              child: Text(
                '${_entries.length} entries',
                style: theme.textTheme.labelLarge,
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // -- Search bar --
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: 'Search words…',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: _searchController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchController.clear();
                          _search();
                        },
                      )
                    : null,
                border: const OutlineInputBorder(),
                isDense: true,
              ),
              onChanged: (_) => _search(),
            ),
          ),

          // -- Language filter chips --
          if (filterChips.isNotEmpty)
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Wrap(
                spacing: 8,
                children: filterChips,
              ),
            ),

          const Divider(height: 1),

          // -- Entry list --
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _entries.isEmpty
                    ? Center(
                        child: Text(
                          'No cached entries found',
                          style: theme.textTheme.bodyLarge?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      )
                    : ListView.builder(
                        itemCount: _entries.length,
                        itemBuilder: (context, index) {
                          final entry = _entries[index];
                          return _CacheEntryTile(
                            entry: entry,
                            onDelete: () => _deleteEntry(entry),
                            onTap: () => _showEntryDetail(entry),
                          );
                        },
                      ),
          ),
        ],
      ),
    );
  }

  void _showEntryDetail(EnrichmentCacheEntry entry) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(entry.word),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              _DetailSection(
                label: 'Languages',
                text:
                    'Def: ${_cap(entry.definitionLanguage)}  •  Ex: ${_cap(entry.examplesLanguage)}',
              ),
              if (entry.warning.isNotEmpty)
                _DetailSection(label: 'Warning', text: entry.warning),
              _DetailSection(label: 'Definition', text: entry.definition),
              _DetailSection(label: 'Examples', text: entry.examples),
              _DetailSection(
                label: 'Cached at',
                text: entry.createdAt.toLocal().toString().split('.').first,
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Close'),
          ),
          TextButton.icon(
            onPressed: () {
              Navigator.pop(ctx);
              _deleteEntry(entry);
            },
            icon: Icon(Icons.delete_outline,
                size: 18, color: Theme.of(ctx).colorScheme.error),
            label: Text('Delete',
                style: TextStyle(color: Theme.of(ctx).colorScheme.error)),
          ),
        ],
      ),
    );
  }
}

class _DetailSection extends StatelessWidget {
  const _DetailSection({required this.label, required this.text});
  final String label;
  final String text;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: theme.textTheme.labelSmall?.copyWith(
                color: theme.colorScheme.primary,
              )),
          const SizedBox(height: 2),
          SelectableText(
            text.isEmpty ? '(empty)' : text,
            style: theme.textTheme.bodyMedium,
          ),
        ],
      ),
    );
  }
}

class _CacheEntryTile extends StatelessWidget {
  const _CacheEntryTile({
    required this.entry,
    required this.onDelete,
    required this.onTap,
  });

  final EnrichmentCacheEntry entry;
  final VoidCallback onDelete;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final preview = entry.definition.length > 80
        ? '${entry.definition.substring(0, 80)}…'
        : entry.definition;

    return ListTile(
      title: Row(
        children: [
          Expanded(
            child: Text(entry.word,
                style: const TextStyle(fontWeight: FontWeight.w600)),
          ),
          if (entry.warning.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(left: 4),
              child: Icon(Icons.warning_amber,
                  size: 16, color: theme.colorScheme.error),
            ),
        ],
      ),
      subtitle: Text(
        preview.isEmpty ? '(no definition)' : preview,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
      ),
      trailing: IconButton(
        icon: Icon(Icons.delete_outline,
            size: 20, color: theme.colorScheme.error),
        tooltip: 'Delete entry',
        onPressed: onDelete,
      ),
      onTap: onTap,
    );
  }
}

// ---------------------------------------------------------------------------
// AI Model tile and picker (Android only)
// ---------------------------------------------------------------------------

class _AiModelTile extends ConsumerWidget {
  const _AiModelTile();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final settings = ref.watch(settingsProvider);
    final registry = ref.watch(modelRegistryProvider);

    return FutureBuilder<List<ModelInfo>>(
      future: registry.listModels(),
      builder: (context, snapshot) {
        final models = snapshot.data;
        final active = models?.firstWhere(
          (m) => m.id == settings.activeModelId,
          orElse: () => models.isNotEmpty ? models.first :
              const ModelInfo(
                id: '',
                name: 'Loading…',
                description: '',
                modelUrl: '',
                mmprojUrl: '',
                modelFilename: '',
                mmprojFilename: '',
                modelSizeBytes: 0,
                mmprojSizeBytes: 0,
                sha256Model: '',
                sha256Mmproj: '',
                supportsVision: false,
                contextSize: 4096,
              ),
        );

        return ListTile(
          leading: const Icon(Icons.model_training),
          title: Text(active?.name ?? 'Loading…'),
          subtitle: Text(
            active?.description ?? '',
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          trailing: const Icon(Icons.chevron_right),
          onTap: () => _showModelPicker(context, ref),
        );
      },
    );
  }

  void _showModelPicker(BuildContext context, WidgetRef ref) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.4,
        maxChildSize: 0.9,
        expand: false,
        builder: (ctx, scrollCtrl) => _ModelPickerSheet(
          scrollController: scrollCtrl,
        ),
      ),
    );
  }
}

class _ModelPickerSheet extends ConsumerStatefulWidget {
  const _ModelPickerSheet({required this.scrollController});
  final ScrollController scrollController;

  @override
  ConsumerState<_ModelPickerSheet> createState() => _ModelPickerSheetState();
}

class _ModelPickerSheetState extends ConsumerState<_ModelPickerSheet> {
  bool _downloading = false;
  double? _downloadProgress;
  String? _downloadFile;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final settings = ref.watch(settingsProvider);
    final registry = ref.read(modelRegistryProvider);

    return FutureBuilder<List<ModelInfo>>(
      future: registry.listModels(),
      builder: (context, modelsSnapshot) {
        return FutureBuilder<int>(
          future: registry.totalDiskUsageBytes(),
          builder: (context, diskSnapshot) {
            final models = modelsSnapshot.data ?? [];
            final diskMb = diskSnapshot.data != null
                ? (diskSnapshot.data! / (1024 * 1024)).toStringAsFixed(0)
                : null;

            return Column(
              children: [
                // Drag handle
                Container(
                  margin: const EdgeInsets.only(top: 8),
                  width: 36,
                  height: 4,
                  decoration: BoxDecoration(
                    color: theme.colorScheme.outlineVariant,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(
                    'Select AI Model',
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                const Divider(height: 1),
                Expanded(
                  child: ListView.builder(
                    controller: widget.scrollController,
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: models.length + (diskMb != null ? 1 : 0),
                    itemBuilder: (ctx, i) {
                      if (i == models.length) {
                        return Padding(
                          padding: const EdgeInsets.all(16),
                          child: Text(
                            'Total disk usage: $diskMb MB',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        );
                      }
                      return _ModelCard(
                        model: models[i],
                        isActive: models[i].id == settings.activeModelId,
                        isDownloading: _downloading,
                        downloadProgress: _downloadProgress,
                        downloadFile: _downloadFile,
                        onSelect: () => _selectModel(models[i]),
                        onDownload: () => _downloadModel(models[i]),
                        onDelete: () => _deleteModel(models[i]),
                      );
                    },
                  ),
                ),
              ],
            );
          },
        );
      },
    );
  }

  Future<void> _selectModel(ModelInfo model) async {
    final notifier = ref.read(settingsProvider.notifier);
    final currentId = ref.read(settingsProvider).activeModelId;

    if (model.id == currentId) {
      Navigator.of(context).pop();
      return;
    }

    final registry = ref.read(modelRegistryProvider);
    final isDownloaded = await registry.isDownloaded(model);

    if (!isDownloaded) {
      if (!mounted) return;
      final proceed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Model not downloaded'),
          content: Text(
            '${model.name} is not on this device yet. '
            'Download it now (~${(model.totalSizeBytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB)?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Download'),
            ),
          ],
        ),
      );
      if (proceed != true) return;
      await _downloadModel(model);
      if (!mounted) return;
    }

    // Update active model in settings.
    await notifier.update((s) => s..activeModelId = model.id);

    // Restart the server with the new model.
    final llama = ref.read(llamaCppAndroidProvider);
    llama.setActiveModel(model);
    await llama.stopServer();
    await llama.startServer();

    if (mounted) Navigator.of(context).pop();
  }

  Future<void> _downloadModel(ModelInfo model) async {
    setState(() {
      _downloading = true;
      _downloadProgress = 0;
    });

    final downloader = ref.read(modelDownloadProvider);
    try {
      await downloader.downloadModel(
        model,
        onProgress: (downloaded, total, file) {
          if (!mounted) return;
          setState(() {
            _downloadProgress = total > 0 ? downloaded / total : null;
            _downloadFile = file;
          });
        },
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Download failed: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _downloading = false;
          _downloadProgress = null;
          _downloadFile = null;
        });
      }
    }
  }

  Future<void> _deleteModel(ModelInfo model) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Delete ${model.name}?'),
        content: const Text(
          'This will free up disk space. You can re-download it later.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(ctx).colorScheme.error,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    final registry = ref.read(modelRegistryProvider);
    await registry.deleteModel(model);
    if (mounted) setState(() {});
  }
}

class _ModelCard extends StatelessWidget {
  const _ModelCard({
    required this.model,
    required this.isActive,
    required this.isDownloading,
    this.downloadProgress,
    this.downloadFile,
    required this.onSelect,
    required this.onDownload,
    required this.onDelete,
  });

  final ModelInfo model;
  final bool isActive;
  final bool isDownloading;
  final double? downloadProgress;
  final String? downloadFile;
  final VoidCallback onSelect;
  final VoidCallback onDownload;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final sizeGb = (model.totalSizeBytes / (1024 * 1024 * 1024))
        .toStringAsFixed(1);

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      color: isActive
          ? theme.colorScheme.primaryContainer.withValues(alpha: 0.5)
          : null,
      child: InkWell(
        onTap: isDownloading ? null : onSelect,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      model.name,
                      style: theme.textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  if (isActive)
                    Chip(
                      label: const Text('Active'),
                      visualDensity: VisualDensity.compact,
                      backgroundColor: theme.colorScheme.primary,
                      labelStyle: TextStyle(
                        color: theme.colorScheme.onPrimary,
                        fontSize: 11,
                      ),
                      padding: EdgeInsets.zero,
                    ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                model.description,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                children: [
                  ...model.tags.map((tag) {
                    return Chip(
                      label: Text(tag),
                      visualDensity: VisualDensity.compact,
                      padding: EdgeInsets.zero,
                      labelStyle: const TextStyle(fontSize: 10),
                    );
                  }),
                  Chip(
                    label: Text('$sizeGb GB'),
                    visualDensity: VisualDensity.compact,
                    padding: EdgeInsets.zero,
                    labelStyle: const TextStyle(fontSize: 10),
                  ),
                  if (model.supportsVision)
                    Chip(
                      label: const Text('Vision'),
                      visualDensity: VisualDensity.compact,
                      padding: EdgeInsets.zero,
                      labelStyle: const TextStyle(fontSize: 10),
                    ),
                ],
              ),
              if (isDownloading) ...[
                const SizedBox(height: 12),
                LinearProgressIndicator(
                  value: downloadProgress,
                  minHeight: 6,
                  borderRadius: BorderRadius.circular(3),
                ),
                const SizedBox(height: 4),
                Text(
                  'Downloading ${downloadFile ?? '…'}',
                  style: theme.textTheme.bodySmall,
                ),
              ] else ...[
                const SizedBox(height: 8),
                FutureBuilder<bool>(
                  future: ModelRegistryService().isDownloaded(model),
                  builder: (context, snap) {
                    final downloaded = snap.data ?? false;
                    return Row(
                      children: [
                        if (downloaded) ...[
                          Icon(Icons.check_circle,
                              size: 16, color: theme.colorScheme.primary),
                          const SizedBox(width: 4),
                          Text('Downloaded',
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: theme.colorScheme.primary,
                              )),
                          const Spacer(),
                          if (!isActive)
                            TextButton(
                              onPressed: onDelete,
                              child: Text(
                                'Delete',
                                style: TextStyle(
                                  color: theme.colorScheme.error,
                                ),
                              ),
                            ),
                        ] else ...[
                          Icon(Icons.download,
                              size: 16,
                              color: theme.colorScheme.onSurfaceVariant),
                          const SizedBox(width: 4),
                          Text('Not downloaded',
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: theme.colorScheme.onSurfaceVariant,
                              )),
                          const Spacer(),
                          FilledButton(
                            onPressed: onDownload,
                            child: const Text('Download'),
                          ),
                        ],
                      ],
                    );
                  },
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Battery optimization tile (Android only)
// ---------------------------------------------------------------------------

/// Live status tile that shows whether the app is whitelisted from Doze /
/// battery optimisation, and on tap opens the system dialog to fix it.
/// Reflects state changes immediately on app resume thanks to the
/// invalidation in [didChangeAppLifecycleState].
class _BatteryOptimizationTile extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final asyncOk = ref.watch(batteryOptimizationDisabledProvider);

    return asyncOk.when(
      loading: () => const ListTile(
        leading: SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
        title: Text('Battery optimization'),
        subtitle: Text('Checking…'),
      ),
      error: (_, _) => ListTile(
        leading: Icon(Icons.error_outline, color: theme.colorScheme.error),
        title: const Text('Battery optimization'),
        subtitle: const Text('Could not read system state'),
      ),
      data: (ok) => ListTile(
        leading: Icon(
          ok ? Icons.check_circle : Icons.warning_amber_rounded,
          color: ok
              ? theme.colorScheme.primary
              : theme.colorScheme.error,
        ),
        title: const Text('Battery optimization'),
        subtitle: Text(
          ok
              ? 'Disabled — long OCR sessions are protected from Doze.'
              : 'Enabled — Android may interrupt long OCR sessions.',
        ),
        trailing: ok ? null : const Icon(Icons.chevron_right),
        onTap: ok
            ? null
            : () async {
                await SystemChannel.requestIgnoreBatteryOptimizations();
                // The state refreshes automatically on resume via
                // _OcrToAnkiAppState.didChangeAppLifecycleState.
              },
      ),
    );
  }
}
