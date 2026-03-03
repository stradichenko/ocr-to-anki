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
          _TextFieldTile(
            label: 'Definition language',
            value: settings.definitionLanguage,
            onChanged: (v) =>
                notifier.update((s) => s..definitionLanguage = v),
          ),
          _TextFieldTile(
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
          _TextFieldTile(
            label: 'Default deck',
            value: settings.defaultDeck,
            onChanged: (v) => notifier.update((s) => s..defaultDeck = v),
          ),
          _TextFieldTile(
            label: 'Default note model',
            value: settings.defaultModel,
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
