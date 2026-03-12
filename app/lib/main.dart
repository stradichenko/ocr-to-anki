import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/providers.dart';
import 'screens/screens.dart';

void main() {
  runApp(const ProviderScope(child: OcrToAnkiApp()));
}

class OcrToAnkiApp extends ConsumerWidget {
  const OcrToAnkiApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final themeMode = ref.watch(themeModeProvider);
    final colorSeed = ref.watch(colorSeedProvider);

    return MaterialApp(
      title: 'OCR to Anki',
      debugShowCheckedModeBanner: false,
      themeMode: themeMode,
      theme: ThemeData(
        colorSchemeSeed: colorSeed,
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: colorSeed,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: const _ServerStartupGate(),
      routes: {
        '/processing': (_) => const ProcessingScreen(),
        '/review': (_) => const ReviewScreen(),
        '/settings': (_) => const SettingsScreen(),
        '/history': (_) => const HistoryScreen(),
      },
    );
  }
}

/// Shows a loading / error screen while the backend is booting.
/// Once the server is healthy, it shows the normal [HomeScreen].
class _ServerStartupGate extends ConsumerWidget {
  const _ServerStartupGate();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final startup = ref.watch(serverStartupProvider);
    final theme = Theme.of(context);

    return switch (startup.status) {
      ServerStatus.starting => Scaffold(
          body: Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const CircularProgressIndicator(),
                const SizedBox(height: 24),
                Text(
                  startup.message,
                  style: theme.textTheme.titleMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                Text(
                  'Initialising vision & language models…',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ),
      // ---- Model download prompt ----
      ServerStatus.modelsNeeded => Scaffold(
          body: Center(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.download_rounded,
                      size: 56, color: theme.colorScheme.primary),
                  const SizedBox(height: 20),
                  Text(
                    'Model download required',
                    style: theme.textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'OCR‑to‑Anki needs a ~3.2 GB language model '
                    '(Gemma 3 4B) to run.\n'
                    'The download is a one‑time setup and will be '
                    'saved locally.',
                    textAlign: TextAlign.center,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 24),
                  FilledButton.icon(
                    onPressed: () => ref
                        .read(serverStartupProvider.notifier)
                        .acceptDownload(),
                    icon: const Icon(Icons.download),
                    label: const Text('Download now'),
                  ),
                ],
              ),
            ),
          ),
        ),
      // ---- Download in progress ----
      ServerStatus.downloading => Scaffold(
          body: Center(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 48),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (startup.totalBytes > 0) ...[
                    LinearProgressIndicator(
                      value: startup.downloadProgress,
                      minHeight: 8,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      '${_megabytes(startup.downloadedBytes)} / '
                      '${_megabytes(startup.totalBytes)} MB',
                      style: theme.textTheme.titleMedium,
                    ),
                  ] else ...[
                    const LinearProgressIndicator(minHeight: 8),
                    const SizedBox(height: 16),
                  ],
                  const SizedBox(height: 8),
                  Text(
                    startup.message,
                    textAlign: TextAlign.center,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ServerStatus.error => Scaffold(
          body: Center(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.error_outline,
                      size: 48, color: theme.colorScheme.error),
                  const SizedBox(height: 16),
                  Text('Backend failed to start',
                      style: theme.textTheme.titleLarge),
                  const SizedBox(height: 12),
                  Container(
                    constraints: const BoxConstraints(maxHeight: 260),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: SingleChildScrollView(
                      child: SelectableText(
                        startup.message,
                        style: theme.textTheme.bodySmall?.copyWith(
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      FilledButton.icon(
                        onPressed: () =>
                            ref.read(serverStartupProvider.notifier).retry(),
                        icon: const Icon(Icons.refresh),
                        label: const Text('Retry'),
                      ),
                      const SizedBox(width: 12),
                      OutlinedButton.icon(
                        onPressed: () {
                          Clipboard.setData(
                              ClipboardData(text: startup.message));
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Error copied to clipboard'),
                              duration: Duration(seconds: 2),
                            ),
                          );
                        },
                        icon: const Icon(Icons.copy),
                        label: const Text('Copy'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ServerStatus.ready => const HomeScreen(),
    };
  }

  static String _megabytes(int bytes) =>
      (bytes / (1024 * 1024)).toStringAsFixed(1);
}
