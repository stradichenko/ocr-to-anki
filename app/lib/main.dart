import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/providers.dart';
import 'screens/screens.dart';

void main() {
  runApp(const ProviderScope(child: OcrToAnkiApp()));
}

class OcrToAnkiApp extends ConsumerStatefulWidget {
  const OcrToAnkiApp({super.key});

  @override
  ConsumerState<OcrToAnkiApp> createState() => _OcrToAnkiAppState();
}

class _OcrToAnkiAppState extends ConsumerState<OcrToAnkiApp>
    with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // On Android, proactively restart llama-server when the app is
    // foregrounded in case Doze mode killed it while backgrounded.
    if (Platform.isAndroid && state == AppLifecycleState.resumed) {
      final llama = ref.read(llamaCppAndroidProvider);
      llama.ensureServerRunning().catchError((_) {
        // Silently ignore — the next inference call will retry anyway.
      });
    }
  }

  @override
  Widget build(BuildContext context) {
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
      // ---- Python downloading (silent, automatic) ----
      ServerStatus.downloadingPython => Scaffold(
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
                    'Setting up — downloading Python runtime…',
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
      // ---- llama.cpp binary downloading (automatic) ----
      ServerStatus.downloadingLlama => Scaffold(
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
                    'Downloading vision engine (one-time setup)…',
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
      // ---- Download in progress (Python or models) ----
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
          body: SafeArea(
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 720),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Icon(Icons.error_outline,
                          size: 48, color: theme.colorScheme.error),
                      const SizedBox(height: 16),
                      Text(
                        'Backend failed to start',
                        style: theme.textTheme.titleLarge,
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 12),
                      Container(
                        constraints: const BoxConstraints(maxHeight: 200),
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.surfaceContainerHighest,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: SingleChildScrollView(
                          child: SelectableText(
                            startup.message,
                            style: theme.textTheme.bodyMedium,
                          ),
                        ),
                      ),
                      if (startup.technicalDetail != null) ...[
                        const SizedBox(height: 12),
                        Theme(
                          data: theme.copyWith(
                            dividerColor: Colors.transparent,
                          ),
                          child: ExpansionTile(
                            tilePadding:
                                const EdgeInsets.symmetric(horizontal: 8),
                            childrenPadding: EdgeInsets.zero,
                            title: Text(
                              'Technical details',
                              style: theme.textTheme.labelLarge,
                            ),
                            children: [
                              Container(
                                constraints:
                                    const BoxConstraints(maxHeight: 320),
                                padding: const EdgeInsets.all(12),
                                decoration: BoxDecoration(
                                  color: theme.colorScheme.surfaceContainerLow,
                                  borderRadius: BorderRadius.circular(8),
                                  border: Border.all(
                                    color: theme.colorScheme.outlineVariant,
                                  ),
                                ),
                                child: SingleChildScrollView(
                                  child: SelectableText(
                                    startup.technicalDetail!,
                                    style: theme.textTheme.bodySmall?.copyWith(
                                      fontFamily: 'monospace',
                                      fontFamilyFallback: const [
                                        'Courier',
                                        'monospace'
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                      const SizedBox(height: 16),
                      Wrap(
                        alignment: WrapAlignment.center,
                        spacing: 12,
                        runSpacing: 8,
                        children: [
                          FilledButton.icon(
                            onPressed: () => ref
                                .read(serverStartupProvider.notifier)
                                .retry(),
                            icon: const Icon(Icons.refresh),
                            label: const Text('Retry'),
                          ),
                          OutlinedButton.icon(
                            onPressed: () {
                              final diagnostics = _buildDiagnostics(startup);
                              Clipboard.setData(
                                  ClipboardData(text: diagnostics));
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content:
                                      Text('Diagnostics copied to clipboard'),
                                  duration: Duration(seconds: 2),
                                ),
                              );
                            },
                            icon: const Icon(Icons.copy),
                            label: const Text('Copy diagnostics'),
                          ),
                          OutlinedButton.icon(
                            onPressed: () {
                              Navigator.of(context).pushNamed('/settings');
                            },
                            icon: const Icon(Icons.settings),
                            label: const Text('Settings'),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ServerStatus.ready => const HomeScreen(),
    };
  }

  static String _megabytes(int bytes) =>
      (bytes / (1024 * 1024)).toStringAsFixed(1);

  /// Build a diagnostics blob the user can paste into a bug report.
  static String _buildDiagnostics(ServerStartupState startup) {
    final buf = StringBuffer()
      ..writeln('=== OCR-to-Anki diagnostics ===')
      ..writeln('Time:    ${DateTime.now().toIso8601String()}')
      ..writeln('OS:      ${Platform.operatingSystem} '
          '${Platform.operatingSystemVersion}')
      ..writeln('Locale:  ${Platform.localeName}')
      ..writeln('Status:  ${startup.status.name}')
      ..writeln()
      ..writeln('--- User-facing message ---')
      ..writeln(startup.message)
      ..writeln()
      ..writeln('--- Technical details ---')
      ..writeln(startup.technicalDetail ?? '(none captured)');
    return buf.toString();
  }
}
