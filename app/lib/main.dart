import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/providers.dart';
import 'screens/screens.dart';
import 'services/foreground_task_service.dart';
import 'services/system_channel.dart';
import 'utils/responsive.dart';

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
    if (Platform.isAndroid) {
      // Eagerly construct the share-intent handler so a cold-launch
      // share intent is captured before HomeScreen mounts.
      ref.read(shareIntentHandlerProvider);
      // Enable edge-to-edge drawing behind status and navigation bars.
      SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!Platform.isAndroid) return;

    final llama = ref.read(llamaCppAndroidProvider);

    if (state == AppLifecycleState.resumed) {
      // Proactively restart llama-server when foregrounded in case Doze
      // mode killed it while backgrounded.
      llama.ensureServerRunning().catchError((_) {
        // Silently ignore — the next inference call will retry anyway.
      });
      // Refresh OS-level grant state — the user may have toggled
      // notifications or battery optimisation in system settings while
      // we were backgrounded.
      ref.invalidate(notificationsGrantedProvider);
      ref.invalidate(batteryOptimizationDisabledProvider);
    } else if (state == AppLifecycleState.paused) {
      // Keep the foreground service alive while the model server is running
      // so Android does not kill the process when the user backgrounds the
      // app mid-process.
      if (llama.isServerRunning) {
        ForegroundTaskService.start(detail: 'AI model ready').catchError((_) {});
      }
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

/// Navigation destinations for the adaptive layout.
const _kNavDestinations = [
  NavigationRailDestination(
    icon: Icon(Icons.home_outlined),
    selectedIcon: Icon(Icons.home),
    label: Text('Home'),
  ),
  NavigationRailDestination(
    icon: Icon(Icons.history_outlined),
    selectedIcon: Icon(Icons.history),
    label: Text('History'),
  ),
  NavigationRailDestination(
    icon: Icon(Icons.settings_outlined),
    selectedIcon: Icon(Icons.settings),
    label: Text('Settings'),
  ),
];

/// Adaptive scaffold for medium/expanded screens.
///
/// Shows a [NavigationRail] on the left and a detail pane on the right.
/// The detail pane is driven by [detailScreenProvider] when processing or
/// review is active, otherwise by the rail selection.
class _AdaptiveLayout extends ConsumerStatefulWidget {
  const _AdaptiveLayout();

  @override
  ConsumerState<_AdaptiveLayout> createState() => _AdaptiveLayoutState();
}

class _AdaptiveLayoutState extends ConsumerState<_AdaptiveLayout> {
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final detail = ref.watch(detailScreenProvider);

    Widget body;
    if (detail == DetailScreen.processing) {
      body = const ProcessingScreen();
    } else if (detail == DetailScreen.review) {
      body = const ReviewScreen();
    } else {
      body = switch (_selectedIndex) {
        0 => const HomeScreen(),
        1 => const HistoryScreen(),
        2 => const SettingsScreen(),
        _ => const HomeScreen(),
      };
    }

    return Scaffold(
      body: Row(
        children: [
          NavigationRail(
            backgroundColor: theme.colorScheme.surfaceContainerLow,
            selectedIndex:
                detail != DetailScreen.none ? null : _selectedIndex,
            onDestinationSelected: (index) {
              ref.read(detailScreenProvider.notifier).clear();
              setState(() => _selectedIndex = index);
            },
            destinations: _kNavDestinations,
            labelType: NavigationRailLabelType.all,
          ),
          const VerticalDivider(thickness: 1, width: 1),
          Expanded(child: body),
        ],
      ),
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

    // One-time Android system prompts at first transition to ready.
    // Fires the OS POST_NOTIFICATIONS dialog and the battery-optimisation
    // exemption dialog, persisting the asked flags so we never re-prompt
    // across launches.
    ref.listen<ServerStartupState>(serverStartupProvider, (prev, next) {
      if (!Platform.isAndroid) return;
      if (next.status != ServerStatus.ready) return;
      if (prev?.status == ServerStatus.ready) return;
      _maybePromptFirstRunPermissions(context, ref);
    });

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
      ServerStatus.ready =>
          useTwoPane(context) ? const _AdaptiveLayout() : const HomeScreen(),
    };
  }

  static String _megabytes(int bytes) =>
      (bytes / (1024 * 1024)).toStringAsFixed(1);

  /// Run the one-time Android first-launch system prompts.  Called from
  /// the build-time `ref.listen` on the first transition to
  /// [ServerStatus.ready].  Persists the "asked" flags so the OS dialogs
  /// never show twice across launches, regardless of the user's choice.
  ///
  /// The order is intentional: notifications first (cheap, no-op on
  /// non-Doze devices), battery-optimisation second (heavier, only if
  /// Android can actually kill us).
  static Future<void> _maybePromptFirstRunPermissions(
    BuildContext context,
    WidgetRef ref,
  ) async {
    final settings = ref.read(settingsProvider);

    // -- 1) POST_NOTIFICATIONS (Android 13+) --------------------------------
    if (!settings.notificationsPermissionAsked) {
      await SystemChannel.requestPostNotifications();
      await ref.read(settingsProvider.notifier).update((s) {
        s.notificationsPermissionAsked = true;
        return s;
      });
      ref.invalidate(notificationsGrantedProvider);
    }

    if (!context.mounted) return;

    // -- 2) Battery-optimisation whitelist ----------------------------------
    final updated = ref.read(settingsProvider);
    if (updated.batteryOptimizationPromptShown) return;

    final batteryOk = await SystemChannel.isBatteryOptimizationDisabled();
    if (!batteryOk && context.mounted) {
      await showDialog<void>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Keep OCR running in background?'),
          content: const Text(
            'Long OCR jobs can take a minute on a phone CPU. To prevent '
            'Android from interrupting them when the screen turns off, '
            'allow this app to skip battery optimisation.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Not now'),
            ),
            FilledButton(
              onPressed: () {
                Navigator.pop(ctx);
                SystemChannel.requestIgnoreBatteryOptimizations();
              },
              child: const Text('Allow'),
            ),
          ],
        ),
      );
    }
    await ref.read(settingsProvider.notifier).update((s) {
      s.batteryOptimizationPromptShown = true;
      return s;
    });
    ref.invalidate(batteryOptimizationDisabledProvider);
  }

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
