import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/providers.dart';
import 'screens/screens.dart';
import 'services/foreground_task_service.dart';
import 'services/system_channel.dart';
import 'utils/responsive.dart';
import 'widgets/startup_overlay.dart';

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

/// Gate that shows the normal app UI immediately while the backend boots in
/// the background.  A small banner indicates init progress; the full overlay
/// is only shown when the user tries to process before init is complete.
class _ServerStartupGate extends ConsumerStatefulWidget {
  const _ServerStartupGate();

  @override
  ConsumerState<_ServerStartupGate> createState() => _ServerStartupGateState();
}

class _ServerStartupGateState extends ConsumerState<_ServerStartupGate> {
  @override
  Widget build(BuildContext context) {
    final startup = ref.watch(serverStartupProvider);
    final theme = Theme.of(context);

    // One-time Android system prompts at first transition to ready.
    ref.listen<ServerStartupState>(serverStartupProvider, (prev, next) {
      if (!Platform.isAndroid) return;
      if (next.status != ServerStatus.ready) return;
      if (prev?.status == ServerStatus.ready) return;
      _maybePromptFirstRunPermissions(context, ref);
    });

    final body = useTwoPane(context)
        ? const _AdaptiveLayout()
        : const HomeScreen();

    // Error banner — non-blocking, user can dismiss or retry.
    if (startup.status == ServerStatus.error) {
      return Scaffold(
        body: Column(
          children: [
            MaterialBanner(
              leading: Icon(Icons.error_outline, color: theme.colorScheme.error),
              content: Text(startup.message),
              actions: [
                TextButton(
                  onPressed: () =>
                      ref.read(serverStartupProvider.notifier).retry(),
                  child: const Text('Retry'),
                ),
                TextButton(
                  onPressed: () {
                    final diagnostics = StartupOverlay.buildDiagnostics(startup);
                    Clipboard.setData(ClipboardData(text: diagnostics));
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Diagnostics copied to clipboard'),
                        duration: Duration(seconds: 2),
                      ),
                    );
                  },
                  child: const Text('Copy diagnostics'),
                ),
              ],
            ),
            Expanded(child: body),
          ],
        ),
      );
    }

    // Starting / downloading — show a subtle progress banner.
    if (startup.status == ServerStatus.starting ||
        startup.status == ServerStatus.downloading ||
        startup.status == ServerStatus.downloadingPython ||
        startup.status == ServerStatus.downloadingLlama) {
      return Scaffold(
        body: Column(
          children: [
            Container(
              color: theme.colorScheme.primaryContainer,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: SafeArea(
                bottom: false,
                child: Row(
                  children: [
                    SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: theme.colorScheme.onPrimaryContainer,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        startup.message,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onPrimaryContainer,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    if (startup.totalBytes > 0)
                      Text(
                        '${(startup.downloadProgress * 100).toStringAsFixed(0)}%',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onPrimaryContainer,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                  ],
                ),
              ),
            ),
            Expanded(child: body),
          ],
        ),
      );
    }

    // Ready — normal UI.
    return body;
  }

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

}
