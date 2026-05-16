import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/providers.dart';

/// Full-screen overlay shown when the user tries to start processing before
/// the background backend init is complete.  Auto-dismisses when the backend
/// reaches [ServerStatus.ready].
class StartupOverlay extends ConsumerWidget {
  const StartupOverlay({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final startup = ref.watch(serverStartupProvider);
    final theme = Theme.of(context);

    // Auto-dismiss when ready.
    ref.listen<ServerStartupState>(serverStartupProvider, (prev, next) {
      if (next.status == ServerStatus.ready &&
          prev?.status != ServerStatus.ready) {
        Navigator.of(context).pop(true);
      }
    });

    return Scaffold(
      backgroundColor: theme.colorScheme.surface.withValues(alpha: 0.95),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 720),
              child: _buildBody(context, ref, startup, theme),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBody(BuildContext context, WidgetRef ref,
      ServerStartupState startup, ThemeData theme) {
    switch (startup.status) {
      case ServerStatus.starting:
      case ServerStatus.downloadingPython:
      case ServerStatus.downloadingLlama:
      case ServerStatus.downloading:
        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 24),
            Text(
              startup.message,
              style: theme.textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
            if (startup.totalBytes > 0) ...[
              const SizedBox(height: 16),
              LinearProgressIndicator(
                value: startup.downloadProgress,
                minHeight: 8,
                borderRadius: BorderRadius.circular(4),
              ),
              const SizedBox(height: 8),
              Text(
                '${_megabytes(startup.downloadedBytes)} / '
                '${_megabytes(startup.totalBytes)} MB',
                style: theme.textTheme.bodyMedium,
              ),
            ],
            const SizedBox(height: 24),
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Go back'),
            ),
          ],
        );
      case ServerStatus.error:
        return Column(
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
                data: theme.copyWith(dividerColor: Colors.transparent),
                child: ExpansionTile(
                  tilePadding: const EdgeInsets.symmetric(horizontal: 8),
                  childrenPadding: EdgeInsets.zero,
                  title: Text(
                    'Technical details',
                    style: theme.textTheme.labelLarge,
                  ),
                  children: [
                    Container(
                      constraints: const BoxConstraints(maxHeight: 320),
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
                  onPressed: () =>
                      ref.read(serverStartupProvider.notifier).retry(),
                  icon: const Icon(Icons.refresh),
                  label: const Text('Retry'),
                ),
                OutlinedButton.icon(
                  onPressed: () {
                    final diagnostics = buildDiagnostics(startup);
                    Clipboard.setData(ClipboardData(text: diagnostics));
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Diagnostics copied to clipboard'),
                        duration: Duration(seconds: 2),
                      ),
                    );
                  },
                  icon: const Icon(Icons.copy),
                  label: const Text('Copy diagnostics'),
                ),
                TextButton(
                  onPressed: () => Navigator.of(context).pop(false),
                  child: const Text('Go back'),
                ),
              ],
            ),
          ],
        );
      case ServerStatus.ready:
        return const Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Almost ready…'),
          ],
        );
    }
  }

  /// Push the overlay and wait for the backend to become ready or for the
  /// user to dismiss it.  Returns `true` if ready, `false` if dismissed.
  static Future<bool> show(BuildContext context) async {
    final result = await Navigator.of(context).push<bool>(
      MaterialPageRoute(
        builder: (_) => const StartupOverlay(),
        fullscreenDialog: true,
      ),
    );
    return result ?? false;
  }

  static String _megabytes(int bytes) =>
      (bytes / (1024 * 1024)).toStringAsFixed(1);

  static String buildDiagnostics(ServerStartupState startup) {
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
