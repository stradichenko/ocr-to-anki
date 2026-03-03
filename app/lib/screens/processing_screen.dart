import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/providers.dart';

class ProcessingScreen extends ConsumerStatefulWidget {
  const ProcessingScreen({super.key});

  @override
  ConsumerState<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends ConsumerState<ProcessingScreen> {
  Timer? _elapsedTimer;
  Duration _elapsed = Duration.zero;
  final ScrollController _logScroll = ScrollController();

  @override
  void initState() {
    super.initState();
    _elapsedTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      final start = ref.read(processingProvider).startTime;
      if (start != null) {
        setState(() => _elapsed = DateTime.now().difference(start));
      }
    });
  }

  @override
  void dispose() {
    _elapsedTimer?.cancel();
    _logScroll.dispose();
    super.dispose();
  }

  void _scrollLogToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_logScroll.hasClients) {
        _logScroll.animateTo(
          _logScroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  String _formatDuration(Duration d) {
    final m = d.inMinutes;
    final s = d.inSeconds % 60;
    return m > 0 ? '${m}m ${s}s' : '${s}s';
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(processingProvider);
    final theme = Theme.of(context);
    final isFinished =
        state.phase == ProcessingPhase.done ||
        state.phase == ProcessingPhase.error;

    // Auto-scroll log when it updates.
    if (state.activityLog.isNotEmpty) _scrollLogToBottom();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Processing'),
        leading: isFinished
            ? IconButton(
                icon: const Icon(Icons.arrow_back),
                onPressed: () => Navigator.of(context).pop(),
              )
            : null,
        automaticallyImplyLeading: false,
        actions: [
          // Live elapsed timer
          if (state.startTime != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Center(
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isFinished ? Icons.timer_off : Icons.timer,
                      size: 16,
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _formatDuration(_elapsed),
                      style: theme.textTheme.labelLarge?.copyWith(
                        fontFeatures: [const FontFeature.tabularFigures()],
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 600),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Phase indicator
                _PhaseIndicator(phase: state.phase),
                const SizedBox(height: 16),

                // Progress bar
                LinearProgressIndicator(
                  value: state.phase == ProcessingPhase.idle
                      ? null
                      : state.progress,
                  minHeight: 8,
                  borderRadius: BorderRadius.circular(4),
                ),
                const SizedBox(height: 8),

                // Status message
                Text(
                  state.statusMessage,
                  style: theme.textTheme.bodyLarge?.copyWith(
                    fontWeight: FontWeight.w500,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),

                // Activity log
                if (state.activityLog.isNotEmpty)
                  Expanded(
                    child: Card(
                      color: theme.colorScheme.surfaceContainerLowest,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Padding(
                            padding:
                                const EdgeInsets.fromLTRB(12, 10, 12, 0),
                            child: Row(
                              children: [
                                Icon(Icons.terminal,
                                    size: 14,
                                    color:
                                        theme.colorScheme.onSurfaceVariant),
                                const SizedBox(width: 6),
                                Text(
                                  'Activity',
                                  style:
                                      theme.textTheme.labelSmall?.copyWith(
                                    color:
                                        theme.colorScheme.onSurfaceVariant,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const Divider(height: 8),
                          Expanded(
                            child: ListView.builder(
                              controller: _logScroll,
                              padding: const EdgeInsets.fromLTRB(
                                  12, 0, 12, 8),
                              itemCount: state.activityLog.length,
                              itemBuilder: (_, i) {
                                final line = state.activityLog[i];
                                final isError = line.startsWith('ERROR:');
                                final isLast =
                                    i == state.activityLog.length - 1;
                                return Padding(
                                  padding: const EdgeInsets.symmetric(
                                      vertical: 2),
                                  child: Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      if (!isFinished && isLast)
                                        Padding(
                                          padding: const EdgeInsets.only(
                                              right: 6, top: 2),
                                          child: SizedBox(
                                            width: 10,
                                            height: 10,
                                            child:
                                                CircularProgressIndicator(
                                              strokeWidth: 1.5,
                                              color:
                                                  theme.colorScheme.primary,
                                            ),
                                          ),
                                        )
                                      else
                                        Padding(
                                          padding: const EdgeInsets.only(
                                              right: 6, top: 3),
                                          child: Icon(
                                            isError
                                                ? Icons.close
                                                : Icons
                                                    .check_circle_outline,
                                            size: 12,
                                            color: isError
                                                ? theme.colorScheme.error
                                                : theme.colorScheme.primary
                                                    .withValues(alpha: 0.6),
                                          ),
                                        ),
                                      Expanded(
                                        child: Text(
                                          line,
                                          style: theme.textTheme.bodySmall
                                              ?.copyWith(
                                            color: isError
                                                ? theme.colorScheme.error
                                                : theme.colorScheme
                                                    .onSurface
                                                    .withValues(alpha: 0.8),
                                            fontFamily: 'monospace',
                                            height: 1.4,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                );
                              },
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                // Error display
                if (state.error != null) ...[
                  const SizedBox(height: 12),
                  _CopyableErrorCard(error: state.error!),
                ],

                // Results summary when done
                if (state.phase == ProcessingPhase.done) ...[
                  const SizedBox(height: 12),
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Results',
                              style: theme.textTheme.titleMedium),
                          const SizedBox(height: 8),
                          Text('Words found: ${state.words.length}'),
                          Text(
                              'Cards enriched: ${state.enrichedWords.length}'),
                          Text('Total time: ${_formatDuration(_elapsed)}'),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: () {
                      Navigator.of(context)
                          .pushReplacementNamed('/review');
                    },
                    icon: const Icon(Icons.rate_review),
                    label: const Text('Review Cards'),
                  ),
                ],

                // Retry button on error
                if (state.phase == ProcessingPhase.error) ...[
                  const SizedBox(height: 12),
                  OutlinedButton.icon(
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.refresh),
                    label: const Text('Go Back'),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _CopyableErrorCard extends StatelessWidget {
  const _CopyableErrorCard({required this.error});

  final String error;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      color: theme.colorScheme.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.error_outline,
                    color: theme.colorScheme.onErrorContainer),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Error',
                    style: theme.textTheme.titleSmall?.copyWith(
                      color: theme.colorScheme.onErrorContainer,
                    ),
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.copy,
                      size: 18,
                      color: theme.colorScheme.onErrorContainer),
                  tooltip: 'Copy error',
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: error));
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Error copied to clipboard'),
                        duration: Duration(seconds: 2),
                      ),
                    );
                  },
                ),
              ],
            ),
            const SizedBox(height: 8),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: theme.colorScheme.error.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(8),
              ),
              child: SelectableText(
                error,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onErrorContainer,
                  fontFamily: 'monospace',
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PhaseIndicator extends StatelessWidget {
  const _PhaseIndicator({required this.phase});

  final ProcessingPhase phase;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    final steps = [
      ('Crop', ProcessingPhase.cropping, Icons.crop),
      ('OCR', ProcessingPhase.ocr, Icons.document_scanner),
      ('Enrich', ProcessingPhase.enriching, Icons.auto_awesome),
      ('Done', ProcessingPhase.done, Icons.check_circle),
    ];

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: steps.map((step) {
        final isActive = phase == step.$2;
        final isDone = phase.index > step.$2.index;
        final isError = phase == ProcessingPhase.error;

        Color color;
        if (isError) {
          color = theme.colorScheme.error.withValues(alpha: 0.5);
        } else if (isDone) {
          color = theme.colorScheme.primary;
        } else if (isActive) {
          color = theme.colorScheme.primary;
        } else {
          color = theme.colorScheme.onSurface.withValues(alpha: 0.3);
        }

        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(step.$3, color: color, size: 28),
            const SizedBox(height: 4),
            Text(
              step.$1,
              style: theme.textTheme.labelSmall?.copyWith(color: color),
            ),
          ],
        );
      }).toList(),
    );
  }
}
