import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/providers.dart';

class ProcessingScreen extends ConsumerWidget {
  const ProcessingScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(processingProvider);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Processing'),
        leading: state.phase == ProcessingPhase.done ||
                state.phase == ProcessingPhase.error
            ? IconButton(
                icon: const Icon(Icons.arrow_back),
                onPressed: () => Navigator.of(context).pop(),
              )
            : null,
        automaticallyImplyLeading: false,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 600),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Phase indicator
                _PhaseIndicator(phase: state.phase),
                const SizedBox(height: 24),

                // Progress bar
                LinearProgressIndicator(
                  value: state.phase == ProcessingPhase.idle
                      ? null
                      : state.progress,
                  minHeight: 8,
                  borderRadius: BorderRadius.circular(4),
                ),
                const SizedBox(height: 16),

                // Status message
                Text(
                  state.statusMessage,
                  style: theme.textTheme.bodyLarge,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 32),

                // Error display
                if (state.error != null) ...[
                  _CopyableErrorCard(error: state.error!),
                  const SizedBox(height: 16),
                ],

                // Results summary when done
                if (state.phase == ProcessingPhase.done) ...[
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Results',
                              style: theme.textTheme.titleMedium),
                          const SizedBox(height: 8),
                          Text(
                              'Words found: ${state.words.length}'),
                          Text(
                              'Cards enriched: ${state.enrichedWords.length}'),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  FilledButton.icon(
                    onPressed: () {
                      Navigator.of(context).pushReplacementNamed('/review');
                    },
                    icon: const Icon(Icons.rate_review),
                    label: const Text('Review Cards'),
                  ),
                ],

                // Retry button on error
                if (state.phase == ProcessingPhase.error)
                  OutlinedButton.icon(
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.refresh),
                    label: const Text('Go Back'),
                  ),
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
