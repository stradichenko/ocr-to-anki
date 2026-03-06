import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/anki_note.dart';
import '../models/models.dart';
import '../providers/providers.dart';

class ReviewScreen extends ConsumerStatefulWidget {
  const ReviewScreen({super.key});

  @override
  ConsumerState<ReviewScreen> createState() => _ReviewScreenState();
}

class _ReviewScreenState extends ConsumerState<ReviewScreen> {
  late List<_EditableCard> _cards;
  bool _initialised = false;
  bool _reEnriching = false;
  int? _retryingIndex;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialised) {
      final state = ref.read(processingProvider);
      _cards = state.enrichedWords
          .map((e) {
            // Clean up any leftover DEF: UNKNOWN text from the LLM.
            final cleanDef = (e.definition.toUpperCase().contains('UNKNOWN'))
                ? ''
                : e.definition;
            final cleanEx = (e.warning == 'not_found') ? '' : e.examples;
            return _EditableCard(
              word: e.word,
              definition: cleanDef,
              examples: cleanEx,
              warning: e.warning,
              selected: e.warning != 'not_found', // deselect unknowns
            );
          })
          .toList();
      _initialised = true;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final warningCount = _cards.where((c) => c.warning.isNotEmpty).length;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Review Cards'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          TextButton.icon(
            onPressed: _cards.any((c) => c.selected) ? _export : null,
            icon: const Icon(Icons.upload),
            label: const Text('Export to Anki'),
          ),
        ],
      ),
      body: _cards.isEmpty
          ? Center(
              child: Text('No cards to review.',
                  style: theme.textTheme.bodyLarge),
            )
          : Column(
              children: [
                // Warning banner
                if (warningCount > 0)
                  MaterialBanner(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 8),
                    leading:
                        Icon(Icons.warning_amber, color: Colors.orange[700]),
                    content: Text(
                      '$warningCount card(s) have warnings – '
                      'the model may not have recognized these words. '
                      'Review or deselect them before exporting.',
                    ),
                    actions: [
                      TextButton(
                        onPressed: _reEnriching ? null : _reEnrichFailed,
                        child: _reEnriching
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                    strokeWidth: 2),
                              )
                            : const Text('Re-enrich failed'),
                      ),
                      TextButton(
                        onPressed: () {
                          setState(() {
                            for (var i = 0; i < _cards.length; i++) {
                              if (_cards[i].warning.isNotEmpty) {
                                _cards[i] =
                                    _cards[i].copyWith(selected: false);
                              }
                            }
                          });
                        },
                        child: const Text('Deselect all warnings'),
                      ),
                    ],
                  ),
                // Card list
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: _cards.length,
                    itemBuilder: (context, index) {
                      final card = _cards[index];
                      return _CardTile(
                        card: card,
                        onChanged: (updated) =>
                            setState(() => _cards[index] = updated),
                        onRemoved: () =>
                            setState(() => _cards.removeAt(index)),
                        isRetrying: _retryingIndex == index,
                        onRetry: () => _retryOneWord(index),
                      );
                    },
                  ),
                ),
              ],
            ),
      floatingActionButton: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          FloatingActionButton.extended(
            heroTag: 'tsv',
            onPressed: _exportTsv,
            icon: const Icon(Icons.file_download),
            label: const Text('Save TSV'),
          ),
          const SizedBox(width: 12),
          FloatingActionButton.extended(
            heroTag: 'json',
            onPressed: _exportJson,
            icon: const Icon(Icons.save_alt),
            label: const Text('Save JSON'),
          ),
        ],
      ),
    );
  }

  Future<void> _export() async {
    final selected = _cards.where((c) => c.selected).toList();
    if (selected.isEmpty) return;

    final notes = selected
        .map((c) => AnkiNote(
              front: c.word,
              definition: c.definition,
              examples: c.examples,
            )..composeBack())
        .toList();

    final service = ref.read(ankiExportServiceProvider);

    // Show loading.
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => const Center(child: CircularProgressIndicator()),
    );

    try {
      final result = await service.importNotes(notes);
      if (mounted) Navigator.of(context).pop(); // close dialog

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Exported ${result.success}/${result.total} cards to Anki',
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) Navigator.of(context).pop(); // close spinner
      if (mounted) {
        _showCopyableError(context, 'Export to Anki Failed', e.toString());
      }
    }
  }

  void _exportJson() {
    final selected = _cards.where((c) => c.selected).toList();
    if (selected.isEmpty) return;

    final notes = selected
        .map((c) => AnkiNote(
              front: c.word,
              definition: c.definition,
              examples: c.examples,
            )..composeBack())
        .toList();

    final service = ref.read(ankiExportServiceProvider);
    final json = service.exportToJson(notes);

    showDialog(
      context: context,
      builder: (_) => _CopyableJsonDialog(json: json),
    );
  }

  /// Export selected cards as a TSV file and save with file picker.
  Future<void> _exportTsv() async {
    final selected = _cards.where((c) => c.selected).toList();
    if (selected.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No cards selected')),
      );
      return;
    }

    final notes = selected
        .map((c) => AnkiNote(
              front: c.word,
              definition: c.definition,
              examples: c.examples,
            )..composeBack())
        .toList();

    final service = ref.read(ankiExportServiceProvider);
    final tsv = service.exportToTsv(notes);

    try {
      final path = await FilePicker.platform.saveFile(
        dialogTitle: 'Save Anki TSV file',
        fileName: 'anki_cards.txt',
        type: FileType.custom,
        allowedExtensions: ['txt', 'tsv'],
      );

      if (path == null) return; // user cancelled

      await File(path).writeAsString(tsv);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved ${notes.length} card(s) to $path')),
        );
      }
    } catch (e) {
      if (mounted) {
        _showCopyableError(context, 'TSV Export Failed', e.toString());
      }
    }
  }

  /// Re-enrich only the cards that have a not_found warning (empty definition).
  Future<void> _reEnrichFailed() async {
    // Collect indices and words that need re-enrichment.
    final failedIndices = <int>[];
    final failedWords = <String>[];
    for (var i = 0; i < _cards.length; i++) {
      if (_cards[i].warning == 'not_found' && _cards[i].definition.isEmpty) {
        failedIndices.add(i);
        failedWords.add(_cards[i].word);
      }
    }
    if (failedWords.isEmpty) return;

    setState(() => _reEnriching = true);

    try {
      final inference = ref.read(inferenceServiceProvider);
      final settings = ref.read(settingsProvider);

      final results = await inference.enrichWords(
        words: failedWords,
        definitionLanguage: settings.definitionLanguage,
        examplesLanguage: settings.examplesLanguage,
        chunkSize: 1,
        chunkTimeout: const Duration(minutes: 10),
      );

      if (!mounted) return;

      // Merge results back into the card list.
      var recovered = 0;
      for (var ri = 0; ri < results.length; ri++) {
        final r = results[ri];
        if (ri < failedIndices.length && r.definition.isNotEmpty) {
          final idx = failedIndices[ri];
          _cards[idx] = _cards[idx].copyWith(
            definition: r.definition,
            examples: r.examples,
            warning: r.warning,
            selected: r.warning != 'not_found',
          );
          recovered++;
        }
      }

      setState(() => _reEnriching = false);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Re-enriched $recovered/${failedWords.length} word(s)',
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() => _reEnriching = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Re-enrich failed: $e')),
        );
      }
    }
  }

  /// Retry enrichment for a single card at [index].
  Future<void> _retryOneWord(int index) async {
    if (_retryingIndex != null) return; // already retrying another

    final card = _cards[index];
    setState(() => _retryingIndex = index);

    try {
      final inference = ref.read(inferenceServiceProvider);
      final settings = ref.read(settingsProvider);

      final results = await inference.enrichWords(
        words: [card.word],
        definitionLanguage: settings.definitionLanguage,
        examplesLanguage: settings.examplesLanguage,
        chunkSize: 1,
        chunkTimeout: const Duration(minutes: 10),
      );

      if (!mounted) return;

      if (results.isNotEmpty && results.first.definition.isNotEmpty) {
        final r = results.first;
        _cards[index] = _cards[index].copyWith(
          definition: r.definition,
          examples: r.examples,
          warning: r.warning,
          selected: r.warning != 'not_found',
        );
      }

      setState(() => _retryingIndex = null);
    } catch (e) {
      if (mounted) {
        setState(() => _retryingIndex = null);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Retry failed for "${card.word}": $e')),
        );
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Copyable error dialog (reusable)
// ---------------------------------------------------------------------------

void _showCopyableError(BuildContext context, String title, String error) {
  showDialog(
    context: context,
    builder: (dialogCtx) {
      final theme = Theme.of(dialogCtx);
      return AlertDialog(
        icon: Icon(Icons.error_outline, color: theme.colorScheme.error),
        title: Text(title),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: theme.colorScheme.errorContainer.withValues(alpha: 0.3),
                borderRadius: BorderRadius.circular(8),
              ),
              child: SelectableText(
                error,
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 12,
                  color: theme.colorScheme.onErrorContainer,
                ),
              ),
            ),
          ],
        ),
        actions: [
          TextButton.icon(
            onPressed: () {
              Clipboard.setData(ClipboardData(text: error));
              ScaffoldMessenger.of(dialogCtx).showSnackBar(
                const SnackBar(
                  content: Text('Error copied to clipboard'),
                  duration: Duration(seconds: 2),
                ),
              );
            },
            icon: const Icon(Icons.copy, size: 16),
            label: const Text('Copy'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(dialogCtx).pop(),
            child: const Text('Close'),
          ),
        ],
      );
    },
  );
}

// ---------------------------------------------------------------------------
// Copyable JSON export dialog
// ---------------------------------------------------------------------------

class _CopyableJsonDialog extends StatelessWidget {
  const _CopyableJsonDialog({required this.json});
  final String json;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AlertDialog(
      title: Row(
        children: [
          const Expanded(child: Text('Exported JSON')),
          IconButton(
            icon: const Icon(Icons.copy),
            tooltip: 'Copy to clipboard',
            onPressed: () {
              Clipboard.setData(ClipboardData(text: json));
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('JSON copied to clipboard'),
                  duration: Duration(seconds: 2),
                ),
              );
            },
          ),
        ],
      ),
      content: SizedBox(
        width: double.maxFinite,
        child: SingleChildScrollView(
          child: SelectableText(
            json,
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: 12,
              color: theme.colorScheme.onSurface,
            ),
          ),
        ),
      ),
      actions: [
        FilledButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Close'),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Editable card model (local to this screen)
// ---------------------------------------------------------------------------

class _EditableCard {
  _EditableCard({
    required this.word,
    required this.definition,
    required this.examples,
    required this.selected,
    this.warning = '',
  });

  String word;
  String definition;
  String examples;
  bool selected;
  String warning;

  _EditableCard copyWith({
    String? word,
    String? definition,
    String? examples,
    bool? selected,
    String? warning,
  }) =>
      _EditableCard(
        word: word ?? this.word,
        definition: definition ?? this.definition,
        examples: examples ?? this.examples,
        selected: selected ?? this.selected,
        warning: warning ?? this.warning,
      );
}

// ---------------------------------------------------------------------------
// Card tile widget
// ---------------------------------------------------------------------------

class _CardTile extends StatelessWidget {
  const _CardTile({
    required this.card,
    required this.onChanged,
    required this.onRemoved,
    this.onRetry,
    this.isRetrying = false,
  });

  final _EditableCard card;
  final ValueChanged<_EditableCard> onChanged;
  final VoidCallback onRemoved;
  final VoidCallback? onRetry;
  final bool isRetrying;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hasWarning = card.warning.isNotEmpty;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      shape: hasWarning
          ? RoundedRectangleBorder(
              side: BorderSide(color: Colors.orange.shade300, width: 1.5),
              borderRadius: BorderRadius.circular(12),
            )
          : null,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header row
            Row(
              children: [
                Checkbox(
                  value: card.selected,
                  onChanged: (v) =>
                      onChanged(card.copyWith(selected: v ?? true)),
                ),
                Expanded(
                  child: Text(
                    card.word,
                    style: theme.textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.bold),
                  ),
                ),
                if (hasWarning) ...[
                  Tooltip(
                    message: _warningLabel(card.warning),
                    child: Icon(Icons.warning_amber,
                        size: 20, color: Colors.orange[700]),
                  ),
                  const SizedBox(width: 4),
                ],
                if (onRetry != null && card.warning == 'not_found')
                  isRetrying
                      ? const Padding(
                          padding: EdgeInsets.all(8),
                          child: SizedBox(
                            width: 18,
                            height: 18,
                            child:
                                CircularProgressIndicator(strokeWidth: 2),
                          ),
                        )
                      : IconButton(
                          icon: const Icon(Icons.refresh, size: 20),
                          onPressed: onRetry,
                          tooltip: 'Retry enrichment',
                        ),
                IconButton(
                  icon: const Icon(Icons.delete_outline),
                  onPressed: onRemoved,
                  tooltip: 'Remove',
                ),
              ],
            ),

            // Warning chip
            if (hasWarning) ...[
              Padding(
                padding: const EdgeInsets.only(left: 48, bottom: 8),
                child: Chip(
                  avatar: Icon(Icons.warning_amber,
                      size: 16, color: Colors.orange[700]),
                  label: Text(
                    _warningLabel(card.warning),
                    style: theme.textTheme.labelSmall,
                  ),
                  backgroundColor: Colors.orange.withValues(alpha: 0.12),
                  visualDensity: VisualDensity.compact,
                ),
              ),
            ],

            const Divider(),

            // Definition
            Text('Definition', style: theme.textTheme.labelMedium),
            const SizedBox(height: 4),
            TextFormField(
              initialValue: card.definition,
              maxLines: null,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                isDense: true,
              ),
              onChanged: (v) => onChanged(card.copyWith(definition: v)),
            ),
            const SizedBox(height: 12),

            // Examples
            Text('Examples', style: theme.textTheme.labelMedium),
            const SizedBox(height: 4),
            TextFormField(
              initialValue: card.examples,
              maxLines: null,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                isDense: true,
              ),
              onChanged: (v) => onChanged(card.copyWith(examples: v)),
            ),
          ],
        ),
      ),
    );
  }

  String _warningLabel(String warning) {
    switch (warning) {
      case 'not_found':
        return 'Word not recognized – definition may be hallucinated or missing';
      case 'truncated':
        return 'Definition appears truncated (generation cut off)';
      default:
        return 'Unknown warning';
    }
  }
}
