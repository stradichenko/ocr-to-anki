import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/anki_note.dart';
import '../providers/providers.dart';

class ReviewScreen extends ConsumerStatefulWidget {
  const ReviewScreen({super.key});

  @override
  ConsumerState<ReviewScreen> createState() => _ReviewScreenState();
}

class _ReviewScreenState extends ConsumerState<ReviewScreen> {
  late List<_EditableCard> _cards;
  bool _initialised = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialised) {
      final state = ref.read(processingProvider);
      _cards = state.enrichedWords
          .map((e) => _EditableCard(
                word: e.word,
                definition: e.definition,
                examples: e.examples,
                selected: true,
              ))
          .toList();
      _initialised = true;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Review Cards'),
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
          : ListView.builder(
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
                );
              },
            ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _exportJson,
        icon: const Icon(Icons.save_alt),
        label: const Text('Save JSON'),
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
      if (mounted) Navigator.of(context).pop();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Export failed: $e'),
            backgroundColor: Theme.of(context).colorScheme.error,
          ),
        );
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
      builder: (_) => AlertDialog(
        title: const Text('Exported JSON'),
        content: SingleChildScrollView(
          child: SelectableText(
            json,
            style: const TextStyle(
              fontFamily: 'monospace',
              fontSize: 12,
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
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
  });

  String word;
  String definition;
  String examples;
  bool selected;

  _EditableCard copyWith({
    String? word,
    String? definition,
    String? examples,
    bool? selected,
  }) =>
      _EditableCard(
        word: word ?? this.word,
        definition: definition ?? this.definition,
        examples: examples ?? this.examples,
        selected: selected ?? this.selected,
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
  });

  final _EditableCard card;
  final ValueChanged<_EditableCard> onChanged;
  final VoidCallback onRemoved;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
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
                IconButton(
                  icon: const Icon(Icons.delete_outline),
                  onPressed: onRemoved,
                  tooltip: 'Remove',
                ),
              ],
            ),
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
              onChanged: (v) =>
                  onChanged(card.copyWith(definition: v)),
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
              onChanged: (v) =>
                  onChanged(card.copyWith(examples: v)),
            ),
          ],
        ),
      ),
    );
  }
}
