import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../database/database.dart';
import '../providers/providers.dart';

class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final db = ref.watch(databaseProvider);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('History')),
      body: FutureBuilder<List<ProcessingSession>>(
        future: db.getAllSessions(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }

          final sessions = snapshot.data ?? [];
          if (sessions.isEmpty) {
            return Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.history,
                      size: 64,
                      color: theme.colorScheme.onSurface.withValues(alpha: 0.3)),
                  const SizedBox(height: 16),
                  Text('No processing history yet.',
                      style: theme.textTheme.bodyLarge),
                ],
              ),
            );
          }

          return ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: sessions.length,
            itemBuilder: (context, index) {
              final session = sessions[index];
              return _SessionTile(session: session);
            },
          );
        },
      ),
      // Stats at the bottom.
      bottomNavigationBar: FutureBuilder<_Stats>(
        future: _loadStats(db),
        builder: (context, snap) {
          final stats = snap.data;
          if (stats == null) return const SizedBox.shrink();

          return Container(
            padding: const EdgeInsets.all(16),
            color: theme.colorScheme.surfaceContainerHighest,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _StatChip(label: 'Sessions', value: '${stats.sessions}'),
                _StatChip(label: 'Words', value: '${stats.words}'),
                _StatChip(label: 'Exported', value: '${stats.exported}'),
              ],
            ),
          );
        },
      ),
    );
  }

  Future<_Stats> _loadStats(AppDatabase db) async {
    final sessions = await db.totalSessionCount();
    final words = await db.totalWordCount();
    final exported = await db.totalExportedCount();
    return _Stats(sessions: sessions, words: words, exported: exported);
  }
}

class _Stats {
  const _Stats({
    required this.sessions,
    required this.words,
    required this.exported,
  });
  final int sessions;
  final int words;
  final int exported;
}

class _StatChip extends StatelessWidget {
  const _StatChip({required this.label, required this.value});
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(value,
            style: theme.textTheme.titleLarge
                ?.copyWith(fontWeight: FontWeight.bold)),
        Text(label, style: theme.textTheme.labelSmall),
      ],
    );
  }
}

class _SessionTile extends ConsumerWidget {
  const _SessionTile({required this.session});
  final ProcessingSession session;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final db = ref.watch(databaseProvider);

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ExpansionTile(
        leading: Icon(
          session.context == 'highlighted' ? Icons.highlight : Icons.text_fields,
          color: theme.colorScheme.primary,
        ),
        title: Text(
          session.imagePath,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        subtitle: Text(
          '${session.context}'
          '${session.highlightColor != null ? ' (${session.highlightColor})' : ''}'
          ' - ${_formatDate(session.createdAt)}',
          style: theme.textTheme.bodySmall,
        ),
        children: [
          FutureBuilder<List<WordEntry>>(
            future: db.getWordsForSession(session.id),
            builder: (context, snap) {
              final words = snap.data ?? [];
              if (words.isEmpty) {
                return const Padding(
                  padding: EdgeInsets.all(16),
                  child: Text('No words in this session.'),
                );
              }
              return Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Wrap(
                  spacing: 8,
                  runSpacing: 4,
                  children: words.map((w) {
                    return Chip(
                      label: Text(w.word),
                      avatar: w.exported
                          ? const Icon(Icons.check_circle,
                              size: 16, color: Colors.green)
                          : null,
                    );
                  }).toList(),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-'
        '${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:'
        '${dt.minute.toString().padLeft(2, '0')}';
  }
}
