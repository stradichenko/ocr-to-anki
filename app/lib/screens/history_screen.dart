import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../database/database.dart';
import '../models/models.dart';
import '../providers/providers.dart';

class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final db = ref.watch(databaseProvider);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('History'),
        actions: [
          IconButton(
            icon: const Icon(Icons.download),
            tooltip: 'Export all benchmarks as JSON',
            onPressed: () => _exportBenchmarks(context, db),
          ),
        ],
      ),
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
                if (stats.avgTotalS > 0)
                  _StatChip(
                    label: 'Avg time',
                    value: '${stats.avgTotalS.toStringAsFixed(0)}s',
                  ),
              ],
            ),
          );
        },
      ),
    );
  }

  Future<void> _exportBenchmarks(
      BuildContext context, AppDatabase db) async {
    final sessions = await db.getAllSessions();
    final benchmarks = sessions
        .where((s) => s.benchmarkJson.isNotEmpty)
        .map((s) {
      final bench = BenchmarkData.fromJsonString(s.benchmarkJson);
      return {
        'session_id': s.id,
        'image': s.imagePath,
        'date': s.createdAt.toIso8601String(),
        ...bench.toJson(),
      };
    }).toList();

    if (benchmarks.isEmpty) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No benchmark data to export')),
        );
      }
      return;
    }

    final json =
        const JsonEncoder.withIndent('  ').convert(benchmarks);
    await Clipboard.setData(ClipboardData(text: json));
    if (context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content:
              Text('${benchmarks.length} benchmark(s) copied to clipboard'),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  Future<_Stats> _loadStats(AppDatabase db) async {
    final sessions = await db.totalSessionCount();
    final words = await db.totalWordCount();
    final exported = await db.totalExportedCount();
    // Compute average total time from all sessions.
    final allSessions = await db.getAllSessions();
    double avgTotalS = 0;
    if (allSessions.isNotEmpty) {
      final total = allSessions.fold<double>(
        0,
        (sum, s) => sum + s.ocrElapsedS + s.enrichElapsedS,
      );
      avgTotalS = total / allSessions.length;
    }
    return _Stats(
        sessions: sessions,
        words: words,
        exported: exported,
        avgTotalS: avgTotalS);
  }
}

class _Stats {
  const _Stats({
    required this.sessions,
    required this.words,
    required this.exported,
    this.avgTotalS = 0,
  });
  final int sessions;
  final int words;
  final int exported;
  final double avgTotalS;
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
    final hasBenchmark = session.benchmarkJson.isNotEmpty;

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
          ' - ${_formatDate(session.createdAt)}'
          '${session.ocrElapsedS > 0 ? ' • ${session.ocrElapsedS.toStringAsFixed(1)}s OCR' : ''}'
          '${session.enrichElapsedS > 0 ? ' + ${session.enrichElapsedS.toStringAsFixed(1)}s enrich' : ''}',
          style: theme.textTheme.bodySmall,
        ),
        children: [
          // Benchmark section
          if (hasBenchmark) _BenchmarkSection(session: session),

          // Words
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

// ---------------------------------------------------------------------------
// Benchmark section inside session tile
// ---------------------------------------------------------------------------

class _BenchmarkSection extends StatelessWidget {
  const _BenchmarkSection({required this.session});
  final ProcessingSession session;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bench = BenchmarkData.fromJsonString(session.benchmarkJson);

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.speed, size: 16, color: theme.colorScheme.primary),
              const SizedBox(width: 6),
              Text('Benchmark',
                  style: theme.textTheme.labelLarge
                      ?.copyWith(color: theme.colorScheme.primary)),
              const Spacer(),
              IconButton(
                icon: const Icon(Icons.copy, size: 16),
                tooltip: 'Copy benchmark JSON',
                visualDensity: VisualDensity.compact,
                onPressed: () {
                  Clipboard.setData(
                      ClipboardData(text: bench.toJsonString()));
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Benchmark copied to clipboard'),
                      duration: Duration(seconds: 2),
                    ),
                  );
                },
              ),
            ],
          ),
          const Divider(height: 8),
          _BenchmarkTable(bench: bench),
        ],
      ),
    );
  }
}

class _BenchmarkTable extends StatelessWidget {
  const _BenchmarkTable({required this.bench});
  final BenchmarkData bench;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final labelStyle = theme.textTheme.bodySmall
        ?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    final valueStyle = theme.textTheme.bodySmall?.copyWith(
      fontFamily: 'monospace',
      fontWeight: FontWeight.w500,
    );

    final rows = <(String, String)>[
      ('Total time', '${bench.totalElapsedS.toStringAsFixed(1)}s'),
      ('OCR time', '${bench.ocrElapsedS.toStringAsFixed(1)}s'),
      if (bench.perCropOcrS.length > 1)
        ('  Per-crop OCR',
            bench.perCropOcrS.map((s) => '${s.toStringAsFixed(1)}s').join(', ')),
      ('Enrichment time', '${bench.enrichElapsedS.toStringAsFixed(1)}s'),
      if (bench.enrichedWordCount > 0)
        ('  Avg per word', '${bench.avgEnrichPerWordS.toStringAsFixed(1)}s'),
      if (bench.cropCount > 0)
        ('Crop detection', '${bench.cropElapsedS.toStringAsFixed(1)}s (${bench.cropCount} crops)'),
      ('Image size', bench.imageSizeFormatted),
      ('Words (raw → unique)',
          '${bench.rawWordCount} → ${bench.uniqueWordCount}'),
      ('Enriched', '${bench.enrichedWordCount}'),
      if (bench.totalWarnings > 0)
        ('Warnings',
            '${bench.warningNotFoundCount} not found, ${bench.warningTruncatedCount} truncated'),
      if (bench.backend.isNotEmpty) ('Backend', bench.backend),
      if (bench.definitionLanguage.isNotEmpty)
        ('Languages', '${bench.definitionLanguage} / ${bench.examplesLanguage}'),
    ];

    return Table(
      columnWidths: const {
        0: IntrinsicColumnWidth(),
        1: FlexColumnWidth(),
      },
      defaultVerticalAlignment: TableCellVerticalAlignment.baseline,
      textBaseline: TextBaseline.alphabetic,
      children: rows.map((r) {
        return TableRow(
          children: [
            Padding(
              padding: const EdgeInsets.only(right: 12, bottom: 2),
              child: Text(r.$1, style: labelStyle),
            ),
            Text(r.$2, style: valueStyle),
          ],
        );
      }).toList(),
    );
  }
}
