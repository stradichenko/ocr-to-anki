import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import '../models/models.dart';
import '../providers/providers.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  OcrContext _context = OcrContext.handwrittenOrPrinted;
  HighlightColor _selectedColor = HighlightColor.orange;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('OCR to Anki'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'History',
            onPressed: () => Navigator.of(context).pushNamed('/history'),
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: 'Settings',
            onPressed: () => Navigator.of(context).pushNamed('/settings'),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 600),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // -- Context selector -----------------------------------------
                Text('OCR Context', style: theme.textTheme.titleMedium),
                const SizedBox(height: 8),
                SegmentedButton<OcrContext>(
                  segments: const [
                    ButtonSegment(
                      value: OcrContext.handwrittenOrPrinted,
                      label: Text('Handwritten / Printed'),
                      icon: Icon(Icons.text_fields),
                    ),
                    ButtonSegment(
                      value: OcrContext.highlighted,
                      label: Text('Highlighted'),
                      icon: Icon(Icons.highlight),
                    ),
                  ],
                  selected: {_context},
                  onSelectionChanged: (v) =>
                      setState(() => _context = v.first),
                ),
                const SizedBox(height: 16),

                // -- Color picker (only for highlighted context) ---------------
                AnimatedCrossFade(
                  duration: const Duration(milliseconds: 200),
                  crossFadeState: _context == OcrContext.highlighted
                      ? CrossFadeState.showFirst
                      : CrossFadeState.showSecond,
                  firstChild: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Highlight Colour',
                          style: theme.textTheme.titleSmall),
                      const SizedBox(height: 8),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: HighlightColor.values.map((c) {
                          return ChoiceChip(
                            label: Text(c.label),
                            selected: _selectedColor == c,
                            selectedColor: _chipColor(c),
                            onSelected: (_) =>
                                setState(() => _selectedColor = c),
                          );
                        }).toList(),
                      ),
                      const SizedBox(height: 16),
                    ],
                  ),
                  secondChild: const SizedBox.shrink(),
                ),

                // -- Image upload area ----------------------------------------
                const SizedBox(height: 8),
                _ImageDropZone(
                  onImageSelected: (bytes, name) =>
                      _startProcessing(bytes, name),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Color _chipColor(HighlightColor c) {
    const map = {
      HighlightColor.yellow: Colors.yellow,
      HighlightColor.orange: Colors.orange,
      HighlightColor.red: Colors.red,
      HighlightColor.green: Colors.green,
      HighlightColor.blue: Colors.blue,
      HighlightColor.purple: Colors.purple,
    };
    return (map[c] ?? Colors.grey).withValues(alpha: 0.3);
  }

  Future<void> _startProcessing(Uint8List bytes, String filename) async {
    final notifier = ref.read(processingProvider.notifier);
    notifier.reset();

    // Navigate to processing screen immediately.
    if (mounted) {
      Navigator.of(context).pushNamed('/processing');
    }

    await notifier.processImage(
      imageBytes: bytes,
      filename: filename,
      context: _context,
      highlightColor:
          _context == OcrContext.highlighted ? _selectedColor : null,
    );
  }
}

// ---------------------------------------------------------------------------
// Image drop zone widget
// ---------------------------------------------------------------------------

class _ImageDropZone extends StatelessWidget {
  const _ImageDropZone({required this.onImageSelected});

  final void Function(Uint8List bytes, String name) onImageSelected;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: theme.colorScheme.outline.withValues(alpha: 0.3)),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: () => _pickImage(context),
        child: Container(
          height: 200,
          alignment: Alignment.center,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.add_photo_alternate_outlined,
                  size: 48, color: theme.colorScheme.primary),
              const SizedBox(height: 12),
              Text('Tap to select an image',
                  style: theme.textTheme.bodyLarge),
              const SizedBox(height: 4),
              Text(
                'Supports JPG, PNG, BMP, TIFF',
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: theme.colorScheme.onSurfaceVariant),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _pickImage(BuildContext context) async {
    // Try system file picker first, fall back to image_picker.
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        withData: true,
      );
      if (result != null && result.files.single.bytes != null) {
        onImageSelected(
          result.files.single.bytes!,
          result.files.single.name,
        );
        return;
      }
      // If bytes are null (mobile), read from path.
      if (result != null && result.files.single.path != null) {
        final file = File(result.files.single.path!);
        final bytes = await file.readAsBytes();
        onImageSelected(bytes, result.files.single.name);
        return;
      }
    } catch (_) {
      // Fall back to image_picker.
    }

    final picker = ImagePicker();
    final xfile = await picker.pickImage(source: ImageSource.gallery);
    if (xfile != null) {
      final bytes = await xfile.readAsBytes();
      onImageSelected(bytes, xfile.name);
    }
  }
}
