import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:desktop_drop/desktop_drop.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import '../models/models.dart';
import '../providers/providers.dart';
import '../services/highlight_detector.dart';

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
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 600),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // -- Header row with actions --------------------------------
                  Row(
                    children: [
                      Expanded(
                        child: Text('OCR to Anki',
                            style: theme.textTheme.headlineSmall),
                      ),
                      IconButton(
                        icon: const Icon(Icons.history),
                        tooltip: 'History',
                        onPressed: () =>
                            Navigator.of(context).pushNamed('/history'),
                      ),
                      IconButton(
                        icon: const Icon(Icons.settings),
                        tooltip: 'Settings',
                        onPressed: () =>
                            Navigator.of(context).pushNamed('/settings'),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
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
    // Show bounding-box preview for highlighted mode.
    if (_context == OcrContext.highlighted) {
      final proceed = await _showBoundingBoxPreview(bytes);
      if (!proceed || !mounted) return;
    }

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

  /// Run highlight detection and show a preview dialog with bounding-box
  /// overlays.  Returns `true` if the user wants to proceed.
  Future<bool> _showBoundingBoxPreview(Uint8List bytes) async {
    final detector = ref.read(highlightDetectorProvider);
    final boxes = detector.detectBoxes(
      imageBytes: bytes,
      color: _selectedColor,
    );

    if (boxes.isEmpty) {
      if (!mounted) return false;
      final proceed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('No highlights detected'),
          content: const Text(
            'No highlighted regions were found for the selected colour. '
            'Process the full image instead?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Process Full Image'),
            ),
          ],
        ),
      );
      return proceed ?? false;
    }

    // Decode to get natural dimensions for overlay scaling.
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final naturalWidth = frame.image.width.toDouble();
    final naturalHeight = frame.image.height.toDouble();
    frame.image.dispose();

    if (!mounted) return false;

    return await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: Text('${boxes.length} region(s) detected'),
            content: SizedBox(
              width: 500,
              height: 400,
              child: _BoundingBoxPreview(
                imageBytes: bytes,
                boxes: boxes,
                naturalWidth: naturalWidth,
                naturalHeight: naturalHeight,
                highlightColor: _selectedColor,
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('Cancel'),
              ),
              FilledButton.icon(
                onPressed: () => Navigator.pop(ctx, true),
                icon: const Icon(Icons.play_arrow),
                label: const Text('Process'),
              ),
            ],
          ),
        ) ??
        false;
  }
}

// ---------------------------------------------------------------------------
// Image drop zone widget (supports click + drag-and-drop)
// ---------------------------------------------------------------------------

class _ImageDropZone extends StatefulWidget {
  const _ImageDropZone({required this.onImageSelected});

  final void Function(Uint8List bytes, String name) onImageSelected;

  @override
  State<_ImageDropZone> createState() => _ImageDropZoneState();
}

class _ImageDropZoneState extends State<_ImageDropZone> {
  bool _isDragging = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return DropTarget(
      onDragEntered: (_) => setState(() => _isDragging = true),
      onDragExited: (_) => setState(() => _isDragging = false),
      onDragDone: (details) async {
        setState(() => _isDragging = false);
        if (details.files.isEmpty) return;
        final xfile = details.files.first;
        final bytes = await xfile.readAsBytes();
        widget.onImageSelected(Uint8List.fromList(bytes), xfile.name);
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: _isDragging
                ? theme.colorScheme.primary
                : theme.colorScheme.outline.withValues(alpha: 0.3),
            width: _isDragging ? 2.0 : 1.0,
          ),
          color: _isDragging
              ? theme.colorScheme.primary.withValues(alpha: 0.08)
              : theme.colorScheme.surfaceContainerLow,
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () => _pickImage(context),
          child: SizedBox(
            height: 200,
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    _isDragging
                        ? Icons.file_download
                        : Icons.add_photo_alternate_outlined,
                    size: 48,
                    color: _isDragging
                        ? theme.colorScheme.primary
                        : theme.colorScheme.onSurfaceVariant,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _isDragging
                        ? 'Drop image here'
                        : 'Tap or drag an image here',
                    style: theme.textTheme.bodyLarge?.copyWith(
                      color: _isDragging
                          ? theme.colorScheme.primary
                          : null,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Supports JPG, PNG, BMP, TIFF',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
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
        widget.onImageSelected(
          result.files.single.bytes!,
          result.files.single.name,
        );
        return;
      }
      // If bytes are null (mobile), read from path.
      if (result != null && result.files.single.path != null) {
        final file = File(result.files.single.path!);
        final bytes = await file.readAsBytes();
        widget.onImageSelected(bytes, result.files.single.name);
        return;
      }
    } catch (_) {
      // Fall back to image_picker.
    }

    final picker = ImagePicker();
    final xfile = await picker.pickImage(source: ImageSource.gallery);
    if (xfile != null) {
      final bytes = await xfile.readAsBytes();
      widget.onImageSelected(bytes, xfile.name);
    }
  }
}

// ---------------------------------------------------------------------------
// Bounding-box overlay preview
// ---------------------------------------------------------------------------

class _BoundingBoxPreview extends StatelessWidget {
  const _BoundingBoxPreview({
    required this.imageBytes,
    required this.boxes,
    required this.naturalWidth,
    required this.naturalHeight,
    required this.highlightColor,
  });

  final Uint8List imageBytes;
  final List<HighlightBBox> boxes;
  final double naturalWidth;
  final double naturalHeight;
  final HighlightColor highlightColor;

  @override
  Widget build(BuildContext context) {
    return FittedBox(
      fit: BoxFit.contain,
      child: SizedBox(
        width: naturalWidth,
        height: naturalHeight,
        child: Stack(
          children: [
            Image.memory(
              imageBytes,
              fit: BoxFit.fill,
              width: naturalWidth,
              height: naturalHeight,
            ),
            CustomPaint(
              size: Size(naturalWidth, naturalHeight),
              painter: _BoundingBoxPainter(
                boxes: boxes,
                color: _highlightToColor(highlightColor),
              ),
            ),
          ],
        ),
      ),
    );
  }

  static Color _highlightToColor(HighlightColor c) {
    const map = {
      HighlightColor.yellow: Colors.yellow,
      HighlightColor.orange: Colors.orange,
      HighlightColor.red: Colors.red,
      HighlightColor.green: Colors.green,
      HighlightColor.blue: Colors.blue,
      HighlightColor.purple: Colors.purple,
    };
    return map[c] ?? Colors.orange;
  }
}

class _BoundingBoxPainter extends CustomPainter {
  _BoundingBoxPainter({required this.boxes, required this.color});

  final List<HighlightBBox> boxes;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final strokePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final fillPaint = Paint()
      ..color = color.withValues(alpha: 0.15)
      ..style = PaintingStyle.fill;

    for (final box in boxes) {
      final rect = Rect.fromLTWH(
        box.x.toDouble(),
        box.y.toDouble(),
        box.w.toDouble(),
        box.h.toDouble(),
      );
      canvas.drawRect(rect, fillPaint);
      canvas.drawRect(rect, strokePaint);
    }
  }

  @override
  bool shouldRepaint(covariant _BoundingBoxPainter old) =>
      boxes != old.boxes || color != old.color;
}
