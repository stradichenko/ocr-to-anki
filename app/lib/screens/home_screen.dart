import 'dart:io';
import 'dart:math';
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
  HighlightColor _selectedPresetColor = HighlightColor.orange;
  bool _useCustomColor = false;
  HsvRange? _customHsvRange;
  final List<({Uint8List bytes, String name})> _imageQueue = [];
  final ScrollController _queueScrollCtrl = ScrollController();

  @override
  void dispose() {
    _queueScrollCtrl.dispose();
    super.dispose();
  }

  /// The HSV range to use for highlight detection.
  HsvRange? get _effectiveHsvRange {
    if (_context != OcrContext.highlighted) return null;
    if (_useCustomColor) return _customHsvRange;
    return HsvRange.fromPreset(_selectedPresetColor);
  }

  bool get _canStartProcessing {
    if (_imageQueue.isEmpty) return false;
    if (_context == OcrContext.highlighted &&
        _useCustomColor &&
        _customHsvRange == null) {
      return false;
    }
    return true;
  }

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
                        children: [
                          ...HighlightColor.values.map((c) {
                            return ChoiceChip(
                              label: Text(c.label),
                              selected:
                                  !_useCustomColor &&
                                  _selectedPresetColor == c,
                              selectedColor: _chipColor(c),
                              onSelected: (_) => setState(() {
                                _useCustomColor = false;
                                _selectedPresetColor = c;
                              }),
                            );
                          }),
                          ChoiceChip(
                            avatar: _customHsvRange != null
                                ? Container(
                                    width: 18,
                                    height: 18,
                                    decoration: BoxDecoration(
                                      color: _hsvRangeToColor(
                                          _customHsvRange!),
                                      shape: BoxShape.circle,
                                      border:
                                          Border.all(color: Colors.grey),
                                    ),
                                  )
                                : const Icon(Icons.colorize, size: 18),
                            label: const Text('Custom'),
                            selected: _useCustomColor,
                            onSelected: (_) =>
                                setState(() => _useCustomColor = true),
                          ),
                        ],
                      ),
                      // Custom colour actions
                      if (_useCustomColor) ...[
                        const SizedBox(height: 8),
                        if (_imageQueue.isEmpty)
                          Text(
                            'Add an image first to sample a custom colour.',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          )
                        else
                          OutlinedButton.icon(
                            icon: const Icon(Icons.colorize),
                            label: Text(_customHsvRange != null
                                ? 'Re-sample Colour'
                                : 'Pick Colour from Image'),
                            onPressed: _showColorSampler,
                          ),
                        if (_customHsvRange != null) ...[
                          const SizedBox(height: 4),
                          _SampledColorInfo(range: _customHsvRange!),
                        ],
                      ],
                      const SizedBox(height: 16),
                    ],
                  ),
                  secondChild: const SizedBox.shrink(),
                ),

                // -- Image upload area ----------------------------------------
                const SizedBox(height: 8),
                _ImageDropZone(
                  onImagesSelected: (images) => setState(() {
                    _imageQueue.addAll(images);
                  }),
                ),

                // -- Queued image thumbnails ----------------------------------
                if (_imageQueue.isNotEmpty) ...[
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Text(
                        '${_imageQueue.length} image(s) queued',
                        style: theme.textTheme.titleSmall,
                      ),
                      const Spacer(),
                      TextButton.icon(
                        icon: const Icon(Icons.clear_all, size: 18),
                        label: const Text('Clear'),
                        onPressed: () =>
                            setState(() => _imageQueue.clear()),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  SizedBox(
                    height: 110,
                    child: Scrollbar(
                    controller: _queueScrollCtrl,
                    thumbVisibility: true,
                    child: ListView.separated(
                      controller: _queueScrollCtrl,
                      scrollDirection: Axis.horizontal,
                      padding: const EdgeInsets.only(bottom: 10),
                      itemCount: _imageQueue.length,
                      separatorBuilder: (_, __) =>
                          const SizedBox(width: 8),
                      itemBuilder: (ctx, i) {
                        return Stack(
                          children: [
                            ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: Image.memory(
                                _imageQueue[i].bytes,
                                width: 100,
                                height: 100,
                                fit: BoxFit.cover,
                              ),
                            ),
                            Positioned(
                              top: 2,
                              right: 2,
                              child: Material(
                                color: Colors.black54,
                                shape: const CircleBorder(),
                                child: InkWell(
                                  customBorder: const CircleBorder(),
                                  onTap: () => setState(
                                      () => _imageQueue.removeAt(i)),
                                  child: const Padding(
                                    padding: EdgeInsets.all(2),
                                    child: Icon(Icons.close,
                                        size: 16,
                                        color: Colors.white),
                                  ),
                                ),
                              ),
                            ),
                            Positioned(
                              bottom: 2,
                              left: 4,
                              right: 4,
                              child: Text(
                                _imageQueue[i].name,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(
                                  fontSize: 10,
                                  color: Colors.white,
                                  shadows: [
                                    Shadow(
                                        blurRadius: 4,
                                        color: Colors.black),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  FilledButton.icon(
                    onPressed:
                        _canStartProcessing ? _startBatchProcessing : null,
                    icon: const Icon(Icons.play_arrow),
                    label: Text(
                      'Start Processing'
                      '${_imageQueue.length > 1 ? " (${_imageQueue.length} images)" : ""}',
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
      ),
    );
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

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

  /// Convert an [HsvRange] centre to a displayable [Color].
  static Color _hsvRangeToColor(HsvRange range) {
    final hue = (range.hueCenter * 2).clamp(0.0, 360.0);
    final sat = ((range.satMin + range.satMax) / 2 / 255).clamp(0.0, 1.0);
    final val = ((range.valMin + range.valMax) / 2 / 255).clamp(0.0, 1.0);
    return HSVColor.fromAHSV(1, hue, sat, val).toColor();
  }

  /// The overlay colour used for bounding-box previews.
  Color get _overlayColor {
    if (_useCustomColor && _customHsvRange != null) {
      return _hsvRangeToColor(_customHsvRange!);
    }
    const map = {
      HighlightColor.yellow: Colors.yellow,
      HighlightColor.orange: Colors.orange,
      HighlightColor.red: Colors.red,
      HighlightColor.green: Colors.green,
      HighlightColor.blue: Colors.blue,
      HighlightColor.purple: Colors.purple,
    };
    return map[_selectedPresetColor] ?? Colors.orange;
  }

  /// Open the colour-sampler dialog on the first queued image.
  Future<void> _showColorSampler() async {
    if (_imageQueue.isEmpty) return;
    final detector = ref.read(highlightDetectorProvider);
    final result = await showDialog<HsvRange>(
      context: context,
      builder: (_) => _ColorSamplerDialog(
        imageBytes: _imageQueue.first.bytes,
        detector: detector,
      ),
    );
    if (result != null && mounted) {
      setState(() => _customHsvRange = result);
    }
  }

  Future<void> _startBatchProcessing() async {
    if (_imageQueue.isEmpty) return;

    // Show bounding-box preview for highlighted mode (first image as sample).
    if (_context == OcrContext.highlighted && _effectiveHsvRange != null) {
      final proceed =
          await _showBoundingBoxPreview(_imageQueue.first.bytes);
      if (!proceed || !mounted) return;
    }

    final images = List<({Uint8List bytes, String name})>.of(_imageQueue);
    setState(() => _imageQueue.clear());

    final notifier = ref.read(processingProvider.notifier);
    notifier.reset();

    if (mounted) {
      Navigator.of(context).pushNamed('/processing');
    }

    await notifier.processImages(
      images: images,
      context: _context,
      hsvRange: _effectiveHsvRange,
    );
  }

  /// Run highlight detection and show a preview dialog with bounding-box
  /// overlays.  Returns `true` if the user wants to proceed.
  Future<bool> _showBoundingBoxPreview(Uint8List bytes) async {
    final detector = ref.read(highlightDetectorProvider);
    final range = _effectiveHsvRange;
    if (range == null) return true;
    final rawBoxes = detector.detectBoxes(
      imageBytes: bytes,
      color: range,
    );

    if (rawBoxes.isEmpty) {
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

    // Apply the same padding that detectAndCrop uses so the preview
    // accurately shows what the OCR will actually receive.
    final pad = detector.padding;
    final boxes = rawBoxes.map((b) {
      final x = (b.x - pad).clamp(0, naturalWidth.toInt() - 1);
      final y = (b.y - pad).clamp(0, naturalHeight.toInt() - 1);
      final x2 = (b.x + b.w + pad).clamp(0, naturalWidth.toInt());
      final y2 = (b.y + b.h + pad).clamp(0, naturalHeight.toInt());
      return HighlightBBox(x, y, x2 - x, y2 - y);
    }).toList();

    if (!mounted) return false;

    final moreImages = _imageQueue.length > 1
        ? ' (showing preview for first image; '
          '${_imageQueue.length - 1} more queued)'
        : '';

    return await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: Text('${boxes.length} region(s) detected$moreImages'),
            content: SizedBox(
              width: 500,
              height: 400,
              child: _BoundingBoxPreview(
                imageBytes: bytes,
                boxes: boxes,
                naturalWidth: naturalWidth,
                naturalHeight: naturalHeight,
                overlayColor: _overlayColor,
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
// Image drop zone widget (supports click + drag-and-drop, multi-select)
// ---------------------------------------------------------------------------

class _ImageDropZone extends StatefulWidget {
  const _ImageDropZone({required this.onImagesSelected});

  final void Function(List<({Uint8List bytes, String name})>) onImagesSelected;

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
        final images = <({Uint8List bytes, String name})>[];
        for (final xfile in details.files) {
          final bytes = await xfile.readAsBytes();
          images.add((bytes: Uint8List.fromList(bytes), name: xfile.name));
        }
        if (images.isNotEmpty) widget.onImagesSelected(images);
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
          onTap: () => _pickImages(context),
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
                        ? 'Drop image(s) here'
                        : 'Tap or drag image(s) here',
                    style: theme.textTheme.bodyLarge?.copyWith(
                      color: _isDragging
                          ? theme.colorScheme.primary
                          : null,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Supports JPG, PNG, BMP, TIFF — multiple files allowed',
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

  Future<void> _pickImages(BuildContext context) async {
    // Try system file picker first (multi-select).
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        withData: true,
        allowMultiple: true,
      );
      // User cancelled — result is null. Return immediately so we
      // don't fall through to the image_picker fallback.
      if (result == null) return;

      if (result.files.isNotEmpty) {
        final images = <({Uint8List bytes, String name})>[];
        for (final file in result.files) {
          Uint8List? bytes = file.bytes;
          if (bytes == null && file.path != null) {
            bytes = await File(file.path!).readAsBytes();
          }
          if (bytes != null) {
            images.add((bytes: bytes, name: file.name));
          }
        }
        if (images.isNotEmpty) {
          widget.onImagesSelected(images);
          return;
        }
      }
    } catch (_) {
      // Fall back to image_picker only on exception.
    }

    final picker = ImagePicker();
    final xfile = await picker.pickImage(source: ImageSource.gallery);
    if (xfile != null) {
      final bytes = await xfile.readAsBytes();
      widget.onImagesSelected([(bytes: bytes, name: xfile.name)]);
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
    required this.overlayColor,
  });

  final Uint8List imageBytes;
  final List<HighlightBBox> boxes;
  final double naturalWidth;
  final double naturalHeight;
  final Color overlayColor;

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
                color: overlayColor,
              ),
            ),
          ],
        ),
      ),
    );
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

// ---------------------------------------------------------------------------
// Custom colour sampler dialog (button-controlled zoom + scrollbars)
// ---------------------------------------------------------------------------

class _ColorSamplerDialog extends StatefulWidget {
  const _ColorSamplerDialog({
    required this.imageBytes,
    required this.detector,
  });

  final Uint8List imageBytes;
  final HighlightDetector detector;

  @override
  State<_ColorSamplerDialog> createState() => _ColorSamplerDialogState();
}

class _ColorSamplerDialogState extends State<_ColorSamplerDialog> {
  Offset? _dragStart;
  Offset? _dragEnd;
  HsvRange? _sampledRange;

  double _imageWidth = 0;
  double _imageHeight = 0;

  // Zoom
  double _zoomScale = 1.0;
  static const double _minZoom = 1.0;
  static const double _maxZoom = 8.0;
  static const double _zoomStep = 0.5;

  // Scroll controllers for navigating the zoomed image.
  final ScrollController _hScrollCtrl = ScrollController();
  final ScrollController _vScrollCtrl = ScrollController();

  @override
  void initState() {
    super.initState();
    _loadImageDimensions();
  }

  @override
  void dispose() {
    _hScrollCtrl.dispose();
    _vScrollCtrl.dispose();
    super.dispose();
  }

  void _zoomIn() {
    setState(() {
      _zoomScale = (_zoomScale + _zoomStep).clamp(_minZoom, _maxZoom);
      _dragStart = null;
      _dragEnd = null;
    });
  }

  void _zoomOut() {
    setState(() {
      _zoomScale = (_zoomScale - _zoomStep).clamp(_minZoom, _maxZoom);
      _dragStart = null;
      _dragEnd = null;
    });
  }

  void _zoomReset() {
    setState(() {
      _zoomScale = 1.0;
      _dragStart = null;
      _dragEnd = null;
    });
  }

  Future<void> _loadImageDimensions() async {
    final codec = await ui.instantiateImageCodec(widget.imageBytes);
    final frame = await codec.getNextFrame();
    if (mounted) {
      setState(() {
        _imageWidth = frame.image.width.toDouble();
        _imageHeight = frame.image.height.toDouble();
      });
    }
    frame.image.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Sample Highlight Colour'),
      content: SizedBox(
        width: 600,
        height: 600,
        child: Column(
          children: [
            Text(
              'Draw a rectangle over a highlighted region to sample its colour.\n'
              'Use the zoom buttons to magnify; scroll or drag the scrollbars to navigate.',
              style: Theme.of(context).textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            // -- Zoom controls -----------------------------------------------
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  icon: const Icon(Icons.zoom_out),
                  tooltip: 'Zoom out',
                  onPressed: _zoomScale > _minZoom ? _zoomOut : null,
                ),
                Text('${(_zoomScale * 100).round()}%',
                    style: Theme.of(context).textTheme.bodyMedium),
                IconButton(
                  icon: const Icon(Icons.zoom_in),
                  tooltip: 'Zoom in',
                  onPressed: _zoomScale < _maxZoom ? _zoomIn : null,
                ),
                const SizedBox(width: 8),
                TextButton.icon(
                  icon: const Icon(Icons.fit_screen, size: 18),
                  label: const Text('Reset'),
                  onPressed: _zoomScale != 1.0 ? _zoomReset : null,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Expanded(
              child: _imageWidth > 0
                  ? ClipRect(
                      child: LayoutBuilder(
                        builder: (context, constraints) {
                          return _buildScrollableImage(constraints);
                        },
                      ),
                    )
                  : const Center(child: CircularProgressIndicator()),
            ),
            if (_sampledRange != null) ...[
              const SizedBox(height: 12),
              _SampledColorInfo(range: _sampledRange!),
            ],
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context, null),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: _sampledRange != null
              ? () => Navigator.pop(context, _sampledRange)
              : null,
          child: const Text('Use This Colour'),
        ),
      ],
    );
  }

  Widget _buildScrollableImage(BoxConstraints constraints) {
    final aspect = _imageWidth / _imageHeight;
    double baseW = constraints.maxWidth;
    double baseH = baseW / aspect;
    if (baseH > constraints.maxHeight) {
      baseH = constraints.maxHeight;
      baseW = baseH * aspect;
    }

    final displayW = baseW * _zoomScale;
    final displayH = baseH * _zoomScale;

    return RawScrollbar(
        controller: _vScrollCtrl,
        thumbVisibility: true,
        thumbColor: Colors.grey.shade600,
        radius: const Radius.circular(4),
        thickness: 8,
        child: RawScrollbar(
          controller: _hScrollCtrl,
          thumbVisibility: true,
          thumbColor: Colors.grey.shade600,
          radius: const Radius.circular(4),
          thickness: 8,
          notificationPredicate: (n) => n.depth == 1,
          child: SingleChildScrollView(
            controller: _vScrollCtrl,
            physics: const ClampingScrollPhysics(),
            child: SingleChildScrollView(
              controller: _hScrollCtrl,
              scrollDirection: Axis.horizontal,
              physics: const ClampingScrollPhysics(),
              child: SizedBox(
                width: displayW,
                height: displayH,
                child: GestureDetector(
                  onPanStart: (d) {
                    setState(() {
                      _dragStart = d.localPosition;
                      _dragEnd = d.localPosition;
                      _sampledRange = null;
                    });
                  },
                  onPanUpdate: (d) {
                    final clamped = Offset(
                      d.localPosition.dx.clamp(0.0, displayW),
                      d.localPosition.dy.clamp(0.0, displayH),
                    );
                    setState(() => _dragEnd = clamped);
                  },
                  onPanEnd: (_) => _sampleRegion(displayW, displayH),
                  child: Stack(
                    children: [
                      Image.memory(
                        widget.imageBytes,
                        width: displayW,
                        height: displayH,
                        fit: BoxFit.fill,
                      ),
                      if (_dragStart != null && _dragEnd != null)
                        CustomPaint(
                          size: Size(displayW, displayH),
                          painter: _SelectionRectPainter(
                            start: _dragStart!,
                            end: _dragEnd!,
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
    );
  }

  void _sampleRegion(double displayW, double displayH) {
    if (_dragStart == null || _dragEnd == null) return;
    if (_imageWidth == 0 || _imageHeight == 0) return;

    final scaleX = _imageWidth / displayW;
    final scaleY = _imageHeight / displayH;

    final x0 = (min(_dragStart!.dx, _dragEnd!.dx) * scaleX).round();
    final y0 = (min(_dragStart!.dy, _dragEnd!.dy) * scaleY).round();
    final x1 = (max(_dragStart!.dx, _dragEnd!.dx) * scaleX).round();
    final y1 = (max(_dragStart!.dy, _dragEnd!.dy) * scaleY).round();

    if (x1 - x0 < 5 || y1 - y0 < 5) return; // Too small

    try {
      final range = widget.detector.sampleHsvRange(
        imageBytes: widget.imageBytes,
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
      );
      setState(() => _sampledRange = range);
    } catch (_) {
      // Sampling failed — ignore.
    }
  }
}

// ---------------------------------------------------------------------------
// Selection rectangle painter (for the colour sampler)
// ---------------------------------------------------------------------------

class _SelectionRectPainter extends CustomPainter {
  _SelectionRectPainter({required this.start, required this.end});

  final Offset start;
  final Offset end;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Rect.fromPoints(start, end);
    canvas.drawRect(
      rect,
      Paint()
        ..color = Colors.white.withValues(alpha: 0.3)
        ..style = PaintingStyle.fill,
    );
    canvas.drawRect(
      rect,
      Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0,
    );
  }

  @override
  bool shouldRepaint(covariant _SelectionRectPainter old) =>
      start != old.start || end != old.end;
}

// ---------------------------------------------------------------------------
// Sampled colour info row
// ---------------------------------------------------------------------------

class _SampledColorInfo extends StatelessWidget {
  const _SampledColorInfo({required this.range});

  final HsvRange range;

  @override
  Widget build(BuildContext context) {
    final hue = (range.hueCenter * 2).clamp(0.0, 360.0);
    final sat = ((range.satMin + range.satMax) / 2 / 255).clamp(0.0, 1.0);
    final val = ((range.valMin + range.valMax) / 2 / 255).clamp(0.0, 1.0);
    final color = HSVColor.fromAHSV(1.0, hue, sat, val).toColor();

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 24,
          height: 24,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: Colors.grey),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          'H: ${range.hueCenter.round()}\u00b1${range.hueRange.round()}  '
          'S: ${range.satMin.round()}-${range.satMax.round()}  '
          'V: ${range.valMin.round()}-${range.valMax.round()}',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}
