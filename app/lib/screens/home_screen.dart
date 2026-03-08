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
  /// Currently selected preset colour for highlight detection.
  /// `null` means no global colour — images are OCR'd as full images unless
  /// they have a per-image colour override.
  HighlightColor? _selectedPresetColor;
  bool _useCustomColor = false;
  HsvRange? _customHsvRange;
  final List<ImageEntry> _imageQueue = [];
  final ScrollController _queueScrollCtrl = ScrollController();

  /// Optional region-of-interest crop (applied to ALL images before colour
  /// detection).  Stored in natural-image coordinates.
  HighlightBBox? _cropRegion;

  @override
  void dispose() {
    _queueScrollCtrl.dispose();
    super.dispose();
  }

  /// The global HSV range for highlight detection.
  /// Returns `null` when no colour is selected (→ plain OCR).
  HsvRange? get _effectiveHsvRange {
    if (_useCustomColor) return _customHsvRange;
    if (_selectedPresetColor != null) {
      return HsvRange.fromPreset(_selectedPresetColor!);
    }
    return null;
  }

  bool get _canStartProcessing {
    if (_imageQueue.isEmpty) return false;
    // Custom mode selected but not yet sampled → block
    if (_useCustomColor && _customHsvRange == null) return false;
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
                // -- Highlight colour picker (always visible) -----------------
                Text('Highlight Colour', style: theme.textTheme.titleMedium),
                const SizedBox(height: 4),
                Text(
                  'Select a colour to detect highlighted regions, or leave '
                  'empty to OCR full images.',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    // "None" chip — no highlight detection
                    ChoiceChip(
                      avatar: const Icon(Icons.text_fields, size: 18),
                      label: const Text('None'),
                      selected: !_useCustomColor &&
                          _selectedPresetColor == null,
                      onSelected: (_) => setState(() {
                        _useCustomColor = false;
                        _selectedPresetColor = null;
                      }),
                    ),
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

                // -- Image upload area ----------------------------------------
                _ImageDropZone(
                  onImagesSelected: (images) => setState(() {
                    _imageQueue.addAll(
                      images.map((img) => ImageEntry(
                        bytes: img.bytes,
                        name: img.name,
                      )),
                    );
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
                        final entry = _imageQueue[i];
                        return GestureDetector(
                          onTap: () => _showPerImageSettings(i),
                          child: Stack(
                            children: [
                              ClipRRect(
                                borderRadius: BorderRadius.circular(8),
                                child: Image.memory(
                                  entry.bytes,
                                  width: 100,
                                  height: 100,
                                  fit: BoxFit.cover,
                                ),
                              ),
                              // -- Per-image badge indicators --
                              if (entry.hasCrop || entry.hasColorOverride)
                                Positioned(
                                  top: 2,
                                  left: 2,
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      if (entry.hasCrop)
                                        _badge(
                                          icon: Icons.crop,
                                          color: theme.colorScheme.primary,
                                        ),
                                      if (entry.hasCrop && entry.hasColorOverride)
                                        const SizedBox(width: 2),
                                      if (entry.hasColorOverride)
                                        _badge(
                                          icon: Icons.palette,
                                          color: _hsvRangeToColor(entry.hsvOverride!),
                                        ),
                                    ],
                                  ),
                                ),
                              // -- Close button --
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
                              // -- Filename --
                              Positioned(
                                bottom: 2,
                                left: 4,
                                right: 4,
                                child: Text(
                                  entry.name,
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
                          ),
                        );
                      },
                    ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  // -- Crop region indicator ----------------------------------
                  if (_cropRegion != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        children: [
                          Icon(Icons.crop, size: 16,
                              color: theme.colorScheme.primary),
                          const SizedBox(width: 6),
                          Expanded(
                            child: Text(
                              'Global crop: ${_cropRegion!.w}×${_cropRegion!.h} '
                              'at (${_cropRegion!.x}, ${_cropRegion!.y})',
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: theme.colorScheme.primary,
                              ),
                            ),
                          ),
                          TextButton(
                            onPressed: () =>
                                setState(() => _cropRegion = null),
                            child: const Text('Clear'),
                          ),
                        ],
                      ),
                    ),
                  // -- Action buttons -----------------------------------------
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed:
                              _canStartProcessing ? _startBatchProcessing : null,
                          icon: const Icon(Icons.play_arrow),
                          label: Text(
                            'Start Processing'
                            '${_imageQueue.length > 1 ? " (${_imageQueue.length} images)" : ""}',
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton.icon(
                        icon: const Icon(Icons.crop, size: 18),
                        label: const Text('Crop All'),
                        onPressed: _imageQueue.isNotEmpty
                            ? _showCropRegionDialog
                            : null,
                      ),
                    ],
                  ),
                ],
                // -- Add Words Directly (always available) --------------------
                const SizedBox(height: 24),
                const Divider(),
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  icon: const Icon(Icons.edit_note),
                  label: const Text('Add Words Directly'),
                  onPressed: _showManualWordEntry,
                ),
                const SizedBox(height: 4),
                Text(
                  'Skip images — type words for enrichment.',
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

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  /// Small circular badge shown on a thumbnail to indicate a per-image
  /// override.
  Widget _badge({required IconData icon, required Color color}) {
    return Container(
      padding: const EdgeInsets.all(2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.85),
        shape: BoxShape.circle,
        border: Border.all(color: Colors.white, width: 1),
      ),
      child: Icon(icon, size: 12, color: Colors.white),
    );
  }

  /// Show a bottom sheet to configure per-image crop and colour for the
  /// image at [index].
  Future<void> _showPerImageSettings(int index) async {
    final entry = _imageQueue[index];
    final detector = ref.read(highlightDetectorProvider);

    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) {
        return _PerImageSettingsSheet(
          entry: entry,
          globalColor: _effectiveHsvRange,
          detector: detector,
          onCropSet: (box) {
            setState(() => _imageQueue[index].cropRegion = box);
            Navigator.of(ctx).pop();
          },
          onCropCleared: () {
            setState(() => _imageQueue[index].cropRegion = null);
            Navigator.of(ctx).pop();
          },
          onColorSet: (hsv) {
            setState(() => _imageQueue[index].hsvOverride = hsv);
            Navigator.of(ctx).pop();
          },
          onColorCleared: () {
            setState(() => _imageQueue[index].hsvOverride = null);
            Navigator.of(ctx).pop();
          },
        );
      },
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

  /// Show the crop-region dialog to let the user draw a sub-region on the
  /// first queued image.  The crop is applied to every image before colour
  /// detection.
  Future<void> _showCropRegionDialog() async {
    if (_imageQueue.isEmpty) return;
    final result = await showDialog<HighlightBBox>(
      context: context,
      builder: (_) => _CropRegionDialog(imageBytes: _imageQueue.first.bytes),
    );
    if (result != null && mounted) {
      setState(() => _cropRegion = result);
    }
  }

  /// Navigate to a word-review screen where the user types words manually,
  /// then runs enrichment without any image processing.
  Future<void> _showManualWordEntry() async {
    final notifier = ref.read(processingProvider.notifier);
    notifier.reset();

    if (mounted) {
      Navigator.of(context).pushNamed('/processing');
    }

    // Start with an empty word list — the word-review UI lets the user add.
    await notifier.processWordsOnly([]);
  }

  Future<void> _startBatchProcessing() async {
    if (_imageQueue.isEmpty) return;

    // Build ImageEntry list. Apply the global crop region to images that
    // don't have their own per-image crop, and per-image crop to those that
    // do.  Per-image colour overrides are already set on each entry.
    final images = _imageQueue.map((entry) {
      final crop = entry.cropRegion ?? _cropRegion;
      if (crop != null) {
        final cropped = HighlightDetector.cropRegion(
          imageBytes: entry.bytes,
          region: crop,
        );
        return ImageEntry(
          bytes: cropped,
          name: entry.name,
          // Crop already applied — don't carry the region forward.
          hsvOverride: entry.hsvOverride,
        );
      }
      return entry;
    }).toList();

    // Show bounding-box preview for any images that will use highlight
    // detection (global colour or per-image colour override).
    final anyHighlighted = _effectiveHsvRange != null ||
        images.any((e) => e.hsvOverride != null);
    Map<int, List<HighlightBBox>>? confirmedBoxes;
    if (anyHighlighted) {
      confirmedBoxes = await _showBatchBoundingBoxPreview(images);
      if (confirmedBoxes == null || !mounted) return;
    }

    setState(() => _imageQueue.clear());

    final notifier = ref.read(processingProvider.notifier);
    notifier.reset();

    if (mounted) {
      Navigator.of(context).pushNamed('/processing');
    }

    await notifier.processImages(
      images: images,
      hsvRange: _effectiveHsvRange,
      confirmedBoxes: confirmedBoxes,
    );
  }

  /// Detect highlights on ALL images and show a paginated preview dialog.
  ///
  /// Returns a map of image-index → confirmed bounding boxes, or `null`
  /// if the user cancelled.
  Future<Map<int, List<HighlightBBox>>?> _showBatchBoundingBoxPreview(
    List<ImageEntry> images,
  ) async {
    final detector = ref.read(highlightDetectorProvider);
    final pad = detector.padding;
    final previewImages = <_PreviewImageData>[];

    // Detect boxes for every image.
    var totalBoxes = 0;
    for (final entry in images) {
      final range = entry.hsvOverride ?? _effectiveHsvRange;
      if (range == null) {
        // Shouldn't happen — we already checked _effectiveHsvRange != null.
        continue;
      }

      final rawBoxes = detector.detectBoxes(
        imageBytes: entry.bytes,
        color: range,
      );

      // Decode to get natural dimensions.
      final codec = await ui.instantiateImageCodec(entry.bytes);
      final frame = await codec.getNextFrame();
      final natW = frame.image.width.toDouble();
      final natH = frame.image.height.toDouble();
      frame.image.dispose();

      // Apply padding to match what detectAndCrop produces.
      final boxes = rawBoxes.map((b) {
        final x = (b.x - pad).clamp(0, natW.toInt() - 1);
        final y = (b.y - pad).clamp(0, natH.toInt() - 1);
        final x2 = (b.x + b.w + pad).clamp(0, natW.toInt());
        final y2 = (b.y + b.h + pad).clamp(0, natH.toInt());
        return HighlightBBox(x, y, x2 - x, y2 - y);
      }).toList();

      totalBoxes += boxes.length;

      // Resolve overlay colour for this image.
      final overlayColor = entry.hsvOverride != null
          ? _hsvRangeToColor(entry.hsvOverride!)
          : _overlayColor;

      previewImages.add(_PreviewImageData(
        imageBytes: entry.bytes,
        boxes: boxes,
        naturalWidth: natW,
        naturalHeight: natH,
        overlayColor: overlayColor,
        name: entry.name,
      ));
    }

    // If NO image has any boxes, ask whether to process full images.
    if (totalBoxes == 0) {
      if (!mounted) return null;
      final proceed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('No highlights detected'),
          content: Text(
            images.length == 1
                ? 'No highlighted regions were found for the selected colour. '
                  'Process the full image instead?'
                : 'No highlighted regions were found in any of the '
                  '${images.length} images. Process full images instead?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Process Full Image(s)'),
            ),
          ],
        ),
      );
      if (proceed != true) return null;
      // Return empty lists — signals "use full image" for each.
      return {
        for (var i = 0; i < images.length; i++) i: <HighlightBBox>[],
      };
    }

    if (!mounted) return null;

    return await showDialog<Map<int, List<HighlightBBox>>>(
      context: context,
      builder: (ctx) => _BoundingBoxPreviewDialog(
        images: previewImages,
      ),
    );
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

/// Data for a single image in the batch bounding-box preview.
class _PreviewImageData {
  _PreviewImageData({
    required this.imageBytes,
    required this.boxes,
    required this.naturalWidth,
    required this.naturalHeight,
    required this.overlayColor,
    required this.name,
  });

  final Uint8List imageBytes;
  final List<HighlightBBox> boxes;
  final double naturalWidth;
  final double naturalHeight;
  final Color overlayColor;
  final String name;
}

// ---------------------------------------------------------------------------
// Batch bounding-box preview dialog (multi-image, prev/next navigation)
// ---------------------------------------------------------------------------

class _BoundingBoxPreviewDialog extends StatefulWidget {
  const _BoundingBoxPreviewDialog({
    required this.images,
  });

  final List<_PreviewImageData> images;

  @override
  State<_BoundingBoxPreviewDialog> createState() =>
      _BoundingBoxPreviewDialogState();
}

class _BoundingBoxPreviewDialogState extends State<_BoundingBoxPreviewDialog> {
  int _currentIndex = 0;

  // Zoom — per-image
  final List<double> _zoomScales = [];
  static const double _minZoom = 1.0;
  static const double _maxZoom = 8.0;
  static const double _zoomStep = 0.5;

  // Scroll controllers (shared, reset on page change)
  final ScrollController _hScrollCtrl = ScrollController();
  final ScrollController _vScrollCtrl = ScrollController();

  // Box removal tracking — per image
  late final List<Set<int>> _removedIndices;
  int _toggleVersion = 0;

  @override
  void initState() {
    super.initState();
    _zoomScales.addAll(List.filled(widget.images.length, 1.0));
    _removedIndices =
        List.generate(widget.images.length, (_) => <int>{});
  }

  @override
  void dispose() {
    _hScrollCtrl.dispose();
    _vScrollCtrl.dispose();
    super.dispose();
  }

  _PreviewImageData get _current => widget.images[_currentIndex];
  double get _zoom => _zoomScales[_currentIndex];

  void _zoomIn() => setState(() =>
      _zoomScales[_currentIndex] =
          (_zoom + _zoomStep).clamp(_minZoom, _maxZoom));

  void _zoomOut() => setState(() =>
      _zoomScales[_currentIndex] =
          (_zoom - _zoomStep).clamp(_minZoom, _maxZoom));

  void _zoomReset() =>
      setState(() => _zoomScales[_currentIndex] = 1.0);

  void _restoreAll() => setState(() {
        _removedIndices[_currentIndex].clear();
        _toggleVersion++;
      });

  void _goTo(int idx) {
    if (idx < 0 || idx >= widget.images.length) return;
    setState(() => _currentIndex = idx);
    // Reset scroll position when switching images.
    _hScrollCtrl.jumpTo(0);
    _vScrollCtrl.jumpTo(0);
  }

  int _activeCountFor(int idx) =>
      widget.images[idx].boxes.length - _removedIndices[idx].length;

  int get _totalActive {
    var c = 0;
    for (var i = 0; i < widget.images.length; i++) {
      c += _activeCountFor(i);
    }
    return c;
  }

  List<HighlightBBox> _activeBoxesFor(int idx) => [
        for (var i = 0; i < widget.images[idx].boxes.length; i++)
          if (!_removedIndices[idx].contains(i))
            widget.images[idx].boxes[i],
      ];

  Map<int, List<HighlightBBox>> get _result => {
        for (var i = 0; i < widget.images.length; i++)
          i: _activeBoxesFor(i),
      };

  void _handleTap(Offset localPos, double displayW, double displayH) {
    final img = _current;
    final scaleX = img.naturalWidth / displayW;
    final scaleY = img.naturalHeight / displayH;
    final natX = localPos.dx * scaleX;
    final natY = localPos.dy * scaleY;

    for (var i = img.boxes.length - 1; i >= 0; i--) {
      final b = img.boxes[i];
      if (natX >= b.x && natX <= b.x + b.w &&
          natY >= b.y && natY <= b.y + b.h) {
        setState(() {
          if (_removedIndices[_currentIndex].contains(i)) {
            _removedIndices[_currentIndex].remove(i);
          } else {
            _removedIndices[_currentIndex].add(i);
          }
          _toggleVersion++;
        });
        return;
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final img = _current;
    final activeHere = _activeCountFor(_currentIndex);
    final totalHere = img.boxes.length;
    final multi = widget.images.length > 1;

    final titleText = multi
        ? 'Image ${_currentIndex + 1}/${widget.images.length}: '
          '$activeHere/$totalHere region(s)'
        : '$activeHere/$totalHere region(s)';

    return AlertDialog(
      title: Text(titleText),
      content: SizedBox(
        width: 600,
        height: 520,
        child: Column(
          children: [
            // -- Image name + navigation --
            if (multi)
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.chevron_left),
                      tooltip: 'Previous image',
                      onPressed: _currentIndex > 0
                          ? () => _goTo(_currentIndex - 1)
                          : null,
                    ),
                    Expanded(
                      child: Column(
                        children: [
                          Text(
                            img.name,
                            style: theme.textTheme.bodyMedium
                                ?.copyWith(fontWeight: FontWeight.w600),
                            overflow: TextOverflow.ellipsis,
                          ),
                          const SizedBox(height: 2),
                          // Dot indicators
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: List.generate(
                              widget.images.length,
                              (i) => Container(
                                width: i == _currentIndex ? 10 : 6,
                                height: i == _currentIndex ? 10 : 6,
                                margin:
                                    const EdgeInsets.symmetric(horizontal: 2),
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: i == _currentIndex
                                      ? theme.colorScheme.primary
                                      : theme.colorScheme.outlineVariant,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.chevron_right),
                      tooltip: 'Next image',
                      onPressed:
                          _currentIndex < widget.images.length - 1
                              ? () => _goTo(_currentIndex + 1)
                              : null,
                    ),
                  ],
                ),
              ),
            Text(
              'Tap a region to remove/restore it.\n'
              'Use zoom to inspect details; scroll to navigate.',
              style: theme.textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            // -- Zoom controls --
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  icon: const Icon(Icons.zoom_out),
                  tooltip: 'Zoom out',
                  onPressed: _zoom > _minZoom ? _zoomOut : null,
                ),
                Text('${(_zoom * 100).round()}%',
                    style: theme.textTheme.bodyMedium),
                IconButton(
                  icon: const Icon(Icons.zoom_in),
                  tooltip: 'Zoom in',
                  onPressed: _zoom < _maxZoom ? _zoomIn : null,
                ),
                const SizedBox(width: 8),
                TextButton.icon(
                  icon: const Icon(Icons.fit_screen, size: 18),
                  label: const Text('Reset'),
                  onPressed: _zoom != 1.0 ? _zoomReset : null,
                ),
                if (_removedIndices[_currentIndex].isNotEmpty) ...[
                  const SizedBox(width: 12),
                  TextButton.icon(
                    icon: const Icon(Icons.restore, size: 18),
                    label: const Text('Restore All'),
                    onPressed: _restoreAll,
                  ),
                ],
              ],
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ClipRect(
                child: LayoutBuilder(
                  builder: (context, constraints) =>
                      _buildScrollableImage(constraints),
                ),
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context, null),
          child: const Text('Cancel'),
        ),
        FilledButton.icon(
          onPressed: _totalActive > 0
              ? () => Navigator.pop(context, _result)
              : null,
          icon: const Icon(Icons.play_arrow),
          label: Text(multi
              ? 'Process $_totalActive region(s) across ${widget.images.length} images'
              : 'Process $_totalActive region(s)'),
        ),
      ],
    );
  }

  Widget _buildScrollableImage(BoxConstraints constraints) {
    final img = _current;
    final aspect = img.naturalWidth / img.naturalHeight;
    double baseW = constraints.maxWidth;
    double baseH = baseW / aspect;
    if (baseH > constraints.maxHeight) {
      baseH = constraints.maxHeight;
      baseW = baseH * aspect;
    }

    final displayW = baseW * _zoom;
    final displayH = baseH * _zoom;

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
            child: GestureDetector(
              onTapUp: (d) =>
                  _handleTap(d.localPosition, displayW, displayH),
              child: SizedBox(
                width: displayW,
                height: displayH,
                child: Stack(
                  children: [
                    Image.memory(
                      img.imageBytes,
                      width: displayW,
                      height: displayH,
                      fit: BoxFit.fill,
                    ),
                    CustomPaint(
                      size: Size(displayW, displayH),
                      painter: _BoundingBoxPainter(
                        boxes: img.boxes,
                        color: img.overlayColor,
                        removedIndices: _removedIndices[_currentIndex],
                        naturalWidth: img.naturalWidth,
                        naturalHeight: img.naturalHeight,
                        toggleVersion: _toggleVersion,
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
}

class _BoundingBoxPainter extends CustomPainter {
  _BoundingBoxPainter({
    required this.boxes,
    required this.color,
    required this.removedIndices,
    required this.naturalWidth,
    required this.naturalHeight,
    required this.toggleVersion,
  });

  final List<HighlightBBox> boxes;
  final Color color;
  final Set<int> removedIndices;
  final double naturalWidth;
  final double naturalHeight;
  final int toggleVersion;

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / naturalWidth;
    final scaleY = size.height / naturalHeight;

    final activeStroke = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final activeFill = Paint()
      ..color = color.withValues(alpha: 0.15)
      ..style = PaintingStyle.fill;

    final removedStroke = Paint()
      ..color = Colors.red.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final removedXPaint = Paint()
      ..color = Colors.red.withValues(alpha: 0.4)
      ..strokeWidth = 2.0;

    for (var i = 0; i < boxes.length; i++) {
      final b = boxes[i];
      final rect = Rect.fromLTWH(
        b.x * scaleX,
        b.y * scaleY,
        b.w * scaleX,
        b.h * scaleY,
      );

      if (removedIndices.contains(i)) {
        canvas.drawRect(rect, removedStroke);
        canvas.drawLine(rect.topLeft, rect.bottomRight, removedXPaint);
        canvas.drawLine(rect.topRight, rect.bottomLeft, removedXPaint);
      } else {
        canvas.drawRect(rect, activeFill);
        canvas.drawRect(rect, activeStroke);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _BoundingBoxPainter old) =>
      boxes != old.boxes ||
      color != old.color ||
      toggleVersion != old.toggleVersion;
}

// ---------------------------------------------------------------------------
// Crop region dialog — draw a rectangle to crop all images before processing
// ---------------------------------------------------------------------------

class _CropRegionDialog extends StatefulWidget {
  const _CropRegionDialog({required this.imageBytes});
  final Uint8List imageBytes;

  @override
  State<_CropRegionDialog> createState() => _CropRegionDialogState();
}

class _CropRegionDialogState extends State<_CropRegionDialog> {
  Offset? _dragStart;
  Offset? _dragEnd;
  double _imageWidth = 0;
  double _imageHeight = 0;

  // Zoom
  double _zoomScale = 1.0;
  static const double _minZoom = 1.0;
  static const double _maxZoom = 8.0;
  static const double _zoomStep = 0.5;

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

  void _zoomIn() =>
      setState(() => _zoomScale = (_zoomScale + _zoomStep).clamp(_minZoom, _maxZoom));
  void _zoomOut() =>
      setState(() => _zoomScale = (_zoomScale - _zoomStep).clamp(_minZoom, _maxZoom));
  void _zoomReset() => setState(() => _zoomScale = 1.0);

  HighlightBBox? _computeRegion(double displayW, double displayH) {
    if (_dragStart == null || _dragEnd == null) return null;
    if (_imageWidth == 0 || _imageHeight == 0) return null;

    final scaleX = _imageWidth / displayW;
    final scaleY = _imageHeight / displayH;

    final x0 = (min(_dragStart!.dx, _dragEnd!.dx) * scaleX).round();
    final y0 = (min(_dragStart!.dy, _dragEnd!.dy) * scaleY).round();
    final x1 = (max(_dragStart!.dx, _dragEnd!.dx) * scaleX).round();
    final y1 = (max(_dragStart!.dy, _dragEnd!.dy) * scaleY).round();

    if (x1 - x0 < 10 || y1 - y0 < 10) return null;
    return HighlightBBox(x0, y0, x1 - x0, y1 - y0);
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Select Crop Region'),
      content: SizedBox(
        width: 600,
        height: 550,
        child: Column(
          children: [
            Text(
              'Draw a rectangle on the image to select the region of interest.\n'
              'Only this region will be used for processing.',
              style: Theme.of(context).textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
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
                        builder: (context, constraints) =>
                            _buildScrollableImage(constraints),
                      ),
                    )
                  : const Center(child: CircularProgressIndicator()),
            ),
            if (_dragStart != null && _dragEnd != null) ...[
              const SizedBox(height: 8),
              Text(
                'Region selected — confirm or redraw.',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).colorScheme.primary),
              ),
            ],
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context, null),
          child: const Text('Cancel'),
        ),
        FilledButton.icon(
          onPressed: _dragStart != null && _dragEnd != null
              ? () {
                  final region = _computeRegion(
                    _lastDisplayW > 0 ? _lastDisplayW : _imageWidth,
                    _lastDisplayH > 0 ? _lastDisplayH : _imageHeight,
                  );
                  Navigator.pop(context, region);
                }
              : null,
          icon: const Icon(Icons.crop),
          label: const Text('Set Crop Region'),
        ),
      ],
    );
  }

  // Track the last display dimensions for region computation.
  double _lastDisplayW = 0;
  double _lastDisplayH = 0;

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
    _lastDisplayW = displayW;
    _lastDisplayH = displayH;

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
                  });
                },
                onPanUpdate: (d) {
                  final clamped = Offset(
                    d.localPosition.dx.clamp(0.0, displayW),
                    d.localPosition.dy.clamp(0.0, displayH),
                  );
                  setState(() => _dragEnd = clamped);
                },
                onPanEnd: (_) {},
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
                        painter: _CropRegionPainter(
                          start: _dragStart!,
                          end: _dragEnd!,
                          displayW: displayW,
                          displayH: displayH,
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
}

/// Paints a semi-transparent overlay outside the selected crop region,
/// with a bright border around the selected area.
class _CropRegionPainter extends CustomPainter {
  _CropRegionPainter({
    required this.start,
    required this.end,
    required this.displayW,
    required this.displayH,
  });

  final Offset start;
  final Offset end;
  final double displayW;
  final double displayH;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Rect.fromPoints(start, end);

    // Dim everything outside the selection.
    final dimPaint = Paint()
      ..color = Colors.black.withValues(alpha: 0.5)
      ..style = PaintingStyle.fill;

    // Top
    canvas.drawRect(
        Rect.fromLTRB(0, 0, size.width, rect.top), dimPaint);
    // Bottom
    canvas.drawRect(
        Rect.fromLTRB(0, rect.bottom, size.width, size.height), dimPaint);
    // Left
    canvas.drawRect(
        Rect.fromLTRB(0, rect.top, rect.left, rect.bottom), dimPaint);
    // Right
    canvas.drawRect(
        Rect.fromLTRB(rect.right, rect.top, size.width, rect.bottom), dimPaint);

    // Selection border
    canvas.drawRect(
      rect,
      Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0,
    );
  }

  @override
  bool shouldRepaint(covariant _CropRegionPainter old) =>
      start != old.start || end != old.end;
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
  /// Drag coordinates normalised to 0-1 (fraction of displayed image).
  Offset? _dragStartNorm;
  Offset? _dragEndNorm;
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
      _dragStartNorm = null;
      _dragEndNorm = null;
    });
  }

  void _zoomOut() {
    setState(() {
      _zoomScale = (_zoomScale - _zoomStep).clamp(_minZoom, _maxZoom);
      _dragStartNorm = null;
      _dragEndNorm = null;
    });
  }

  void _zoomReset() {
    setState(() {
      _zoomScale = 1.0;
      _dragStartNorm = null;
      _dragEndNorm = null;
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
            ] else ...[
              // Reserve the same vertical space so the image area doesn't
              // resize when the colour info appears.
              const SizedBox(height: 12),
              const SizedBox(height: 24),
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
                      _dragStartNorm = Offset(
                        d.localPosition.dx / displayW,
                        d.localPosition.dy / displayH,
                      );
                      _dragEndNorm = _dragStartNorm;
                      _sampledRange = null;
                    });
                  },
                  onPanUpdate: (d) {
                    final norm = Offset(
                      (d.localPosition.dx / displayW).clamp(0.0, 1.0),
                      (d.localPosition.dy / displayH).clamp(0.0, 1.0),
                    );
                    setState(() => _dragEndNorm = norm);
                  },
                  onPanEnd: (_) => _sampleRegion(),
                  child: Stack(
                    children: [
                      Image.memory(
                        widget.imageBytes,
                        width: displayW,
                        height: displayH,
                        fit: BoxFit.fill,
                      ),
                      if (_dragStartNorm != null && _dragEndNorm != null)
                        CustomPaint(
                          size: Size(displayW, displayH),
                          painter: _SelectionRectPainter(
                            start: Offset(
                              _dragStartNorm!.dx * displayW,
                              _dragStartNorm!.dy * displayH,
                            ),
                            end: Offset(
                              _dragEndNorm!.dx * displayW,
                              _dragEndNorm!.dy * displayH,
                            ),
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

  void _sampleRegion() {
    if (_dragStartNorm == null || _dragEndNorm == null) return;
    if (_imageWidth == 0 || _imageHeight == 0) return;

    final x0 = (min(_dragStartNorm!.dx, _dragEndNorm!.dx) * _imageWidth).round();
    final y0 = (min(_dragStartNorm!.dy, _dragEndNorm!.dy) * _imageHeight).round();
    final x1 = (max(_dragStartNorm!.dx, _dragEndNorm!.dx) * _imageWidth).round();
    final y1 = (max(_dragStartNorm!.dy, _dragEndNorm!.dy) * _imageHeight).round();

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

// ---------------------------------------------------------------------------
// Per-image settings bottom sheet
// ---------------------------------------------------------------------------

class _PerImageSettingsSheet extends StatelessWidget {
  const _PerImageSettingsSheet({
    required this.entry,
    required this.globalColor,
    required this.detector,
    required this.onCropSet,
    required this.onCropCleared,
    required this.onColorSet,
    required this.onColorCleared,
  });

  final ImageEntry entry;
  final HsvRange? globalColor;
  final HighlightDetector detector;
  final void Function(HighlightBBox) onCropSet;
  final VoidCallback onCropCleared;
  final void Function(HsvRange) onColorSet;
  final VoidCallback onColorCleared;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom + 16,
        left: 16,
        right: 16,
        top: 16,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // -- Header with thumbnail --
          Row(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.memory(
                  entry.bytes,
                  width: 56,
                  height: 56,
                  fit: BoxFit.cover,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  entry.name,
                  style: theme.textTheme.titleMedium,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          const Divider(),

          // -- Per-image crop --
          ListTile(
            leading: Icon(
              Icons.crop,
              color: entry.hasCrop ? theme.colorScheme.primary : null,
            ),
            title: Text(entry.hasCrop
                ? 'Crop: ${entry.cropRegion!.w}\u00d7${entry.cropRegion!.h}'
                : 'Set Crop Region'),
            subtitle: entry.hasCrop
                ? Text(
                    'at (${entry.cropRegion!.x}, ${entry.cropRegion!.y})',
                    style: theme.textTheme.bodySmall,
                  )
                : const Text('Draw a region on this image'),
            trailing: entry.hasCrop
                ? IconButton(
                    icon: const Icon(Icons.clear),
                    tooltip: 'Clear crop',
                    onPressed: onCropCleared,
                  )
                : null,
            onTap: () async {
              final result = await showDialog<HighlightBBox>(
                context: context,
                builder: (_) =>
                    _CropRegionDialog(imageBytes: entry.bytes),
              );
              if (result != null) onCropSet(result);
            },
          ),

          // -- Per-image colour --
          const Divider(),
          ListTile(
              leading: Icon(
                Icons.palette,
                color: entry.hasColorOverride
                    ? _HomeScreenState._hsvRangeToColor(entry.hsvOverride!)
                    : null,
              ),
              title: Text(entry.hasColorOverride
                  ? 'Custom Colour: ${entry.hsvOverride!.label}'
                  : 'Set Custom Colour'),
              subtitle: entry.hasColorOverride
                  ? null
                  : Text(
                      globalColor != null
                          ? 'Using global: ${globalColor!.label}'
                          : 'No colour set',
                      style: theme.textTheme.bodySmall,
                    ),
              trailing: entry.hasColorOverride
                  ? IconButton(
                      icon: const Icon(Icons.clear),
                      tooltip: 'Use global colour',
                      onPressed: onColorCleared,
                    )
                  : null,
              onTap: () async {
                final result = await showDialog<HsvRange>(
                  context: context,
                  builder: (_) => _ColorSamplerDialog(
                    imageBytes: entry.bytes,
                    detector: detector,
                  ),
                );
                if (result != null) onColorSet(result);
              },
            ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
