import 'dart:typed_data';

import '../services/highlight_detector.dart';
import 'highlight_color.dart';

/// Represents a single queued image with optional per-image overrides.
///
/// When [cropRegion] or [hsvOverride] are `null`, the global settings
/// from `_HomeScreenState` are used.  When set, they override globals for
/// this image only.
class ImageEntry {
  ImageEntry({
    required this.bytes,
    required this.name,
    this.cropRegion,
    this.hsvOverride,
    this.termLanguage,
  });

  /// Raw image bytes.
  Uint8List bytes;

  /// Display name (usually the original filename).
  final String name;

  /// Optional per-image crop region. Overrides the global crop when set.
  HighlightBBox? cropRegion;

  /// Optional per-image HSV colour range. Overrides the global highlight
  /// colour when set.
  HsvRange? hsvOverride;

  /// Optional per-image term language (e.g. 'french', 'spanish').
  /// When set, overrides the global termLanguage for words from this image.
  String? termLanguage;

  /// Whether this image has any per-image override applied.
  bool get hasOverrides =>
      cropRegion != null || hsvOverride != null || termLanguage != null;

  /// Whether a per-image crop region is set.
  bool get hasCrop => cropRegion != null;

  /// Whether a per-image colour override is set.
  bool get hasColorOverride => hsvOverride != null;

  /// Whether a per-image term language is set.
  bool get hasLanguageOverride => termLanguage != null;

  /// Create a copy with selectively updated fields.
  ImageEntry copyWith({
    Uint8List? bytes,
    String? name,
    HighlightBBox? cropRegion,
    HsvRange? hsvOverride,
    String? termLanguage,
    bool clearCrop = false,
    bool clearHsvOverride = false,
    bool clearTermLanguage = false,
  }) {
    return ImageEntry(
      bytes: bytes ?? this.bytes,
      name: name ?? this.name,
      cropRegion: clearCrop ? null : (cropRegion ?? this.cropRegion),
      hsvOverride:
          clearHsvOverride ? null : (hsvOverride ?? this.hsvOverride),
      termLanguage:
          clearTermLanguage ? null : (termLanguage ?? this.termLanguage),
    );
  }
}
