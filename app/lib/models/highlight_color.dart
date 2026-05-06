/// Highlight color definitions matching the Python backend's HSV ranges.
///
/// Each color maps to an HSV range used for detecting highlighted text
/// regions in images. The ranges mirror [AdaptiveHighlightCropper.BASE_COLORS]
/// from `src/preprocessing/highlight_cropper.py`.
enum HighlightColor {
  yellow(
    label: 'Yellow',
    hueCenter: 25,
    hueRange: 15,
    satMin: 80,
    satMax: 255,
    valMin: 120,
    valMax: 255,
  ),
  orange(
    label: 'Orange',
    hueCenter: 12,
    hueRange: 10,
    satMin: 100,
    satMax: 255,
    valMin: 120,
    valMax: 255,
  ),
  red(
    label: 'Red',
    hueCenter: 0,
    hueRange: 10,
    satMin: 100,
    satMax: 255,
    valMin: 100,
    valMax: 255,
  ),
  green(
    label: 'Green',
    hueCenter: 55,
    hueRange: 20,
    satMin: 60,
    satMax: 255,
    valMin: 100,
    valMax: 255,
  ),
  blue(
    label: 'Blue',
    hueCenter: 105,
    hueRange: 20,
    satMin: 60,
    satMax: 255,
    valMin: 100,
    valMax: 255,
  ),
  purple(
    label: 'Purple',
    hueCenter: 135,
    hueRange: 20,
    satMin: 60,
    satMax: 255,
    valMin: 100,
    valMax: 255,
  );

  const HighlightColor({
    required this.label,
    required this.hueCenter,
    required this.hueRange,
    required this.satMin,
    required this.satMax,
    required this.valMin,
    required this.valMax,
  });

  final String label;

  /// Centre hue in OpenCV scale (0-180).
  final double hueCenter;

  /// Acceptable range around [hueCenter].
  final double hueRange;

  final double satMin;
  final double satMax;
  final double valMin;
  final double valMax;
}

/// An arbitrary HSV colour range for highlight detection.
///
/// Preset ranges come from [HighlightColor] enum values via
/// [HsvRange.fromPreset]; custom ranges can be sampled from an image
/// region using [HighlightDetector.sampleHsvRange].
class HsvRange {
  const HsvRange({
    required this.label,
    required this.hueCenter,
    required this.hueRange,
    required this.satMin,
    required this.satMax,
    required this.valMin,
    required this.valMax,
  });

  /// Build from a preset [HighlightColor].
  factory HsvRange.fromPreset(HighlightColor c) => HsvRange(
        label: c.label,
        hueCenter: c.hueCenter,
        hueRange: c.hueRange,
        satMin: c.satMin,
        satMax: c.satMax,
        valMin: c.valMin,
        valMax: c.valMax,
      );

  final String label;

  /// Centre hue in OpenCV scale (0-180).
  final double hueCenter;

  /// Acceptable range around [hueCenter].
  final double hueRange;

  final double satMin;
  final double satMax;
  final double valMin;
  final double valMax;

  Map<String, dynamic> toJson() => {
        'label': label,
        'hueCenter': hueCenter,
        'hueRange': hueRange,
        'satMin': satMin,
        'satMax': satMax,
        'valMin': valMin,
        'valMax': valMax,
      };

  factory HsvRange.fromJson(Map<String, dynamic> json) => HsvRange(
        label: json['label'] as String,
        hueCenter: (json['hueCenter'] as num).toDouble(),
        hueRange: (json['hueRange'] as num).toDouble(),
        satMin: (json['satMin'] as num).toDouble(),
        satMax: (json['satMax'] as num).toDouble(),
        valMin: (json['valMin'] as num).toDouble(),
        valMax: (json['valMax'] as num).toDouble(),
      );
}
