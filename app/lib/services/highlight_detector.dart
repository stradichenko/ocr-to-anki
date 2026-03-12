import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../models/highlight_color.dart';

/// Pure-Dart port of `src/preprocessing/highlight_cropper.py`.
///
/// Uses the `image` package instead of OpenCV. Converts each pixel to HSV and
/// builds a mask for the requested highlight colour, then finds bounding boxes
/// of connected regions via a simple flood-fill label pass.
class HighlightDetector {
  HighlightDetector({
    this.colorTolerance = 25,
    this.minArea = 200,
    this.padding = 0,
    this.mergeNearby = true,
    this.mergeDistance = 10,
    this.adaptiveMode = false,
  });

  final int colorTolerance;
  final int minArea;
  final int padding;
  final bool mergeNearby;
  final int mergeDistance;
  final bool adaptiveMode;

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Detect highlighted regions and return cropped sub-images.
  ///
  /// Returns a list of PNG-encoded byte arrays, one per detected region.
  List<Uint8List> detectAndCrop({
    required Uint8List imageBytes,
    required HsvRange color,
  }) {
    final image = img.decodeImage(imageBytes);
    if (image == null) return [];

    final mask = _buildMask(image, color);
    final boxes = _findBoundingBoxes(mask, image.width, image.height);

    final merged = mergeNearby ? _mergeNearbyBoxes(boxes) : boxes;

    final results = <Uint8List>[];
    for (final box in merged) {
      final x0 = (box.x - padding).clamp(0, image.width - 1);
      final y0 = (box.y - padding).clamp(0, image.height - 1);
      final x1 = (box.x + box.w + padding).clamp(0, image.width);
      final y1 = (box.y + box.h + padding).clamp(0, image.height);

      final cropped = img.copyCrop(
        image,
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
      );
      results.add(Uint8List.fromList(img.encodePng(cropped)));
    }

    return results;
  }

  /// Crop specific bounding boxes from an image **without** re-detecting.
  ///
  /// The provided [boxes] are used as-is (no additional padding is applied).
  /// Use this when boxes already include padding (e.g. from the bounding-box
  /// preview).
  List<Uint8List> cropBoxes({
    required Uint8List imageBytes,
    required List<HighlightBBox> boxes,
  }) {
    final image = img.decodeImage(imageBytes);
    if (image == null) return [];

    final results = <Uint8List>[];
    for (final box in boxes) {
      final x0 = box.x.clamp(0, image.width - 1);
      final y0 = box.y.clamp(0, image.height - 1);
      final w = box.w.clamp(1, image.width - x0);
      final h = box.h.clamp(1, image.height - y0);

      final cropped = img.copyCrop(image, x: x0, y: y0, width: w, height: h);
      results.add(Uint8List.fromList(img.encodePng(cropped)));
    }

    return results;
  }

  /// Stitch a list of crop images into a single vertical montage.
  ///
  /// A 4 px red separator is drawn between crops so the vision model
  /// clearly sees the boundary.  Returns the montage as PNG bytes.
  /// This lets us OCR all crops in a single inference call instead of
  /// launching a new subprocess per crop.
  ///
  /// When [maxWidth] is > 0 each crop is downscaled so its width does
  /// not exceed that value.  This dramatically reduces the number of
  /// vision tokens and speeds up prompt eval on iGPUs.
  static Uint8List buildMontage(List<Uint8List> crops, {int maxWidth = 0}) {
    if (crops.length == 1 && maxWidth <= 0) return crops.first;

    var images = crops
        .map((b) => img.decodeImage(b))
        .whereType<img.Image>()
        .toList();
    if (images.isEmpty) return crops.first;

    // Downscale each crop if requested.
    if (maxWidth > 0) {
      images = images.map((im) {
        if (im.width <= maxWidth) return im;
        final scale = maxWidth / im.width;
        return img.copyResize(
          im,
          width: maxWidth,
          height: (im.height * scale).round(),
          interpolation: img.Interpolation.average,
        );
      }).toList();
    }

    if (images.length == 1) {
      return Uint8List.fromList(img.encodePng(images.first));
    }

    const sep = 4;
    final totalW = images.map((i) => i.width).reduce(max);
    final totalH = images.fold<int>(0, (s, i) => s + i.height) +
        sep * (images.length - 1);

    final canvas = img.Image(width: totalW, height: totalH);
    // Fill white
    img.fill(canvas, color: img.ColorRgba8(255, 255, 255, 255));

    var yOff = 0;
    for (var i = 0; i < images.length; i++) {
      img.compositeImage(canvas, images[i], dstY: yOff);
      yOff += images[i].height;
      if (i < images.length - 1) {
        // Red separator line
        img.fillRect(canvas,
            x1: 0, y1: yOff, x2: totalW, y2: yOff + sep,
            color: img.ColorRgba8(255, 0, 0, 255));
        yOff += sep;
      }
    }

    return Uint8List.fromList(img.encodePng(canvas));
  }

  /// Crop the given image to [region] and return the result as PNG bytes.
  static Uint8List cropRegion({
    required Uint8List imageBytes,
    required HighlightBBox region,
  }) {
    final image = img.decodeImage(imageBytes);
    if (image == null) return imageBytes;

    final x0 = region.x.clamp(0, image.width - 1);
    final y0 = region.y.clamp(0, image.height - 1);
    final w = region.w.clamp(1, image.width - x0);
    final h = region.h.clamp(1, image.height - y0);

    final cropped = img.copyCrop(image, x: x0, y: y0, width: w, height: h);
    return Uint8List.fromList(img.encodePng(cropped));
  }

  /// Return just the bounding boxes (without cropping) for visualisation.
  List<HighlightBBox> detectBoxes({
    required Uint8List imageBytes,
    required HsvRange color,
  }) {
    final image = img.decodeImage(imageBytes);
    if (image == null) return [];

    final mask = _buildMask(image, color);
    final boxes = _findBoundingBoxes(mask, image.width, image.height);

    return mergeNearby ? _mergeNearbyBoxes(boxes) : boxes;
  }

  /// Sample pixels inside a rectangle and return an [HsvRange] describing
  /// the dominant colour.  Used by the custom colour-picker UI.
  HsvRange sampleHsvRange({
    required Uint8List imageBytes,
    required int x,
    required int y,
    required int w,
    required int h,
    String label = 'Custom',
  }) {
    final image = img.decodeImage(imageBytes);
    if (image == null) throw ArgumentError('Failed to decode image');

    final x0 = x.clamp(0, image.width - 1);
    final y0 = y.clamp(0, image.height - 1);
    final x1 = (x + w).clamp(0, image.width);
    final y1 = (y + h).clamp(0, image.height);

    final hues = <double>[];
    final sats = <double>[];
    final vals = <double>[];

    for (var py = y0; py < y1; py++) {
      for (var px = x0; px < x1; px++) {
        final pixel = image.getPixel(px, py);
        final hsv =
            _rgbToHsv(pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt());
        hues.add(hsv.h);
        sats.add(hsv.s);
        vals.add(hsv.v);
      }
    }

    if (hues.isEmpty) throw StateError('No pixels in selection');

    hues.sort();
    sats.sort();
    vals.sort();

    // Use median for hue centre, IQR-based spread + padding for range.
    final hueCenter = hues[hues.length ~/ 2];
    final hueQ1 = hues[hues.length ~/ 4];
    final hueQ3 = hues[3 * hues.length ~/ 4];
    final hueRange = ((hueQ3 - hueQ1) / 2 + 10).clamp(5.0, 90.0);

    return HsvRange(
      label: label,
      hueCenter: hueCenter,
      hueRange: hueRange,
      satMin: (sats.first - 20).clamp(0, 255),
      satMax: (sats.last + 20).clamp(0, 255),
      valMin: (vals.first - 30).clamp(0, 255),
      valMax: (vals.last + 20).clamp(0, 255),
    );
  }

  // ---------------------------------------------------------------------------
  // HSV mask construction
  // ---------------------------------------------------------------------------

  /// Build a binary mask (same dimensions as [image]) where `true` means the
  /// pixel falls within the HSV range for [color].
  List<bool> _buildMask(img.Image image, HsvRange color) {
    final w = image.width;
    final h = image.height;
    final mask = List<bool>.filled(w * h, false);

    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final pixel = image.getPixel(x, y);
        final r = pixel.r.toInt();
        final g = pixel.g.toInt();
        final b = pixel.b.toInt();

        final hsv = _rgbToHsv(r, g, b);

        if (_inRange(hsv, color)) {
          mask[y * w + x] = true;
        }
      }
    }

    // Simple morphological open: erode then dilate (3x3 kernel, 2 iterations).
    var cleaned = _erode(mask, w, h, iterations: 2);
    cleaned = _dilate(cleaned, w, h, iterations: 2);
    // Close: dilate then erode (3x3 kernel, 1 iteration).
    cleaned = _dilate(cleaned, w, h, kernelSize: 3, iterations: 1);
    cleaned = _erode(cleaned, w, h, kernelSize: 3, iterations: 1);

    return cleaned;
  }

  /// Convert RGB (0-255) to HSV in OpenCV scale: H 0-180, S 0-255, V 0-255.
  _Hsv _rgbToHsv(int r, int g, int b) {
    final rf = r / 255.0;
    final gf = g / 255.0;
    final bf = b / 255.0;

    final cmax = max(rf, max(gf, bf));
    final cmin = min(rf, min(gf, bf));
    final delta = cmax - cmin;

    double h = 0;
    if (delta != 0) {
      if (cmax == rf) {
        h = 60 * (((gf - bf) / delta) % 6);
      } else if (cmax == gf) {
        h = 60 * ((bf - rf) / delta + 2);
      } else {
        h = 60 * ((rf - gf) / delta + 4);
      }
    }
    if (h < 0) h += 360;

    final s = cmax == 0 ? 0.0 : delta / cmax;
    final v = cmax;

    // Convert to OpenCV scale.
    return _Hsv(
      h: (h / 2).roundToDouble(), // 0-180
      s: (s * 255).roundToDouble(), // 0-255
      v: (v * 255).roundToDouble(), // 0-255
    );
  }

  bool _inRange(_Hsv hsv, HsvRange range) {
    final tolerance =
        adaptiveMode ? colorTolerance.toDouble() : range.hueRange;
    final hLow = range.hueCenter - tolerance;
    final hHigh = range.hueCenter + tolerance;

    bool hueOk;
    if (hLow < 0) {
      // Wraps below 0 (e.g. red: centre 0, range 10 → 170-180 ∪ 0-10).
      hueOk = hsv.h >= (hLow + 180) || hsv.h <= hHigh;
    } else if (hHigh > 180) {
      // Wraps above 180.
      hueOk = hsv.h >= hLow || hsv.h <= (hHigh - 180);
    } else {
      hueOk = hsv.h >= hLow && hsv.h <= hHigh;
    }

    if (!hueOk) return false;
    if (hsv.s < range.satMin || hsv.s > range.satMax) return false;
    if (hsv.v < range.valMin || hsv.v > range.valMax) return false;
    return true;
  }

  // ---------------------------------------------------------------------------
  // Morphological ops (simple box kernels on a flat bool list)
  // ---------------------------------------------------------------------------

  List<bool> _erode(List<bool> mask, int w, int h,
      {int kernelSize = 3, int iterations = 1}) {
    var current = mask;
    final r = kernelSize ~/ 2;
    for (var i = 0; i < iterations; i++) {
      final next = List<bool>.filled(w * h, false);
      for (var y = r; y < h - r; y++) {
        for (var x = r; x < w - r; x++) {
          var allSet = true;
          outer:
          for (var ky = -r; ky <= r; ky++) {
            for (var kx = -r; kx <= r; kx++) {
              if (!current[(y + ky) * w + (x + kx)]) {
                allSet = false;
                break outer;
              }
            }
          }
          next[y * w + x] = allSet;
        }
      }
      current = next;
    }
    return current;
  }

  List<bool> _dilate(List<bool> mask, int w, int h,
      {int kernelSize = 3, int iterations = 1}) {
    var current = mask;
    final r = kernelSize ~/ 2;
    for (var i = 0; i < iterations; i++) {
      final next = List<bool>.filled(w * h, false);
      for (var y = r; y < h - r; y++) {
        for (var x = r; x < w - r; x++) {
          if (current[y * w + x]) {
            for (var ky = -r; ky <= r; ky++) {
              for (var kx = -r; kx <= r; kx++) {
                next[(y + ky) * w + (x + kx)] = true;
              }
            }
          }
        }
      }
      current = next;
    }
    return current;
  }

  // ---------------------------------------------------------------------------
  // Connected-component labelling → bounding boxes
  // ---------------------------------------------------------------------------

  List<HighlightBBox> _findBoundingBoxes(List<bool> mask, int w, int h) {
    final labels = List<int>.filled(w * h, 0);
    var nextLabel = 1;
    final boxMap = <int, HighlightBBox>{};

    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        if (mask[y * w + x] && labels[y * w + x] == 0) {
          // BFS flood fill.
          final label = nextLabel++;
          final queue = <int>[y * w + x];
          var minX = x, maxX = x, minY = y, maxY = y;
          var area = 0;

          while (queue.isNotEmpty) {
            final idx = queue.removeLast();
            if (labels[idx] != 0) continue;
            labels[idx] = label;
            area++;

            final px = idx % w;
            final py = idx ~/ w;
            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;

            // 4-connected neighbours.
            if (px > 0 && mask[idx - 1] && labels[idx - 1] == 0) {
              queue.add(idx - 1);
            }
            if (px < w - 1 && mask[idx + 1] && labels[idx + 1] == 0) {
              queue.add(idx + 1);
            }
            if (py > 0 && mask[idx - w] && labels[idx - w] == 0) {
              queue.add(idx - w);
            }
            if (py < h - 1 && mask[idx + w] && labels[idx + w] == 0) {
              queue.add(idx + w);
            }
          }

          if (area >= minArea) {
            final bw = maxX - minX + 1;
            final bh = maxY - minY + 1;
            final aspectRatio = bw / bh;
            if (aspectRatio > 0.1 && aspectRatio < 10) {
              boxMap[label] = HighlightBBox(minX, minY, bw, bh);
            }
          }
        }
      }
    }

    return boxMap.values.toList();
  }

  // ---------------------------------------------------------------------------
  // Merge nearby boxes
  // ---------------------------------------------------------------------------

  List<HighlightBBox> _mergeNearbyBoxes(List<HighlightBBox> boxes) {
    if (boxes.length <= 1) return boxes;

    final used = <int>{};
    final merged = <HighlightBBox>[];

    for (var i = 0; i < boxes.length; i++) {
      if (used.contains(i)) continue;
      used.add(i);

      var current = boxes[i];

      for (var j = i + 1; j < boxes.length; j++) {
        if (used.contains(j)) continue;

        final other = boxes[j];
        final xDist = max(0, max(current.x, other.x) -
            min(current.x + current.w, other.x + other.w));
        final yDist = max(0, max(current.y, other.y) -
            min(current.y + current.h, other.y + other.h));
        final dist = sqrt(xDist * xDist + yDist * yDist);

        if (dist <= mergeDistance) {
          used.add(j);
          final nx = min(current.x, other.x);
          final ny = min(current.y, other.y);
          final nx2 = max(current.x + current.w, other.x + other.w);
          final ny2 = max(current.y + current.h, other.y + other.h);
          current = HighlightBBox(nx, ny, nx2 - nx, ny2 - ny);
        }
      }

      merged.add(current);
    }

    return merged;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class _Hsv {
  const _Hsv({required this.h, required this.s, required this.v});
  final double h;
  final double s;
  final double v;
}

class HighlightBBox {
  const HighlightBBox(this.x, this.y, this.w, this.h);
  final int x;
  final int y;
  final int w;
  final int h;
}
