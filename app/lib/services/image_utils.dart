import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Normalize image orientation by reading EXIF data and applying the
/// appropriate rotation / flip so the image displays upright.
///
/// Returns the normalized bytes as JPEG (quality 95). If no EXIF orientation
/// tag is present, returns the original bytes unchanged.
Future<Uint8List> normalizeOrientation(Uint8List bytes) async {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) return bytes;

  // No orientation tag → nothing to do.
  if (!decoded.exif.imageIfd.hasOrientation ||
      decoded.exif.imageIfd.orientation == 1) {
    return bytes;
  }

  final normalized = img.bakeOrientation(decoded);
  return Uint8List.fromList(img.encodeJpg(normalized, quality: 95));
}
