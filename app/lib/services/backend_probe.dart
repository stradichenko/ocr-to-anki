/// Helpers for probing llama.cpp backend binaries.
///
/// Backend detection runs the shipped binaries themselves (`--list-devices`)
/// rather than probing GPU libraries from Dart. Dart-side checks mislead in
/// both directions: vendor libraries are invisible to the app's linker
/// namespace until declared in the manifest, and a library that loads does
/// not mean the backend can use it — a no-op symbol table reads exactly like
/// "no devices present". The binary is the only source of truth.
library;

/// Output of `llama.cpp --list-devices` when no usable device was found.
const String _noDeviceLine = '(none)';

/// Marker line that starts the device list in `--list-devices` output.
const String _devicesMarker = 'Available devices:';

/// llama.cpp log lines, e.g. `0.00.059.114 E ggml_vulkan: Error: ...`.
///
/// Log lines go to stderr while the device list goes to stdout, but anything
/// that ends up in one combined buffer puts log lines after the device list.
final RegExp _logLine = RegExp(r'^\d{2}\.\d{2}\.\d{2}\.\d{3} [IWE] ');

/// True if `--list-devices` output names at least one device.
///
/// Only the lines directly after the `Available devices:` marker count. The
/// first `(none)` placeholder settles it as false, and the first llama.cpp
/// log line ends the list — both guards exist because a combined
/// stdout+stderr buffer carries error lines after an empty device list, and
/// reading those as devices is exactly how a Vulkan 1.1 device once got
/// reported as Vulkan-capable.
bool listDevicesFoundDevice(String output) {
  final markerAt = output.indexOf(_devicesMarker);
  if (markerAt < 0) return false;

  for (final line in output.substring(markerAt + _devicesMarker.length).split('\n')) {
    final trimmed = line.trim();
    if (trimmed.isEmpty) continue;
    if (trimmed == _noDeviceLine) return false;
    if (_logLine.hasMatch(trimmed)) return false;
    return true;
  }
  return false;
}

/// First error-looking line of backend output, for probe diagnostics.
///
/// Prefers lines tagged as errors by llama.cpp's log format (`E ...`),
/// falls back to the first non-empty line, and truncates to a single
/// diagnostic-sized line.
String firstBackendErrorLine(String output) {
  final lines =
      output.split('\n').map((line) => line.trim()).where((line) => line.isNotEmpty).toList();
  if (lines.isEmpty) return '(no output)';

  final errorLine = lines.firstWhere(
    (line) => line.contains(' E ') || line.toLowerCase().contains('error'),
    orElse: () => lines.first,
  );
  return errorLine.length > 400 ? '${errorLine.substring(0, 397)}...' : errorLine;
}
