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

/// True if `--list-devices` output names at least one device.
///
/// Everything after the `Available devices:` marker counts, except empty
/// lines and the `(none)` placeholder. Output before the marker (loading
/// logs, backend errors) never counts as a device.
bool listDevicesFoundDevice(String output) {
  final markerAt = output.indexOf(_devicesMarker);
  if (markerAt < 0) return false;

  final after = output.substring(markerAt + _devicesMarker.length);
  return after
      .split('\n')
      .map((line) => line.trim())
      .any((line) => line.isNotEmpty && line != _noDeviceLine);
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
  return errorLine.length > 160 ? '${errorLine.substring(0, 157)}...' : errorLine;
}
