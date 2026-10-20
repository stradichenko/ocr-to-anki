import 'package:flutter_test/flutter_test.dart';

import 'package:ocr_to_anki/services/backend_probe.dart';

void main() {
  group('listDevicesFoundDevice', () {
    test('accepts output that lists a device', () {
      const out = '''
0.00.100.000 I ggml_opencl: found 1 platform.
Available devices:
  OpenCL 0: QUALCOMM Adreno(TM)
''';
      expect(listDevicesFoundDevice(out), isTrue);
    });

    test('rejects the (none) placeholder', () {
      const out = '''
0.00.008.376 E ggml_opencl: could find any OpenCL devices.
Available devices:
  (none)
''';
      expect(listDevicesFoundDevice(out), isFalse);
    });

    test('rejects output without the device marker', () {
      const out = 'error: unknown argument --list-devices\n';
      expect(listDevicesFoundDevice(out), isFalse);
    });

    test('rejects empty output', () {
      expect(listDevicesFoundDevice(''), isFalse);
    });

    test('does not count load logs as devices', () {
      const out = '''
0.00.001.000 I ggml_vulkan: found 1 Vulkan devices:
0.00.001.100 E ggml_vulkan: Error: Vulkan 1.2 required.
Available devices:
  (none)
''';
      expect(listDevicesFoundDevice(out), isFalse);
    });

    test('accepts multiple devices', () {
      const out = '''
Available devices:
  OpenCL 0: QUALCOMM Adreno(TM)
  OpenCL 1: Mali-G78
''';
      expect(listDevicesFoundDevice(out), isTrue);
    });
  });

  group('firstBackendErrorLine', () {
    test('prefers the llama.cpp error line', () {
      const out = '''
0.00.005.000 I srv  llama_server: initializing ...
0.00.059.114 E ggml_vulkan: Error: Vulkan 1.2 required.
''';
      expect(firstBackendErrorLine(out),
          contains('ggml_vulkan: Error: Vulkan 1.2 required.'));
    });

    test('falls back to the first non-empty line', () {
      const out = '\nlibOpenCL.so not found\n';
      expect(firstBackendErrorLine(out), 'libOpenCL.so not found');
    });

    test('truncates very long lines', () {
      final out = 'E ${'x' * 500}';
      final line = firstBackendErrorLine(out);
      expect(line.length, 160);
      expect(line.endsWith('...'), isTrue);
    });

    test('reports no output distinctly', () {
      expect(firstBackendErrorLine(''), '(no output)');
    });
  });
}
