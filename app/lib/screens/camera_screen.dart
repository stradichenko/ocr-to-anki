import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/system_channel.dart';

/// Full-screen in-app camera with capture, flash toggle, and camera switch.
/// Returns the captured image bytes via [onImageCaptured].
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, this.onImageCaptured});

  /// Called when the user captures an image. If null, the screen pops
  /// and returns the bytes via Navigator.pop.
  final void Function(Uint8List bytes)? onImageCaptured;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _capturing = false;
  int _cameraIndex = 0;
  FlashMode _flashMode = FlashMode.off;

  /// Whether the camera permission has been permanently denied.
  /// When true we show a full-screen prompt instead of the camera preview.
  bool _permissionDenied = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      ctrl.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    if (!Platform.isAndroid) return;

    final granted = await SystemChannel.isCameraGranted();
    if (!granted) {
      final didGrant = await SystemChannel.requestCameraPermission();
      if (!didGrant) {
        if (mounted) {
          setState(() => _permissionDenied = true);
        }
        return;
      }
    }

    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        if (mounted) Navigator.pop(context);
        return;
      }

      await _controller?.dispose();
      _controller = CameraController(
        _cameras[_cameraIndex],
        ResolutionPreset.high,
        enableAudio: false,
      );
      await _controller!.initialize();
      await _controller!.setFlashMode(_flashMode);

      if (mounted) setState(() {});
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Camera error: $e')),
        );
        Navigator.pop(context);
      }
    }
  }

  Future<void> _capture() async {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized || _capturing) return;
    setState(() => _capturing = true);

    try {
      final file = await ctrl.takePicture();
      final bytes = await file.readAsBytes();

      if (widget.onImageCaptured != null) {
        widget.onImageCaptured!(bytes);
      } else if (mounted) {
        Navigator.pop(context, bytes);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Capture failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  void _toggleFlash() {
    final next = _flashMode == FlashMode.off
        ? FlashMode.auto
        : _flashMode == FlashMode.auto
            ? FlashMode.torch
            : FlashMode.off;
    setState(() => _flashMode = next);
    _controller?.setFlashMode(next);
  }

  void _switchCamera() {
    if (_cameras.length < 2) return;
    _cameraIndex = (_cameraIndex + 1) % _cameras.length;
    _initCamera();
  }

  Future<void> _openGallery() async {
    final picker = ImagePicker();
    final files = await picker.pickMultiImage();
    if (!mounted) return;
    if (files.isNotEmpty) {
      final allBytes = <Uint8List>[];
      for (final f in files) {
        final b = await f.readAsBytes();
        allBytes.add(b);
      }
      if (mounted) Navigator.pop(context, allBytes);
    }
  }

  IconData get _flashIcon => switch (_flashMode) {
    FlashMode.auto => Icons.flash_auto,
    FlashMode.torch => Icons.flash_on,
    _ => Icons.flash_off,
  };

  @override
  Widget build(BuildContext context) {
    final ctrl = _controller;

    // Permission permanently denied — show a full-screen prompt instead of
    // a SnackBar that would persist after popping.
    if (_permissionDenied) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.videocam_off,
                      size: 64, color: Colors.white70),
                  const SizedBox(height: 24),
                  const Text(
                    'Camera permission required',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'OCR to Anki needs camera access to capture images '
                    'of vocabulary lists.',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 32),
                  FilledButton.icon(
                    onPressed: () {
                      SystemChannel.openAppDetailsSettings();
                    },
                    icon: const Icon(Icons.settings),
                    label: const Text('Open Settings'),
                  ),
                  const SizedBox(height: 12),
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Go Back'),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            // Preview
            if (ctrl != null && ctrl.value.isInitialized)
              CameraPreview(ctrl)
            else
              const Center(child: CircularProgressIndicator()),

            // Top controls
            Positioned(
              top: 8,
              left: 8,
              right: 8,
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: () => Navigator.pop(context),
                  ),
                  const Spacer(),
                  if (_cameras.length > 1)
                    IconButton(
                      icon: const Icon(
                        Icons.flip_camera_ios,
                        color: Colors.white,
                      ),
                      onPressed: _switchCamera,
                    ),
                  IconButton(
                    icon: Icon(_flashIcon, color: Colors.white),
                    onPressed: _toggleFlash,
                  ),
                ],
              ),
            ),

            // Bottom controls
            Positioned(
              bottom: 32,
              left: 24,
              right: 24,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // Gallery shortcut
                  IconButton(
                    icon: const Icon(Icons.photo_library, color: Colors.white),
                    onPressed: _openGallery,
                  ),

                  // Capture button
                  GestureDetector(
                    onTap: _capturing ? null : _capture,
                    child: Container(
                      width: 72,
                      height: 72,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 4),
                      ),
                      child: Center(
                        child: Container(
                          width: 60,
                          height: 60,
                          decoration: const BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.white,
                          ),
                          child: _capturing
                              ? const Padding(
                                  padding: EdgeInsets.all(16),
                                  child: CircularProgressIndicator(
                                    strokeWidth: 3,
                                  ),
                                )
                              : const SizedBox.shrink(),
                        ),
                      ),
                    ),
                  ),

                  // Spacer to balance layout
                  const SizedBox(width: 48),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
