import 'dart:convert';
import 'dart:io';

import 'package:archive/archive.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Information about an available update.
class UpdateInfo {
  const UpdateInfo({
    required this.hasUpdate,
    required this.currentVersion,
    required this.latestVersion,
    this.downloadUrl = '',
    this.releaseNotes = '',
    this.publishedAt = '',
  });

  final bool hasUpdate;
  final String currentVersion;
  final String latestVersion;
  final String downloadUrl;
  final String releaseNotes;
  final String publishedAt;
}

/// Progress of a download operation.
class DownloadProgress {
  const DownloadProgress({
    required this.downloadedBytes,
    required this.totalBytes,
    required this.fileName,
  });

  final int downloadedBytes;
  final int totalBytes;
  final String fileName;

  double get fraction => totalBytes > 0 ? downloadedBytes / totalBytes : 0;
}

/// Service that checks for app updates on GitHub Releases and applies them.
///
/// Desktop path: uses the FastAPI backend as a proxy.
/// Android path: queries GitHub API directly and installs the downloaded APK.
class UpdateService {
  UpdateService({
    required this.serverUrl,
    required this.currentVersion,
    this.isAndroid = false,
  });

  /// URL of the FastAPI backend (used for `/update/check` proxy on desktop).
  final String serverUrl;

  /// Current app version from pubspec.yaml (e.g. "0.4.4").
  final String currentVersion;

  /// Whether this is the Android variant (uses direct GitHub API + APK install).
  final bool isAndroid;

  static const _githubApiLatest =
      'https://api.github.com/repos/stradichenko/ocr-to-anki/releases/latest';

  /// HTTP client reused across calls.
  final _client = http.Client();

  /// Check whether a newer release is available.
  ///
  /// On desktop, proxies through the FastAPI backend.
  /// On Android, queries the GitHub API directly.
  Future<UpdateInfo> checkForUpdate() async {
    if (isAndroid) {
      return _checkGitHubRelease();
    }
    return _checkViaBackend();
  }

  /// Desktop path: ask the FastAPI backend to compare versions.
  Future<UpdateInfo> _checkViaBackend() async {
    final uri = Uri.parse(
      '$serverUrl/update/check?current_version=$currentVersion',
    );
    final response = await _client.get(uri).timeout(
      const Duration(seconds: 30),
    );

    if (response.statusCode != 200) {
      throw Exception('Update check failed: ${response.statusCode}');
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    return UpdateInfo(
      hasUpdate: body['has_update'] == true,
      currentVersion: body['current_version'] as String? ?? currentVersion,
      latestVersion: body['latest_version'] as String? ?? currentVersion,
      downloadUrl: body['download_url'] as String? ?? '',
      releaseNotes: body['release_notes'] as String? ?? '',
      publishedAt: body['published_at'] as String? ?? '',
    );
  }

  /// Android path: query GitHub Releases API directly.
  Future<UpdateInfo> _checkGitHubRelease() async {
    final response = await _client
        .get(
          Uri.parse(_githubApiLatest),
          headers: {'Accept': 'application/vnd.github+json'},
        )
        .timeout(const Duration(seconds: 30));

    if (response.statusCode != 200) {
      throw Exception('GitHub API error: ${response.statusCode}');
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    final tag = body['tag_name'] as String? ?? '';
    final latestVersion = tag.startsWith('v') ? tag.substring(1) : tag;

    // Find the Android zip asset.
    final assets = body['assets'] as List<dynamic>? ?? [];
    String downloadUrl = '';
    for (final a in assets) {
      final name = (a as Map<String, dynamic>)['name'] as String? ?? '';
      if (name.contains('android') && name.endsWith('.zip')) {
        downloadUrl = a['browser_download_url'] as String? ?? '';
        break;
      }
    }

    final hasUpdate = _versionCompare(latestVersion, currentVersion) > 0;

    return UpdateInfo(
      hasUpdate: hasUpdate,
      currentVersion: currentVersion,
      latestVersion: latestVersion,
      downloadUrl: downloadUrl,
      releaseNotes: body['body'] as String? ?? '',
      publishedAt: body['published_at'] as String? ?? '',
    );
  }

  /// Compare two semver strings. Returns >0 if [a] > [b], 0 if equal, <0 otherwise.
  int _versionCompare(String a, String b) {
    final aParts = a.split('+').first.split('.');
    final bParts = b.split('+').first.split('.');
    for (var i = 0; i < aParts.length && i < bParts.length; i++) {
      final an = int.tryParse(aParts[i]) ?? 0;
      final bn = int.tryParse(bParts[i]) ?? 0;
      if (an != bn) return an - bn;
    }
    return aParts.length - bParts.length;
  }

  /// Download the update archive, reporting progress via [onProgress].
  ///
  /// Returns the path to the downloaded archive file.
  Future<String> downloadUpdate(
    String url, {
    required void Function(DownloadProgress) onProgress,
  }) async {
    final tmpDir = await getTemporaryDirectory();
    final fileName = p.basename(Uri.parse(url).path);
    final destPath = p.join(tmpDir.path, 'ocr-to-anki-update', fileName);
    final destFile = File(destPath);
    await destFile.parent.create(recursive: true);

    final request = http.Request('GET', Uri.parse(url));
    final streamed = await _client.send(request);

    if (streamed.statusCode != 200) {
      throw Exception('Download failed: HTTP ${streamed.statusCode}');
    }

    final total = int.parse(
      streamed.headers['content-length'] ?? '0',
    );
    final sink = destFile.openWrite();
    var downloaded = 0;

    try {
      await for (final chunk in streamed.stream) {
        sink.add(chunk);
        downloaded += chunk.length;
        onProgress(DownloadProgress(
          downloadedBytes: downloaded,
          totalBytes: total,
          fileName: fileName,
        ));
      }
    } finally {
      await sink.close();
    }

    return destPath;
  }

  /// Extract the Android zip and return the path to the APK inside.
  Future<String> extractAndroidApk(String zipPath) async {
    final bytes = await File(zipPath).readAsBytes();
    final archive = ZipDecoder().decodeBytes(bytes);

    final tmpDir = await getTemporaryDirectory();
    final extractDir = Directory(p.join(tmpDir.path, 'ocr-to-anki-update', 'extracted'));
    await extractDir.create(recursive: true);

    String? apkPath;
    for (final file in archive) {
      final outPath = p.join(extractDir.path, file.name);
      if (file.isFile) {
        final outFile = File(outPath);
        await outFile.parent.create(recursive: true);
        await outFile.writeAsBytes(file.content as List<int>);
        if (file.name.endsWith('.apk')) {
          apkPath = outPath;
        }
      }
    }

    if (apkPath == null) {
      throw Exception('No APK found inside the downloaded archive');
    }
    return apkPath;
  }

  /// Apply a downloaded update.
  ///
  /// Desktop: spawns the platform-specific updater script and exits.
  /// Android: triggers the system APK installer via MethodChannel.
  Future<void> applyUpdate(String archivePath, {String? appDir}) async {
    if (isAndroid) {
      final apkPath = await extractAndroidApk(archivePath);
      // The Android side copies the APK to its own cache dir and
      // launches the install intent.  We just need to pass the path.
      await _installApkOnAndroid(apkPath);
      return;
    }

    final targetDir = appDir ?? _detectAppDir();
    final updateDir = p.join(
      (await getTemporaryDirectory()).path,
      'ocr-to-anki-update',
    );

    if (Platform.isWindows) {
      final batchPath = p.join(updateDir, 'apply-update.bat');
      await _writeWindowsUpdater(batchPath, archivePath, targetDir);
      await Process.start('cmd', ['/c', batchPath],
          mode: ProcessStartMode.detached);
    } else {
      final scriptPath = p.join(updateDir, 'apply-update.sh');
      await _writeUnixUpdater(scriptPath, archivePath, targetDir);
      await Process.start('/bin/bash', [scriptPath],
          mode: ProcessStartMode.detached);
    }

    // Give the updater script a moment to start before we exit.
    await Future<void>.delayed(const Duration(seconds: 1));
    exit(0);
  }

  /// Ask MainActivity.kt to install the APK.
  Future<void> _installApkOnAndroid(String apkPath) async {
    const channel = MethodChannel(
      'com.ocrtoanki.ocr_to_anki/system',
    );
    await channel.invokeMethod('installApk', {'path': apkPath});
  }

  /// Auto-detect the application directory from the executable path.
  String _detectAppDir() {
    final exe = Platform.resolvedExecutable;
    // Linux bundle:  …/ocr-to-anki/ocr_to_anki
    // macOS bundle:  …/ocr_to_anki.app/Contents/MacOS/ocr_to_anki
    // Windows:       …/ocr-to-anki/ocr_to_anki.exe
    final dir = p.dirname(exe);
    if (Platform.isMacOS) {
      // Walk up to the .app bundle root.
      var current = dir;
      while (current != p.rootPrefix(current)) {
        if (p.basename(current).endsWith('.app')) {
          return p.dirname(current); // parent of .app
        }
        current = p.dirname(current);
      }
    }
    return dir;
  }

  Future<void> _writeUnixUpdater(
    String path,
    String archivePath,
    String targetDir,
  ) async {
    final isTarGz = archivePath.endsWith('.tar.gz');
    final extractCmd = isTarGz
        ? 'tar xzf "\$ARCHIVE" -C "\$EXTRACT_DIR"'
        : 'unzip -q "\$ARCHIVE" -d "\$EXTRACT_DIR"';

    final script = '''
#!/usr/bin/env bash
set -euo pipefail
ARCHIVE="$archivePath"
TARGET_DIR="$targetDir"
EXTRACT_DIR="\$(dirname "\$ARCHIVE")/extracted"
PID=$pid

echo "[updater] Waiting for app (pid \$PID) to exit..."
while kill -0 \$PID 2>/dev/null; do
  sleep 1
done

echo "[updater] Extracting update..."
rm -rf "\$EXTRACT_DIR"
mkdir -p "\$EXTRACT_DIR"
$extractCmd

# Find the extracted bundle root (first subdirectory).
EXTRACTED_ROOT="\$(find "\$EXTRACT_DIR" -maxdepth 1 -type d | tail -n +2 | head -n 1)"
if [ -z "\$EXTRACTED_ROOT" ]; then
  EXTRACTED_ROOT="\$EXTRACT_DIR"
fi

echo "[updater] Replacing app in \$TARGET_DIR..."
# Preserve user data and settings by not touching known subdirectories.
BACKUP_DIR="\$TARGET_DIR/.update-\$(date +%s)"
mkdir -p "\$BACKUP_DIR"
for item in "\$TARGET_DIR"/*; do
  name="\$(basename "\$item")"
  if [ "\$name" = "data" ] || [ "\$name" = "backend" ]; then
    continue
  fi
  mv "\$item" "\$BACKUP_DIR/"
done

# Copy new files in.
cp -a "\$EXTRACTED_ROOT"/* "\$TARGET_DIR/" || true

# Cleanup.
rm -rf "\$EXTRACT_DIR" "\$BACKUP_DIR" "\$ARCHIVE"

echo "[updater] Restarting app..."
if [ -f "\$TARGET_DIR/run.sh" ]; then
  nohup "\$TARGET_DIR/run.sh" >/dev/null 2>&1 &
elif [ -d "\$TARGET_DIR/ocr_to_anki.app" ]; then
  open "\$TARGET_DIR/ocr_to_anki.app"
else
  echo "[updater] Please restart the app manually."
fi
'''
;

    final file = File(path);
    await file.writeAsString(script);
    await Process.run('chmod', ['+x', path]);
  }

  Future<void> _writeWindowsUpdater(
    String path,
    String archivePath,
    String targetDir,
  ) async {
    final script = '''
@echo off
setlocal
set "ARCHIVE=$archivePath"
set "TARGET_DIR=$targetDir"
set "EXTRACT_DIR=%TEMP%\\ocr-to-anki-update\\extracted"
set "PID=$pid"

echo [updater] Waiting for app to exit...
:waitloop
tasklist | findstr " %PID% " >nul 2>&1
if %errorlevel% == 0 (
  timeout /t 1 /nobreak >nul
  goto waitloop
)

echo [updater] Extracting update...
if exist "%EXTRACT_DIR%" rmdir /s /q "%EXTRACT_DIR%"
mkdir "%EXTRACT_DIR%"
powershell -Command "Expand-Archive -Path '%ARCHIVE%' -DestinationPath '%EXTRACT_DIR%' -Force"

echo [updater] Replacing app files...
set "BACKUP_DIR=%TARGET_DIR%\\.update-backup-%random%"
mkdir "%BACKUP_DIR%"
for /d %%D in ("%TARGET_DIR%\\*") do (
  if /I not "%%~nxD"=="data" if /I not "%%~nxD"=="backend" (
    move "%%D" "%BACKUP_DIR%" >nul 2>&1
  )
)
for %%F in ("%TARGET_DIR%\\*") do (
  move "%%F" "%BACKUP_DIR%" >nul 2>&1
)

xcopy /s /e /y /q "%EXTRACT_DIR%\\*" "%TARGET_DIR%\\" >nul 2>&1

rmdir /s /q "%EXTRACT_DIR%" 2>nul
rmdir /s /q "%BACKUP_DIR%" 2>nul
del "%ARCHIVE%" 2>nul

echo [updater] Starting updated app...
if exist "%TARGET_DIR%\\ocr_to_anki.exe" (
  start "" "%TARGET_DIR%\\ocr_to_anki.exe"
)
endlocal
'''
;

    await File(path).writeAsString(script);
  }

  /// Dispose the HTTP client.
  void dispose() => _client.close();
}
