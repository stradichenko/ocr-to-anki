# Building Release Binaries

OCR-to-Anki can produce release binaries for **Linux**, **macOS**, and **Windows**.
All build workflows live within the Nix flake to ensure reproducibility.

> **Cross-compilation is NOT supported.** Flutter desktop compiles native C/C++
> code using the host toolchain — you must build on each target platform natively.
> Use CI/CD (GitHub Actions) to build all 3 from a single push.

## Architecture

The app is a two-tier system:

| Component | Technology | Binary |
|---|---|---|
| **GUI** | Flutter (Dart) | Platform-native executable |
| **Backend** | Python FastAPI + llama.cpp | Python bundle or system Python |

Both components must be distributed together for the app to function.

---

## Quick Reference

| Goal | Command |
|---|---|
| Build Linux (Nix derivation) | `nix build .#flutter-app` |
| Build Linux (script) | `nix develop .#flutter --command ./scripts/build-flutter.sh linux` |
| Build macOS (on Mac) | `nix develop .#flutter --command ./scripts/build-flutter.sh macos` |
| Build Windows (on Windows) | `./scripts/build-flutter.sh windows` |
| Bundle Python backend | `nix develop --command ./scripts/bundle-backend.sh` |
| Full Nix bundle (GUI + backend) | `nix build .#bundle` |
| CI/CD (all platforms) | Push a `v*` tag or run workflow manually |

### Cross-compilation matrix

| Build host ↓ \ Target → | Linux | macOS | Windows |
|---|---|---|---|
| **Linux** | ✅ Nix | ❌ Needs Xcode + macOS SDK | ❌ Needs MSVC |
| **macOS** | ❌ Needs GTK3 libs | ✅ Nix | ❌ Needs MSVC |
| **Windows** | ❌ | ❌ | ✅ Native MSVC |

---

## 1. Compiling for Linux

### Prerequisites

- **Nix** with flakes enabled (`experimental-features = nix-command flakes`)
- That's it — Nix provides Flutter, Dart, GTK3, CMake, clang, and all libs.

### Option A: Pure Nix derivation (recommended — fully reproducible)

```bash
nix build .#flutter-app

# Binary lands in ./result/bin/ocr-to-anki
./result/bin/ocr-to-anki
```

This runs inside the Nix sandbox — identical output every time. The resulting
binary is wrapped with `LD_LIBRARY_PATH` pointing to Nix-store GTK3/GL libs.

### Option B: Interactive build in the Flutter dev shell

```bash
# 1. Enter the Flutter shell (provides flutter, cmake, clang, GTK3…)
nix develop .#flutter

# 2. Build
cd app
flutter pub get
flutter build linux --release

# 3. Output
ls -lh build/linux/x64/release/bundle/
#   ocr_to_anki          ← 24 KB launcher (loads libapp.so)
#   lib/libapp.so         ← 8.5 MB (your Dart code, AOT-compiled)
#   lib/libflutter_linux_gtk.so ← 20 MB (Flutter engine)
#   lib/libsqlite3_flutter_libs_plugin.so
#   lib/libdesktop_drop_plugin.so
#   lib/libfile_selector_linux_plugin.so
#   data/icudtl.dat
#   data/flutter_assets/
```

### Option C: Packaged tarball (for distribution)

```bash
nix develop .#flutter --command ./scripts/build-flutter.sh linux

# Output: output/release/ocr-to-anki-v0.1.0-linux-x86_64.tar.gz (~33 MB)
# Contains: Flutter bundle + Python backend source + run.sh launcher
```

### Option D: Complete bundle (GUI + backend, everything Nix-managed)

```bash
nix build .#bundle

# result/bin/ocr-to-anki  ← single launcher that starts backend + GUI
```

### Running the release on another Linux system

```bash
tar xzf ocr-to-anki-v0.1.0-linux-x86_64.tar.gz
cd ocr-to-anki-v0.1.0-linux-x86_64

# The Flutter binary needs GTK3 at runtime.
# On Ubuntu/Debian:
sudo apt install libgtk-3-0 libblkid1 liblzma5

# On Fedora:
sudo dnf install gtk3

# On NixOS: already available, or use the Nix bundle which includes everything.

# Start the app:
./run.sh
# — or directly: ./ocr_to_anki
```

---

## 2. Compiling for macOS

> **Must be run on a macOS machine.** Apple's SDK license prohibits redistribution,
> so cross-compiling from Linux is not possible.

### Prerequisites

- **Nix** with flakes enabled (install via [Determinate Systems installer](https://zero-to-nix.com/start/install))
- **Xcode** (for the macOS toolchain — `xcode-select --install` at minimum)

### Build steps

```bash
# 1. Enter the Flutter shell (provides flutter, cmake, ninja, Apple SDK frameworks)
nix develop .#flutter

# 2. Build
cd app
flutter pub get
flutter build macos --release

# 3. Output — a native .app bundle
ls build/macos/Build/Products/Release/
#   ocr_to_anki.app/    ← double-click to run
```

### Packaged distribution

```bash
nix develop .#flutter --command ./scripts/build-flutter.sh macos

# If hdiutil is available (macOS): output/release/ocr-to-anki-v0.1.0-macos-arm64.dmg
# Otherwise:                       output/release/ocr-to-anki-v0.1.0-macos-arm64.zip
```

### Notes

- **Primary target**: Apple Silicon (arm64). Intel Macs work via Rosetta 2.
- **Code signing**: Not configured. First launch requires:
  System Settings → Privacy & Security → "Open Anyway", or:
  ```bash
  xattr -cr /path/to/ocr_to_anki.app
  ```
- **App Store**: Would require adding a signing identity + provisioning profile
  to the Xcode project in `app/macos/Runner.xcodeproj`.

---

## 3. Compiling for Windows

> **Must be run on a Windows machine.** Flutter for Windows requires the MSVC
> toolchain (Visual Studio Build Tools), which only runs on Windows.

### Prerequisites

- **Flutter SDK** 3.x ([install guide](https://docs.flutter.dev/get-started/install/windows/desktop))
- **Visual Studio 2022** with "Desktop development with C++" workload
  (or just the Build Tools: `vs_buildtools.exe --add Microsoft.VisualStudio.Workload.VCTools`)
- **Python 3.11+** (for the backend)
- **Git** (for Flutter and dependency management)

> Nix is not used on Windows. The Flutter SDK + VS Build Tools are the native approach.

### Build steps

```powershell
# 1. Get dependencies
cd app
flutter pub get

# 2. Build
flutter build windows --release

# 3. Output
dir build\windows\x64\runner\Release\
#   ocr_to_anki.exe      ← main executable
#   flutter_windows.dll   ← Flutter engine
#   *.dll                 ← plugin DLLs
#   data\                 ← assets + ICU data
```

### Using the build script (Git Bash or WSL)

```bash
./scripts/build-flutter.sh windows
# Output: output/release/ocr-to-anki-v0.1.0-windows-x86_64.zip
```

### Notes

- **Minimum**: Windows 10 version 1903+
- **Visual C++ Redistributable**: Flutter bundles it automatically.
- **Code signing**: Not configured. Windows SmartScreen may warn on first run.
  For production, sign with a code-signing certificate via `signtool`.
- **MSIX packaging**: For Microsoft Store distribution:
  ```bash
  flutter pub add msix
  flutter pub run msix:create
  ```

---

## 4. Nix Package Outputs

The flake exposes several package targets:

| Package | Command | Description |
|---|---|---|
| `default` | `nix build` | Python CLI tool (legacy GTK4 version) |
| `flutter-app` | `nix build .#flutter-app` | Flutter Linux desktop binary |
| `backend` | `nix build .#backend` | Nix-wrapped Python backend (with all deps) |
| `bundle` | `nix build .#bundle` | Complete distribution (Flutter app + backend + launcher) |
| `dockerImage` | `nix build .#dockerImage` | Docker image for server deployment |

### Full bundle structure

```
result/
├── bin/ocr-to-anki          ← launcher (starts backend, then GUI)
└── opt/ocr-to-anki/
    ├── ocr_to_anki           ← Flutter binary
    ├── lib/                  ← Flutter runtime libraries
    ├── data/                 ← Flutter assets
    ├── backend-lib/          ← Python backend source
    └── ocr-to-anki-backend   ← Backend launcher
```

---

## 5. CI/CD (GitHub Actions)

The workflow at `.github/workflows/build.yml` builds for all three platforms
using platform-native runners:

| Platform | Runner | Build method |
|---|---|---|
| Linux | `ubuntu-24.04` | Nix (`nix develop .#flutter`) |
| macOS | `macos-14` (Apple Silicon) | Nix (`nix develop .#flutter`) |
| Windows | `windows-latest` | `subosito/flutter-action` + `setup-python` |

### Triggering a release

```bash
git tag v0.1.0
git push origin v0.1.0
```

This creates a **draft GitHub Release** with artifacts for all platforms.
You can also trigger manually from the Actions tab ("Run workflow").

### Artifacts produced

| Platform | Format | Size (approx.) | Contents |
|---|---|---|---|
| Linux | `.tar.gz` | ~33 MB | Flutter bundle + backend + `run.sh` |
| macOS | `.zip` | ~35 MB | `.app` bundle + backend |
| Windows | `.zip` | ~35 MB | `.exe` + DLLs + backend |

---

## 6. Python Backend Bundling

For fully self-contained distributions (no Python install required), use PyInstaller:

```bash
nix develop --command ./scripts/bundle-backend.sh

# Output: output/backend/dist/ocr-to-anki-backend/
#   └── ocr-to-anki-backend    ← standalone executable (~50 MB)
```

The CI/CD workflow does this automatically for each platform.

---

## 7. Runtime Dependencies

Even with bundled binaries, users still need:

| Dependency | How to get it | Why |
|---|---|---|
| **llama.cpp** | `nix profile install nixpkgs#llama-cpp` or build from source | Runs the LLM inference engine |
| **Gemma 3 4B model** | `./scripts/setup-llama-cpp.sh` (~2.4 GB) | Vision + language model |
| **Vision projector** | `./scripts/setup-llama-cpp.sh` (~812 MB) | Image understanding |
| **Python 3.11+** | System package manager (unless backend is PyInstaller-bundled) | Runs the FastAPI backend |
| **GTK3** (Linux only) | `apt install libgtk-3-0` / `dnf install gtk3` | Flutter Linux runtime |

The app shows a startup error screen if the backend cannot find `llama-server`.

---

## Dev Shell Reference

| Shell | Command | Purpose |
|---|---|---|
| Default | `nix develop` | Python backend development |
| Flutter | `nix develop .#flutter` | Flutter app build & development |
| CUDA | `nix develop .#cuda` | NVIDIA GPU backend builds |
| SYCL | `nix develop .#sycl` | Intel GPU backend builds |
| CUDA | `nix develop .#cuda` | NVIDIA GPU backend builds |
| SYCL | `nix develop .#sycl` | Intel GPU backend builds |
