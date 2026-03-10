#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# build-flutter.sh — Build OCR-to-Anki Flutter release for the current platform
#
# Usage:
#   ./scripts/build-flutter.sh              # auto-detect platform
#   ./scripts/build-flutter.sh linux        # force Linux build
#   ./scripts/build-flutter.sh macos        # force macOS build
#   ./scripts/build-flutter.sh windows      # force Windows build (Windows only)
#
# Must be run inside:  nix develop .#flutter
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$PROJECT_ROOT/app"
OUTPUT_DIR="$PROJECT_ROOT/output/release"
VERSION=$(grep '^version:' "$APP_DIR/pubspec.yaml" | head -1 | awk '{print $2}' | cut -d+ -f1)

# ── Platform detection ────────────────────────────────────────────────────────
detect_platform() {
  case "${1:-auto}" in
    linux)   echo "linux" ;;
    macos)   echo "macos" ;;
    windows) echo "windows" ;;
    auto)
      case "$OSTYPE" in
        linux*)  echo "linux" ;;
        darwin*) echo "macos" ;;
        msys*|cygwin*|win*) echo "windows" ;;
        *) echo "ERROR: Unknown OS: $OSTYPE" >&2; exit 1 ;;
      esac
      ;;
    *) echo "ERROR: Unknown platform: $1" >&2; exit 1 ;;
  esac
}

PLATFORM=$(detect_platform "${1:-auto}")

echo "╔══════════════════════════════════════════════════╗"
echo "║  OCR-to-Anki  v${VERSION}  —  Building for ${PLATFORM}"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Verify we're in the Nix flutter shell ─────────────────────────────────────
if ! command -v flutter &>/dev/null; then
  echo "ERROR: flutter not found in PATH."
  echo "Run:   nix develop .#flutter"
  exit 1
fi

# ── Ensure dependencies ──────────────────────────────────────────────────────
echo "→ Fetching dependencies…"
cd "$APP_DIR"
flutter pub get

# ── Build ─────────────────────────────────────────────────────────────────────
echo "→ Building Flutter $PLATFORM release…"

case "$PLATFORM" in
  linux)
    flutter build linux --release
    BUNDLE_DIR="$APP_DIR/build/linux/x64/release/bundle"
    if [ ! -d "$BUNDLE_DIR" ]; then
      # Some Flutter versions use a different path
      BUNDLE_DIR=$(find "$APP_DIR/build/linux" -path "*/release/bundle" -type d | head -1)
    fi
    ;;
  macos)
    flutter build macos --release
    BUNDLE_DIR="$APP_DIR/build/macos/Build/Products/Release"
    ;;
  windows)
    flutter build windows --release
    BUNDLE_DIR="$APP_DIR/build/windows/x64/runner/Release"
    if [ ! -d "$BUNDLE_DIR" ]; then
      BUNDLE_DIR=$(find "$APP_DIR/build/windows" -path "*/runner/Release" -type d | head -1)
    fi
    ;;
esac

if [ -z "${BUNDLE_DIR:-}" ] || [ ! -d "$BUNDLE_DIR" ]; then
  echo "ERROR: Build output not found." >&2
  exit 1
fi

echo "→ Build output: $BUNDLE_DIR"

# ── Package into output/release ───────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

ARCHIVE_NAME="ocr-to-anki-v${VERSION}-${PLATFORM}-$(uname -m)"

case "$PLATFORM" in
  linux)
    DEST="$OUTPUT_DIR/$ARCHIVE_NAME"
    rm -rf "$DEST"
    mkdir -p "$DEST"
    cp -r "$BUNDLE_DIR/"* "$DEST/"
    
    # Bundle the Python backend source alongside
    mkdir -p "$DEST/backend"
    cp -r "$PROJECT_ROOT/src" "$DEST/backend/"
    cp -r "$PROJECT_ROOT/config" "$DEST/backend/"
    cp "$PROJECT_ROOT/requirements.txt" "$DEST/backend/"
    
    # Create a launcher script
    cat > "$DEST/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OCR_TO_ANKI_ROOT="$DIR/backend"
exec "$DIR/ocr_to_anki" "$@"
EOF
    chmod +x "$DEST/run.sh"
    
    # Create tarball
    cd "$OUTPUT_DIR"
    tar czf "${ARCHIVE_NAME}.tar.gz" "$ARCHIVE_NAME"
    echo ""
    echo "✓ Linux bundle:  $OUTPUT_DIR/${ARCHIVE_NAME}.tar.gz"
    echo "  Size: $(du -h "${ARCHIVE_NAME}.tar.gz" | cut -f1)"
    ;;
    
  macos)
    DEST="$OUTPUT_DIR/$ARCHIVE_NAME"
    rm -rf "$DEST"
    mkdir -p "$DEST"
    
    # Copy the .app bundle
    cp -R "$BUNDLE_DIR/ocr_to_anki.app" "$DEST/" 2>/dev/null || \
    cp -R "$BUNDLE_DIR/"*.app "$DEST/"
    
    # Bundle backend alongside the .app
    mkdir -p "$DEST/backend"
    cp -r "$PROJECT_ROOT/src" "$DEST/backend/"
    cp -r "$PROJECT_ROOT/config" "$DEST/backend/"
    cp "$PROJECT_ROOT/requirements.txt" "$DEST/backend/"
    
    # Create a DMG if hdiutil is available, otherwise zip
    cd "$OUTPUT_DIR"
    if command -v hdiutil &>/dev/null; then
      hdiutil create -volname "OCR to Anki" -srcfolder "$DEST" \
        -ov -format UDZO "${ARCHIVE_NAME}.dmg"
      echo ""
      echo "✓ macOS bundle:  $OUTPUT_DIR/${ARCHIVE_NAME}.dmg"
      echo "  Size: $(du -h "${ARCHIVE_NAME}.dmg" | cut -f1)"
    else
      zip -qr "${ARCHIVE_NAME}.zip" "$ARCHIVE_NAME"
      echo ""
      echo "✓ macOS bundle:  $OUTPUT_DIR/${ARCHIVE_NAME}.zip"
      echo "  Size: $(du -h "${ARCHIVE_NAME}.zip" | cut -f1)"
    fi
    ;;
    
  windows)
    DEST="$OUTPUT_DIR/$ARCHIVE_NAME"
    rm -rf "$DEST"
    mkdir -p "$DEST"
    cp -r "$BUNDLE_DIR/"* "$DEST/"
    
    # Bundle backend alongside
    mkdir -p "$DEST/backend"
    cp -r "$PROJECT_ROOT/src" "$DEST/backend/"
    cp -r "$PROJECT_ROOT/config" "$DEST/backend/"
    cp "$PROJECT_ROOT/requirements.txt" "$DEST/backend/"
    
    cd "$OUTPUT_DIR"
    if command -v zip &>/dev/null; then
      zip -qr "${ARCHIVE_NAME}.zip" "$ARCHIVE_NAME"
      echo ""
      echo "✓ Windows bundle: $OUTPUT_DIR/${ARCHIVE_NAME}.zip"
      echo "  Size: $(du -h "${ARCHIVE_NAME}.zip" | cut -f1)"
    else
      echo ""
      echo "✓ Windows bundle: $DEST/"
    fi
    ;;
esac

echo ""
echo "Done! The build includes:"
echo "  • Flutter desktop app (release mode)"
echo "  • Python backend source (requires Python 3.11+ at runtime)"
echo ""
echo "Note: The end user needs:"
echo "  • Python 3.11+ with deps from requirements.txt"
echo "  • llama-cpp (llama-server + llama-mtmd-cli)"
echo "  • Gemma 3 4B model + vision projector"
