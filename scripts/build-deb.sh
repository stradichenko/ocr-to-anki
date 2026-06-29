#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# build-deb.sh — Build a Debian package for OCR-to-Anki
#
# Usage:
#   ./scripts/build-deb.sh [source-dir] [version]
#
# Arguments:
#   source-dir  Path to the assembled Linux bundle directory
#               (default: output/release/ocr-to-anki-v$(pubspec-version)-linux-x86_64)
#   version     Package version (default: extracted from app/pubspec.yaml)
#
# Must be run on a Debian/Ubuntu system with dpkg-deb available.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERSION=$(grep '^version:' "$PROJECT_ROOT/app/pubspec.yaml" | head -1 | awk '{print $2}' | cut -d+ -f1)
DEFAULT_SOURCE="$PROJECT_ROOT/output/release/ocr-to-anki-v${VERSION}-linux-x86_64"

SOURCE_DIR="${1:-$DEFAULT_SOURCE}"
VERSION="${2:-$VERSION}"

PKG_NAME="ocr-to-anki"
ARCH="amd64"
DEB_DIR="$PROJECT_ROOT/output/release/${PKG_NAME}_${VERSION}_${ARCH}"
DEB_FILE="${DEB_DIR}.deb"

if ! command -v dpkg-deb &>/dev/null; then
  echo "ERROR: dpkg-deb not found. Install with: sudo apt install dpkg-dev" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "ERROR: Source bundle not found: $SOURCE_DIR" >&2
  echo "Build the Linux bundle first, e.g.: nix develop .#flutter --command ./scripts/build-flutter.sh linux" >&2
  exit 1
fi

echo "→ Building Debian package ${PKG_NAME}_${VERSION}_${ARCH}.deb"
echo "  Source: $SOURCE_DIR"

# Clean previous build
rm -rf "$DEB_DIR"
mkdir -p "$DEB_DIR/DEBIAN"
mkdir -p "$DEB_DIR/opt/$PKG_NAME"
mkdir -p "$DEB_DIR/usr/share/applications"
mkdir -p "$DEB_DIR/usr/bin"
mkdir -p "$DEB_DIR/usr/share/icons/hicolor/scalable/apps"
mkdir -p "$DEB_DIR/usr/share/metainfo"

# Copy the assembled bundle
cp -r "$SOURCE_DIR/"* "$DEB_DIR/opt/$PKG_NAME/"

# Ensure launcher is executable
chmod +x "$DEB_DIR/opt/$PKG_NAME/run.sh" 2>/dev/null || true
chmod +x "$DEB_DIR/opt/$PKG_NAME/ocr_to_anki" 2>/dev/null || true

# Install the canonical SVG icon for the .desktop entry
ICON_SRC="$PROJECT_ROOT/icon.svg"
if [[ -f "$ICON_SRC" ]]; then
  cp "$ICON_SRC" "$DEB_DIR/usr/share/icons/hicolor/scalable/apps/ocr-to-anki.svg"
fi

# Install AppStream metadata with the current release version/date
META_SRC="$PROJECT_ROOT/scripts/linux/ocr-to-anki.metainfo.xml"
if [[ -f "$META_SRC" ]]; then
  TODAY=$(date +%Y-%m-%d)
  sed -e "s/version=\"[^\"]*/version=\"$VERSION/" \
      -e "s/date=\"[^\"]*/date=\"$TODAY/" \
      "$META_SRC" > "$DEB_DIR/usr/share/metainfo/ocr-to-anki.metainfo.xml"
fi

# Create a system launcher symlink
ln -sf "/opt/$PKG_NAME/run.sh" "$DEB_DIR/usr/bin/ocr-to-anki"

# Desktop entry (refer to the installed icon by name so themes can resolve it)
cat > "$DEB_DIR/usr/share/applications/ocr-to-anki.desktop" <<EOF
[Desktop Entry]
Name=OCR to Anki
Comment=Cross-platform OCR to Anki flashcard generator
Exec=/opt/$PKG_NAME/run.sh
Icon=ocr-to-anki
Type=Application
Categories=Education;Utility;
Terminal=false
EOF

# Control file
cat > "$DEB_DIR/DEBIAN/control" <<EOF
Package: $PKG_NAME
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Depends: libgtk-3-0, python3 (>= 3.11)
Maintainer: stradichenko <stradichenko@gmail.com>
Description: OCR to Anki
 Cross-platform application for extracting vocabulary from images
 and creating Anki flashcards using local LLM inference.
EOF

# Post-install script: ensure launcher is executable
cat > "$DEB_DIR/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
chmod +x /opt/ocr-to-anki/run.sh /opt/ocr-to-anki/ocr_to_anki 2>/dev/null || true
EOF
chmod 755 "$DEB_DIR/DEBIAN/postinst"

# Build the package
dpkg-deb --build --root-owner-group "$DEB_DIR"

# Clean up the staging directory
rm -rf "$DEB_DIR"

echo ""
echo "✓ Debian package: $DEB_FILE"
echo "  Size: $(du -h "$DEB_FILE" | cut -f1)"
