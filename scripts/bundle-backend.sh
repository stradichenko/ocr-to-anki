#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# bundle-backend.sh — Bundle the Python FastAPI backend as a standalone binary
#
# Uses PyInstaller to create a single-directory distribution that includes
# Python, all pip dependencies, and the application source code.
#
# Usage:
#   ./scripts/bundle-backend.sh
#
# Must be run inside:  nix develop  (the default shell with Python)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output/backend"

echo "╔══════════════════════════════════════════════════╗"
echo "║  Bundling Python backend with PyInstaller        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Verify Python environment ─────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not in PATH. Run: nix develop"
  exit 1
fi

# ── Install PyInstaller if needed ─────────────────────────────────────────────
if ! python3 -c "import PyInstaller" 2>/dev/null; then
  echo "→ Installing PyInstaller…"
  pip install --user pyinstaller
fi

# ── Create PyInstaller spec ───────────────────────────────────────────────────
cd "$PROJECT_ROOT"

echo "→ Running PyInstaller…"
python3 -m PyInstaller \
  --name ocr-to-anki-backend \
  --distpath "$OUTPUT_DIR/dist" \
  --workpath "$OUTPUT_DIR/build" \
  --specpath "$OUTPUT_DIR" \
  --noconfirm \
  --clean \
  --collect-all uvicorn \
  --collect-all fastapi \
  --collect-all pydantic \
  --hidden-import uvicorn.logging \
  --hidden-import uvicorn.loops.auto \
  --hidden-import uvicorn.protocols.http.auto \
  --hidden-import uvicorn.protocols.websockets.auto \
  --hidden-import uvicorn.lifespan.on \
  --hidden-import src.api.app \
  --hidden-import src.api.models \
  --add-data "src:src" \
  --add-data "config:config" \
  --paths src \
  src/api/app.py

echo ""
echo "→ Backend binary:"
ls -lh "$OUTPUT_DIR/dist/ocr-to-anki-backend/" | head -20

BUNDLE_SIZE=$(du -sh "$OUTPUT_DIR/dist/ocr-to-anki-backend/" | cut -f1)
echo ""
echo "✓ Backend bundled: $OUTPUT_DIR/dist/ocr-to-anki-backend/"
echo "  Total size: $BUNDLE_SIZE"
echo ""
echo "Test with:"
echo "  $OUTPUT_DIR/dist/ocr-to-anki-backend/ocr-to-anki-backend --host 127.0.0.1 --port 8000"
