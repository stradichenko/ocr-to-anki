#!/usr/bin/env bash
#
# Regenerate every derived asset from the two SVG sources.
#
#   github_logo.svg   the README logo
#   icon.svg          the app icon, derived from the same artwork in a square frame
#
# The platform icons, the .ico and the raster fallbacks were previously made by
# hand, so a change to the artwork meant someone redoing each of them. This
# script makes that reproducible: edit github_logo.svg, run this, and every
# derived file follows.
#
#   scripts/generate-icons.sh
#
# Needs rsvg-convert and ImageMagick. Both are available from nixpkgs, so run it
# as:
#
#   nix shell nixpkgs#librsvg nixpkgs#imagemagick -c scripts/generate-icons.sh
#
# Files written:
#   icon.svg                     rebuilt from github_logo.svg
#   github_logo.png, icon.png    raster fallbacks for anything that cannot draw SVG
#   app/macos/Runner/Assets.xcassets/AppIcon.appiconset/*.png
#   app/android/app/src/main/res/mipmap-*/ic_launcher.png
#   app/windows/runner/resources/app_icon.ico

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOGO_SVG="github_logo.svg"
ICON_SVG="icon.svg"

# Square canvas for the app icon, and the margin left around the artwork.
#
# No margin: the artwork is trimmed to its own bounds and then fills the canvas
# edge to edge. A padding ring here reads as dead space on a launcher.
ICON_CANVAS=1024
ICON_MARGIN_FRACTION=0.0
# Background behind the artwork. The logo is drawn for a light surface and its
# detail is lost on a dark launcher without one.
ICON_BACKGROUND="#ffffff"

for tool in rsvg-convert magick; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "missing: $tool (see the header for the nix shell invocation)" >&2
        exit 1
    }
done

[ -f "$LOGO_SVG" ] || { echo "missing $LOGO_SVG" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Measure the artwork inside the logo canvas, so it can be centred in a square
# ---------------------------------------------------------------------------
echo ":: measuring the artwork in $LOGO_SVG"

VIEWBOX=$(python3 - "$LOGO_SVG" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r'viewBox="([^"]+)"', t)
if not m:
    sys.exit("no viewBox in the source SVG")
print(m.group(1))
PY
)
read -r VBX VBY VBW VBH <<<"$VIEWBOX"
echo "   viewBox: $VBX $VBY $VBW $VBH"

# Render wide and trim, then map the pixel bounds back into viewBox units.
PROBE_WIDTH=2000
rsvg-convert -w "$PROBE_WIDTH" -b none "$LOGO_SVG" -o /tmp/gi_probe.png
BBOX=$(magick identify -format '%@' /tmp/gi_probe.png)   # WxH+X+Y of the non-transparent area
read -r ART_W_PX ART_H_PX ART_X_PX ART_Y_PX <<<"$(python3 - "$BBOX" <<'PY'
import re, sys
m = re.match(r'(\d+)x(\d+)\+(\d+)\+(\d+)', sys.argv[1])
if not m:
    sys.exit(f"unexpected trim result: {sys.argv[1]}")
print(*m.groups())
PY
)"

SCALE_TO_VB=$(python3 -c "print($VBW / $PROBE_WIDTH)")
ART_X=$(python3 -c "print($ART_X_PX * $SCALE_TO_VB)")
ART_Y=$(python3 -c "print($ART_Y_PX * $SCALE_TO_VB)")
ART_W=$(python3 -c "print($ART_W_PX * $SCALE_TO_VB)")
ART_H=$(python3 -c "print($ART_H_PX * $SCALE_TO_VB)")
echo "   artwork: ${ART_W}x${ART_H} at (${ART_X}, ${ART_Y}) in viewBox units"

# Fit that artwork into the square canvas with a margin, preserving aspect.
read -r FIT_SCALE FIT_TX FIT_TY <<<"$(python3 - "$ICON_CANVAS" "$ICON_MARGIN_FRACTION" "$ART_X" "$ART_Y" "$ART_W" "$ART_H" <<'PY'
import sys
canvas, margin_f, ax, ay, aw, ah = (float(v) for v in sys.argv[1:])
margin = canvas * margin_f
avail = canvas - 2 * margin
scale = min(avail / aw, avail / ah)
tx = (canvas - aw * scale) / 2 - ax * scale
ty = (canvas - ah * scale) / 2 - ay * scale
print(f"{scale:.6f} {tx:.4f} {ty:.4f}")
PY
)"
echo "   fitting at scale ${FIT_SCALE}, offset (${FIT_TX}, ${FIT_TY})"

# ---------------------------------------------------------------------------
# Rebuild icon.svg from the logo, wrapped in the square frame
# ---------------------------------------------------------------------------
echo ":: writing $ICON_SVG"
python3 - "$LOGO_SVG" "$ICON_SVG" "$ICON_CANVAS" "$ICON_BACKGROUND" \
         "$FIT_TX" "$FIT_TY" "$FIT_SCALE" <<'PY'
import sys
import xml.etree.ElementTree as ET

src, dst, canvas, background, tx, ty, scale = sys.argv[1:8]
canvas = int(canvas)

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)

tree = ET.parse(src)
root = tree.getroot()

# Everything the renderer needs, minus editor state that only Inkscape reads.
drop = {"namedview", "metadata"}
kept = [c for c in list(root) if c.tag.split("}")[-1] not in drop]

out = ET.Element(f"{{{SVG}}}svg", {
    "width": str(canvas),
    "height": str(canvas),
    "viewBox": f"0 0 {canvas} {canvas}",
})
ET.SubElement(out, f"{{{SVG}}}rect", {
    "x": "0", "y": "0", "width": str(canvas), "height": str(canvas),
    "fill": background,
})
wrapped = ET.SubElement(out, f"{{{SVG}}}g", {
    "transform": f"translate({tx},{ty}) scale({scale})",
})
for child in kept:
    wrapped.append(child)

ET.ElementTree(out).write(dst, encoding="utf-8", xml_declaration=True)
print(f"   {dst}: {len(kept)} top-level element(s) re-framed")
PY

# ---------------------------------------------------------------------------
# Raster fallbacks
# ---------------------------------------------------------------------------
echo ":: writing the raster fallbacks"
rsvg-convert -w 1600 -b none "$LOGO_SVG" -o github_logo.png
rsvg-convert -w 1024 -b none "$ICON_SVG" -o icon.png
echo "   github_logo.png, icon.png"

# ---------------------------------------------------------------------------
# Platform icons
# ---------------------------------------------------------------------------
echo ":: writing the platform icons"
ICONSET="app/macos/Runner/Assets.xcassets/AppIcon.appiconset"
if [ -d "$ICONSET" ]; then
    for size in 16 32 64 128 256 512 1024; do
        rsvg-convert -w "$size" -h "$size" -b none "$ICON_SVG" -o "$ICONSET/app_icon_${size}.png"
    done
    echo "   macOS appiconset: $(ls "$ICONSET"/*.png | wc -l) files"
fi

# Android launcher densities: 48/72/96/144/192 at the standard multipliers.
declare -A ANDROID=(
    [mdpi]=48 [hdpi]=72 [xhdpi]=96 [xxhdpi]=144 [xxxhdpi]=192
)
for density in "${!ANDROID[@]}"; do
    dir="app/android/app/src/main/res/mipmap-$density"
    [ -d "$dir" ] || continue
    rsvg-convert -w "${ANDROID[$density]}" -h "${ANDROID[$density]}" -b none "$ICON_SVG" -o "$dir/ic_launcher.png"
done
echo "   android mipmaps: ${#ANDROID[@]} densities"

WIN_ICO="app/windows/runner/resources/app_icon.ico"
if [ -d "$(dirname "$WIN_ICO")" ]; then
    TMPDIR_ICO=$(mktemp -d)
    for size in 16 24 32 48 64 128 256; do
        rsvg-convert -w "$size" -h "$size" -b none "$ICON_SVG" -o "$TMPDIR_ICO/$size.png"
    done
    magick "$TMPDIR_ICO"/16.png "$TMPDIR_ICO"/24.png "$TMPDIR_ICO"/32.png \
           "$TMPDIR_ICO"/48.png "$TMPDIR_ICO"/64.png "$TMPDIR_ICO"/128.png \
           "$TMPDIR_ICO"/256.png "$WIN_ICO"
    rm -rf "$TMPDIR_ICO"
    echo "   windows app_icon.ico"
fi

# iOS shares the appiconset layout if it is ever added.
IOS_ICONSET="app/ios/Runner/Assets.xcassets/AppIcon.appiconset"
if [ -d "$IOS_ICONSET" ]; then
    for f in "$IOS_ICONSET"/*.png; do
        [ -e "$f" ] || continue
        size=$(magick identify -format '%w' "$f" 2>/dev/null || echo 0)
        [ "$size" -gt 0 ] && rsvg-convert -w "$size" -h "$size" -b none "$ICON_SVG" -o "$f"
    done
    echo "   iOS appiconset refreshed"
fi

echo
echo "Done. Rebuilt from $LOGO_SVG and $ICON_SVG."