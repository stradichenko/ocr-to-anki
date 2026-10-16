#!/usr/bin/env bash
# Report a GGUF model's architecture, then try to load it with llama.cpp.
#
# Written because the app shipped a default model whose architecture its pinned
# llama.cpp predated, so it downloaded three gigabytes and then failed to load.
# Run this against a model and a candidate runtime before pairing the two.
#
#   scripts/llama-compat-check.sh /path/to/model.gguf
#   scripts/llama-compat-check.sh /path/to/model.gguf /path/to/llama-cli
#
# With no binary given it uses llama-cli from nixpkgs.

set -uo pipefail

MODEL="${1:-}"
BIN="${2:-}"
[ -n "$MODEL" ] || { echo "usage: $0 <model.gguf> [llama-cli]" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "no such model: $MODEL" >&2; exit 2; }

echo "=== model ==="
echo "  path : $MODEL"
echo "  size : $(stat -c%s "$MODEL") bytes"

# The GGUF header carries general.architecture as plain text near the start, so
# it can be read without loading any weights. That is what tells us which
# runtime feature set the file needs.
echo "=== declared architecture ==="
python3 - "$MODEL" <<'PY' || echo "  (could not parse the GGUF header)"
import struct, sys

path = sys.argv[1]
with open(path, "rb") as fh:
    if fh.read(4) != b"GGUF":
        print("  not a GGUF file"); sys.exit(1)
    version, = struct.unpack("<I", fh.read(4))
    _, _ = struct.unpack("<QQ", fh.read(16))  # tensor count, kv count
    print(f"  gguf version: {version}")

    TYPES = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",
             7:"bool",8:"str",9:"array",10:"u64",11:"i64",12:"f64"}

    def read_string():
        n, = struct.unpack("<Q", fh.read(8))
        return fh.read(n).decode("utf-8", "replace")

    def skip_value(t, depth=0):
        if t == 8:
            read_string()
        elif t in (0, 1, 7):
            fh.read(1)
        elif t in (2, 3):
            fh.read(2)
        elif t in (4, 5, 6):
            fh.read(4)
        elif t in (10, 11, 12):
            fh.read(8)
        elif t == 9 and depth == 0:
            et, = struct.unpack("<I", fh.read(4))
            n, = struct.unpack("<Q", fh.read(8))
            for _ in range(n):
                skip_value(et, depth + 1)
        else:
            raise ValueError(f"unknown value type {t}")

    found = False
    while not found:
        key = read_string()
        vtype, = struct.unpack("<I", fh.read(4))
        if key == "general.architecture" and vtype == 8:
            print(f"  general.architecture: {read_string()}")
            found = True
        else:
            skip_value(vtype)
PY

echo "=== load attempt ==="
if [ -n "$BIN" ]; then
    LAUNCH=("$BIN")
else
    command -v llama-cli >/dev/null 2>&1 || {
        echo "  llama-cli not on PATH and no binary given" >&2; exit 3; }
    LAUNCH=(llama-cli)
fi

OUT="$(timeout 900 "${LAUNCH[@]}" -m "$MODEL" -p hi -n 1 2>&1 | tr -d '\r')"
RC=$?

# llama.cpp draws a progress bar and a banner when it has a terminal, so the
# useful lines are not always present. Absence of the architecture error plus a
# clean exit is the reliable signal; the banner is corroboration.
if grep -qi "unknown model architecture" <<<"$OUT"; then
    echo "  RESULT: LOAD FAILED - the runtime does not know this architecture"
    grep -i "unknown model architecture" <<<"$OUT" | head -2 | sed 's/^/    /'
    exit 1
fi

if grep -qi "error loading model\|failed to load model" <<<"$OUT"; then
    echo "  RESULT: LOAD FAILED"
    grep -i "error" <<<"$OUT" | head -3 | sed 's/^/    /'
    exit 1
fi

if [ "$RC" -eq 0 ] && grep -q "llama.cpp\|build:" <<<"$OUT"; then
    echo "  RESULT: LOADED OK (exit 0, runtime recognised the architecture)"
    grep -iE "arch|n_ctx|n_layer|load time|model size" <<<"$OUT" | head -6 | sed 's/^/    /'
    exit 0
fi

if grep -qi "load time\|model loaded" <<<"$OUT"; then
    echo "  RESULT: LOADED OK"
    grep -iE "arch|n_ctx|load time" <<<"$OUT" | head -6 | sed 's/^/    /'
    exit 0
fi

echo "  RESULT: inconclusive (exit $RC)"
grep -v '^\s*[|/\\-]*\s*$' <<<"$OUT" | head -8 | sed 's/^/    /'
exit 3