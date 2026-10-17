#!/usr/bin/env python3
"""Sweep the vision token budget and report encode cost against output quality.

The image token budget is the only encoder-side control llama.cpp exposes, and
it only applies to projectors that implement one. On the hardware this project
targets, the vision encode dominated a whole OCR pass, so the question is
whether cutting the budget cuts the encode, and what it costs in accuracy.

The flag is accepted but ignored by projectors without a budget, so a run at
"unset" is included as the control: if every budget produces the same encode
time as unset, the projector is not honouring the flag at all.

Run inside a shell that provides a working llama-mtmd-cli, with a GPU available:

    nix shell nixpkgs#vulkan-loader nixpkgs#stdenv.cc.cc.lib -c \\
        python tests/benchmark_vision_tokens.py \\
        --model ~/.cache/llama.cpp/models/gemma-4-e2b-it-q4_0.gguf \\
        --mmproj ~/.cache/llama.cpp/models/mmproj-gemma-4-e2b-f16.gguf \\
        --image  /tmp/page768.jpg \\
        --binary /path/to/llama-mtmd-cli \\
        --expect amour bouleaux pencha

Writes JSON next to this file unless --out is given. Not a pytest file: it needs
a model, a projector and a working vision backend.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backends.mtmd_cli import DEFAULT_VISION_SYSTEM_PROMPT  # noqa: E402

PROMPT = "Extract all visible text from this image. List each word or phrase, one per line."

# b11577 writes "mtmd batch encoding done in N ms"; older builds wrote
# "image slice encoded in N ms". Accept either.
_ENCODE_RE = re.compile(r"(?:encoding done|encoded) in (\d+) ms")
_LOAD_RE = re.compile(r"load time = \+?(\d+) ms")


def run_once(binary, model, mmproj, image, budget, max_tokens, timeout):
    cmd = [
        str(binary),
        "-m", str(model),
        "--mmproj", str(mmproj),
        "--image", str(image),
        "-p", PROMPT,
        "-sys", DEFAULT_VISION_SYSTEM_PROMPT,
        "--jinja",
        "--ctx-size", "8192",
        "--threads", "2",
        "--temp", "0.1",
        "--top-k", "40",
        "--top-p", "0.9",
        "-ngl", "-1",
        "-n", str(max_tokens),
    ]
    if budget is not None:
        cmd.extend(["--image-max-tokens", str(budget)])

    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall = time.monotonic() - started

    stderr = proc.stderr or ""
    encode_ms = int(_ENCODE_RE.search(stderr).group(1)) if _ENCODE_RE.search(stderr) else None
    load_ms = int(_LOAD_RE.search(stderr).group(1)) if _LOAD_RE.search(stderr) else None

    return {
        "budget": budget,
        "wall_s": round(wall, 2),
        "encode_ms": encode_ms,
        "load_ms": load_ms,
        "exit_code": proc.returncode,
        "text": (proc.stdout or "").strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument(
        "--budgets", nargs="*", type=int, default=[None, 1120, 560, 280, 140, 70],
        help="token budgets to try; omit values to include the unset control",
    )
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument(
        "--expect", nargs="*", default=[],
        help="words the correct answer contains, for a rough accuracy signal",
    )
    ap.add_argument("--out", default=str(ROOT / "tests" / "benchmark_vision_tokens.json"))
    args = ap.parse_args()

    for label, path in (("model", args.model), ("mmproj", args.mmproj),
                        ("image", args.image), ("binary", args.binary)):
        if not Path(path).expanduser().exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2

    print("=== Vision image-token sweep ===")
    print(f"  image : {args.image}")
    print(f"  budgets: {args.budgets}  (None = flag omitted, the control)")
    if args.expect:
        print(f"  expected words present in a correct answer: {args.expect}")
    print()

    results = []
    for budget in args.budgets:
        label = "unset" if budget is None else str(budget)
        print(f"--- budget {label} ---")
        try:
            r = run_once(
                args.binary, Path(args.model).expanduser(), Path(args.mmproj).expanduser(),
                args.image, budget, args.max_tokens, args.timeout,
            )
        except subprocess.TimeoutExpired:
            print("    TIMEOUT")
            results.append({"budget": budget, "timeout": True})
            continue

        hits = [w for w in args.expect if w.lower() in r["text"].lower()]
        r["expected_hits"] = hits
        r["expected_total"] = len(args.expect)
        r["word_lines"] = len([l for l in r["text"].splitlines() if l.strip()])
        results.append(r)

        print(f"    wall={r['wall_s']}s encode={r['encode_ms']}ms load={r['load_ms']}ms "
              f"exit={r['exit_code']} lines={r['word_lines']}")
        if args.expect:
            print(f"    expected words found: {len(hits)}/{len(args.expect)} {hits}")
        print(f"    first line: {r['text'].splitlines()[0][:100] if r['text'] else '(empty)'}")
        print()

    base = next((r for r in results if r.get("budget") is None and not r.get("timeout")), None)
    if base and base.get("encode_ms"):
        print("=== Encode cost vs the unset control ===")
        for r in results:
            if r.get("timeout") or not r.get("encode_ms"):
                continue
            label = "unset" if r["budget"] is None else str(r["budget"])
            ratio = r["encode_ms"] / base["encode_ms"]
            acc = (f", expected {len(r.get('expected_hits', []))}/{r.get('expected_total', 0)}"
                   if args.expect else "")
            print(f"  budget {label:>5}: encode {r['encode_ms']:>7}ms "
                  f"({ratio:4.2f}x of unset){acc}")

    with open(args.out, "w") as fh:
        json.dump({"image": args.image, "max_tokens": args.max_tokens,
                   "expected": args.expect, "results": results},
                  fh, indent=2, ensure_ascii=False)
    print()
    print(f"Written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())