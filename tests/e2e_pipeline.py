#!/usr/bin/env python3
"""Run one image through the real backend, end to end.

Every other check in this repository tests a component. This one starts the
actual FastAPI app, posts an image to /pipeline/image-to-cards, and reports what
came back: the OCR text, the cards, and where the time went.

It exists because reasoning about this stack produced confident wrong answers
twice. What the app really does when the pieces meet is worth measuring rather
than inferring.

Run inside a shell that provides the vision backend's libraries, the Python
dependencies, and a model:

    nix shell --impure --expr '...' -c python tests/e2e_pipeline.py \\
        --image data/images/handwritten.jpeg \\
        --binary ~/.cache/llama.cpp/bin/llama-mtmd-cli-vulkan

Starts the backend on a private port, waits for health, runs one request, and
shuts it down. Not a pytest file: it needs a model and a GPU backend.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wait_for_health(port: int, timeout: float = 240) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=10
            ) as resp:
                return json.load(resp)
        except Exception as exc:  # server not up yet
            last = exc
            time.sleep(2)
    raise TimeoutError(f"backend never became healthy: {last}")


def post_image(port: int, image: Path, timeout: int) -> dict:
    boundary = "----ocrtoanki"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; '
        + f'filename="{image.name}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        image.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/pipeline/image-to-cards",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--binary", default="", help="vision binary to force")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default=str(ROOT / "tests" / "e2e_pipeline_result.json"))
    args = ap.parse_args()

    image = Path(args.image)
    if not image.exists():
        print(f"image not found: {image}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    if args.binary:
        # A forced binary is how the caller pins the run to a known build
        # rather than whichever one discovery happens to prefer.
        env["OCR_TO_ANKI_VISION_BINARY"] = args.binary

    print("=== starting the backend ===")
    print(f"  image : {image}")
    print(f"  models: {env.get('LLAMA_CPP_MODELS', '(default)')}")
    print(f"  port  : {args.port}")

    log = open("/tmp/e2e_backend.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.app:app",
         "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
    )

    try:
        health = wait_for_health(args.port)
        print()
        print("=== health ===")
        for key in ("status", "backend", "model", "vision_available",
                    "text_available", "context_size", "enrich_batch_size",
                    "image_max_tokens"):
            print(f"  {key:18}: {health.get(key)}")

        if not health.get("vision_available"):
            print("\nVision backend unavailable; the pipeline cannot run.")
            return 1

        print()
        print("=== running one image through the pipeline ===")
        started = time.monotonic()
        try:
            result = post_image(args.port, image, args.timeout)
        except urllib.error.HTTPError as exc:
            print(f"  HTTP {exc.code}: {exc.read()[:400].decode(errors='replace')}")
            return 1
        wall = time.monotonic() - started

        print(f"  total wall        : {wall:.1f}s")
        print(f"  ocr backend       : {result.get('ocr_backend')}")
        print(f"  ocr elapsed       : {result.get('ocr_elapsed_s')}s")
        print(f"  enrich elapsed    : {result.get('enrich_elapsed_s')}s")
        stages = result.get("stages") or {}
        if stages:
            print(f"  batches           : {stages.get('batches')} "
                  f"(+{stages.get('retry_batches')} retry), "
                  f"batch size {stages.get('batch_size')}")
            print(f"  prompt / decode ms: {stages.get('prompt_ms')} / "
                  f"{stages.get('predicted_ms')}")

        print()
        print("=== OCR text ===")
        text = (result.get("ocr_text") or "").strip()
        print("  " + (text.replace("\n", "\n  ") if text else "(empty)"))
        if "<|channel>" in text:
            print("  WARNING: a reasoning channel survived into the OCR text")

        cards = result.get("cards") or []
        print()
        print(f"=== cards ({len(cards)}) ===")
        for i, card in enumerate(cards, 1):
            word = card.get("word", "")
            definition = (card.get("definition") or "").replace("\n", " ")
            warning = card.get("warning") or ""
            source = card.get("source") or ""
            flag = f" [{warning}]" if warning else ""
            src = f" ({source})" if source else ""
            print(f"  {i:>2}. {word}{flag}{src}")
            print(f"      {definition[:100]}")

        ok = [c for c in cards if c.get("definition")]
        print()
        print(f"  cards with a definition: {len(ok)}/{len(cards)}")

        with open(args.out, "w") as fh:
            json.dump({"wall_s": round(wall, 2), "health": health, **result},
                      fh, indent=2, ensure_ascii=False)
        print(f"  written: {args.out}")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())