#!/usr/bin/env python3
"""Test consecutive vision requests to llama-server with unique request IDs."""

import json
import base64
import urllib.request
import time
import uuid
import sys

SERVER_URL = "http://127.0.0.1:8090"
IMG_PATH = "/home/gespitia/projects/ocr-to-anki/data/cropped_highlights/orange/1d21248b-9dda-4e67-972b-70aa96e35eee_orange_000.png"

def send_vision_request(b64_image: str, prompt_suffix: str = "") -> dict:
    """Send a vision request with a unique request ID."""
    rid = uuid.uuid4().hex[:8]
    payload = {
        "prompt": (
            f"<start_of_turn>user\n"
            f"<start_of_image>\n"
            f"[rid:{rid}] Extract all visible text from this image exactly as written. "
            f"Output ONLY the raw text, nothing else.{prompt_suffix}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        ),
        "image_data": [{"data": b64_image, "id": 10}],
        "temperature": 0.1,
        "n_predict": 256,
        "cache_prompt": False,
        "stop": ["<end_of_turn>", "<eos>"],
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0

    return {
        "tokens": result["tokens_predicted"],
        "prompt_ms": result["timings"]["prompt_ms"],
        "gen_ms": result["timings"]["predicted_ms"],
        "tok_s": result["timings"]["predicted_per_second"],
        "content": result["content"],
        "wall_s": elapsed,
    }


def main():
    n_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Loading image: {IMG_PATH}")
    with open(IMG_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    print(f"Image base64 length: {len(b64)}")

    # Health check
    with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as resp:
        health = json.loads(resp.read())
    print(f"Server health: {health['status']}\n")

    successes = 0
    failures = 0

    for i in range(n_requests):
        try:
            r = send_vision_request(b64)
            status = "OK" if r["tokens"] > 1 else "FAIL"
            if status == "OK":
                successes += 1
            else:
                failures += 1
            content_preview = r["content"][:100].replace("\n", " ")
            print(
                f"REQ {i+1}: {status} "
                f"T={r['tokens']} "
                f"P={r['prompt_ms']:.0f}ms "
                f"G={r['gen_ms']:.0f}ms "
                f"wall={r['wall_s']:.1f}s "
                f"| {content_preview}"
            )
        except Exception as e:
            failures += 1
            print(f"REQ {i+1}: ERROR - {e}")

    print(f"\nResults: {successes}/{n_requests} OK, {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
