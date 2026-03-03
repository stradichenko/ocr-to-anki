#!/usr/bin/env python3
"""
Test consecutive vision requests to llama-server.
Diagnoses the 2nd-request-empty-output bug.
Uses only stdlib (no requests dependency).
"""
import base64, json, time, sys
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

SERVER = "http://127.0.0.1:8090"
IMG = "data/cropped_highlights/orange/handwritten_orange_000.png"  # small crop

def http_get(url, timeout=5):
    return json.loads(urlopen(Request(url), timeout=timeout).read())

def http_post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    return json.loads(urlopen(req, timeout=timeout).read())

def b64_img(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def send_vision_chat(img_b64, label=""):
    """Use /v1/chat/completions with image_url (OAI-compatible)."""
    payload = {
        "model": "gemma3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": "Extract all visible text from this image. List each word."
                    }
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }
    t0 = time.monotonic()
    data = http_post(f"{SERVER}/v1/chat/completions", payload)
    elapsed = time.monotonic() - t0
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"  [{label}] {elapsed:.1f}s | tokens: prompt={usage.get('prompt_tokens','?')} compl={usage.get('completion_tokens','?')}")
    print(f"           output: {content[:200]}")
    return content

def send_vision_completion(img_b64, label=""):
    """Use /completion with image_data (legacy format)."""
    payload = {
        "prompt": "<bos><start_of_turn>user\n<start_of_image>\nExtract all visible text from this image. List each word.<end_of_turn>\n<start_of_turn>model\n",
        "image_data": [{"data": img_b64}],
        "n_predict": 256,
        "temperature": 0.1,
        "stop": ["<end_of_turn>", "<eos>"],
        "cache_prompt": False,
    }
    t0 = time.monotonic()
    data = http_post(f"{SERVER}/completion", payload)
    elapsed = time.monotonic() - t0
    content = data.get("content", "").strip()
    tok_eval = data.get("tokens_evaluated", "?")
    tok_pred = data.get("tokens_predicted", "?")
    print(f"  [{label}] {elapsed:.1f}s | tok_eval={tok_eval} tok_pred={tok_pred}")
    print(f"           output: {content[:200]}")
    return content

# Health check
h = http_get(f"{SERVER}/health")
print(f"Server health: {h}")
print()

img_b64 = b64_img(IMG)
print(f"Image: {IMG} ({len(img_b64)/1024:.1f} KB base64)")
print()

# Test 1: /v1/chat/completions (3 consecutive)
print("=== /v1/chat/completions (OAI-compatible) ===")
for i in range(3):
    try:
        send_vision_chat(img_b64, f"REQ-{i+1}")
    except Exception as e:
        print(f"  [REQ-{i+1}] ERROR: {e}")
    print()

# Test 2: /completion (legacy, 3 consecutive)
print("=== /completion (legacy) ===")
for i in range(3):
    try:
        send_vision_completion(img_b64, f"REQ-{i+1}")
    except Exception as e:
        print(f"  [REQ-{i+1}] ERROR: {e}")
    print()
