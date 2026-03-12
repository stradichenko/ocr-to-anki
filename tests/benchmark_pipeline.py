#!/usr/bin/env python3
"""
Benchmark: OCR-to-Anki pipeline optimization evaluation.

Skips the ~43 min CPU vision test (already measured in prior sessions).
Focus: GPU vision via OpenCL on Intel Gen9 iGPU.
"""

import os, sys, time, subprocess, json, io
from pathlib import Path

# Force unbuffered output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from backends.auto_detect import detect, _opencl_env, _find_binary, Backend

MODELS_DIR = Path(os.getenv("LLAMA_CPP_MODELS", Path.home() / ".cache" / "llama.cpp" / "models"))
MODEL  = MODELS_DIR / "gemma-3-4b-it-q4_0_s.gguf"
MMPROJ = MODELS_DIR / "mmproj-model-f16-4B.gguf"
PROJECT    = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT / "data" / "images"
CROPPED_DIR = PROJECT / "data" / "cropped_highlights" / "orange"
PROMPT = "Extract all visible text from this image. List each word or phrase you can read."

results = []

def run_ocr(binary, image, extra_args=None, env=None, timeout=600, label=""):
    cmd = [str(binary), "-m", str(MODEL), "--mmproj", str(MMPROJ),
           "--image", str(image), "-p", PROMPT, "--jinja", "-ngl", "-1"]
    if extra_args: cmd.extend(extra_args)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        r = {"label": label, "status": "TIMEOUT", "elapsed_s": timeout}
        results.append(r); return r

    elapsed = time.monotonic() - t0
    encode_ms = prompt_eval_tok_s = eval_tok_s = None
    prompt_tokens = eval_tokens = load_ms = None

    if proc.stderr:
        for line in proc.stderr.splitlines():
            if "encoded in" in line:
                try: encode_ms = float(line.split("encoded in")[1].split("ms")[0].strip())
                except: pass
            if "prompt eval time" in line:
                try:
                    prompt_tokens = int(line.split("/")[1].strip().split()[0])
                    prompt_eval_tok_s = float(line.split("tokens per second)")[0].rsplit(",",1)[-1].strip())
                except: pass
            elif "eval time" in line and "prompt" not in line:
                try:
                    eval_tokens = int(line.split("/")[1].strip().split()[0])
                    eval_tok_s = float(line.split("tokens per second)")[0].rsplit(",",1)[-1].strip())
                except: pass
            elif "load time" in line:
                try: load_ms = float(line.split("=")[1].strip().split("ms")[0].strip())
                except: pass

    text = proc.stdout.strip() if proc.returncode == 0 else ""
    r = {
        "label": label,
        "status": "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})",
        "elapsed_s": round(elapsed, 2),
        "load_ms": load_ms, "encode_ms": encode_ms,
        "encode_s": round(encode_ms/1000, 2) if encode_ms else None,
        "prompt_eval_tok_s": prompt_eval_tok_s, "prompt_tokens": prompt_tokens,
        "eval_tok_s": eval_tok_s, "eval_tokens": eval_tokens,
        "output_words": len(text.split()) if text else 0,
        "text_preview": text[:150] if text else "(empty)",
    }
    results.append(r); return r

def pr(r):
    enc = f"{r['encode_s']}s" if r.get('encode_s') else "n/a"
    gen = f"{r['eval_tok_s']:.1f}t/s" if r.get('eval_tok_s') else "n/a"
    print(f"  [{r['status']:>4}] {r['label']:<48s} total={r['elapsed_s']:>6.1f}s  enc={enc:<8s} gen={gen:<10s} toks={r.get('eval_tokens','?')}")

def hdr(title):
    print(f"\n{'─'*72}\n  {title}\n{'─'*72}")

def main():
    print("═"*60)
    print("  OCR Pipeline Benchmark (GPU Vision / OpenCL)")
    print("═"*60)
    print(f"  Model:  {MODEL.name} ({MODEL.stat().st_size/1e9:.2f} GB)")
    print(f"  mmproj: {MMPROJ.name} ({MMPROJ.stat().st_size/1e6:.0f} MB)")
    cpu_n = os.cpu_count() or 4
    half = max(1, cpu_n // 2)
    print(f"  CPUs: {cpu_n}  (default threads: {half})")

    # ── Detection ──────────────────────────────────────────────
    hdr("0. Backend Detection")
    times = []
    for _ in range(5):
        t0 = time.monotonic(); det = detect(); times.append(time.monotonic()-t0)
    print(f"  avg={sum(times)/len(times)*1000:.0f}ms  best={min(times)*1000:.0f}ms")
    print(f"  → {det.recommended_backend.value}: {det.binary_path}")

    ocl = _find_binary(Backend.OPENCL)
    env = _opencl_env()
    if not ocl:
        print("  [ERR] No OpenCL binary"); return

    full  = IMAGES_DIR / "handwritten.jpeg"      # 1133×1348, 145 KB
    small = CROPPED_DIR / "handwritten_orange_000.png"  # ~12 KB
    large = IMAGES_DIR / "20251110_174327.jpg"    # 2706×1330, 706 KB

    gpu = ["--ctx-size","4096","--threads",str(half),"--temp","0.1","--top-k","40","--top-p","0.9","-n","512"]

    # ── 1. GPU Vision baseline ─────────────────────────────────
    hdr("1. GPU Vision Baseline")
    r = run_ocr(ocl, full, gpu, env=env, label="GPU vision baseline (handwritten)")
    pr(r)

    # ── 2. Image size ──────────────────────────────────────────
    hdr("2. Image Size Impact")
    for img, lbl in [(small,"small crop 12KB"), (full,"medium 145KB 1133x1348"), (large,"large 706KB 2706x1330")]:
        if img.exists():
            r = run_ocr(ocl, img, gpu, env=env, label=lbl); pr(r)

    # ── 3. Thread count ────────────────────────────────────────
    hdr("3. Thread Scaling")
    for t in sorted(set([1, 2, half, cpu_n])):
        a = gpu.copy(); a[a.index("--threads")+1] = str(t)
        r = run_ocr(ocl, full, a, env=env, label=f"threads={t}"); pr(r)

    # ── 4. Context size ────────────────────────────────────────
    hdr("4. Context Size")
    for c in [2048, 4096, 8192]:
        a = gpu.copy(); a[a.index("--ctx-size")+1] = str(c)
        r = run_ocr(ocl, full, a, env=env, label=f"ctx={c}"); pr(r)

    # ── 5. Max tokens ──────────────────────────────────────────
    hdr("5. Max Tokens")
    for n in [128, 256, 512, 1024]:
        a = gpu.copy(); a[a.index("-n")+1] = str(n)
        r = run_ocr(ocl, full, a, env=env, label=f"max_tokens={n}"); pr(r)

    # ── 6. Sampling ────────────────────────────────────────────
    hdr("6. Sampling Strategy")
    for temp,k,p,lbl in [(0.0,1,1.0,"greedy"),(0.1,40,0.9,"default"),(0.3,50,0.95,"creative")]:
        a = gpu.copy()
        a[a.index("--temp")+1]=str(temp); a[a.index("--top-k")+1]=str(k); a[a.index("--top-p")+1]=str(p)
        r = run_ocr(ocl, full, a, env=env, label=f"sampling: {lbl}"); pr(r)

    # ── 7. Batch crops ─────────────────────────────────────────
    hdr("7. Sequential Crop Batch")
    crops = sorted(CROPPED_DIR.glob("*.png"))[:5]
    t0_batch = time.monotonic()
    for i,c in enumerate(crops):
        r = run_ocr(ocl, c, gpu, env=env, label=f"crop[{i}] {c.name}"); pr(r)
    batch_total = time.monotonic() - t0_batch
    if crops:
        print(f"  → {len(crops)} crops: {batch_total:.1f}s total, {batch_total/len(crops):.1f}s avg")

    # ── Summary ────────────────────────────────────────────────
    hdr("SUMMARY")
    print(f"  {'#':>2} {'Label':<48s} {'Total':>7s} {'Encode':>8s} {'Gen':>7s} {'Toks':>5s}")
    print(f"  {'─'*2} {'─'*48} {'─'*7} {'─'*8} {'─'*7} {'─'*5}")
    for i,r in enumerate(results):
        enc = f"{r.get('encode_s','')}s" if r.get('encode_s') else "-"
        gen = f"{r['eval_tok_s']:.1f}" if r.get('eval_tok_s') else "-"
        tot = f"{r['elapsed_s']:.1f}s" if r['status']=='OK' else r['status']
        print(f"  {i+1:>2} {r['label']:<48s} {tot:>7s} {enc:>8s} {gen:>7s} {str(r.get('eval_tokens','')):>5s}")

    # Save JSON
    out = PROJECT / "tests" / "benchmark_results.json"
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  → {out}")

    # ── Analysis ───────────────────────────────────────────────
    hdr("OPTIMIZATION ANALYSIS")
    baseline = next((r for r in results if "baseline" in r['label'] and r['status']=='OK'), None)
    if baseline:
        enc_t = baseline.get('encode_s',0) or 0
        gen_rate = baseline.get('eval_tok_s',0) or 0
        gen_toks = baseline.get('eval_tokens',0) or 0
        gen_t = gen_toks / gen_rate if gen_rate else 0
        load_t = (baseline.get('load_ms',0) or 0) / 1000
        total = baseline['elapsed_s']
        overhead = total - enc_t - gen_t - load_t
        print(f"\n  Baseline breakdown (total {total:.1f}s):")
        print(f"    Model load:   {load_t:>6.1f}s  ({load_t/total*100:>4.0f}%)")
        print(f"    Img encode:   {enc_t:>6.1f}s  ({enc_t/total*100:>4.0f}%)")
        print(f"    Text gen:     {gen_t:>6.1f}s  ({gen_t/total*100:>4.0f}%) @ {gen_rate:.1f} tok/s")
        print(f"    Overhead:     {overhead:>6.1f}s  ({overhead/total*100:>4.0f}%)")

    crop_rs = [r for r in results if r['label'].startswith('crop[') and r['status']=='OK']
    if crop_rs:
        loads = [r.get('load_ms',0) or 0 for r in crop_rs]
        avg_load = sum(loads)/len(loads)/1000
        print(f"\n  Batch crop analysis ({len(crop_rs)} crops):")
        print(f"    Avg load time:  {avg_load:.1f}s/crop")
        print(f"    → Server mode could save ~{avg_load*len(crop_rs):.0f}s by loading model once")

    print("\n  Optimization Priorities (estimated impact):")
    print("  ┌─────┬────────────────────────────────────────────┬──────────┐")
    print("  │  #  │ Optimization                               │ Est Gain │")
    print("  ├─────┼────────────────────────────────────────────┼──────────┤")
    print("  │  1  │ [OK] GPU vision (OpenCL) -- already done      │ ~20×     │")
    print("  │  2  │ [ ] Server mode (keep model loaded)         │ ~30-50%  │")
    print("  │  3  │ [ ] Downscale large images before OCR       │ ~20-40%  │")
    print("  │  4  │ [ ] Shorter/tuned prompt                    │ ~5-15%   │")
    print("  │  5  │ [ ] Reduce max_tokens for simple extracts   │ ~10-30%  │")
    print("  │  6  │ [ ] Parallel crop processing (if multi-GPU) │ N/A      │")
    print("  └─────┴────────────────────────────────────────────┴──────────┘")

if __name__ == "__main__":
    main()
