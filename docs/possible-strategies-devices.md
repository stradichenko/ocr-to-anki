# Cross-Platform Target Performance Baseline

## Purpose

This document records only the performance numbers that have actually been
measured for this project, with their source. It is scoped to measurements.
Forward-looking model and runtime strategy lives in a separate state-of-the-art
survey kept outside this repository, at `~/projects/ocr-to-anki-sota/`. It is
held apart deliberately: this repo's `.gitignore` excludes `docs/`, and a
research knowledge base is not aligned with the project's purpose or
presentation.

## Measured numbers

Hardware: Intel Core i3-10110U, UHD Graphics CML GT2 (Gen9.5 iGPU), NixOS.

Source: `README.md` ("Intel iGPU: OpenCL vs Vulkan" section) and the OpenCL
benchmark in `tests/benchmark_results.json` / `tests/benchmark_output.txt`.

| Backend | Vision encoder output | Encode time | Text generation | Binary |
|---------|-----------------------|-------------|-----------------|--------|
| OpenCL | correct | ~2 min (GPU) | 4.1 tok/s | `llama-mtmd-cli-opencl` |
| Vulkan | corrupted | 0.4 s (garbage) | 3.6 tok/s | `llama-mtmd-cli` |
| CPU | correct | ~43 min | 0.7 tok/s | any binary with `--no-mmproj-offload` |

Per-stage breakdown, OpenCL backend, 22 benchmark runs
(`tests/benchmark_results.json`, 2025-11/2026-03 era build b8182/b8292):

- image encode: 103–115 s
- prompt eval: ~2.4–2.7 tok/s over 285 prompt tokens
- generation: ~4.3–4.9 tok/s
- a typical cropped highlight completes in ~145–200 s end to end

CPU mmproj path, from `ocr_stderr.log`:

- image slice encoded in 2,141,938 ms (~35.7 min)
- prompt eval 0.13 tok/s, generation 3.70 tok/s

## What is NOT measured

No mobile, ARM64, NPU, or Apple Silicon measurements exist for this project.
No benchmark has ever been run on a phone, tablet, or any Android device.

Earlier revisions of this file carried a "Modern Mobile Performance Reality"
section with a per-model table (Gemma 2B Q4 "30-50 tokens/sec", Phi-3 Mini
"25-40 tokens/sec", Llama 3.2 1B "50-80 tokens/sec", TinyLlama 1.1B
"30-50 tokens/sec"). Those figures had no source, methodology, hardware, or
date, and they contradicted this repo's own measured results. They have been
removed rather than restated.

## Superseded for strategy

The SOTA survey supersedes this file for forward-looking model and runtime
strategy. This file is retained only as the measured baseline that any new
strategy must beat.