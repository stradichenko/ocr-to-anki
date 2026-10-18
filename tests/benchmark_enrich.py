#!/usr/bin/env python3
"""Measure the enrichment stage against batch size.

Answers the question the batching change was made on faith about: how much of
the enrichment cost is repeated prompt evaluation, and how much is generation.

Every request re-sends the system prompt, the format template, the worked
example and the word list, and prompt caching is deliberately off (see the LCP
workaround in src/backends/llama_cpp_server.py), so the prompt is evaluated
from scratch each time. Sending one word per request pays that cost once per
word; sending six pays it once per six.

Uses the real prompt builders, the real batch-size helper and the real parser,
so the numbers describe the shipping code path rather than an approximation.

Run inside a shell that provides llama-server:

    nix shell nixpkgs#llama-cpp -c python tests/benchmark_enrich.py \
        --model ~/.cache/llama.cpp/models/gemma-4-e2b-it-q4_0.gguf

Writes JSON to tests/benchmark_enrich_results.json unless --out is given.
Not a pytest file: it needs the model and takes minutes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.enrichment import (  # noqa: E402
    DEFAULT_ENRICH_TOKENS_PER_WORD,
    MAX_ENRICH_BATCH_WORDS,
    build_enrich_prompt,
    build_enrich_system,
    compute_batch_size,
    parse_enrich_response,
)
from backends.llama_cpp_server import LlamaCppServer  # noqa: E402

# A fixed list so runs are comparable. Japanese terms with English glosses,
# matching the default deck this app ships with.
DEFAULT_WORDS = [
    "探す", "静か", "扉", "約束", "景色", "努力", "届く", "懐かしい",
    "偶然", "我慢", "成長", "溜息", "輝く", "幼い", "秘密", "訪れる",
    "疑う", "爽やか", "挑戦", "穏やか",
]


def run_one(server, words, def_lang, ex_lang, term_lang, temperature, timeout):
    """One enrichment request over *words*, measured end to end."""
    prompt = build_enrich_prompt(words, def_lang, ex_lang, term_lang)
    system = build_enrich_system(def_lang, ex_lang, term_lang)
    max_tokens = DEFAULT_ENRICH_TOKENS_PER_WORD * len(words)

    started = time.monotonic()
    result = server.generate(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    wall = time.monotonic() - started

    parsed = parse_enrich_response(result["content"], words)
    timings = result.get("timings") or {}
    warnings = {}
    for entry in parsed:
        warnings[entry["warning"]] = warnings.get(entry["warning"], 0) + 1

    return {
        "words": len(words),
        "wall_s": round(wall, 2),
        "prompt_ms": int(timings.get("prompt_ms") or 0),
        "decode_ms": int(timings.get("predicted_ms") or 0),
        "prompt_tokens": int(result.get("tokens_evaluated") or 0),
        "gen_tokens": int(result.get("tokens_predicted") or 0),
        "tok_per_s": round(float(timings.get("predicted_per_second") or 0), 2),
        "max_tokens_budget": max_tokens,
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="path to the GGUF model")
    ap.add_argument("--mmproj", default=None, help="optional vision projector")
    ap.add_argument(
        "--text-only", action="store_true",
        help=(
            "Do not load a projector. Enrichment is text only, and the server "
            "otherwise pairs the active model with whatever projector happens "
            "to be on disk, which fails when comparing two models."
        ),
    )
    ap.add_argument("--server-binary", default=None, help="path to llama-server")
    ap.add_argument("--context-size", type=int, default=8192)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--term-language", default="japanese")
    ap.add_argument("--definition-language", default="english")
    ap.add_argument("--examples-language", default="english")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[1, 2, 3, 6, MAX_ENRICH_BATCH_WORDS],
        help="batch sizes to compare, each measured over the same words",
    )
    ap.add_argument("--out", default=str(ROOT / "tests" / "benchmark_enrich_results.json"))
    ap.add_argument(
        "--words-limit", type=int, default=0,
        help="cap the word list to keep a run short (0 = no cap)",
    )
    args = ap.parse_args()

    word_pool = DEFAULT_WORDS[: args.words_limit] if args.words_limit else DEFAULT_WORDS

    model = Path(args.model).expanduser()
    if not model.exists():
        print(f"Model not found: {model}", file=sys.stderr)
        return 2

    print("=== Enrichment batch-size benchmark ===")
    print(f"  model:      {model.name}")
    print(f"  context:    {args.context_size}")
    print(f"  derived batch size: {compute_batch_size(args.context_size)}")
    print(f"  word list:  {len(DEFAULT_WORDS)} words")
    print()

    # A path that does not exist reads as "no projector" to the server, which
    # is what a text-only run wants.
    mmproj = "/nonexistent-no-mmproj" if args.text_only else args.mmproj
    server = LlamaCppServer(
        model_path=str(model),
        mmproj_path=mmproj,
        binary_path=args.server_binary,
        context_size=args.context_size,
        n_gpu_layers=args.n_gpu_layers,
    )
    print(f"  server binary: {server.binary_path}")
    print("  starting server ...")
    server.start()
    try:
        results = []
        for size in args.batch_sizes:
            # Keep the word count fixed so the batch size, not the work, is
            # what varies between runs.
            words = word_pool[: max(size, 1) * 2][: len(word_pool)]
            chunks = [words[i:i + size] for i in range(0, len(words), size)]

            print(f"--- batch size {size}: {len(chunks)} call(s) over {len(words)} words ---")
            total_wall = 0.0
            total_prompt_ms = 0
            total_decode_ms = 0
            total_prompt_tokens = 0
            total_gen_tokens = 0
            warnings: dict[str, int] = {}

            for idx, chunk in enumerate(chunks, 1):
                entry = run_one(
                    server, chunk, args.definition_language,
                    args.examples_language, args.term_language,
                    args.temperature, args.timeout,
                )
                total_wall += entry["wall_s"]
                total_prompt_ms += entry["prompt_ms"]
                total_decode_ms += entry["decode_ms"]
                total_prompt_tokens += entry["prompt_tokens"]
                total_gen_tokens += entry["gen_tokens"]
                for key, count in entry["warnings"].items():
                    warnings[key] = warnings.get(key, 0) + count
                print(
                    f"    call {idx}/{len(chunks)} ({len(chunk)} words): "
                    f"wall={entry['wall_s']:.1f}s prompt={entry['prompt_ms']}ms "
                    f"decode={entry['decode_ms']}ms gen={entry['gen_tokens']}tok"
                )

            summary = {
                "batch_size": size,
                "calls": len(chunks),
                "words": len(words),
                "total_wall_s": round(total_wall, 2),
                "wall_per_word_s": round(total_wall / len(words), 2),
                "total_prompt_ms": total_prompt_ms,
                "prompt_ms_per_word": round(total_prompt_ms / len(words), 1),
                "total_decode_ms": total_decode_ms,
                "prompt_tokens": total_prompt_tokens,
                "gen_tokens": total_gen_tokens,
                "warnings": warnings,
            }
            results.append(summary)
            print(
                f"  -> {summary['total_wall_s']}s total, "
                f"{summary['wall_per_word_s']}s/word, "
                f"prompt eval {summary['prompt_ms_per_word']}ms/word, "
                f"warnings={warnings or 'none'}"
            )
            print()

        baseline = next((r for r in results if r["batch_size"] == 1), None)
        if baseline:
            print("=== Amortisation vs one word per request ===")
            for r in results:
                speedup = baseline["total_wall_s"] / r["total_wall_s"] if r["total_wall_s"] else 0
                prompt_ratio = (
                    r["prompt_ms_per_word"] / baseline["prompt_ms_per_word"]
                    if baseline["prompt_ms_per_word"] else 0
                )
                print(
                    f"  batch {r['batch_size']:>2}: {speedup:4.2f}x faster overall, "
                    f"prompt cost {prompt_ratio:4.2f}x of baseline per word"
                )

        payload = {
            "model": model.name,
            "context_size": args.context_size,
            "derived_batch_size": compute_batch_size(args.context_size),
            "term_language": args.term_language,
            "definition_language": args.definition_language,
            "examples_language": args.examples_language,
            "results": results,
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print()
        print(f"Written: {args.out}")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())