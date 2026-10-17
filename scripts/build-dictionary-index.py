#!/usr/bin/env python3
"""Build a pruned SQLite dictionary index from a Wiktextract JSONL extract.

The app looks words up in a small local index rather than asking the model to
generate a definition it could have looked up. The source is Wiktextract's
per-language extracts from kaikki.org, which are derived from Wiktionary and
carry glosses in the language of the edition they were extracted from.

Those extracts are hundreds of megabytes to several gigabytes. Parsing one on a
phone is not reasonable, so indices are built here, ahead of time, and the app
downloads the small result. This script is what produces that result, and it is
also what tells us how large the result is.

    # Download the extract for a term language, then build:
    python3 scripts/build-dictionary-index.py \\
        --input kaikki.org-dictionary-Japanese.jsonl \\
        --output jmdict-style-index.sqlite \\
        --term-language Japanese

    # Read a .gz directly, and report without writing:
    python3 scripts/build-dictionary-index.py --input x.jsonl.gz --stats-only

Standard library only, so it runs anywhere the app's backend runs.
"""

import argparse
import gzip
import json
import sqlite3
import sys
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    word_norm  TEXT NOT NULL,
    word       TEXT NOT NULL,
    term_lang  TEXT NOT NULL,
    pos        TEXT,
    gloss      TEXT NOT NULL,
    sense_no   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (word_norm, term_lang, pos, sense_no)
);
CREATE INDEX IF NOT EXISTS idx_entries_word ON entries(word_norm, term_lang);

CREATE TABLE IF NOT EXISTS examples (
    word_norm  TEXT NOT NULL,
    term_lang  TEXT NOT NULL,
    sentence   TEXT NOT NULL,
    translation TEXT,
    source     TEXT,
    PRIMARY KEY (word_norm, term_lang, sentence)
);
CREATE INDEX IF NOT EXISTS idx_examples_word ON examples(word_norm, term_lang);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# How many glosses to keep per word. Wiktionary senses are ordered by
# prominence, so the first couple carry the meaning a learner needs; keeping
# all of them multiplies the index for little gain.
GLOSSES_PER_WORD = 2
# How many example sentences to keep per word.
EXAMPLES_PER_WORD = 3


def normalise(word: str) -> str:
    """Lookup key for a headword."""
    return word.strip().lower()


def open_input(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def build(input_path: Path, output_path: Path, term_language: str, limit: int) -> dict:
    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)
    conn.executescript(SCHEMA)

    stats = {
        "lines_read": 0, "entries": 0, "examples": 0,
        "skipped_no_gloss": 0, "skipped_bad_json": 0,
    }
    started = time.monotonic()

    entry_rows: list[tuple] = []
    example_rows: list[tuple] = []

    def flush():
        if entry_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO entries "
                "(word_norm, word, term_lang, pos, gloss, sense_no) VALUES (?,?,?,?,?,?)",
                entry_rows,
            )
            entry_rows.clear()
        if example_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO examples "
                "(word_norm, term_lang, sentence, translation, source) VALUES (?,?,?,?,?)",
                example_rows,
            )
            example_rows.clear()

    with open_input(input_path) as fh:
        for line in fh:
            if limit and stats["lines_read"] >= limit:
                break
            line = line.strip()
            if not line:
                continue
            stats["lines_read"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_bad_json"] += 1
                continue

            word = (rec.get("word") or "").strip()
            if not word:
                continue
            # Only keep records in the requested language, in case the file
            # mixes them.
            if term_language and (rec.get("lang") or "") != term_language:
                continue

            key = normalise(word)
            pos = rec.get("pos") or ""
            lang = rec.get("lang") or term_language

            glosses = []
            examples = []
            for sense in rec.get("senses") or []:
                for g in sense.get("glosses") or []:
                    if g and g not in glosses:
                        glosses.append(g)
                for ex in sense.get("examples") or []:
                    text = (ex.get("text") or "").strip()
                    if text and text not in examples:
                        examples.append((text, (ex.get("translation") or "").strip()))

            if not glosses:
                stats["skipped_no_gloss"] += 1
                continue

            for i, g in enumerate(glosses[:GLOSSES_PER_WORD]):
                entry_rows.append((key, word, lang, pos, g, i))
            stats["entries"] += 1

            for text, translation in examples[:EXAMPLES_PER_WORD]:
                example_rows.append((key, lang, text, translation, "wiktextract"))
                stats["examples"] += 1

            if stats["lines_read"] % 20000 == 0:
                flush()
                conn.commit()
                elapsed = time.monotonic() - started
                print(f"  {stats['lines_read']:,} lines, {stats['entries']:,} entries, "
                      f"{elapsed:.0f}s", file=sys.stderr)

    flush()
    conn.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [
            ("term_language", term_language),
            ("source", input_path.name),
            ("glosses_per_word", str(GLOSSES_PER_WORD)),
            ("built_unix", str(int(time.time()))),
        ],
    )
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    stats["elapsed_s"] = round(time.monotonic() - started, 1)
    stats["index_bytes"] = output_path.stat().st_size
    stats["input_bytes"] = input_path.stat().st_size
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="Wiktextract .jsonl or .jsonl.gz")
    ap.add_argument("--output", help="SQLite file to write")
    ap.add_argument("--term-language", default="", help='e.g. "Japanese"; empty keeps all')
    ap.add_argument("--limit", type=int, default=0, help="stop after N lines (0 = all)")
    ap.add_argument("--stats-only", action="store_true",
                    help="report sizes without writing an index")
    args = ap.parse_args()

    src = Path(args.input).expanduser()
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 2

    if args.stats_only:
        args.output = "/tmp/_dict_index_stats_only.sqlite"
    if not args.output:
        print("--output is required unless --stats-only is given", file=sys.stderr)
        return 2

    out = Path(args.output)
    print(f"input : {src}  ({src.stat().st_size:,} bytes)")
    print(f"output: {out}")
    print(f"term language filter: {args.term_language or '(none)'}")
    print()

    stats = build(src, out, args.term_language, args.limit)

    print()
    print("=== result ===")
    print(f"  lines read        : {stats['lines_read']:,}")
    print(f"  entries kept      : {stats['entries']:,}")
    print(f"  example sentences : {stats['examples']:,}")
    print(f"  dropped (no gloss): {stats['skipped_no_gloss']:,}")
    print(f"  malformed lines   : {stats['skipped_bad_json']:,}")
    print(f"  elapsed           : {stats['elapsed_s']}s")
    if not args.stats_only:
        ratio = stats["index_bytes"] / stats["input_bytes"] * 100 if stats["input_bytes"] else 0
        print(f"  index size        : {stats['index_bytes']:,} bytes "
              f"({stats['index_bytes'] / 1e6:.1f} MB, {ratio:.1f}% of the extract)")
    return 0


if __name__ == "__main__":
    sys.exit(main())