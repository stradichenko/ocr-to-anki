#!/usr/bin/env python3
"""Check the model registry against what HuggingFace actually serves.

Every entry in ``app/assets/models.json`` records a size and a SHA-256 for each
file. Android verifies both on startup and deletes the files when they do not
match, then re-downloads them, so a stale hash is not a cosmetic problem: it
turns into an endless download loop of several gigabytes per launch.

That is not hypothetical. Both Gemma 4 model entries were stale because the
GGUFs were re-uploaded upstream after the registry was written, and the default
model is one of them.

Run this after touching the registry, and before a release. Exits non-zero on
any mismatch so it can gate CI.

    python3 scripts/verify-model-registry.py
    python3 scripts/verify-model-registry.py --json app/assets/models.json

Standard library only. Needs network access to huggingface.co.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

API = "https://huggingface.co/api/models/{repo}?blobs=true"


def repo_of(url: str) -> str:
    """Extract ``org/name`` from a HuggingFace resolve URL."""
    parts = url.split("/")
    if len(parts) < 5 or "huggingface.co" not in parts[2]:
        raise ValueError(f"not a HuggingFace resolve URL: {url}")
    return f"{parts[3]}/{parts[4]}"


def revision_of(url: str) -> str:
    """The revision segment of a resolve URL ('main' when unpinned)."""
    parts = url.split("/")
    return parts[6] if len(parts) > 6 else "?"


def fetch_repo(repo: str) -> dict:
    with urllib.request.urlopen(API.format(repo=repo), timeout=60) as resp:
        return json.load(resp)


def check(models: list[dict]) -> list[dict]:
    """Return one result per file checked."""
    cache: dict[str, dict] = {}
    results = []

    for model in models:
        for kind, url_key, size_key, hash_key in (
            ("model", "modelUrl", "modelSizeBytes", "sha256Model"),
            ("mmproj", "mmprojUrl", "mmprojSizeBytes", "sha256Mmproj"),
        ):
            url = model[url_key]
            repo = repo_of(url)
            if repo not in cache:
                try:
                    cache[repo] = fetch_repo(repo)
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                    results.append({
                        "model": model["id"], "kind": kind,
                        "error": f"could not query {repo}: {exc}",
                    })
                    continue

            filename = url.rsplit("/", 1)[-1]
            siblings = {s.get("rfilename"): s for s in cache[repo].get("siblings", [])}
            entry = siblings.get(filename)
            if entry is None:
                results.append({
                    "model": model["id"], "kind": kind, "file": filename,
                    "error": "not present upstream",
                })
                continue

            upstream_size = entry.get("size")
            upstream_hash = (entry.get("lfs") or {}).get("sha256") or entry.get("blobId")

            results.append({
                "model": model["id"],
                "kind": kind,
                "file": filename,
                "revision": revision_of(url),
                "pinned": revision_of(url) != "main",
                "size_ok": upstream_size == model[size_key],
                "hash_ok": upstream_hash == model[hash_key],
                "registry_size": model[size_key],
                "upstream_size": upstream_size,
                "registry_hash": model[hash_key],
                "upstream_hash": upstream_hash,
            })

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default="app/assets/models.json",
                    help="path to the registry (default: app/assets/models.json)")
    ap.add_argument("--quiet", action="store_true",
                    help="only print problems")
    args = ap.parse_args()

    with open(args.json) as fh:
        models = json.load(fh)["models"]

    results = check(models)
    problems = 0

    for r in results:
        label = f"{r['model']} / {r['kind']}"
        if "error" in r:
            print(f"ERROR  {label}: {r['error']}")
            problems += 1
            continue

        ok = r["size_ok"] and r["hash_ok"]
        if ok and args.quiet:
            continue

        status = "ok   " if ok else "STALE"
        print(f"{status}  {label}  {r['file']}  (rev {r['revision'][:12]})")
        if not r["size_ok"]:
            print(f"         size registry={r['registry_size']:,} "
                  f"upstream={r['upstream_size']:,}")
        if not r["hash_ok"]:
            print(f"         hash registry={str(r['registry_hash'])[:24]}...")
            print(f"              upstream={str(r['upstream_hash'])[:24]}...")
        if not r["pinned"]:
            print("         note: URL is not pinned to a revision, so this can "
                  "drift again without the registry changing")
        if not ok:
            problems += 1

    print()
    if problems:
        print(f"{problems} problem(s). Update app/assets/models.json, and pin the "
              f"URLs to a revision so the recorded values stay valid.")
        return 1
    print("Registry matches upstream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())