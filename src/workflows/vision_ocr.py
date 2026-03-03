#!/usr/bin/env python3
"""
Vision OCR using llama.cpp Gemma 3 4B (fully offline).

Supports two backends:
  - **server** (default): uses a persistent llama-server process
    (~7 s per crop, eliminates ~53 s subprocess overhead)
  - **cli**: calls llama-mtmd-cli as a subprocess (~199 s per crop)
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_text(
    image_path: str,
    backend: str = "server",
    server_port: int = 8090,
) -> dict:
    """
    Extract text from an image using the vision model.

    Parameters
    ----------
    image_path : str
        Path to an image file.
    backend : str
        ``"server"`` (persistent server, fast) or ``"cli"`` (subprocess).
    server_port : int
        Port for the llama-server when using server backend.
    """
    if backend == "server":
        from backends.llama_cpp_server import LlamaCppServer

        server = LlamaCppServer(port=server_port)
        if not server.is_running():
            print("Starting llama-server (one-time ~30 s warmup)...")
            server.start()

        result = server.run_vision(image_path)
    elif backend == "cli":
        from backends.mtmd_cli import LlamaMtmdCli

        cli = LlamaMtmdCli()
        result = cli.run_vision(image_path)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (use 'server' or 'cli')")

    return {
        "image": image_path,
        "text": result["text"],
        "elapsed_s": result["elapsed_s"],
        "backend": result.get("backend", backend),
        "timestamp": datetime.now().isoformat(),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Vision OCR with Gemma 3 4B")
    parser.add_argument("image", help="Image path to OCR")
    parser.add_argument(
        "--backend",
        choices=["server", "cli"],
        default="server",
        help="Inference backend (default: server)",
    )
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    args = parser.parse_args()

    result = extract_text(args.image, backend=args.backend, server_port=args.port)

    # Save result
    output_dir = Path("data/ocr_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(result['image']).stem}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved: {output_file}")
    print(f"Elapsed: {result['elapsed_s']:.1f}s")
    print(f"\nExtracted text:\n{result['text']}")


if __name__ == "__main__":
    main()
