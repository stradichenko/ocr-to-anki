"""Shared pytest configuration.

Makes ``src/`` importable so tests can reach the backend modules directly, and
keeps the manual diagnostic and benchmark scripts out of collection. Those
scripts need either a live llama-server on port 8090 or the downloaded model
files, so they cannot run as part of a normal test pass.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Manual scripts: diagnostics against a running server, and benchmarks that need
# the model files. None of them define collected test functions.
collect_ignore = [
    "test_server_vision.py",
    "test_consecutive.py",
    "benchmark_pipeline.py",
    "benchmark_enrich.py",
]