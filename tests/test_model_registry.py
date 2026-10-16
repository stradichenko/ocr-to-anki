"""Structural checks on the shipped model registry.

These run offline. They cannot tell whether a hash matches upstream, which is
what scripts/verify-model-registry.py does over the network, but they do catch
the conditions that made that script necessary in the first place.

Both Gemma 4 entries were stale because their GGUFs were re-uploaded upstream
after the registry was written. Android verifies the size and hash on startup
and deletes the files when either differs, then re-downloads them, so the stale
default model turned every launch into a multi-gigabyte download that could
never succeed. Pinning the URLs to a revision is what stops it recurring.
"""

import json
import re
from pathlib import Path

import pytest

REGISTRY = Path(__file__).resolve().parent.parent / "app" / "assets" / "models.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_FIELDS = (
    "id", "name", "description", "modelUrl", "mmprojUrl",
    "modelFilename", "mmprojFilename",
    "modelSizeBytes", "mmprojSizeBytes",
    "sha256Model", "sha256Mmproj",
    "supportsVision", "contextSize", "tags",
)


@pytest.fixture(scope="module")
def models():
    with open(REGISTRY) as fh:
        return json.load(fh)["models"]


class TestStructure:
    def test_registry_is_not_empty(self, models):
        assert models

    def test_every_entry_has_every_field(self, models):
        for m in models:
            missing = [f for f in REQUIRED_FIELDS if f not in m]
            assert not missing, f"{m.get('id')} missing {missing}"

    def test_ids_are_unique(self, models):
        ids = [m["id"] for m in models]
        assert len(ids) == len(set(ids))

    def test_first_entry_is_the_default(self, models):
        # ModelRegistry.defaultModel returns models.first, so the first entry is
        # what a fresh install downloads.
        assert models[0]["id"] == "gemma-4-e2b-q4"


class TestHashes:
    def test_hashes_are_lowercase_sha256(self, models):
        for m in models:
            for key in ("sha256Model", "sha256Mmproj"):
                assert HEX64.match(m[key]), f"{m['id']}.{key} is not a sha256"


class TestSizes:
    def test_sizes_are_positive_and_plausible(self, models):
        for m in models:
            for key in ("modelSizeBytes", "mmprojSizeBytes"):
                # A GGUF is megabytes at the very least; a zero or a tiny
                # value means the field was never filled in.
                assert m[key] > 1024 * 1024, f"{m['id']}.{key} looks unset"


class TestUrls:
    def test_urls_are_pinned_to_a_revision(self, models):
        """The guard against silent drift.

        A '/resolve/main/' URL can change under the app at any time, which
        invalidates the recorded size and hash and breaks Android's startup
        verification. A pinned commit cannot move.
        """
        for m in models:
            for key in ("modelUrl", "mmprojUrl"):
                url = m[key]
                assert "/resolve/" in url, f"{m['id']}.{key} has no resolve segment"
                revision = url.split("/resolve/")[1].split("/")[0]
                assert revision != "main", (
                    f"{m['id']}.{key} is unpinned; a re-upload upstream would "
                    f"invalidate the recorded hash"
                )
                assert len(revision) >= 7, (
                    f"{m['id']}.{key} revision {revision!r} looks too short to "
                    f"identify a commit"
                )

    def test_urls_point_at_huggingface(self, models):
        for m in models:
            for key in ("modelUrl", "mmprojUrl"):
                assert m[key].startswith("https://huggingface.co/")

    def test_url_filename_identifies_the_recorded_filename(self, models):
        """The local name may differ in case, but must be the same file.

        Gemma 4 is stored under a lowercased name while upstream capitalises
        it, and the mmproj is renamed outright. Both are deliberate, so the
        check is that the URL still points at the file the entry names, which
        is what the backend and the setup script key off.
        """
        for m in models:
            url_base = m["modelUrl"].rsplit("/", 1)[-1]
            assert url_base.lower() == m["modelFilename"].lower(), m["id"]


class TestArchitecture:
    def test_every_entry_declares_an_architecture(self, models):
        """The backend refuses a model whose architecture it cannot name."""
        for m in models:
            assert m.get("architecture"), f"{m['id']} has no architecture"

    def test_every_architecture_is_supported_by_the_pinned_runtime(self, models):
        """The guard for the bug this registry shipped with.

        The app pinned a llama.cpp older than the default model's architecture,
        so it downloaded three gigabytes and then failed with "unknown model
        architecture". The declared architectures and the runtime's supported
        set must agree.
        """
        pytest.importorskip("cv2", reason="API module needs OpenCV to import")
        import sys
        sys.path.insert(0, str(REGISTRY.parent.parent / "src"))
        from api.app import SUPPORTED_ARCHITECTURES, arch_supported

        for m in models:
            assert arch_supported(m["architecture"]), (
                f"{m['id']} declares architecture {m['architecture']!r}, which "
                f"the pinned runtime does not list in {sorted(SUPPORTED_ARCHITECTURES)}"
            )

    def test_architecture_lookup_is_case_insensitive(self):
        pytest.importorskip("cv2", reason="API module needs OpenCV to import")
        import sys
        sys.path.insert(0, str(REGISTRY.parent.parent / "src"))
        from api.app import arch_supported

        assert arch_supported("GEMMA4")
        assert arch_supported("  gemma4  ")
        assert not arch_supported("definitely-not-an-architecture")


class TestContextSize:
    def test_context_size_is_usable(self, models):
        for m in models:
            assert m["contextSize"] >= 1024, f"{m['id']} context is too small"

    def test_default_model_context_matches_the_backend_default(self, models):
        """The backend derives its enrichment batch from this number."""
        pytest.importorskip("cv2", reason="API module needs OpenCV to import")
        import sys
        sys.path.insert(0, str(REGISTRY.parent.parent / "src"))
        from api.app import DEFAULT_CONTEXT_SIZE

        assert models[0]["contextSize"] == DEFAULT_CONTEXT_SIZE