"""Tests for the active-model endpoint.

The backend previously hardcoded its model while the shipped Flutter registry
defaulted to a different one, with nothing carrying the choice between them.
These tests pin the endpoint that closes that gap. They need the API module to
import, so they skip themselves when its heavier dependencies are unavailable.
"""

import pytest

pytest.importorskip("cv2", reason="API module needs OpenCV to import")
pytest.importorskip("fastapi", reason="API module needs FastAPI to import")

from fastapi.testclient import TestClient  # noqa: E402

from api import app as app_module  # noqa: E402
from api.app import _DEFAULT_MODEL, DEFAULT_CONTEXT_SIZE  # noqa: E402
from api.enrichment import MAX_ENRICH_BATCH_WORDS  # noqa: E402


DESCRIPTOR = {
    "id": "test-model-q4",
    "model_file": "test-model-q4.gguf",
    "mmproj_file": "test-mmproj.gguf",
    "model_url": "https://example.invalid/test-model-q4.gguf",
    "mmproj_url": "https://example.invalid/test-mmproj.gguf",
    "architecture": "gemma4",
    "context_size": 4096,
}


@pytest.fixture
def client(monkeypatch):
    """A client with the reinit path stubbed out.

    The real one stops and starts llama.cpp servers; these tests are about the
    descriptor, not about process management.
    """
    async def _noop_reinit():
        return {"vision": True, "text": True}

    monkeypatch.setattr(app_module, "_reinit_backends", _noop_reinit)
    monkeypatch.setattr(app_module, "_active_model", dict(_DEFAULT_MODEL))
    yield TestClient(app_module.app)
    # Restore, so one test cannot leak a descriptor into the next.
    app_module._active_model = dict(_DEFAULT_MODEL)


class TestSetModel:
    def test_accepts_a_valid_descriptor(self, client):
        resp = client.post("/config/model", json=DESCRIPTOR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == DESCRIPTOR["id"]
        assert body["context_size"] == 4096

    def test_the_descriptor_takes_effect(self, client):
        client.post("/config/model", json=DESCRIPTOR)
        assert app_module._active_model["model_file"] == "test-model-q4.gguf"
        assert app_module._active_model["mmproj_file"] == "test-mmproj.gguf"

    def test_model_paths_follow_the_descriptor(self, client):
        client.post("/config/model", json=DESCRIPTOR)
        assert app_module._model_path("model").name == "test-model-q4.gguf"
        assert app_module._model_path("mmproj").name == "test-mmproj.gguf"

    def test_reports_the_derived_batch_size(self, client):
        resp = client.post("/config/model", json=DESCRIPTOR)
        assert resp.json()["enrich_batch_size"] == MAX_ENRICH_BATCH_WORDS

    def test_a_smaller_context_shrinks_the_batch(self, client):
        small = dict(DESCRIPTOR, context_size=1024)
        resp = client.post("/config/model", json=small)
        assert resp.status_code == 200
        # At 1024 the derived batch is two, which is why the old hardcoded
        # context silently capped enrichment there.
        assert resp.json()["enrich_batch_size"] == 2


class TestValidation:
    def test_requires_an_id(self, client):
        body = {k: v for k, v in DESCRIPTOR.items() if k != "id"}
        assert client.post("/config/model", json=body).status_code == 400

    def test_requires_every_file_field(self, client):
        for field in ("model_file", "mmproj_file", "model_url", "mmproj_url"):
            body = {k: v for k, v in DESCRIPTOR.items() if k != field}
            resp = client.post("/config/model", json=body)
            assert resp.status_code == 400, f"{field} should be required"

    def test_rejects_a_context_too_small_to_hold_a_word(self, client):
        body = dict(DESCRIPTOR, context_size=128)
        assert client.post("/config/model", json=body).status_code == 400

    def test_a_rejected_request_leaves_the_model_unchanged(self, client):
        before = dict(app_module._active_model)
        client.post("/config/model", json={"id": "broken"})
        assert app_module._active_model == before

    def test_context_defaults_when_omitted(self, client):
        body = {k: v for k, v in DESCRIPTOR.items() if k != "context_size"}
        resp = client.post("/config/model", json=body)
        assert resp.status_code == 200
        assert resp.json()["context_size"] == DEFAULT_CONTEXT_SIZE


class TestArchitectureGuard:
    """The check that stops a download the runtime cannot load.

    The app once shipped a default model whose architecture its pinned
    llama.cpp predated, so it fetched three gigabytes and then failed with
    "unknown model architecture". The refusal has to happen before the
    download, which means at the point the model is chosen.
    """

    def test_rejects_an_architecture_the_runtime_cannot_load(self, client):
        body = dict(DESCRIPTOR, architecture="some-future-architecture")
        resp = client.post("/config/model", json=body)
        assert resp.status_code == 400
        assert "architecture" in resp.json()["detail"]

    def test_error_names_the_supported_set(self, client):
        body = dict(DESCRIPTOR, architecture="some-future-architecture")
        detail = client.post("/config/model", json=body).json()["detail"]
        assert "gemma4" in detail

    def test_accepts_a_supported_architecture(self, client):
        body = dict(DESCRIPTOR, architecture="gemma4")
        assert client.post("/config/model", json=body).status_code == 200

    def test_is_case_insensitive(self, client):
        body = dict(DESCRIPTOR, architecture="GEMMA4")
        assert client.post("/config/model", json=body).status_code == 200

    def test_an_omitted_architecture_is_allowed(self, client):
        """Older clients do not send it, and the default is already checked."""
        body = {k: v for k, v in DESCRIPTOR.items() if k != "architecture"}
        assert client.post("/config/model", json=body).status_code == 200

    def test_a_rejected_architecture_leaves_the_model_unchanged(self, client):
        before = dict(app_module._active_model)
        client.post("/config/model", json=dict(DESCRIPTOR, architecture="nope"))
        assert app_module._active_model == before


class TestGetModel:
    def test_reports_the_default_before_any_change(self, client):
        body = client.get("/config/model").json()
        assert body["id"] == _DEFAULT_MODEL["id"]

    def test_reflects_a_change(self, client):
        client.post("/config/model", json=DESCRIPTOR)
        assert client.get("/config/model").json()["id"] == DESCRIPTOR["id"]


class TestDefaultDescriptor:
    def test_default_matches_the_shipped_registry_entry(self):
        """The migration the endpoint exists to make possible.

        The registry's first entry is the default the Flutter side advertises,
        so the backend has to agree with it rather than with Gemma 3.
        """
        assert _DEFAULT_MODEL["id"] == "gemma-4-e2b-q4"
        assert _DEFAULT_MODEL["context_size"] == DEFAULT_CONTEXT_SIZE
        assert _DEFAULT_MODEL["model_file"].endswith(".gguf")
        assert _DEFAULT_MODEL["mmproj_file"].endswith(".gguf")

    def test_default_is_not_the_old_hardcoded_model(self):
        assert "gemma-3" not in _DEFAULT_MODEL["model_file"]