"""Tests for the shared enrichment loop.

These exercise batching, the retry pass and failure handling against a stub
generation backend, so they need no model files. They do need the full API
module to import, which pulls in OpenCV and FastAPI, so they are skipped
automatically when those are unavailable.
"""

import asyncio

import pytest

pytest.importorskip("cv2", reason="API module needs OpenCV to import")
pytest.importorskip("fastapi", reason="API module needs FastAPI to import")

from api import app as app_module  # noqa: E402
from api.enrichment import (  # noqa: E402
    DEFAULT_ENRICH_TOKENS_PER_WORD,
    MAX_ENRICH_BATCH_WORDS,
)


def _words_in(prompt: str) -> list[str]:
    """Recover the word list the prompt asked about."""
    return [line[2:].strip() for line in prompt.splitlines() if line.startswith("- ")]


def _blocks(words) -> str:
    return "\n".join(
        f"WORD: {w}\nDEF: A definition of the entry.\nEX1: An English example."
        for w in words
    )


class FakeText:
    """Stands in for LlamaCppServer, recording every call it receives."""

    def __init__(self, responder):
        self._responder = responder
        self.calls: list[dict] = []
        self.stopped = 0

    def generate(self, prompt, system=None, max_tokens=0, temperature=0.1, timeout=0):
        self.calls.append(
            {"prompt": prompt, "max_tokens": max_tokens, "timeout": timeout,
             "system": system, "temperature": temperature}
        )
        return {
            "content": self._responder(prompt, len(self.calls)),
            "timings": {"prompt_ms": 100, "predicted_ms": 200,
                        "predicted_per_second": 5.0},
            "tokens_evaluated": 11,
            "tokens_predicted": 22,
            "wall_s": 0.3,
        }

    def stop(self):
        self.stopped += 1


@pytest.fixture
def fake_text(monkeypatch):
    """Install a stub text backend and neuter the server restart path."""
    def _install(responder):
        fake = FakeText(responder)
        monkeypatch.setattr(app_module, "_text", fake)
        monkeypatch.setattr(app_module, "_ensure_text_server", lambda: None)
        return fake
    return _install


def _run(words, **overrides):
    kwargs = dict(
        definition_language="english",
        examples_language="english",
        term_language="auto",
        temperature=0.1,
    )
    kwargs.update(overrides)
    return asyncio.run(app_module._enrich_words(words, **kwargs))


class TestBatching:
    def test_splits_into_context_sized_batches(self, fake_text):
        fake = fake_text(lambda prompt, n: _blocks(_words_in(prompt)))
        words = [f"word{i}" for i in range(13)]

        results, stages = _run(words)
        # 13 words at the cap of 8 gives two calls, not thirteen.
        assert len(fake.calls) == 2
        assert stages["batches"] == 2
        assert stages["batch_size"] == MAX_ENRICH_BATCH_WORDS

    def test_batch_budget_matches_the_words_it_asked_about(self, fake_text):
        fake = fake_text(lambda prompt, n: _blocks(_words_in(prompt)))
        _run([f"word{i}" for i in range(13)])

        for call in fake.calls:
            assert call["max_tokens"] == DEFAULT_ENRICH_TOKENS_PER_WORD * len(
                _words_in(call["prompt"])
            )

    def test_a_single_word_is_one_call(self, fake_text):
        fake = fake_text(lambda prompt, n: _blocks(_words_in(prompt)))
        results, stages = _run(["solo"])
        assert len(fake.calls) == 1
        assert stages["batches"] == 1
        assert results[0]["word"] == "solo"

    def test_results_keep_input_order(self, fake_text):
        fake_text(lambda prompt, n: _blocks(_words_in(prompt)))
        words = [f"word{i}" for i in range(13)]
        results, _ = _run(words)
        assert [r["word"] for r in results] == words


class TestRetryPass:
    def test_a_word_the_model_skipped_is_retried(self, fake_text):
        """The retry pass is what makes a skip survivable."""
        skipped = "word3"

        def responder(prompt, call_no):
            words = _words_in(prompt)
            if call_no == 1:
                # First pass silently omits one word.
                return _blocks([w for w in words if w != skipped])
            return _blocks(words)

        fake = fake_text(responder)
        words = [f"word{i}" for i in range(4)]
        results, stages = _run(words)

        by_word = {r["word"]: r for r in results}
        assert by_word[skipped]["definition"] != ""
        assert by_word[skipped]["warning"] == ""
        assert stages["retry_batches"] >= 1

    def test_a_truncated_definition_is_retried(self, fake_text):
        def responder(prompt, call_no):
            words = _words_in(prompt)
            if call_no == 1:
                return "\n".join(
                    f"WORD: {w}\nDEF: A definition with no full stop\nEX1: An example."
                    for w in words
                )
            return _blocks(words)

        fake = fake_text(responder)
        results, stages = _run(["alpha", "beta"])

        assert all(r["warning"] == "" for r in results)
        assert stages["retry_batches"] >= 1

    def test_no_retry_when_everything_succeeded(self, fake_text):
        fake = fake_text(lambda prompt, n: _blocks(_words_in(prompt)))
        _, stages = _run(["alpha", "beta"])
        assert stages["retry_batches"] == 0


class TestFailureHandling:
    def test_total_failure_marks_every_word_not_found(self, fake_text):
        """This is the guard for the pipeline bug.

        Failed entries used to be written without a warning key at all, which
        the client cache read as a good result and never retried.
        """
        def responder(prompt, call_no):
            raise RuntimeError("backend exploded")

        fake_text(responder)
        results, _ = _run(["alpha", "beta"])

        assert len(results) == 2
        for r in results:
            assert r["warning"] == "not_found"
            assert r["definition"] == ""
            assert "warning" in r

    def test_words_are_never_lost(self, fake_text):
        fake_text(lambda prompt, n: (_ for _ in ()).throw(RuntimeError("boom")))
        words = [f"w{i}" for i in range(5)]
        results, _ = _run(words)
        assert [r["word"] for r in results] == words

    def test_server_is_restarted_once_before_giving_up(self, fake_text):
        fake = fake_text(lambda prompt, n: (_ for _ in ()).throw(RuntimeError("boom")))
        _run(["alpha"])
        assert fake.stopped >= 1


class TestStages:
    def test_accumulates_backend_timings(self, fake_text):
        fake_text(lambda prompt, n: _blocks(_words_in(prompt)))
        _, stages = _run([f"w{i}" for i in range(13)])

        assert stages["prompt_ms"] == 200       # two calls, 100 ms each
        assert stages["predicted_ms"] == 400    # two calls, 200 ms each
        assert stages["tokens_evaluated"] == 22
        assert stages["tokens_predicted"] == 44

    def test_reports_the_batch_size_used(self, fake_text):
        fake_text(lambda prompt, n: _blocks(_words_in(prompt)))
        _, stages = _run(["alpha"])
        assert stages["batch_size"] == MAX_ENRICH_BATCH_WORDS

    def test_no_backend_returns_empty_stages(self, monkeypatch):
        monkeypatch.setattr(app_module, "_text", None)
        results, stages = _run(["alpha"])
        assert stages == {}
        assert results[0]["warning"] == "not_found"