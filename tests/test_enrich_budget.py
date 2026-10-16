"""Unit tests for the enrichment batch sizing.

The batch size is derived from the model context rather than hardcoded, so
these tests are the guard that the two cannot drift apart again.
"""

import pytest

from api.enrichment import (
    DEFAULT_ENRICH_TOKENS_PER_WORD,
    GENERATION_MARGIN_TOKENS,
    MAX_ENRICH_BATCH_WORDS,
    PROMPT_RESERVE_TOKENS,
    compute_batch_size,
    compute_batch_timeout,
)


class TestComputeBatchSize:
    def test_batch_never_overflows_the_context(self):
        """The whole point: prompt plus generation must fit."""
        for ctx in (1024, 2048, 4096, 8192, 16384, 32768):
            size = compute_batch_size(ctx)
            used = PROMPT_RESERVE_TOKENS + GENERATION_MARGIN_TOKENS + (
                size * DEFAULT_ENRICH_TOKENS_PER_WORD
            )
            assert used <= ctx, f"ctx={ctx} size={size} uses {used}"

    def test_never_returns_less_than_one(self):
        for ctx in (0, 1, 64, 128, 300, 575):
            assert compute_batch_size(ctx) >= 1

    def test_respects_the_cap_on_large_contexts(self):
        assert compute_batch_size(131072) == MAX_ENRICH_BATCH_WORDS

    def test_monotonic_in_context(self):
        sizes = [compute_batch_size(ctx) for ctx in range(512, 16384, 512)]
        assert sizes == sorted(sizes)

    def test_regression_at_the_context_sizes_this_project_uses(self):
        """Pinned so a change to the constants is a deliberate act."""
        # The old hardcoded 1024 context produced room for exactly two words.
        assert compute_batch_size(1024) == 2
        # The configured context, which the API previously ignored.
        assert compute_batch_size(4096) == MAX_ENRICH_BATCH_WORDS
        # What the Flutter registry advertises for Gemma 4 E2B.
        assert compute_batch_size(8192) == MAX_ENRICH_BATCH_WORDS

    def test_wider_words_reduce_the_batch(self):
        narrow = compute_batch_size(8192, tokens_per_word=50)
        wide = compute_batch_size(8192, tokens_per_word=2000)
        assert wide <= narrow
        assert wide >= 1

    def test_larger_reserve_reduces_the_batch(self):
        assert compute_batch_size(8192, reserve=7000) < compute_batch_size(8192)

    def test_custom_cap_is_honoured(self):
        assert compute_batch_size(8192, cap=3) == 3


class TestComputeBatchTimeout:
    def test_scales_with_batch_size(self):
        assert compute_batch_timeout(8) > compute_batch_timeout(2)

    def test_has_a_floor(self):
        """A one-word batch must still get a workable window on a slow iGPU."""
        assert compute_batch_timeout(1) >= 600

    def test_always_positive(self):
        for size in (0, 1, 2, 8, 100):
            assert compute_batch_timeout(size) > 0