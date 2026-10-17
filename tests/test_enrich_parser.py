"""Unit tests for parse_enrich_response.

This function is pure and was previously untested. It is the component most at
risk when enrichment batches get wider, because a model that skips or merges a
word must not cause the remaining words to receive each other's definitions.
"""

import pytest

from api.enrichment import (
    parse_enrich_response,
    sanitize_words,
    split_ocr_words,
)


def _block(word, definition, ex1, ex2=None):
    text = f"WORD: {word}\nDEF: {definition}\nEX1: {ex1}\n"
    if ex2:
        text += f"EX2: {ex2}\n"
    return text


class TestWellFormedResponse:
    """The ordinary case: the model echoes every requested word.

    Example sentences here are in English, a different language from the
    vocabulary word, which is the normal case for this app. Writing the
    examples in the source language is what the untranslated warning exists to
    catch, so these fixtures deliberately avoid that.
    """

    def test_single_word_exact_match(self):
        text = _block("casa", "A house.", "I live in a big house.", "My house is small.")
        [result] = parse_enrich_response(text, ["casa"])

        assert result["word"] == "casa"
        assert result["definition"] == "A house."
        assert "I live in a big house." in result["examples"]
        assert "My house is small." in result["examples"]
        assert result["warning"] == ""

    def test_multiple_words_keep_their_own_definitions(self):
        text = (
            _block("uno", "The number one.", "One plus one is two.", "Give me one.")
            + _block("dos", "The number two.", "Two plus two is four.", "Give me two.")
            + _block("tres", "The number three.", "Three is a prime.", "Give me three.")
        )
        results = parse_enrich_response(text, ["uno", "dos", "tres"])

        assert [r["word"] for r in results] == ["uno", "dos", "tres"]
        assert results[0]["definition"] == "The number one."
        assert results[1]["definition"] == "The number two."
        assert results[2]["definition"] == "The number three."
        assert all(r["warning"] == "" for r in results)

    def test_matching_is_case_insensitive(self):
        text = _block("Casa", "A house.", "It is a large house.", None)
        [result] = parse_enrich_response(text, ["casa"])
        assert result["definition"] == "A house."

    def test_accented_label_matches_accented_request(self):
        """The model may add or drop accents; the word itself is unchanged."""
        text = _block("café", "Coffee.", "I drink coffee every morning.", None)
        [result] = parse_enrich_response(text, ["café"])
        assert result["definition"] == "Coffee."


class TestWarnings:
    def test_unknown_marker_becomes_not_found(self):
        text = _block("xyzzy", "UNKNOWN", "", None)
        [result] = parse_enrich_response(text, ["xyzzy"])
        assert result["warning"] == "not_found"
        assert result["definition"] == ""
        assert result["examples"] == ""

    def test_missing_word_becomes_not_found(self):
        """A word the model never mentioned, with no blocks to misalign to."""
        text = _block("uno", "The number one.", "One plus one is two.", None)
        results = parse_enrich_response(text, ["uno", "dos"])
        assert results[0]["warning"] == ""
        assert results[1]["word"] == "dos"
        assert results[1]["warning"] == "not_found"
        assert results[1]["definition"] == ""

    def test_definition_without_terminal_punctuation_is_truncated(self):
        text = _block("casa", "A house with no full stop", "", None)
        [result] = parse_enrich_response(text, ["casa"])
        assert result["warning"] == "truncated"

    def test_untranslated_when_source_word_leaks_into_examples(self):
        """Examples written in the source language are the failure this catches."""
        text = _block("casa", "A house.", "casa casa casa", "la casa")
        [result] = parse_enrich_response(text, ["casa"])
        assert result["warning"] == "untranslated"

    def test_examples_in_the_source_language_are_flagged_but_definition_is_kept(self):
        text = _block("casa", "A house.", "Mi casa es grande.", "Vivo en una casa.")
        [result] = parse_enrich_response(text, ["casa"])
        assert result["warning"] == "untranslated"
        assert result["definition"] == "A house."

    def test_empty_response_marks_every_word_not_found(self):
        results = parse_enrich_response("", ["uno", "dos"])
        assert [r["word"] for r in results] == ["uno", "dos"]
        assert all(r["warning"] == "not_found" for r in results)

    def test_empty_word_list_returns_empty(self):
        assert parse_enrich_response("WORD: casa\nDEF: A house.\n", []) == []


class TestPositionalAlignment:
    """The positional fallback is the risk that grows with batch size.

    When the model returns a block whose label does not match the expected
    word, that block must never be attached to the wrong vocabulary word.
    Attaching a neighbouring word's definition produces a card that looks
    correct and is silently wrong, which is worse than an obvious miss.
    """

    def test_skipped_word_does_not_receive_the_next_words_definition(self):
        # The model returned alpha and gamma but silently skipped beta.
        # Block 1 belongs to gamma, so beta must not inherit it.
        text = (
            _block("alpha", "The first letter.", "Alpha comes first.", None)
            + _block("gamma", "The third letter.", "Gamma comes third.", None)
        )
        results = parse_enrich_response(text, ["alpha", "beta", "gamma"])

        beta = next(r for r in results if r["word"] == "beta")
        alpha = next(r for r in results if r["word"] == "alpha")
        gamma = next(r for r in results if r["word"] == "gamma")

        assert alpha["definition"] == "The first letter."
        assert gamma["definition"] == "The third letter."

        # The regression: beta must be reported as missing, not given gamma's
        # definition by position.
        assert beta["definition"] != "The third letter."
        assert beta["warning"] == "not_found"

    def test_duplicated_word_does_not_displace_the_rest(self):
        # The model emitted alpha twice and never emitted beta. Counting alone
        # cannot catch this, because the block count still matches.
        text = (
            _block("alpha", "The first letter.", "Alpha comes first.", None)
            + _block("alpha", "The first letter again.", "Alpha again.", None)
        )
        results = parse_enrich_response(text, ["alpha", "beta"])

        beta = next(r for r in results if r["word"] == "beta")
        assert beta["definition"] != "The first letter again."
        assert beta["warning"] == "not_found"


class TestRobustness:
    def test_markdown_asterisks_are_stripped(self):
        text = "WORD: casa\nDEF: **A house.**\nEX1: *Mi casa.*\n"
        [result] = parse_enrich_response(text, ["casa"])
        assert "*" not in result["definition"]
        assert "*" not in result["examples"]

    def test_corrected_line_is_ignored(self):
        text = (
            "WORD: casa\nDEF: A house.\n"
            "CORRECTED: kasa\n"
            "EX1: Mi casa.\n"
        )
        [result] = parse_enrich_response(text, ["casa"])
        assert result["definition"] == "A house."

    def test_result_shape_is_stable(self):
        text = _block("casa", "A house.", "Mi casa.", None)
        [result] = parse_enrich_response(text, ["casa"])
        assert set(result) == {
            "word",
            "definition",
            "examples",
            "warning",
            "corrected_word",
        }

    def test_output_length_always_matches_input(self):
        text = _block("uno", "One.", "Uno mas.", None)
        for words in (["uno"], ["uno", "dos"], ["uno", "dos", "tres"]):
            assert len(parse_enrich_response(text, words)) == len(words)

class TestSplitOcrWords:
    """Turning transcribed text into vocabulary words.

    Found by running an image through the real pipeline: the vision model is
    asked for one word per line but transcribes what it sees, so a page whose
    entries are comma-separated came back comma-separated and each line became
    a single flashcard holding three words. Splitting on newlines was the bug.
    """

    def test_one_word_per_line(self):
        assert split_ocr_words("casa\nperro\ngato") == ["casa", "perro", "gato"]

    def test_comma_separated_line_becomes_separate_words(self):
        assert split_ocr_words("proies, repère, pencha,") == [
            "proies", "repère", "pencha",
        ]

    def test_the_real_failure_that_prompted_this(self):
        text = "côta\nl'amont, l'aval,\nproies, repère, pencha,"
        assert split_ocr_words(text) == [
            "côta", "l'amont", "l'aval", "proies", "repère", "pencha",
        ]

    def test_bullets_are_stripped_in_the_splitter(self):
        assert split_ocr_words("- casa\n- perro") == ["casa", "perro"]
        assert split_ocr_words("* casa") == ["casa"]

    def test_numbering_is_left_to_sanitize_words(self):
        """Splitting and cleaning are separate steps, and the pipeline runs
        both. Asserting the pair is what actually happens."""
        assert split_ocr_words("2. perro") == ["2. perro"]
        assert sanitize_words(split_ocr_words("2. perro\n- casa")) == [
            "perro", "casa",
        ]

    def test_other_separators(self):
        assert split_ocr_words("one; two") == ["one", "two"]
        assert split_ocr_words("one\ttwo") == ["one", "two"]

    def test_empty_and_whitespace(self):
        assert split_ocr_words("") == []
        assert split_ocr_words("\n\n  \n") == []

    def test_trailing_punctuation_does_not_create_empty_words(self):
        assert split_ocr_words("casa,,perro,") == ["casa", "perro"]

    def test_accented_and_cjk_entries_survive(self):
        assert split_ocr_words("探す, 静か") == ["探す", "静か"]
        assert split_ocr_words("café, fenêtre") == ["café", "fenêtre"]
