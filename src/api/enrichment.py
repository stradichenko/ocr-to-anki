"""Pure helpers for the enrichment step.

Prompt construction, response parsing and word sanitising for the vocabulary
enrichment stage. Everything here is a pure function over strings and lists:
no model, no HTTP, no FastAPI, no OpenCV. That keeps the parsing logic unit
testable without pulling in the backend's heavier dependencies, which matters
because this is the component most at risk when batches get wider.

The batch-size budget lives here too (see ``compute_batch_size``) so the sizes
and the parsing that depends on them sit together.
"""

import logging
import re
import unicodedata
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch sizing
# ---------------------------------------------------------------------------

# Tokens per word the model is allowed to spend on a definition plus two
# example sentences.
DEFAULT_ENRICH_TOKENS_PER_WORD = 150

# Room for the system prompt, the format template, the worked example, and the
# word list itself. Measured against tokens_evaluated once a real model is
# running; 512 is a conservative starting point.
PROMPT_RESERVE_TOKENS = 512

# Slack so the final word of a batch is not cut off inside its last sentence.
GENERATION_MARGIN_TOKENS = 64

# Upper bound regardless of how much context is available. Format adherence
# degrades over long batches and the parser has less room to recover, so this
# is raised deliberately after measuring rather than set to whatever fits.
MAX_ENRICH_BATCH_WORDS = 8

# Timeout budget for one batch. The integrated GPU is slow at prompt
# evaluation, so the floor is generous and the per-word term covers generation.
ENRICH_TIMEOUT_BASE_S = 240
ENRICH_TIMEOUT_PER_WORD_S = 90
ENRICH_TIMEOUT_FLOOR_S = 600


def compute_batch_size(
    context_size: int,
    tokens_per_word: int = DEFAULT_ENRICH_TOKENS_PER_WORD,
    reserve: int = PROMPT_RESERVE_TOKENS,
    margin: int = GENERATION_MARGIN_TOKENS,
    cap: int = MAX_ENRICH_BATCH_WORDS,
) -> int:
    """How many words fit in one enrichment call at this context size.

    Every word costs ``tokens_per_word`` of generation, and every call also
    carries the same fixed prompt. Sizing the batch from the context is what
    lets the context be raised without the batch silently overflowing it, which
    is how the batch and the context drifted apart before.

    Never returns less than 1: a single word must always be attempted even if
    the context is too small to hold it comfortably.
    """
    usable = context_size - reserve - margin
    if usable <= 0:
        return 1
    return max(1, min(cap, usable // tokens_per_word))


def compute_batch_timeout(
    batch_size: int,
    base: int = ENRICH_TIMEOUT_BASE_S,
    per_word: int = ENRICH_TIMEOUT_PER_WORD_S,
    floor: int = ENRICH_TIMEOUT_FLOOR_S,
) -> int:
    """Seconds to allow one enrichment batch of this size."""
    return max(floor, base + per_word * batch_size)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_enrich_prompt(
    words: list[str],
    def_lang: str = "english",
    ex_lang: str = "english",
    term_lang: str = "auto",
) -> str:
    """Build the user prompt asking for a definition and two examples per word."""
    word_list = "\n".join(f"- {w}" for w in words)
    lang_hint = (
        f"The word is in {term_lang}. "
        if term_lang and term_lang.lower() != "auto"
        else ""
    )
    return (
        f"{lang_hint}"
        f"For each word, give a definition in {def_lang} and 2 example sentences in {ex_lang}.\n"
        f"WORD: must contain the EXACT original word, unchanged. Do NOT translate, correct, or modify it.\n"
        f"No asterisks, no bold, no markdown. Plain text only.\n\n"
        # Without this the model writes {ex_lang} sentences with the source
        # word left inside them, which reads as a half-finished translation
        # ("The assaillant force attacked") rather than an example a learner can
        # use. The English sentence has to stand on its own.
        f"Each {ex_lang} sentence must read naturally in {ex_lang} and must NOT "
        f"contain the original word or a form of it. Do not mix the two languages.\n\n"
        f"If unrecognizable, write DEF: UNKNOWN and skip examples.\n\n"
        f"Format (labels must be in English):\n"
        f"WORD: <original word, unchanged>\n"
        f"DEF: <{def_lang} definition>\n"
        f"EX1: <{ex_lang} sentence>\n"
        f"EX2: <{ex_lang} sentence>\n\n"
        f"Example — word: 'amigo' (spanish), def_lang={def_lang}, ex_lang={ex_lang}:\n"
        f"WORD: amigo\n"
        f"DEF: A friend; a person with whom one has a bond of mutual affection.\n"
        f"EX1: I met a new friend at school today.\n"
        f"EX2: She is my best friend and we always help each other.\n\n"
        f"Words:\n{word_list}"
    )


def build_enrich_system(
    def_lang: str = "english",
    ex_lang: str = "english",
    term_lang: str = "auto",
) -> str:
    """Build a system prompt reinforcing the language and formatting rules."""
    lang_hint = (
        f"The words are in {term_lang}. "
        if term_lang and term_lang.lower() != "auto"
        else ""
    )
    return (
        f"You are a dictionary. {lang_hint}"
        f"For each word, output its definition in "
        f"{def_lang} and two example sentences in {ex_lang}. "
        f"The WORD: line must contain the EXACT original word, unchanged. "
        f"Do NOT translate, correct, or replace the word itself. "
        f"Write the example sentences entirely in {ex_lang}, without the "
        f"original word appearing in them. "
        f"No markdown, no asterisks. "
        f"Use labels: WORD:, DEF:, EX1:, EX2:."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Regex patterns for multilingual format markers the LLM might produce
# when asked for definitions in non-English languages.
_WORD_SPLIT_RE = re.compile(
    r"(?m)^(?:WORD|WORT|MOT|PALABRA|PALAVRA|PAROLA|СЛОВО|単語|단어):\s*",
    re.IGNORECASE,
)
_DEF_RE = re.compile(
    r"^(?:DEF|DEFINITION|BEDEUTUNG|DÉFINITION|DEFINICIÓN|DEFINIÇÃO|DEFINIZIONE):\s*",
    re.IGNORECASE,
)
_EX1_RE = re.compile(
    r"^(?:EX1|BEISPIEL\s*1|EXEMPLE\s*1|EJEMPLO\s*1|EXEMPLO\s*1|ESEMPIO\s*1):\s*",
    re.IGNORECASE,
)
_EX2_RE = re.compile(
    r"^(?:EX2|BEISPIEL\s*2|EXEMPLE\s*2|EJEMPLO\s*2|EXEMPLO\s*2|ESEMPIO\s*2):\s*",
    re.IGNORECASE,
)
_CORRECTED_RE = re.compile(
    r"^(?:CORRECTED|KORRIGIERT|CORRIGÉ|CORREGIDO|CORRIGIDO|CORRETTO):\s*",
    re.IGNORECASE,
)


def _normalize_word(text: str) -> str:
    """Lowercase, strip accents and surrounding punctuation, for comparison."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip(" \t.,;:!?\"'()[]{}·•-")


def _is_close_variant(candidate: str, expected: str) -> bool:
    """True when a block's own WORD: label plausibly refers to the expected word.

    Deliberately narrow. It accepts case and accent differences, which the
    model makes routinely, and simple plural or inflection endings. Anything
    else is treated as a mismatch, because the safe outcome for an unclear
    pairing is a miss the retry pass can pick up, not a definition attached to
    the wrong word.
    """
    a = _normalize_word(candidate)
    b = _normalize_word(expected)
    if not a or not b:
        return False
    if a == b:
        return True
    for suffix in ("s", "es", "n", "en"):
        if a == b + suffix or b == a + suffix:
            return True
    return False


_BULLET_RE = re.compile(r"^[\s*\-\u2022\u00b7#>]+")


def _clean_label(line: str) -> str:
    """Strip list decoration from a line standing in for a WORD: label."""
    return _BULLET_RE.sub("", line).strip().strip("*").strip()


def _split_enrich_entries(text: str) -> list[dict]:
    """Break a response into per-word entries.

    Line-oriented rather than marker-oriented, because the model does not
    reliably repeat the WORD: label. Asked for three words it will often emit:

        WORD: 探す
        DEF: ...
        EX1: ...
        - 静か
        DEF: ...
        扉
        DEF: ...

    Only the first entry carries the label. Splitting on WORD: swallowed
    everything after the first entry into it, and because the last DEF: in a
    block won, the first word was handed the last word's definition — a wrong
    card, with no warning. Entries are therefore started either by a WORD: line
    or by a bare line that turns out to precede a DEF:.
    """
    entries: list[dict] = []
    current: Optional[dict] = None
    pending_label: Optional[str] = None

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if _CORRECTED_RE.match(line):
            continue

        if _WORD_SPLIT_RE.match(line):
            current = {
                "label": _WORD_SPLIT_RE.sub("", line).strip(),
                "definition": "",
                "examples": "",
            }
            entries.append(current)
            pending_label = None
            continue

        if _DEF_RE.match(line):
            if current is None or pending_label is not None:
                current = {
                    "label": pending_label or "",
                    "definition": "",
                    "examples": "",
                }
                entries.append(current)
                pending_label = None
            current["definition"] = _DEF_RE.sub("", line).strip()
            continue

        if _EX1_RE.match(line):
            if current is not None:
                current["examples"] = _EX1_RE.sub("", line).strip()
            continue

        if _EX2_RE.match(line):
            if current is not None:
                ex2 = _EX2_RE.sub("", line).strip()
                if ex2:
                    current["examples"] = (
                        current["examples"] + "\n" + ex2
                        if current["examples"] else ex2
                    )
            continue

        # Not a label. Once the open entry has a definition this is almost
        # certainly the next word, missing its marker.
        if current is None or current["definition"]:
            pending_label = _clean_label(line)

    for entry in entries:
        for key in ("definition", "examples"):
            entry[key] = entry[key].replace("*", "")
    return entries


def parse_enrich_response(text: str, words: list[str]) -> list[dict]:
    """Parse a batched enrichment response into one dict per requested word."""
    entries = _split_enrich_entries(text)

    parsed: dict[str, dict] = {}
    parsed_ordered: list[dict] = []
    for entry in entries:
        key = entry.get("label", "").lower()
        if key and key not in parsed:
            parsed[key] = entry
        parsed_ordered.append(entry)

    # The positional fallback may only be trusted when the block count matches
    # the word count. One skipped or merged word shifts every later block.
    counts_match = len(parsed_ordered) == len(words)

    results = []
    for idx, w in enumerate(words):
        entry = parsed.get(w.lower())
        corrected = ""
        if entry is None and counts_match and idx < len(parsed_ordered):
            candidate = parsed_ordered[idx]
            if _is_close_variant(candidate.get("label", ""), w):
                entry = candidate
                corrected = candidate.get("label", "")
        if entry is None:
            entry = {}

        defn = entry.get("definition", "")
        examples = entry.get("examples", "")

        # The marker is the whole definition, per the prompt's instruction to
        # "write DEF: UNKNOWN" when a word is unrecognisable. It used to be a
        # substring test, so any definition containing the word "unknown" was
        # thrown away as a miss: "To search for something that is lost or
        # unknown." was discarded, and the word was retried and then reported
        # missing despite the model having answered correctly.
        defn_stripped = defn.strip().strip(".!?").strip()
        if defn_stripped.upper() == "UNKNOWN":
            defn = ""
            examples = ""
            warning = "not_found"
        elif defn and not defn.rstrip().endswith(('.', '!', '?', '"', ')')):
            warning = "truncated"
        elif not defn:
            warning = "not_found"
        else:
            warning = ""

        if not warning and len(w) > 2 and examples:
            if re.search(r'\b' + re.escape(w.lower()) + r'\w{0,2}\b', examples.lower()):
                warning = "untranslated"

        results.append({
            "word": w,
            "definition": defn,
            "examples": examples,
            "warning": warning,
            "corrected_word": corrected,
        })
    return results


# ---------------------------------------------------------------------------
# Word sanitising
# ---------------------------------------------------------------------------


# Separators that appear inside a line of transcribed text. A photographed
# vocabulary list often has several entries per line, and the vision model
# transcribes it as it sees it, so a line is not the same thing as a word.
_WORD_SEPARATORS_RE = re.compile(r"[,\u3001;\uff1b|/\t]+")


def split_ocr_words(text: str) -> list[str]:
    """Split transcribed text into individual vocabulary words.

    Splitting on newlines alone is not enough. The vision model is asked for
    one word per line but transcribes what it sees, so a page whose entries are
    comma-separated comes back comma-separated and each line becomes a single
    flashcard holding several words.
    """
    words: list[str] = []
    for line in (text or "").splitlines():
        for part in _WORD_SEPARATORS_RE.split(line):
            part = part.strip().strip("-\u2022\u00b7*#>").strip()
            if part:
                words.append(part)
    return words


def sanitize_words(raw: list[str]) -> list[str]:
    """Clean OCR artefacts from a word list before enrichment."""
    cleaned: list[str] = []
    for w in raw:
        # Strip markdown bullets, numbering, leading punctuation
        w = re.sub(r'^[\s*\-•·#>]+', '', w)
        w = re.sub(r'^\d+\.\s*', '', w)
        w = w.strip()
        if len(w) < 2:
            continue
        # Skip LLM commentary: sentences with >4 words
        if len(w.split()) > 4:
            continue
        cleaned.append(w)
    return cleaned