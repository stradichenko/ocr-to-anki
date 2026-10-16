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


def parse_enrich_response(text: str, words: list[str]) -> list[dict]:
    """Parse a batched enrichment response into one dict per requested word."""
    results = []
    # Split by WORD: markers (supports translated labels)
    blocks = _WORD_SPLIT_RE.split(text)
    parsed: dict[str, dict] = {}
    parsed_ordered: list[dict] = []  # positional fallback, keeps its own label
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        word_key = lines[0].strip()
        body = lines[1] if len(lines) > 1 else ""
        defn = ""
        examples = ""
        for line in body.split("\n"):
            line = line.strip()
            if _CORRECTED_RE.match(line):
                pass  # Ignore any CORRECTED: the model emits
            elif _DEF_RE.match(line):
                defn = _DEF_RE.sub("", line).strip()
            elif _EX1_RE.match(line):
                examples = _EX1_RE.sub("", line).strip()
            elif _EX2_RE.match(line):
                ex2 = _EX2_RE.sub("", line).strip()
                if ex2:
                    examples += "\n" + ex2
        data = {"label": word_key, "definition": defn, "examples": examples}
        # Strip markdown-style asterisks the LLM may insert despite instructions.
        for key in ("definition", "examples"):
            data[key] = data[key].replace("*", "")
        key = word_key.lower()
        if key in parsed:
            # The model repeated a word. Keep the first block; a repeat is a
            # sign the response drifted, and silently overwriting would decide
            # which definition is correct with no evidence either way.
            log.debug("Duplicate WORD label %r in response; keeping first", word_key)
        else:
            parsed[key] = data
        parsed_ordered.append(data)

    # The positional fallback may only be trusted when the block count matches
    # the word count. One skipped or merged word shifts every later block.
    counts_match = len(parsed_ordered) == len(words)

    # Match back to original word list.
    for idx, w in enumerate(words):
        # 1. Exact case-insensitive match on the block's own label.
        entry = parsed.get(w.lower())
        corrected = ""
        # 2. Positional fallback, only when the counts line up AND the block at
        #    this position names a plausible variant of the expected word.
        #    Attaching a neighbour's definition would produce a card that looks
        #    right and is silently wrong.
        if entry is None and counts_match and idx < len(parsed_ordered):
            candidate = parsed_ordered[idx]
            if _is_close_variant(candidate.get("label", ""), w):
                entry = candidate
                corrected = candidate.get("label", "")
        if entry is None:
            entry = {}
        defn = entry.get("definition", "")
        examples = entry.get("examples", "")

        # Detect UNKNOWN marker from LLM
        if (
            defn.strip().upper() == "UNKNOWN"
            or defn.strip().upper().startswith("UNKNOWN")
            or "UNKNOWN" in defn.upper()
        ):
            defn = ""
            examples = ""
            warning = "not_found"
        # Detect truncated definitions (no terminal punctuation)
        elif defn and not defn.rstrip().endswith(('.', '!', '?', '"', ')')):
            warning = "truncated"
        elif not defn:
            warning = "not_found"
        else:
            warning = ""

        # Detect untranslated: original word appears verbatim in examples.
        # Skip very short words (<=2 chars) to avoid false positives.
        if not warning and len(w) > 2 and examples:
            w_lower = w.lower()
            ex_lower = examples.lower()
            # Check if the original word (or a close variant) leaked into examples.
            if re.search(r'\b' + re.escape(w_lower) + r'\w{0,2}\b', ex_lower):
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