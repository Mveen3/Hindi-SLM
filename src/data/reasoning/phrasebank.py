"""
Phrase Bank Construction (Groq / gpt-oss-120b)
==============================================

Builds the *language surface* of the Phase 3 reasoning corpus: entity name
pools, per-attribute vocabulary, and many grammatical variants of every
template slot declared in ``tasks.py``.

Division of labour
------------------
The LLM writes **wording only**.  It never sees a numeric value, never decides
an ordering, and never produces a label — ``generator.py`` samples the values
and derives every answer in Python.  So a hallucination can at worst give us an
awkward sentence (which validation usually catches), never a wrong label.  That
keeps the corpus "synthetic with known ground truth" as the brief requires,
while the phrasing stays natural Hindi / Nepali rather than translated English.

Why templates instead of per-example generation
-----------------------------------------------
The free Groq tier allows 30 requests/min, 8k tokens/min and 200k tokens/day.
Asking the model for 40,000 individual examples is impossible under that budget
and would also hand label correctness to the model.  Asking it for ~200
reusable phrasings costs roughly 14 requests per language, and the combinatorics
of (phrasing x entity x attribute x value x task) then yield an arbitrarily
large corpus offline.  The result is cached in
``<language>/data/reasoning/phrasebank.json``, so the API is touched once.

Validation
----------
Every returned template must satisfy its slot's placeholder contract, contain
no Latin script, contain Devanagari, and render without error against a probe
context.  Anything else is discarded and reported.  The hand-authored seeds in
``seeds.py`` are always merged in first, so the pipeline still produces a
complete corpus with ``--offline`` or a dead API key.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.languages import Language

from . import tasks
from .seeds import SEED_PHRASEBANKS

#: Groq's OpenAI-compatible model id, per the console screenshot.
DEFAULT_MODEL = "openai/gpt-oss-120b"

#: Free-tier limits are 30 req/min and 8K tok/min. Kept well under both — not
#: skimmed close — because the sliding window here and Groq's own bucket are
#: two independent clocks that can disagree by a second or two at the edges.
DEFAULT_RPM = 20
DEFAULT_TPM = 6500

PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
LATIN_RE = re.compile(r"[A-Za-z]")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window limiter for requests-per-minute and tokens-per-minute.

    Groq enforces both, and a 429 on a half-built phrase bank is expensive, so
    we block locally *before* sending, rather than reacting to rejections —
    this is what keeps a normal run from ever seeing the per-minute limit at
    all. Two things beyond the plain sliding window earn their keep:

    - ``settle()`` corrects a reservation down to what a request actually
      cost once Groq reports real usage, so paying for the full
      ``max_tokens`` budget up front doesn't permanently under-use the window.
    - a minimum gap between requests (``60 / requests_per_minute`` seconds)
      spaces requests out evenly instead of letting several land in the same
      instant just because the token budget briefly allowed it — bursts are
      exactly what tips two independent clocks (ours and Groq's) out of sync
      near a window boundary.
    """

    def __init__(self, requests_per_minute: int = DEFAULT_RPM,
                 tokens_per_minute: int = DEFAULT_TPM):
        self.rpm = max(1, requests_per_minute)
        self.tpm = max(500, tokens_per_minute)
        self._min_interval = 60.0 / self.rpm
        self._events: list[tuple[float, int]] = []   # (timestamp, tokens)
        self._last_sent: float | None = None

    def _prune(self, now: float) -> None:
        self._events = [e for e in self._events if now - e[0] < 60.0]

    def acquire(self, estimated_tokens: int) -> None:
        """Block until sending a request of this size stays inside both limits.

        Args:
            estimated_tokens: Prompt plus maximum completion tokens for the
                request about to be sent.  A single request whose estimate
                exceeds the per-minute token budget is charged the whole budget
                rather than being refused, otherwise it could never be sent.
        """
        charge = min(max(1, estimated_tokens), self.tpm)
        while True:
            now = time.monotonic()

            # Even pacing floor: never send two requests closer together than
            # 60/rpm seconds, regardless of how much token budget is free.
            if self._last_sent is not None:
                since_last = now - self._last_sent
                if since_last < self._min_interval:
                    time.sleep(self._min_interval - since_last)
                    continue

            self._prune(now)
            used_requests = len(self._events)
            used_tokens = sum(t for _, t in self._events)

            # With an empty window the request always proceeds: `charge` is
            # capped at the budget, so there is nothing left to wait for.
            if not self._events or (used_requests < self.rpm
                                    and used_tokens + charge <= self.tpm):
                self._events.append((now, charge))
                self._last_sent = now
                return

            oldest = min(event[0] for event in self._events)
            wait = max(0.6, 60.0 - (now - oldest) + 0.4)
            print(f"    [rate-limit] pacing locally — waiting {wait:.0f}s "
                  f"({used_requests}/{self.rpm} req, {used_tokens}/{self.tpm} tok in window) "
                  f"so we never actually reach Groq's limit ...")
            time.sleep(wait)

    def saturate(self) -> None:
        """Treat the current window as fully spent.

        Called after Groq itself returns a 429: our local estimate must have
        under-shot real usage (a shared key, a retried request that still
        billed, clock drift between the two accounting windows), so the safe
        correction is to assume no budget is left until the window rolls over,
        rather than risk immediately tripping the wall a second time.
        """
        now = time.monotonic()
        self._events = [(now, self.tpm)]
        self._last_sent = now

    def settle(self, actual_tokens: int) -> None:
        """Correct the last reservation to what the request actually consumed.

        ``acquire`` has to reserve the full ``max_tokens`` budget before the
        request is sent, but completions typically use far less.  Replacing the
        estimate with the reported usage stops the limiter throttling on tokens
        that were never spent, which roughly halves a full authoring run.
        """
        if not self._events or actual_tokens <= 0:
            return
        timestamp, _ = self._events[-1]
        self._events[-1] = (timestamp, min(actual_tokens, self.tpm))


# ---------------------------------------------------------------------------
# Prompt construction — the authoring contract handed to gpt-oss-120b
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a native-speaker computational linguist building a synthetic \
reasoning dataset for a {display} language model. You write {display} exactly as it is \
written by {display} speakers on the web and in textbooks — never translated-sounding \
{display}, never transliteration, never English.

You are given SENTENCE TEMPLATES to author. A template is one {display} sentence containing \
placeholders written in curly braces, which a Python program later fills in.

ABSOLUTE RULES
1. Reply with a single valid JSON object and nothing else. No markdown fences, no commentary.
2. Copy every placeholder EXACTLY, including the curly braces and the exact spelling and case, \
e.g. {{A}} stays {{A}}. Never translate, rename, re-case or drop a placeholder. Never invent a \
placeholder that is not on the allowed list for that slot.
3. Use ONLY the {script} script plus ordinary punctuation. Not one Latin letter may appear \
outside a placeholder. Do not write any digits — numbers arrive through placeholders.
4. Each template must be grammatical for EVERY possible substitution: any personal name of any \
gender, any city, object, animal or fruit name, and any number. Therefore anchor comparisons on \
the attribute ("X's height is more than Y's height") rather than on the person ("X is taller"), \
and never use a gendered adjective or verb form that agrees with the entity.
5. {{cmp}}, {{cmp_from}} and {{cmp_to}} are filled with the word for "more" on some examples and \
the word for "less" on others. A template containing one of them must therefore read correctly \
BOTH WAYS: it must not contain any comparative word of its own, and must not commit to a \
direction anywhere else in the sentence. Writing "...is {{cmp}}, while B's is less" is wrong, \
because it contradicts itself whenever {{cmp}} is "less".
6. Never write an attribute noun (age, height, price, ...) or a measurement unit as literal text. \
They arrive only through {{noun}} and {{unit}}, because one template is reused for every attribute.
7. Only {{poss}}, {{poss_obl}}, {{whose}}, {{howmuch}} and {{copula}} may carry agreement with the \
attribute noun. Do not add a relative pronoun, demonstrative or adjective of your own that would \
have to inflect for that noun's gender — such a word is correct for some attributes and wrong for \
others.
8. Statements end with the full stop used in {script} ("।"). Questions end with "?".
9. Keep every template short and natural: at most about 25 words.
10. VARY the phrasing across the templates of one slot — different openings, different \
constructions, different politeness levels, different word order. Do not merely reorder one \
sentence. Every template of a slot must be distinct.
11. Templates must not leak the answer, must not add facts that were not given, and must not \
express uncertainty ("about", "roughly").

Correctness of the reasoning is handled by the program. Your only job is natural, reusable, \
grammatical {display} wording."""


def _slot_block(slot: tasks.SlotSpec, seed_examples: list[str], n: int) -> str:
    """Render one slot's authoring spec (contract + worked examples) for the prompt."""
    examples = "\n".join(f"      - {t}" for t in seed_examples[:3])
    return (
        f'  "{slot.name}": {n} templates.\n'
        f"      Meaning: {slot.instruction}\n"
        f"      Placeholders that MUST appear: {', '.join(slot.required)}\n"
        f"      Placeholders that MAY appear: {', '.join(slot.allowed)}\n"
        f"      Existing examples of this slot (match their style, do not copy them):\n"
        f"{examples}\n"
    )


def _templates_user_prompt(display: str, script: str, slots: list[tasks.SlotSpec],
                           seed_templates: dict, n_per_slot: int) -> str:
    """Build the user message asking for several slots' templates at once."""
    blocks = "".join(_slot_block(s, seed_templates.get(s.name, []), n_per_slot) for s in slots)
    keys = ", ".join(f'"{s.name}"' for s in slots)
    return (
        f"Author new {display} sentence templates for the slots below.\n\n"
        f"{blocks}\n"
        f"Placeholder meanings (identical across slots):\n"
        f"  {{E}}, {{A}}, {{B}}, {{C}} = entity names (person, city, object, animal or fruit)\n"
        f"  {{noun}} = the attribute noun (age, height, price, ...)\n"
        f"  {{poss}} = possessive particle joining an entity name to {{noun}}\n"
        f"  {{poss_obl}} = the same possessive in the oblique case, for the entity that is the\n"
        f"      object of the comparison (the one followed by the 'than' postposition)\n"
        f"  {{whose}} = interrogative possessive, 'whose'\n"
        f"  {{howmuch}} = interrogative quantity word, 'how much'\n"
        f"  {{unit}} = measurement unit\n"
        f"  {{copula}} = the 'is/are' verb\n"
        f"  {{cmp}}, {{cmp_from}}, {{cmp_to}} = a comparative word, 'more' or 'less'\n"
        f"  {{v}}, {{vA}}, {{vB}}, {{d}} = numbers\n"
        f"  {{ordered}} = an already comma-joined list of entity names\n"
        f"  {{answer}} = the final answer string\n\n"
        f"Return exactly this JSON shape, with the {n_per_slot} templates as plain strings:\n"
        f"{{{keys}: [...]}}"
    )


def _entities_user_prompt(display: str, script: str, per_pool: int,
                          seed_entities: dict) -> str:
    """Build the user message asking for entity name pools."""
    lines = []
    descriptions = {
        "people": f"common {display} personal first names, a mix of genders",
        "cities": f"well-known cities or towns where {display} is spoken",
        "objects": "everyday household or school objects",
        "animals": "common animals",
        "fruits": "common fruits",
    }
    for pool in tasks.ENTITY_POOLS:
        sample = ", ".join(seed_entities.get(pool, [])[:6])
        lines.append(f'  "{pool}": {per_pool} {descriptions[pool]} (like: {sample})')
    body = "\n".join(lines)
    return (
        f"List entity names for a {display} reasoning dataset.\n\n{body}\n\n"
        f"Rules: {script} script only; single words or short two-word names; no titles, no "
        f"honorifics, no numbers, no duplicates, and none of the example names repeated. "
        f"Every name must be a name real {display} text would use.\n\n"
        f"Return exactly this JSON shape:\n"
        f"{{{', '.join(chr(34) + p + chr(34) for p in tasks.ENTITY_POOLS)}}}"
    )


def _attributes_user_prompt(display: str, script: str, seed_attributes: dict) -> str:
    """Build the user message asking for the per-attribute word set."""
    fields = "\n".join(f"    \"{name}\": {desc}"
                       for name, desc in tasks.ATTRIBUTE_WORD_FIELDS.items())
    attrs = "\n".join(f'  "{a.key}": {a.gloss}' for a in tasks.ATTRIBUTES)
    example_key = tasks.ATTRIBUTES[0].key
    example = json.dumps({example_key: seed_attributes.get(example_key, {})},
                         ensure_ascii=False)
    return (
        f"Give the {display} vocabulary for each measurable attribute below.\n\n"
        f"Attributes:\n{attrs}\n\n"
        f"For each attribute return these fields:\n{fields}\n\n"
        f"Rules: {script} script only. 'poss', 'whose', 'howmuch' and 'copula' must agree "
        f"grammatically with that attribute's own noun. 'more' and 'less' must be invariant "
        f"words that do not inflect for gender or number, so that the same template works with "
        f"any entity name. 'unit' must be the unit a {display} speaker would actually write.\n\n"
        f"Return a JSON object keyed by the attribute ids above, shaped like:\n{example}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

#: A context that fills every placeholder, used to prove a template renders.
PROBE_CONTEXT = {
    "E": "क", "A": "क", "B": "ख", "C": "ग",
    "noun": "न", "poss": "प", "poss_obl": "प", "whose": "क", "howmuch": "क",
    "unit": "उ", "copula": "छ",
    "cmp": "अ", "cmp_from": "अ", "cmp_to": "क",
    "v": "१", "vA": "१", "vB": "२", "d": "३",
    "ordered": "क, ख", "answer": "क",
}


CMP_PLACEHOLDERS = ("{cmp}", "{cmp_from}", "{cmp_to}")

#: Characters that delimit a word for the purpose of lexicon-leak detection.
_TOKEN_SPLIT_RE = re.compile(r"[\s।?,:=%()\-]+")

#: "Smart typography" an LLM habitually reaches for that looks identical to
#: plain ASCII punctuation on screen, but is a different Unicode code point
#: that a Devanagari corpus's tokenizer was never trained on — so it silently
#: becomes <unk> instead of a normal, expected character. Caught by name
#: rather than by code point so the intent (and how it was found) stays clear:
#: a real run authored "कौन‑सा" ("which") with a NON-BREAKING HYPHEN in place
#: of a plain one, and it slipped past every earlier check.
_SMART_TYPOGRAPHY_RE = re.compile(
    "[" + "".join([
        "‑",           # non-breaking hyphen
        "‒–—―",   # figure/en/em dash, horizontal bar
        "‘’‚‛",   # curly single quotes
        "“”„‟",   # curly double quotes
        "…",           # ellipsis
        "  ",     # non-breaking / narrow no-break space
    ]) + "]"
)


def _renders_cleanly(text: str, tokenizer, context: dict | None = None) -> tuple[bool, str]:
    """Does ``text`` tokenize with zero ``<unk>`` under the frozen tokenizer?

    This is the ground-truth defense against exactly the failure the regex
    checks above cannot anticipate: any character the model reaches for that
    was never in the language's training corpus, whatever it turns out to be.
    ``_SMART_TYPOGRAPHY_RE`` catches the common, known offenders even when no
    tokenizer is available yet; this catches everything else, because it asks
    the actual tokenizer instead of guessing at a character list.

    Args:
        text:      The literal template text (or an entity/attribute word).
        tokenizer: A loaded ``tokenizers.Tokenizer``, or ``None`` to skip.
        context:   Placeholder values to render through first, if any.

    Returns:
        ``(True, "")`` if clean (or no tokenizer was given), else
        ``(False, reason)`` naming the offending character.
    """
    if tokenizer is None:
        return True, ""
    rendered = text.format(**context) if context else text
    tokens = tokenizer.encode(rendered).tokens
    if "<unk>" not in tokens:
        return True, ""
    # Bisect down to the actual bad character so the reason is actionable.
    for ch in rendered:
        if "<unk>" in tokenizer.encode(ch).tokens:
            return False, f"produces <unk> under the frozen tokenizer (character {ch!r})"
    return False, "produces <unk> under the frozen tokenizer"


def lexicon_bans(bank: dict) -> dict:
    """Collect the words that must reach a template through a placeholder only.

    Args:
        bank: A phrase bank, whose ``attributes`` block supplies the words.

    Returns:
        ``{"comparatives": {...}, "nouns_units": {...}, "copulas": {...}}``.
        Attribute nouns and units are banned outright, since one template is
        reused for all nine attributes.  Comparatives are banned only where a
        ``{cmp}`` placeholder is present, because a second hard-coded direction
        contradicts the premise the generator meant to state.  Copulas are
        flagged only where a copula sits directly beside ``{copula}``, which
        would render as a doubled verb.
    """
    comparatives, nouns_units, copulas = set(), set(), set()
    for words in (bank.get("attributes") or {}).values():
        for field_name in ("more", "less"):
            if words.get(field_name):
                comparatives.add(words[field_name])
        for field_name in ("noun", "unit"):
            if words.get(field_name):
                nouns_units.add(words[field_name])
        if words.get("copula"):
            copulas.add(words["copula"])
    return {"comparatives": comparatives, "nouns_units": nouns_units, "copulas": copulas}


def _doubles_copula(text: str, copula: str) -> bool:
    """Is a copula written immediately beside the ``{copula}`` placeholder?

    Only adjacency is a fault.  A copula elsewhere in the sentence is usually a
    different clause ("it is known that ... {copula}"), which is fine; a copula
    touching the placeholder renders as a doubled verb ("है है").
    """
    escaped = re.escape(copula)
    return bool(re.search(rf"\{{copula\}}\s*{escaped}(?![^\s।?,])", text)
                or re.search(rf"(?<![^\s।?,]){escaped}\s*\{{copula\}}", text))


def _states_literally(bare: str, word: str) -> bool:
    """Is ``word`` present in ``bare`` as a whole word (or a multi-word phrase)?"""
    if " " in word:
        return word in bare
    return word in set(_TOKEN_SPLIT_RE.split(bare))


def placeholders_of(text: str) -> set[str]:
    """Return the set of ``{placeholder}`` tokens appearing in a template."""
    return set(PLACEHOLDER_RE.findall(text))


def validate_template(slot: tasks.SlotSpec, text: str,
                      bans: dict | None = None, tokenizer=None) -> tuple[bool, str]:
    """Check one template against its slot contract and the lexicon rules.

    Args:
        slot:      The slot the template is meant to fill.
        text:      The candidate template string.
        bans:      Optional output of :func:`lexicon_bans`.  Supplying it
                   rejects templates that state an attribute noun, a unit, or
                   a second comparative direction as literal text — mistakes
                   that satisfy the placeholder contract but silently corrupt
                   the premise.
        tokenizer: Optional loaded tokenizer for this language.  Supplying it
                   rejects templates whose literal text would tokenize to
                   ``<unk>`` — the ground-truth check behind the smart-typography
                   regex below, for whatever that regex did not anticipate.

    Returns:
        ``(True, "")`` if the template is usable, else ``(False, reason)``.
    """
    if not isinstance(text, str):
        return False, "not a string"

    text = text.strip()
    if not (4 <= len(text) <= 400):
        return False, "implausible length"

    found = placeholders_of(text)
    allowed, required = set(slot.allowed), set(slot.required)
    if not required <= found:
        return False, f"missing required {sorted(required - found)}"
    if not found <= allowed:
        return False, f"unexpected placeholder {sorted(found - allowed)}"

    # Braces must all belong to recognised placeholders, or str.format will fail.
    if text.count("{") != len(PLACEHOLDER_RE.findall(text)) or text.count("{") != text.count("}"):
        return False, "malformed braces"

    bare = PLACEHOLDER_RE.sub("", text)
    if LATIN_RE.search(bare):
        return False, "contains Latin script outside placeholders"
    if not DEVANAGARI_RE.search(bare):
        return False, "contains no Devanagari"
    if re.search(r"\d", bare) or re.search(r"[०-९]", bare):
        return False, "contains a hard-coded digit"
    smart_char = _SMART_TYPOGRAPHY_RE.search(bare)
    if smart_char:
        return False, f"contains smart typography {smart_char.group()!r} instead of plain punctuation"

    if bans:
        for word in bans["nouns_units"]:
            if _states_literally(bare, word):
                return False, f"states the attribute word {word!r} literally"
        if any(ph in text for ph in CMP_PLACEHOLDERS):
            for word in bans["comparatives"]:
                if _states_literally(bare, word):
                    return False, (f"hard-codes the comparative {word!r} alongside a "
                                   f"{{cmp}} placeholder")
        if "{copula}" in text:
            for word in bans.get("copulas", ()):
                if _doubles_copula(text, word):
                    return False, f"doubles the copula: {word!r} sits next to {{copula}}"

    try:
        rendered = text.format(**PROBE_CONTEXT)
    except (KeyError, IndexError, ValueError) as exc:
        return False, f"does not render ({exc})"
    if not rendered.strip():
        return False, "renders empty"

    clean, reason = _renders_cleanly(text, tokenizer, PROBE_CONTEXT)
    if not clean:
        return False, reason

    return True, ""


def _clean_name(name, tokenizer=None) -> str | None:
    """Normalise one LLM-proposed entity name, or return None if unusable.

    Args:
        name:      The raw candidate from the model's reply.
        tokenizer: Optional loaded tokenizer — rejects a name that would
                   tokenize to ``<unk>`` (see :func:`_renders_cleanly`).
    """
    if not isinstance(name, str):
        return None
    name = " ".join(name.strip().split())
    if not (1 <= len(name) <= 30):
        return None
    if LATIN_RE.search(name) or re.search(r"[\d०-९]", name):
        return None
    if not DEVANAGARI_RE.search(name):
        return None
    if any(ch in name for ch in "{}[]()\"'|/\\,"):
        return None
    if _SMART_TYPOGRAPHY_RE.search(name):
        return None
    if not _renders_cleanly(name, tokenizer)[0]:
        return None
    return name


def _clean_word(word, tokenizer=None) -> str | None:
    """Normalise one LLM-proposed attribute word, or return None if unusable.

    Args:
        word:      The raw candidate from the model's reply.
        tokenizer: Optional loaded tokenizer — rejects a word that would
                   tokenize to ``<unk>`` (see :func:`_renders_cleanly`).
    """
    if not isinstance(word, str):
        return None
    word = " ".join(word.strip().split())
    if not (1 <= len(word) <= 40):
        return None
    if LATIN_RE.search(word) or re.search(r"[\d०-९]", word):
        return None
    if not DEVANAGARI_RE.search(word):
        return None
    if _SMART_TYPOGRAPHY_RE.search(word):
        return None
    if not _renders_cleanly(word, tokenizer)[0]:
        return None
    return word


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

@dataclass
class LLMSettings:
    """Everything about how we talk to Groq, read from ``reasoning_config.yaml``."""

    model: str = DEFAULT_MODEL
    templates_per_slot: int = 8
    entities_per_pool: int = 30
    slots_per_request: int = 3
    temperature: float = 0.9
    max_output_tokens: int = 1600
    reasoning_effort: str = "low"
    requests_per_minute: int = DEFAULT_RPM
    tokens_per_minute: int = DEFAULT_TPM
    max_retries: int = 4
    #: How many times in a row a per-minute 429 is waited out before giving up.
    #: Each wait is fully recoverable (the window always rolls over), so this
    #: is generous; it only protects against looping forever on something else
    #: masquerading as a rate limit.
    rate_limit_max_waits: int = 10

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMSettings":
        """Build settings from the ``llm:`` block of a reasoning config."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (cfg or {}).items() if k in known})


class QuotaExhausted(Exception):
    """Raised when Groq's rate limiting cannot be waited out inside this run.

    Attributes:
        scope: ``"day"`` if Groq reported an account-wide daily quota (which
               no amount of local waiting can fix this session), or
               ``"minute"`` if the per-minute limit kept rejecting requests
               even after repeatedly waiting the server-reported time.
    """

    def __init__(self, message: str, scope: str):
        super().__init__(message)
        self.scope = scope


#: Groq's 429 error text names the exhausted window explicitly, e.g.
#: "Rate limit reached ... on requests per day (RPD)" vs "... per minute (RPM)".
#: Matching that text is more reliable than headers, which Groq does not
#: always send a day-scoped equivalent of.
_DAY_LIMIT_RE = re.compile(r"per day|requests per day|tokens per day|\bRPD\b|\bTPD\b|daily",
                           re.IGNORECASE)
_MINUTE_LIMIT_RE = re.compile(r"per minute|\bRPM\b|\bTPM\b", re.IGNORECASE)
#: Groq phrases the retry hint as "Please try again in 1h2m3.4s" / "12.5s" / "3m".
_WAIT_HINT_RE = re.compile(
    r"try again in\s+(?:(?P<h>\d+)h)?\s*(?:(?P<m>\d+)m)?\s*(?:(?P<s>[\d.]+)s)?",
    re.IGNORECASE,
)


def _parse_wait_hint(message: str) -> float | None:
    """Extract a "try again in ..." duration from a Groq error message."""
    match = _WAIT_HINT_RE.search(message or "")
    if not match or not any(match.group(g) for g in ("h", "m", "s")):
        return None
    hours, minutes, seconds = match.group("h"), match.group("m"), match.group("s")
    return (int(hours or 0) * 3600) + (int(minutes or 0) * 60) + float(seconds or 0)


def _diagnose_rate_limit(exc: Exception) -> tuple[str, float | None, str]:
    """Classify a Groq 429 as minute- or day-scoped, and find how long to wait.

    Args:
        exc: The caught ``groq.RateLimitError``.

    Returns:
        ``(scope, wait_seconds, message)`` — ``scope`` is ``"day"``,
        ``"minute"`` or ``"unknown"``; ``wait_seconds`` is ``None`` when no
        duration could be determined (the caller then falls back to a fixed
        wait, since only day-scoped limits skip waiting altogether).
    """
    message = str(getattr(exc, "message", None) or exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = f"{message} {json.dumps(body, ensure_ascii=False)}"

    wait_seconds = None
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                wait_seconds = float(retry_after)
            except ValueError:
                wait_seconds = None
    if wait_seconds is None:
        wait_seconds = _parse_wait_hint(message)

    if _DAY_LIMIT_RE.search(message):
        scope = "day"
    elif _MINUTE_LIMIT_RE.search(message):
        scope = "minute"
    else:
        scope = "unknown"
    return scope, wait_seconds, message.strip()


class GroqPhraseAuthor:
    """Thin, rate-limited JSON client for Groq chat completions.

    Args:
        settings: Model and throttling settings.
        api_key:  Groq key.  Defaults to ``GROQ_API_KEY`` / ``GROK_API_KEY``.

    Raises:
        SystemExit: If the ``groq`` package or an API key is unavailable.
    """

    def __init__(self, settings: LLMSettings, api_key: str | None = None):
        try:
            from groq import Groq, RateLimitError
        except ImportError:
            raise SystemExit(
                "The 'groq' package is required for online phrase-bank authoring.\n"
                "  pip install groq       (or run with --offline to use the seed phrase bank)"
            )

        key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
        if not key:
            raise SystemExit(
                "No Groq API key found. Put GROK_API_KEY=... in .env, "
                "or run with --offline to use the seed phrase bank."
            )

        self.settings = settings
        self._client = Groq(api_key=key)
        self._RateLimitError = RateLimitError
        self._limiter = RateLimiter(settings.requests_per_minute, settings.tokens_per_minute)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- low level ----------------------------------------------------------

    def _create(self, system: str, user: str, max_tokens: int, drop_extras: bool = False):
        """Issue one chat completion, degrading gracefully on unsupported kwargs."""
        kwargs = dict(
            model=self.settings.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.settings.temperature,
            max_tokens=max_tokens,
        )
        if not drop_extras:
            kwargs["response_format"] = {"type": "json_object"}
            if self.settings.reasoning_effort:
                kwargs["reasoning_effort"] = self.settings.reasoning_effort
        return self._client.chat.completions.create(**kwargs)

    def chat_json(self, system: str, user: str, max_tokens: int | None = None) -> dict:
        """Send one prompt and parse the reply as a JSON object.

        Three kinds of failure are handled differently, because only one of
        them is actually a bug:

        - A **per-minute** 429 is normal and fully recoverable — Groq tells us
          how long the window has left, we sleep exactly that long, and retry.
          This does not consume any of ``max_retries``, since nothing went
          wrong; it only counts against the much larger ``rate_limit_max_waits``
          safety cap, so a real outage still eventually gives up instead of
          looping forever.
        - A **per-day** 429 cannot be waited out inside this process, so it is
          raised immediately as :class:`QuotaExhausted` for the caller to
          checkpoint progress and stop authoring gracefully.
        - Everything else (connection errors, 5xx, a non-JSON reply) retries
          with exponential backoff against the ordinary ``max_retries`` budget.

        Returns:
            The parsed object, or ``{}`` if every ordinary-error attempt failed.

        Raises:
            QuotaExhausted: On a day-scoped 429, or after ``rate_limit_max_waits``
                consecutive minute-scoped 429s.
        """
        max_tokens = max_tokens or self.settings.max_output_tokens
        estimate = len(system + user) // 3 + max_tokens
        drop_extras = False
        rate_limit_waits = 0

        attempt = 0
        while attempt < self.settings.max_retries:
            self._limiter.acquire(estimate)
            try:
                response = self._create(system, user, max_tokens, drop_extras)
            except TypeError:
                drop_extras = True                      # SDK too old for a kwarg
                continue
            except self._RateLimitError as exc:
                scope, wait, message = _diagnose_rate_limit(exc)

                if scope == "day":
                    raise QuotaExhausted(
                        f"Groq's DAILY quota is exhausted for {self.settings.model} "
                        f"(server said: {message[:220]})",
                        scope="day",
                    ) from exc

                rate_limit_waits += 1
                if rate_limit_waits > self.settings.rate_limit_max_waits:
                    raise QuotaExhausted(
                        f"the per-minute limit rejected {self.settings.rate_limit_max_waits} "
                        f"requests in a row even after waiting each time (last server message: "
                        f"{message[:220]}) — something beyond normal pacing looks wrong",
                        scope="minute",
                    ) from exc

                # A server-reported wait is an exact debt: sleeping it pays it
                # off in full, so the local limiter needs no further penalty —
                # calling saturate() on top would double the wait for nothing.
                # Only an unknown wait (we had to guess) leaves real doubt
                # about how much room is left, which is what saturate() is for.
                known_wait = wait is not None
                wait = wait if known_wait else min(60.0, 5.0 * rate_limit_waits)
                print(f"    [rate-limit] Groq's PER-MINUTE limit was hit (attempt "
                      f"{rate_limit_waits}/{self.settings.rate_limit_max_waits}) — waiting "
                      f"{wait:.0f}s ({'server-reported' if known_wait else 'estimated'}), "
                      f"then retrying automatically. This is expected only under unusually "
                      f"heavy load and does not use up the normal "
                      f"{self.settings.max_retries}-retry error budget.")
                time.sleep(wait + 0.5)
                if not known_wait:
                    self._limiter.saturate()
                continue                                 # does not consume `attempt`
            except Exception as exc:                    # noqa: BLE001 - many SDK error types
                message = str(exc)
                if "response_format" in message or "reasoning_effort" in message:
                    drop_extras = True
                    continue
                attempt += 1
                backoff = min(60.0, 2.0 ** attempt)
                print(f"    [warn] Groq request failed ({message[:140]}); "
                      f"retry {attempt}/{self.settings.max_retries} in {backoff:.0f}s")
                time.sleep(backoff)
                continue

            self.calls += 1
            usage = getattr(response, "usage", None)
            if usage:
                prompt_used = getattr(usage, "prompt_tokens", 0) or 0
                completion_used = getattr(usage, "completion_tokens", 0) or 0
                self.prompt_tokens += prompt_used
                self.completion_tokens += completion_used
                self._limiter.settle(prompt_used + completion_used)

            parsed = _parse_json_object(response.choices[0].message.content or "")
            if parsed:
                return parsed
            attempt += 1
            print(f"    [warn] reply was not a JSON object; "
                  f"retry {attempt}/{self.settings.max_retries}")

        return {}


def _parse_json_object(text: str) -> dict:
    """Parse a JSON object out of a model reply, tolerating fences and prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Phrase bank assembly
# ---------------------------------------------------------------------------

@dataclass
class AuthoringReport:
    """What the authoring pass actually achieved, for the dataset statistics."""

    source: str = "seed"
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    templates_accepted: dict = field(default_factory=dict)
    templates_rejected: dict = field(default_factory=dict)
    rejection_examples: list = field(default_factory=list)
    entities_added: dict = field(default_factory=dict)
    attributes_overridden: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "templates_accepted": self.templates_accepted,
            "templates_rejected": self.templates_rejected,
            "rejection_examples": self.rejection_examples[:12],
            "entities_added": self.entities_added,
            "attributes_overridden": self.attributes_overridden,
        }


def _merge_templates(bank: dict, slot_name: str, candidates, report: AuthoringReport,
                     bans: dict | None = None, tokenizer=None) -> None:
    """Validate candidate templates and append the good ones to the bank."""
    slot = tasks.SLOTS[slot_name]
    existing = bank["templates"].setdefault(slot_name, [])
    seen = {t.strip() for t in existing}

    for candidate in candidates or []:
        text = candidate.strip() if isinstance(candidate, str) else candidate
        ok, reason = (validate_template(slot, text, bans, tokenizer) if isinstance(text, str)
                      else (False, "not a string"))
        if not ok:
            report.templates_rejected[slot_name] = report.templates_rejected.get(slot_name, 0) + 1
            if len(report.rejection_examples) < 40:
                report.rejection_examples.append(
                    {"slot": slot_name, "reason": reason, "template": str(text)[:160]}
                )
            continue
        if text in seen:
            report.templates_rejected[slot_name] = report.templates_rejected.get(slot_name, 0) + 1
            continue
        existing.append(text)
        seen.add(text)
        report.templates_accepted[slot_name] = report.templates_accepted.get(slot_name, 0) + 1


def _save_bank(lang: Language, bank: dict) -> None:
    """Persist the phrase bank exactly as it stands right now.

    Called after every authoring stage — entities, attributes, each template
    batch — not just once at the end, so a run cut short by a daily quota
    limit (or a crash) keeps everything already authored instead of losing it.
    Written atomically (temp file + rename) so an interruption mid-write can
    never leave a half-written, unparseable cache behind.
    """
    cache_path = lang.reasoning_phrasebank
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
    tmp_path.replace(cache_path)


def _author_entities(author: GroqPhraseAuthor, lang: Language, bank: dict,
                     report: AuthoringReport, tokenizer=None) -> None:
    """Ask the model to widen the entity name pools, then merge what validates."""
    print(f"  [{lang.display}] authoring entity name pools ...")
    system = SYSTEM_PROMPT.format(display=lang.display, script="Devanagari")
    prompt = _entities_user_prompt(lang.display, "Devanagari",
                                   author.settings.entities_per_pool, bank["entities"])
    reply = author.chat_json(system, prompt, max_tokens=2000)

    for pool in tasks.ENTITY_POOLS:
        existing = bank["entities"].setdefault(pool, [])
        seen = set(existing)
        added = 0
        for raw in reply.get(pool, []) or []:
            name = _clean_name(raw, tokenizer)
            if name and name not in seen:
                existing.append(name)
                seen.add(name)
                added += 1
        report.entities_added[pool] = added
    print("    added: " + ", ".join(f"{k}+{v}" for k, v in report.entities_added.items()))


def _author_attributes(author: GroqPhraseAuthor, lang: Language, bank: dict,
                       report: AuthoringReport, tokenizer=None) -> None:
    """Ask the model for attribute vocabulary; accept only fully valid word sets.

    An attribute is overridden all-or-nothing: a half-replaced word set would
    mix two grammatical analyses of the same noun and produce broken agreement.
    """
    print(f"  [{lang.display}] authoring attribute vocabulary ...")
    system = SYSTEM_PROMPT.format(display=lang.display, script="Devanagari")
    prompt = _attributes_user_prompt(lang.display, "Devanagari", bank["attributes"])
    reply = author.chat_json(system, prompt, max_tokens=2400)

    for key in tasks.ATTRIBUTE_KEYS:
        proposed = reply.get(key)
        if not isinstance(proposed, dict):
            continue
        cleaned = {}
        for field_name in tasks.ATTRIBUTE_WORD_FIELDS:
            word = _clean_word(proposed.get(field_name), tokenizer)
            if word is None:
                cleaned = {}
                break
            cleaned[field_name] = word
        if cleaned:
            bank["attributes"][key] = cleaned
            report.attributes_overridden.append(key)
    print(f"    replaced vocabulary for {len(report.attributes_overridden)}"
          f"/{len(tasks.ATTRIBUTE_KEYS)} attributes")


def _author_templates(author: GroqPhraseAuthor, lang: Language, bank: dict,
                      report: AuthoringReport, progress: dict, log=None,
                      tokenizer=None) -> None:
    """Ask the model for extra template variants, a few slots per request.

    Checkpointed per batch: each batch's starting index is recorded in
    ``progress["template_batches_done"]`` and the bank is saved to disk right
    after, so a run interrupted partway (daily quota, crash, Ctrl-C) resumes
    at the next batch instead of re-requesting — and re-paying for — slots
    that were already authored.

    Args:
        author:    The Groq client.
        lang:      Language being authored.
        bank:      Phrase bank, modified in place.
        report:    Authoring report, updated in place.
        progress:  This bank's ``authoring_progress`` dict, updated in place.
        log:       Optional ``log_event(lang, message)`` callable for the
                   cross-run append-only history.
        tokenizer: Optional loaded tokenizer — rejects any template that would
                   tokenize to ``<unk>``, e.g. smart-typography punctuation.
    """
    system = SYSTEM_PROMPT.format(display=lang.display, script="Devanagari")
    batch_size = max(1, author.settings.slots_per_request)
    slots = list(tasks.ALL_SLOTS)
    bans = lexicon_bans(bank)
    done_batches = set(progress.get("template_batches_done", []))
    total_batches = -(-len(slots) // batch_size)

    for batch_index, start in enumerate(range(0, len(slots), batch_size), start=1):
        if start in done_batches:
            continue
        batch = slots[start:start + batch_size]
        names = ", ".join(s.name for s in batch)
        print(f"  [{lang.display}] authoring templates for {names} "
              f"(batch {batch_index}/{total_batches}) ...")
        prompt = _templates_user_prompt(lang.display, "Devanagari", batch,
                                        bank["templates"], author.settings.templates_per_slot)
        # ~120 tokens per template covers a 25-word sentence plus the low-effort
        # reasoning gpt-oss emits, while keeping one request inside the
        # per-minute token budget alongside its ~1.8k-token prompt.
        budget = 120 * author.settings.templates_per_slot * len(batch)
        reply = author.chat_json(system, prompt, max_tokens=min(3000, max(700, budget)))
        for slot in batch:
            _merge_templates(bank, slot.name, reply.get(slot.name), report, bans, tokenizer)
            print(f"    {slot.name}: {len(bank['templates'][slot.name])} usable "
                  f"(+{report.templates_accepted.get(slot.name, 0)} new, "
                  f"{report.templates_rejected.get(slot.name, 0)} rejected)")

        done_batches.add(start)
        progress["template_batches_done"] = sorted(done_batches)
        _save_bank(lang, bank)
        if log:
            log(lang, f"template batch {batch_index}/{total_batches} authored ({names})")


def check_coverage(bank: dict, min_per_slot: int = 3) -> list[str]:
    """Report slots or attributes that are too thin to build a dataset from.

    Args:
        bank:         A phrase bank.
        min_per_slot: Minimum templates a splittable slot needs (one per split).

    Returns:
        A list of human-readable problems; empty means the bank is usable.
    """
    problems = []
    for slot in tasks.ALL_SLOTS:
        count = len(bank.get("templates", {}).get(slot.name, []))
        needed = min_per_slot if slot.splittable else 1
        if count < needed:
            problems.append(f"slot '{slot.name}' has {count} templates, needs >= {needed}")

    for key in tasks.ATTRIBUTE_KEYS:
        words = bank.get("attributes", {}).get(key, {})
        missing = [f for f in tasks.ATTRIBUTE_WORD_FIELDS if not words.get(f)]
        if missing:
            problems.append(f"attribute '{key}' is missing words {missing}")

    for pool in tasks.ENTITY_POOLS:
        names = bank.get("entities", {}).get(pool, [])
        if len(names) < 10:
            problems.append(f"entity pool '{pool}' has {len(names)} names, needs >= 10")

    return problems


def _validate_seed_templates(bank: dict, tokenizer=None) -> list[tuple[str, str, str]]:
    """Apply the template rules to the hand-authored seeds, dropping failures.

    The seeds are held to the same contract as LLM output — including the
    real-tokenizer ``<unk>`` check when a tokenizer is available — so that a
    typo in ``seeds.py`` cannot quietly enter the corpus.

    Args:
        bank:      The seed phrase bank, modified in place.
        tokenizer: Optional loaded tokenizer for this language.

    Returns:
        ``(slot, reason, template)`` for every template removed.
    """
    bans = lexicon_bans(bank)
    dropped = []
    for slot in tasks.ALL_SLOTS:
        kept = []
        for template in bank["templates"].get(slot.name, []):
            ok, reason = validate_template(slot, template, bans, tokenizer)
            if ok:
                kept.append(template)
            else:
                dropped.append((slot.name, reason, template))
        bank["templates"][slot.name] = kept
    return dropped


def sanitize_bank(bank: dict, tokenizer) -> list[tuple[str, str, str]]:
    """Re-check every template and entity name already in a bank, in place.

    Used to clean a *cached* bank without spending any more Groq quota — the
    exact situation after a template (or entity name) that passed the
    regex-based checks turns out to tokenize to ``<unk>`` (smart typography
    the regexes didn't anticipate; see ``_SMART_TYPOGRAPHY_RE``). Attribute
    words are checked and reported but never auto-dropped here, since removing
    one field of a required set would break that attribute's grammar outright
    rather than degrade gracefully — ``check_coverage`` catches that case and
    tells the caller to ``--refresh-phrasebank`` instead.

    Args:
        bank:      The phrase bank to sanitize, modified in place.
        tokenizer: Loaded tokenizer for this language. If ``None``, nothing is
                   checked and an empty list is returned.

    Returns:
        ``(kind, item, reason)`` for everything removed or merely flagged —
        ``kind`` is ``"template:<slot>"``, ``"entity:<pool>"`` or
        ``"attribute:<key>.<field>"`` (flagged only, not removed).
    """
    if tokenizer is None:
        return []
    bans = lexicon_bans(bank)
    removed = []

    for slot in tasks.ALL_SLOTS:
        kept = []
        for template in bank.get("templates", {}).get(slot.name, []):
            ok, reason = validate_template(slot, template, bans, tokenizer)
            if ok:
                kept.append(template)
            else:
                removed.append((f"template:{slot.name}", template, reason))
        bank.setdefault("templates", {})[slot.name] = kept

    for pool in tasks.ENTITY_POOLS:
        names = bank.get("entities", {}).get(pool, [])
        kept, dropped_here = [], []
        for name in names:
            clean, reason = _renders_cleanly(name, tokenizer)
            if clean:
                kept.append(name)
            else:
                dropped_here.append((name, reason))
        bank.setdefault("entities", {})[pool] = kept
        removed.extend((f"entity:{pool}", name, reason) for name, reason in dropped_here)

    for key, words in bank.get("attributes", {}).items():
        for field_name, word in (words or {}).items():
            if not word:
                continue
            clean, reason = _renders_cleanly(word, tokenizer)
            if not clean:
                removed.append((f"attribute:{key}.{field_name}", word, reason))

    return removed


def _empty_progress() -> dict:
    """A fresh ``authoring_progress`` record for a phrase bank."""
    return {"entities_done": False, "attributes_done": False, "template_batches_done": []}


def _log_path(lang: Language):
    """Where this language's cross-run authoring history is kept."""
    return lang.reasoning_dir / "authoring_log.txt"


def log_event(lang: Language, message: str) -> None:
    """Append one timestamped line to this language's authoring log.

    Opened in **append** mode and never truncated or overwritten — unlike
    ``phrasebank.json`` (which is rewritten on every checkpoint but always
    grows, see ``_save_bank``), this file is pure history. Authoring a corpus
    can easily take several separate invocations because of Groq's daily
    quota, and each run's console output disappears once the terminal session
    ends; this file is where the full story survives across all of them, so
    re-running the same command after a quota resets, minutes or days later,
    never loses the record of what already happened.
    """
    path = _log_path(lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def _quota_exhausted_banner(lang: Language, exc: QuotaExhausted, cache_path) -> str:
    """A hard-to-miss block explaining what was hit and exactly what to do next."""
    bar = "!" * 70
    if exc.scope == "day":
        headline = "GROQ'S DAILY QUOTA IS EXHAUSTED"
        explanation = (
            "  This is the ACCOUNT-WIDE DAILY limit (200K tokens/day on the free tier).\n"
            "  It resets on a rolling 24h basis and cannot be waited out inside this run,\n"
            "  so authoring has stopped here for now."
        )
    else:
        headline = "GROQ'S PER-MINUTE LIMIT KEPT REJECTING REQUESTS"
        explanation = (
            "  Waited the server-reported time and retried automatically several times,\n"
            "  but it kept getting rejected — stopping rather than looping forever."
        )
    return (
        f"\n{bar}\n"
        f"  {headline}  —  {lang.display} ({lang.model_label})\n"
        f"{bar}\n"
        f"{explanation}\n"
        f"  Server said: {str(exc)[:280]}\n"
        f"\n"
        f"  Nothing is lost — everything authored so far IS SAVED to:\n"
        f"    {cache_path}\n"
        f"\n"
        f"  Just re-run the exact same command"
        f"{' later (after the daily quota resets)' if exc.scope == 'day' else ' in a minute or two'}:\n"
        f"    python main.py reasoning-data --lang {lang.key}\n"

        f"  It will pick up authoring exactly where it stopped — already-authored\n"
        f"  entities, attributes and template batches are skipped, so no quota already\n"
        f"  spent is wasted.\n"
        f"\n"
        f"  For right now, generation continues below using what was authored so far,\n"
        f"  layered on top of the always-present hand-authored seed phrase bank — so\n"
        f"  this run still produces a complete, usable dataset.\n"
        f"{bar}\n"
    )


def build_phrasebank(lang: Language, llm_config: dict | None = None,
                     offline: bool = False, refresh: bool = False,
                     api_key: str | None = None) -> tuple[dict, AuthoringReport]:
    """Load or build this language's phrase bank.

    The seed bank is always the starting point; online authoring only ever adds
    templates and names on top of it, so the corpus degrades gracefully rather
    than failing when the API is unavailable — and the same is true mid-way
    through authoring: entities, attributes and each batch of templates are
    checkpointed to disk as they are produced (see ``_save_bank``), so a run
    stopped by Groq's daily quota, a per-minute limit that would not clear, or
    a crash never loses work.  A later run with the same arguments resumes
    exactly where the previous one stopped, at no extra API cost.

    Args:
        lang:       Which model's language to build for.
        llm_config: The ``llm:`` block of ``reasoning_config.yaml``.
        offline:    Skip the API entirely and use the hand-authored seeds.
        refresh:    Ignore any cached phrase bank and author a fresh one from
                    scratch (re-spending the API calls a previous run already paid for).
        api_key:    Explicit Groq key, overriding the environment.

    Returns:
        ``(phrasebank, report)``.

    Raises:
        SystemExit: If the resulting bank cannot support dataset generation.
    """
    from .generator import load_tokenizer            # local import: avoids a module cycle
    tokenizer = load_tokenizer(lang)
    if tokenizer is None:
        print(f"  [{lang.display}] [warn] {lang.tokenizer_path} unavailable — authored text "
              f"cannot be checked against it, so a smart-typography character could still slip "
              f"through as <unk>. Train the Phase 1 tokenizer first if possible.")

    cache_path = lang.reasoning_phrasebank
    resuming = False

    if cache_path.exists() and not refresh:
        with open(cache_path, encoding="utf-8") as f:
            bank = json.load(f)
        # Caches written before this checkpointing existed have no "complete"
        # key; treat them as complete rather than as an interrupted run.
        is_complete = bank.get("complete", True)

        if is_complete or offline:
            report = AuthoringReport(source=bank.get("source", "cache") + " (cached)")
            note = "" if is_complete else " (incomplete — using as-is in --offline mode)"
            print(f"  [{lang.display}] reusing cached phrase bank {cache_path}{note}")

            # A cache can predate a validation rule added since it was written
            # (exactly what happened here: smart-typography punctuation that
            # passed the old regex checks but tokenizes to <unk>). Re-checking
            # against the real tokenizer costs no API quota and fixes it in place.
            sanitized = sanitize_bank(bank, tokenizer)
            if sanitized:
                print(f"  [{lang.display}] sanitizing cached phrase bank — removed "
                      f"{sum(1 for k, *_ in sanitized if not k.startswith('attribute:'))} "
                      f"item(s) that would tokenize to <unk>:")
                for kind, item, reason in sanitized[:20]:
                    print(f"    {kind}: {reason} -> {item!r}")
                if len(sanitized) > 20:
                    print(f"    ... and {len(sanitized) - 20} more")
                _save_bank(lang, bank)
                log_event(lang, f"SANITIZED cached bank — removed {len(sanitized)} "
                                f"item(s) that would tokenize to <unk>")

            problems = check_coverage(bank)
            if problems:
                raise SystemExit(
                    f"Cached phrase bank at {cache_path} is unusable:\n  - "
                    + "\n  - ".join(problems)
                    + "\nDelete it or re-run with --refresh-phrasebank."
                )
            return bank, report

        progress = bank.setdefault("authoring_progress", _empty_progress())
        resuming = True
        print(f"  [{lang.display}] resuming interrupted authoring from {cache_path} "
              f"(entities {'done' if progress['entities_done'] else 'pending'}, "
              f"attributes {'done' if progress['attributes_done'] else 'pending'}, "
              f"{len(progress['template_batches_done'])} template batch(es) already done)")
        report = AuthoringReport(source=bank.get("source", "seed"))
        log_event(lang, f"RESUME run — entities={'done' if progress['entities_done'] else 'pending'}, "
                        f"attributes={'done' if progress['attributes_done'] else 'pending'}, "
                        f"{len(progress['template_batches_done'])} template batch(es) already done")
    else:
        bank = copy.deepcopy(SEED_PHRASEBANKS[lang.key])
        bank["language"] = lang.key
        bank["model_label"] = lang.model_label
        report = AuthoringReport(source="seed")

        dropped = _validate_seed_templates(bank, tokenizer)
        if dropped:
            print(f"  [{lang.display}] dropped {len(dropped)} invalid seed template(s):")
            for slot_name, reason, template in dropped:
                print(f"    {slot_name}: {reason} -> {template}")

    if not offline:
        settings = LLMSettings.from_config((llm_config or {}))
        author = GroqPhraseAuthor(settings, api_key=api_key)
        progress = bank.setdefault("authoring_progress", _empty_progress())
        if not resuming:
            print(f"  [{lang.display}] authoring with {settings.model} via Groq "
                  f"(<= {settings.requests_per_minute} req/min, "
                  f"{settings.tokens_per_minute} tok/min, paced locally so the "
                  f"per-minute limit is never actually reached)")
            log_event(lang, "START run — authoring from the seed phrase bank")

        try:
            if not progress["entities_done"]:
                _author_entities(author, lang, bank, report, tokenizer)
                progress["entities_done"] = True
                _save_bank(lang, bank)
                log_event(lang, "entity name pools authored")
            else:
                print(f"  [{lang.display}] entity pools already authored — skipping")

            if not progress["attributes_done"]:
                _author_attributes(author, lang, bank, report, tokenizer)
                progress["attributes_done"] = True
                _save_bank(lang, bank)
                log_event(lang, "attribute vocabulary authored")
            else:
                print(f"  [{lang.display}] attribute vocabulary already authored — skipping")

            _author_templates(author, lang, bank, report, progress, log=log_event,
                              tokenizer=tokenizer)
            bank["complete"] = True
            log_event(lang, f"COMPLETE — phrase bank fully authored "
                            f"({author.calls} API call(s) this run)")

        except QuotaExhausted as exc:
            bank["complete"] = False
            _save_bank(lang, bank)
            print(_quota_exhausted_banner(lang, exc, cache_path))
            log_event(lang, f"STOPPED — {exc.scope}-scoped quota hit after {author.calls} "
                            f"API call(s) this run: {str(exc)[:200]}")

        report.source = "llm+seed" if author.calls or resuming else report.source
        report.api_calls = author.calls
        report.prompt_tokens = author.prompt_tokens
        report.completion_tokens = author.completion_tokens
        bank["llm_model"] = settings.model
        print(f"  [{lang.display}] {author.calls} API call(s) this run, "
              f"{author.prompt_tokens + author.completion_tokens:,} tokens used this run")
    else:
        bank["complete"] = bank.get("complete", True)

    bank["source"] = report.source
    bank["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    problems = check_coverage(bank)
    if problems:
        raise SystemExit("Phrase bank is unusable:\n  - " + "\n  - ".join(problems))

    _save_bank(lang, bank)
    print(f"  [{lang.display}] phrase bank saved to {cache_path}"
          + ("" if bank["complete"] else " (partial — will resume authoring next run)"))
    return bank, report
