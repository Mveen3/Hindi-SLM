"""
Synthetic Reasoning Task Specifications (Phase 3)
=================================================

Declarative definition of the comparative / transitive reasoning tasks used to
finetune Model H and Model L.  This module is deliberately **language-neutral**:
it knows the *logic* of each task (what is given, what must be deduced, what the
ground-truth answer is) but not a single Hindi or Nepali word.  Every surface
string comes from that language's phrase bank (``phrasebank.py``) and is
rendered by the ``TaskContext`` supplied by ``generator.py``.

Why the split matters
---------------------
The project brief requires a *synthetic* finetuning set whose labels we know
exactly.  So the ground truth here is never produced by a language model: for
every example the generator samples hidden numeric values, derives the answer
by ordinary Python comparison / sorting, and only then asks the phrase bank to
render the facts and the question.  An LLM contributes phrasing variety, never
labels.

Task families (15 tasks over 7 question groups)
-----------------------------------------------
Value-fact tasks   — numeric attributes are stated outright:
    compare_two_more / compare_two_less, compare_equal, numeric_difference,
    order_ascending / order_descending, middle_entity, verify_claim
Relation-fact tasks — only pairwise inequalities are stated, so the answer
                      requires chaining them (the transitive core of the brief):
    transitive_relation, transitive_largest, transitive_smallest,
    chain_relation, chain_largest, chain_smallest, mixed_direction_chain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Numerals
# ---------------------------------------------------------------------------

DEVANAGARI_DIGITS = "०१२३४५६७८९"
_ASCII_TO_DEVANAGARI = str.maketrans("0123456789", DEVANAGARI_DIGITS)


def format_number(value: int, numeral_script: str = "ascii") -> str:
    """Render an integer in the numeral script the corpus actually uses.

    Hindi web text overwhelmingly uses ASCII digits while Nepali web text
    overwhelmingly uses Devanagari digits, and the two Phase 1 tokenizers show
    exactly that skew in their vocabularies.  Matching the corpus keeps the
    finetuning data inside the pretrained vocabulary instead of fragmenting
    every number into rare single-character tokens.

    Args:
        value:          The integer to render.
        numeral_script: ``"ascii"`` (0-9) or ``"devanagari"`` (०-९).

    Returns:
        The number as a string in the requested script.
    """
    text = str(value)
    if numeral_script == "devanagari":
        return text.translate(_ASCII_TO_DEVANAGARI)
    return text


# ---------------------------------------------------------------------------
# Attributes: the quantities entities are compared on
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttributeSpec:
    """A comparable numeric property, e.g. age or price.

    The *numbers* live here (they must be reproducible and label-safe); the
    *words* for this attribute live in the phrase bank under the same ``key``.

    Attributes:
        key:   Stable identifier, also the phrase-bank lookup key.
        pool:  Which entity pool this attribute applies to ("people", "cities", ...).
        low:   Smallest plausible value.
        high:  Largest plausible value.
        step:  Value granularity (prices move in 5s, ages in 1s).
        gloss: English gloss, used only when prompting the LLM for wording.
    """

    key: str
    pool: str
    low: int
    high: int
    step: int = 1
    gloss: str = ""


ATTRIBUTES: tuple[AttributeSpec, ...] = (
    AttributeSpec("age",         "people",  4,   90,  1,  "age of a person in years"),
    AttributeSpec("height",      "people",  95,  195, 1,  "height of a person in centimetres"),
    AttributeSpec("weight",      "people",  18,  110, 1,  "body weight of a person in kilograms"),
    AttributeSpec("price",       "objects", 25,  4000, 5, "price of an everyday object in rupees"),
    AttributeSpec("quantity",    "fruits",  2,   80,  1,  "quantity of a fruit in kilograms"),
    AttributeSpec("distance",    "cities",  10,  900, 5,  "distance to a city in kilometres"),
    AttributeSpec("population",  "cities",  2,   95,  1,  "population of a city in hundred-thousands (lakh)"),
    AttributeSpec("speed",       "animals", 6,   115, 1,  "running speed of an animal in km per hour"),
    AttributeSpec("temperature", "cities",  2,   45,  1,  "temperature of a city in degrees Celsius"),
)

ATTRIBUTE_KEYS: tuple[str, ...] = tuple(a.key for a in ATTRIBUTES)

#: Entity pools the attributes draw their subjects from.
ENTITY_POOLS: tuple[str, ...] = ("people", "cities", "objects", "animals", "fruits")

#: Words the phrase bank must supply for every attribute, with the grammatical
#: role each one plays.  Kept explicit because Hindi possessives and
#: interrogatives agree with the *noun's* gender, so they cannot be hard-coded.
ATTRIBUTE_WORD_FIELDS: dict[str, str] = {
    "noun":    "the attribute noun itself (e.g. the word for 'age')",
    "poss":    "possessive particle joining an entity name to the noun, agreeing with the noun",
    "poss_obl": ("the same possessive in the oblique case, i.e. the form it takes when the "
                 "phrase is followed by a comparative postposition"),
    "whose":   "interrogative possessive ('whose') agreeing with the noun",
    "howmuch": "interrogative quantity word ('how much') agreeing with the noun",
    "unit":    "measurement unit for the values",
    "copula":  "the 'is/are' copula agreeing with the noun",
    "more":    "an invariant comparative word meaning 'more/greater'",
    "less":    "an invariant comparative word meaning 'less/smaller'",
}


# ---------------------------------------------------------------------------
# Template slots: the placeholder contract every phrasing must satisfy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotSpec:
    """One kind of sentence the phrase bank must provide variants of.

    Attributes:
        name:        Slot identifier used everywhere else.
        required:    Placeholders every template for this slot MUST contain.
        allowed:     Full set of placeholders a template MAY contain.
        instruction:  What the sentence has to express, for the LLM prompt.
        splittable:  If True, the template pool is partitioned across
                     train/val/test so the test set uses unseen phrasings.
    """

    name: str
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    instruction: str
    splittable: bool = False


_ENTITY_PH = ("{E}", "{A}", "{B}", "{C}")

#: Sentences that state the premises, and the questions asked about them.
#: These are the slots held out across splits — the test set is graded on
#: phrasings the model never saw during finetuning.
PROMPT_SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec(
        "fact",
        required=("{E}", "{noun}", "{v}"),
        allowed=("{E}", "{poss}", "{noun}", "{v}", "{unit}", "{copula}"),
        instruction="State that entity {E} has the numeric value {v} {unit} for the attribute {noun}.",
        splittable=True,
    ),
    SlotSpec(
        "rel_fact",
        required=("{A}", "{B}", "{noun}", "{cmp}"),
        allowed=("{A}", "{B}", "{poss}", "{poss_obl}", "{noun}", "{cmp}", "{copula}"),
        instruction=("State that {A}'s {noun} is {cmp} than {B}'s {noun}, without revealing any "
                     "numbers. {cmp} is filled with the comparative word for 'more' or 'less'."),
        splittable=True,
    ),
    SlotSpec(
        "q_which_of_two",
        required=("{A}", "{B}", "{noun}", "{cmp}"),
        allowed=("{A}", "{B}", "{poss}", "{noun}", "{whose}", "{cmp}", "{copula}"),
        instruction="Ask which of the two entities {A} and {B} has the {cmp} {noun}. The answer is one name.",
        splittable=True,
    ),
    SlotSpec(
        "q_extreme",
        required=("{noun}", "{cmp}"),
        allowed=("{poss}", "{noun}", "{whose}", "{cmp}", "{copula}"),
        instruction=("Ask which one of the entities mentioned above has the most {cmp} {noun} "
                     "(a superlative question). Do not name any entity. The answer is one name."),
        splittable=True,
    ),
    SlotSpec(
        "q_equality",
        required=("{A}", "{B}", "{noun}"),
        allowed=("{A}", "{B}", "{poss}", "{noun}", "{copula}"),
        instruction=("Ask whether {A} and {B} have an equal {noun}. It must read as a yes/no "
                     "question, because the answer is the word for yes or no."),
        splittable=True,
    ),
    SlotSpec(
        "q_difference",
        required=("{A}", "{B}", "{noun}"),
        allowed=("{A}", "{B}", "{poss}", "{poss_obl}", "{noun}", "{howmuch}", "{cmp}", "{copula}"),
        instruction=("Ask by how much {A}'s {noun} exceeds {B}'s {noun}. The answer is a number "
                     "followed by the unit."),
        splittable=True,
    ),
    SlotSpec(
        "q_ordering",
        required=("{noun}", "{cmp_from}", "{cmp_to}"),
        allowed=("{poss}", "{noun}", "{cmp_from}", "{cmp_to}", "{copula}"),
        instruction=("Instruct the reader to arrange all the entities mentioned above in order of "
                     "{noun}, going from {cmp_from} to {cmp_to}. Do not name any entity."),
        splittable=True,
    ),
    SlotSpec(
        "q_middle",
        required=("{noun}",),
        allowed=("{poss}", "{noun}", "{whose}", "{copula}"),
        instruction=("Ask which of the three entities above is in the middle by {noun} — neither "
                     "the largest nor the smallest. Do not name any entity. The answer is one name."),
        splittable=True,
    ),
    SlotSpec(
        "q_verify",
        required=("{A}", "{B}", "{noun}", "{cmp}"),
        allowed=("{A}", "{B}", "{poss}", "{poss_obl}", "{noun}", "{cmp}", "{copula}"),
        instruction=("Ask whether the claim '{A}'s {noun} is {cmp} than {B}'s {noun}' is correct. "
                     "It must read as a yes/no question."),
        splittable=True,
    ),
)

#: Sentences that make up the chain-of-thought.  These are *shared* across
#: splits: they are the reasoning style we are teaching, not a generalisation
#: axis being tested, and the test metric only scores the final answer.
STEP_SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec(
        "fact_step",
        required=("{E}", "{noun}", "{v}"),
        allowed=("{E}", "{poss}", "{noun}", "{v}", "{unit}"),
        instruction="Restate compactly, as a working note, that {E}'s {noun} is {v} {unit}.",
    ),
    SlotSpec(
        "rel_step",
        required=("{A}", "{B}", "{noun}", "{cmp}"),
        allowed=("{A}", "{B}", "{poss}", "{poss_obl}", "{noun}", "{cmp}", "{copula}"),
        instruction="Note as a reasoning step that {A}'s {noun} is {cmp} than {B}'s {noun}.",
    ),
    SlotSpec(
        "deduce_step",
        required=("{A}", "{B}", "{C}", "{noun}", "{cmp}"),
        allowed=("{A}", "{B}", "{C}", "{poss}", "{poss_obl}", "{noun}", "{cmp}", "{copula}"),
        instruction=("Express the transitive deduction: because {A}'s {noun} is {cmp} than {B}'s "
                     "and {B}'s is {cmp} than {C}'s, therefore {A}'s {noun} is {cmp} than {C}'s."),
    ),
    SlotSpec(
        "flip_step",
        required=("{A}", "{B}", "{noun}", "{cmp_from}", "{cmp_to}"),
        allowed=("{A}", "{B}", "{poss}", "{poss_obl}", "{noun}", "{cmp_from}", "{cmp_to}", "{copula}"),
        instruction=("Express reversing an inequality: '{A}'s {noun} is {cmp_from} than {B}'s' is "
                     "the same as saying '{B}'s {noun} is {cmp_to} than {A}'s'."),
    ),
    SlotSpec(
        "diff_step",
        required=("{vA}", "{vB}", "{d}"),
        allowed=("{vA}", "{vB}", "{d}", "{unit}", "{noun}"),
        instruction="Show the subtraction {vA} minus {vB} giving {d} {unit}.",
    ),
    SlotSpec(
        "order_step",
        required=("{ordered}",),
        allowed=("{ordered}", "{poss}", "{noun}"),
        instruction="Present the finished ordering, where {ordered} is an already comma-joined list of names.",
    ),
    SlotSpec(
        "equal_step",
        required=("{noun}",),
        allowed=("{A}", "{B}", "{poss}", "{noun}", "{v}", "{unit}", "{copula}"),
        instruction="Note that {A} and {B} have the same {noun}, namely {v} {unit}, so they are equal.",
    ),
    SlotSpec(
        "conclusion_step",
        required=("{answer}",),
        allowed=("{answer}",),
        instruction="Close the reasoning by declaring that the answer is {answer}.",
    ),
)

ALL_SLOTS: tuple[SlotSpec, ...] = PROMPT_SLOTS + STEP_SLOTS
SLOTS: dict[str, SlotSpec] = {slot.name: slot for slot in ALL_SLOTS}
SPLITTABLE_SLOTS: tuple[str, ...] = tuple(s.name for s in ALL_SLOTS if s.splittable)


# ---------------------------------------------------------------------------
# Task builders
#
# Every builder takes a TaskContext (see generator.TaskContext) exposing:
#   ctx.rng                     seeded Random
#   ctx.names(n)                n distinct entity names from this split's pool
#   ctx.values(n, min_gap=)     n distinct attribute values, descending
#   ctx.more / ctx.less         comparative words for the chosen attribute
#   ctx.fact(name, value)       rendered value premise
#   ctx.rel(a, b, cmp)          rendered inequality premise ("a's X is cmp than b's")
#   ctx.question(slot, **kw)    rendered question
#   ctx.step(kind, **kw)        rendered reasoning step
#   ctx.num(value)              number in the corpus' numeral script
#   ctx.unit                    unit word
#   ctx.yes / ctx.no            in-language yes / no answers
#   ctx.shuffled(seq)           a shuffled copy
# and returns (premises: list[str], question: str, steps: list[str], answer: str).
# ---------------------------------------------------------------------------

def _compare_two(ctx, want_more: bool):
    """Two stated values; ask which entity is greater (or smaller)."""
    a, b = ctx.names(2)
    va, vb = ctx.values(2)                      # descending, so va > vb
    if ctx.rng.random() < 0.5:                  # decouple answer from name order
        a, b = b, a
        va, vb = vb, va

    hi_name, lo_name = (a, b) if va > vb else (b, a)
    answer = hi_name if want_more else lo_name

    premises = ctx.shuffled([ctx.fact(a, va), ctx.fact(b, vb)])
    question = ctx.question("q_which_of_two", A=a, B=b, cmp=ctx.more if want_more else ctx.less)
    steps = [
        ctx.step("fact_step", E=a, v=va),
        ctx.step("fact_step", E=b, v=vb),
        ctx.step("rel_step", A=hi_name, B=lo_name, cmp=ctx.more),
        ctx.step("conclusion_step", answer=answer),
    ]
    return premises, question, steps, answer


def compare_two_more(ctx):
    """Which of two entities has the greater value."""
    return _compare_two(ctx, want_more=True)


def compare_two_less(ctx):
    """Which of two entities has the smaller value."""
    return _compare_two(ctx, want_more=False)


def compare_equal(ctx):
    """Are two stated values equal?  Balanced yes / no."""
    a, b = ctx.names(2)
    equal = ctx.rng.random() < 0.5
    if equal:
        va = vb = ctx.values(1)[0]
    else:
        va, vb = ctx.values(2)
        if ctx.rng.random() < 0.5:
            va, vb = vb, va

    answer = ctx.yes if equal else ctx.no
    premises = ctx.shuffled([ctx.fact(a, va), ctx.fact(b, vb)])
    question = ctx.question("q_equality", A=a, B=b)

    steps = [ctx.step("fact_step", E=a, v=va), ctx.step("fact_step", E=b, v=vb)]
    if equal:
        steps.append(ctx.step("equal_step", A=a, B=b, v=va))
    else:
        hi_name, lo_name = (a, b) if va > vb else (b, a)
        steps.append(ctx.step("rel_step", A=hi_name, B=lo_name, cmp=ctx.more))
    steps.append(ctx.step("conclusion_step", answer=answer))
    return premises, question, steps, answer


def numeric_difference(ctx):
    """By how much does one stated value exceed the other?"""
    a, b = ctx.names(2)
    va, vb = ctx.values(2)                      # va > vb, so "a exceeds b" holds
    diff = va - vb
    answer = f"{ctx.num(diff)} {ctx.unit}".strip()

    premises = ctx.shuffled([ctx.fact(a, va), ctx.fact(b, vb)])
    question = ctx.question("q_difference", A=a, B=b, cmp=ctx.more)
    steps = [
        ctx.step("fact_step", E=a, v=va),
        ctx.step("fact_step", E=b, v=vb),
        ctx.step("diff_step", vA=va, vB=vb, d=diff),
        ctx.step("conclusion_step", answer=answer),
    ]
    return premises, question, steps, answer


def _relation_chain(ctx, n: int, mixed: bool):
    """Build a hidden total order and present it as pairwise inequalities.

    Args:
        ctx:   Task context.
        n:     Number of entities in the chain.
        mixed: If True, roughly half the premises are phrased with the "less"
               comparative, so the model must normalise direction before chaining.

    Returns:
        ``(ranked, premises, flip_steps, normalised)`` where ``ranked`` is in
        strictly descending order, and ``normalised`` holds the adjacent index
        pairs whose direction a flip step has already settled.
    """
    # names() samples a random permutation, so the draw order *is* the hidden
    # total order: ranked[0] is the largest, ranked[-1] the smallest. No numeric
    # values are needed (and none are stated) for a relation-premise task.
    ranked = ctx.names(n)

    premises, flips, normalised = [], [], set()
    for i in range(n - 1):
        hi, lo = ranked[i], ranked[i + 1]
        if mixed and ctx.rng.random() < 0.5:
            premises.append(ctx.rel(lo, hi, ctx.less))
            flips.append(ctx.step("flip_step", A=lo, B=hi,
                                  cmp_from=ctx.less, cmp_to=ctx.more))
            normalised.add(i)
        else:
            premises.append(ctx.rel(hi, lo, ctx.more))
    return ranked, ctx.shuffled(premises), flips, normalised


def _chain_steps(ctx, ranked, i: int, j: int, normalised=()):
    """Reasoning steps that chain ``ranked[i] > ... > ranked[j]`` transitively.

    Args:
        ctx:        Task context.
        ranked:     Entities in descending order.
        i, j:       Endpoints of the sub-chain to walk.
        normalised: Adjacent pairs a flip step has already stated in the
                    "more" direction.  Restating them would repeat the same
                    sentence twice in a row.
    """
    steps = [ctx.step("rel_step", A=ranked[k], B=ranked[k + 1], cmp=ctx.more)
             for k in range(i, j) if k not in normalised]
    for k in range(i + 1, j):
        steps.append(ctx.step("deduce_step", A=ranked[i], B=ranked[k],
                              C=ranked[k + 1], cmp=ctx.more))
    return steps


def _transitive_relation(ctx, n: int, mixed: bool = False):
    """Ask about a pair that is never stated directly, forcing multi-hop chaining."""
    ranked, premises, flips, normalised = _relation_chain(ctx, n, mixed)

    # Only non-adjacent pairs require a deduction, so restrict the query to them.
    pairs = [(i, j) for i in range(n) for j in range(i + 2, n)]
    i, j = ctx.rng.choice(pairs)
    want_more = ctx.rng.random() < 0.5
    answer = ranked[i] if want_more else ranked[j]

    x, y = ctx.shuffled([ranked[i], ranked[j]])
    question = ctx.question("q_which_of_two", A=x, B=y,
                            cmp=ctx.more if want_more else ctx.less)
    steps = flips + _chain_steps(ctx, ranked, i, j, normalised)
    steps.append(ctx.step("conclusion_step", answer=answer))
    return premises, question, steps, answer


def _transitive_extreme(ctx, n: int, want_more: bool, mixed: bool = False):
    """Ask for the largest / smallest entity given only pairwise inequalities."""
    ranked, premises, flips, normalised = _relation_chain(ctx, n, mixed)
    answer = ranked[0] if want_more else ranked[-1]

    question = ctx.question("q_extreme", cmp=ctx.more if want_more else ctx.less)
    steps = flips + _chain_steps(ctx, ranked, 0, n - 1, normalised)
    steps.append(ctx.step("conclusion_step", answer=answer))
    return premises, question, steps, answer


def transitive_relation(ctx):
    """A > B, B > C: how do A and C relate?  (3 entities, 2 premises)"""
    return _transitive_relation(ctx, n=3)


def transitive_largest(ctx):
    """Largest of three entities from two chained inequalities."""
    return _transitive_extreme(ctx, n=3, want_more=True)


def transitive_smallest(ctx):
    """Smallest of three entities from two chained inequalities."""
    return _transitive_extreme(ctx, n=3, want_more=False)


def chain_relation(ctx):
    """Multi-hop relation query over a 4-5 entity inequality chain."""
    return _transitive_relation(ctx, n=ctx.pick_n([4, 5]))


def chain_largest(ctx):
    """Largest over a 4-5 entity inequality chain."""
    return _transitive_extreme(ctx, n=ctx.pick_n([4, 5]), want_more=True)


def chain_smallest(ctx):
    """Smallest over a 4-5 entity inequality chain."""
    return _transitive_extreme(ctx, n=ctx.pick_n([4, 5]), want_more=False)


def mixed_direction_chain(ctx):
    """Chain whose premises mix 'more' and 'less' phrasing before chaining."""
    n = ctx.pick_n([3, 4, 5])
    if ctx.rng.random() < 0.5:
        return _transitive_relation(ctx, n=n, mixed=True)
    return _transitive_extreme(ctx, n=n, want_more=ctx.rng.random() < 0.5, mixed=True)


def _ordering(ctx, ascending: bool):
    """Sort 3-4 entities by their stated values."""
    n = ctx.pick_n([3, 4])
    names = ctx.names(n)
    values = ctx.values(n)                      # descending
    pairs = list(zip(names, values))            # names[i] holds values[i]

    ordered = [name for name, _ in sorted(pairs, key=lambda p: p[1], reverse=not ascending)]
    answer = ", ".join(ordered)

    premises = ctx.shuffled([ctx.fact(name, value) for name, value in pairs])
    question = ctx.question(
        "q_ordering",
        cmp_from=ctx.less if ascending else ctx.more,
        cmp_to=ctx.more if ascending else ctx.less,
    )
    steps = [ctx.step("fact_step", E=name, v=value) for name, value in pairs]
    steps.append(ctx.step("order_step", ordered=answer))
    steps.append(ctx.step("conclusion_step", answer=answer))
    return premises, question, steps, answer


def order_ascending(ctx):
    """Arrange 3-4 entities from smallest to largest."""
    return _ordering(ctx, ascending=True)


def order_descending(ctx):
    """Arrange 3-4 entities from largest to smallest."""
    return _ordering(ctx, ascending=False)


def middle_entity(ctx):
    """Which of three stated values is neither the largest nor the smallest."""
    names = ctx.names(3)
    values = ctx.values(3)                      # descending
    pairs = list(zip(names, values))
    ranked = [name for name, _ in sorted(pairs, key=lambda p: p[1], reverse=True)]
    answer = ranked[1]

    premises = ctx.shuffled([ctx.fact(name, value) for name, value in pairs])
    question = ctx.question("q_middle")
    steps = [ctx.step("fact_step", E=name, v=value) for name, value in pairs]
    steps.append(ctx.step("order_step", ordered=", ".join(ranked)))
    steps.append(ctx.step("conclusion_step", answer=answer))
    return premises, question, steps, answer


def verify_claim(ctx):
    """Is a stated comparative claim true?  Balanced yes / no."""
    a, b = ctx.names(2)
    va, vb = ctx.values(2)
    if ctx.rng.random() < 0.5:
        a, b = b, a
        va, vb = vb, va

    claim_more = ctx.rng.random() < 0.5         # claim: a is {more/less} than b
    claim_true = (va > vb) if claim_more else (va < vb)
    answer = ctx.yes if claim_true else ctx.no

    premises = ctx.shuffled([ctx.fact(a, va), ctx.fact(b, vb)])
    question = ctx.question("q_verify", A=a, B=b,
                            cmp=ctx.more if claim_more else ctx.less)
    hi_name, lo_name = (a, b) if va > vb else (b, a)
    steps = [
        ctx.step("fact_step", E=a, v=va),
        ctx.step("fact_step", E=b, v=vb),
        ctx.step("rel_step", A=hi_name, B=lo_name, cmp=ctx.more),
        ctx.step("conclusion_step", answer=answer),
    ]
    return premises, question, steps, answer


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSpec:
    """One reasoning task: its builder, its premise style, and its weight."""

    id: str
    builder: Callable
    fact_kind: str          # "value" (numbers stated) or "relation" (inequalities stated)
    hops: int               # deduction depth, for the dataset statistics table
    weight: float           # relative sampling frequency
    description: str


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("compare_two_more",     compare_two_more,     "value",    1, 1.0,
             "Two stated values; which entity is greater."),
    TaskSpec("compare_two_less",     compare_two_less,     "value",    1, 1.0,
             "Two stated values; which entity is smaller."),
    TaskSpec("compare_equal",        compare_equal,        "value",    1, 0.8,
             "Two stated values; are they equal (yes/no)."),
    TaskSpec("numeric_difference",   numeric_difference,   "value",    1, 0.8,
             "Two stated values; by how much one exceeds the other."),
    TaskSpec("verify_claim",         verify_claim,         "value",    1, 0.8,
             "Verify a comparative claim against stated values (yes/no)."),
    TaskSpec("middle_entity",        middle_entity,        "value",    2, 0.8,
             "Three stated values; which is in the middle."),
    TaskSpec("order_ascending",      order_ascending,      "value",    2, 0.9,
             "Sort 3-4 entities from smallest to largest."),
    TaskSpec("order_descending",     order_descending,     "value",    2, 0.9,
             "Sort 3-4 entities from largest to smallest."),
    TaskSpec("transitive_relation",  transitive_relation,  "relation", 2, 1.3,
             "A>B, B>C: relation between A and C."),
    TaskSpec("transitive_largest",   transitive_largest,   "relation", 2, 1.1,
             "Largest of three from chained inequalities."),
    TaskSpec("transitive_smallest",  transitive_smallest,  "relation", 2, 1.1,
             "Smallest of three from chained inequalities."),
    TaskSpec("chain_relation",       chain_relation,       "relation", 3, 1.2,
             "Multi-hop relation over a 4-5 entity chain."),
    TaskSpec("chain_largest",        chain_largest,        "relation", 3, 1.0,
             "Largest over a 4-5 entity chain."),
    TaskSpec("chain_smallest",       chain_smallest,       "relation", 3, 1.0,
             "Smallest over a 4-5 entity chain."),
    TaskSpec("mixed_direction_chain", mixed_direction_chain, "relation", 3, 1.2,
             "Chain with mixed more/less premises requiring normalisation."),
)

TASK_IDS: tuple[str, ...] = tuple(t.id for t in TASKS)
TASK_BY_ID: dict[str, TaskSpec] = {t.id: t for t in TASKS}
