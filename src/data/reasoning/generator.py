"""
Synthetic Reasoning Dataset Generation (Phase 3)
================================================

Turns a language's phrase bank into its finetuning corpus: train / validation /
test JSONL splits of comparative and transitive reasoning problems, plus the
statistics table the report needs.

Ground truth
------------
Every example starts from hidden numeric values sampled here.  The answer is
computed by plain Python comparison or sorting *before* any wording is chosen,
so the label is correct by construction and independent of the phrasing (and of
the LLM that authored the phrasing).

Leakage control
---------------
Two independent generalisation axes are held out, both partitioned disjointly
across the three splits:

* **Entity names** — the names a test question talks about were never seen
  during finetuning, so the model cannot memorise "राम is older than श्याम".
* **Premise and question phrasings** — the ``fact``, ``rel_fact`` and ``q_*``
  template pools are partitioned too, so the test set asks in sentence patterns
  the model never trained on.  Chain-of-thought *step* phrasings are shared on
  purpose: they are the output style being taught, and the metric scores only
  the final answer.

Numeric values are drawn from the same ranges in every split (holding those out
would test extrapolation, not reasoning), and exact prompt de-duplication is
applied globally so no prompt can appear in two splits.

Invoked through ``python main.py reasoning-data``.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass

from src.languages import Language

from . import tasks
from .tasks import TASK_BY_ID, TASKS, format_number

SPLITS = ("train", "val", "test")

#: Smallest entity pool / template pool a split may end up with.
MIN_NAMES_PER_SPLIT = 3
MIN_TEMPLATES_PER_SPLIT = 1

_MULTISPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([।?,:%])")


# ---------------------------------------------------------------------------
# Deterministic disjoint partitioning
# ---------------------------------------------------------------------------

def partition(items: list, fractions: dict, seed: int, floor: int) -> dict:
    """Split a list into disjoint per-split lists, giving every split a floor.

    Pure proportional splitting starves the small splits (15% of a 12-name pool
    is one name, too few to build a 3-entity comparison), so each split is first
    guaranteed ``floor`` items and only the remainder is shared out by fraction.

    Args:
        items:     The pool to partition.  Not mutated; duplicates are dropped.
        fractions: Split name -> share of the remainder.
        seed:      Seed for the shuffle, so the partition is reproducible.
        floor:     Minimum items per split.

    Returns:
        ``{split: [items]}`` with pairwise-disjoint values.

    Raises:
        SystemExit: If the pool is too small to give every split its floor.
    """
    # De-duplicate first: a name or template listed twice would otherwise be
    # dealt into two different splits and quietly break disjointness.
    shuffled = list(dict.fromkeys(items))
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    if total < floor * len(fractions):
        raise SystemExit(
            f"Cannot partition {total} items across {len(fractions)} splits with a "
            f"floor of {floor} each — the phrase bank needs more entries."
        )

    counts = {name: floor for name in fractions}
    remainder = total - floor * len(fractions)
    weight_sum = sum(fractions.values()) or 1.0

    assigned = 0
    names = list(fractions)
    for name in names[:-1]:
        take = int(remainder * fractions[name] / weight_sum)
        counts[name] += take
        assigned += take
    counts[names[-1]] += remainder - assigned

    out, cursor = {}, 0
    for name in names:
        out[name] = shuffled[cursor:cursor + counts[name]]
        cursor += counts[name]
    return out


# ---------------------------------------------------------------------------
# Per-split resources
# ---------------------------------------------------------------------------

@dataclass
class SplitResources:
    """The entity names and templates one split is allowed to use.

    Attributes:
        split:     Split name.
        entities:  ``{pool: [names]}`` — disjoint from the other splits.
        templates: ``{slot: [templates]}`` — held-out slots are disjoint from
                   the other splits; step slots are shared.
    """

    split: str
    entities: dict
    templates: dict


def build_split_resources(bank: dict, entity_fractions: dict,
                          template_fractions: dict, seed: int) -> dict:
    """Partition the phrase bank's names and templates across the three splits.

    Args:
        bank:               The language's phrase bank.
        entity_fractions:   Split shares for entity names.
        template_fractions: Split shares for held-out template slots.
        seed:               Base seed; each pool/slot gets a derived seed so one
                            split does not systematically get the first names.

    Returns:
        ``{split: SplitResources}``.
    """
    entities = {name: {} for name in SPLITS}
    for index, pool in enumerate(tasks.ENTITY_POOLS):
        parts = partition(bank["entities"][pool], entity_fractions,
                          seed + 101 * (index + 1), MIN_NAMES_PER_SPLIT)
        for split in SPLITS:
            entities[split][pool] = parts[split]

    templates = {name: {} for name in SPLITS}
    for index, slot in enumerate(tasks.ALL_SLOTS):
        pool = bank["templates"][slot.name]
        if slot.splittable:
            parts = partition(pool, template_fractions,
                              seed + 977 * (index + 1), MIN_TEMPLATES_PER_SPLIT)
        else:
            parts = {split: list(pool) for split in SPLITS}   # shared reasoning style
        for split in SPLITS:
            templates[split][slot.name] = parts[split]

    return {split: SplitResources(split, entities[split], templates[split])
            for split in SPLITS}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def tidy(text: str) -> str:
    """Normalise whitespace so templates with optional placeholders read cleanly.

    An empty ``{unit}`` or a template that puts two placeholders side by side
    would otherwise leave double spaces or a space before the danda.
    """
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


class TaskContext:
    """Everything a task builder needs to make one example, in one language.

    Holds the split's entity pool and template pool, the chosen attribute, and
    the seeded RNG.  Task builders in ``tasks.py`` talk only to this object, so
    they stay language-agnostic and label-safe.

    Args:
        rng:            Seeded RNG driving every sampling decision.
        bank:           The language's phrase bank.
        resources:      The current split's allowed names and templates.
        attribute:      The attribute being compared in this example.
        numeral_script: ``"ascii"`` or ``"devanagari"``.
    """

    def __init__(self, rng: random.Random, bank: dict, resources: SplitResources,
                 attribute: tasks.AttributeSpec, numeral_script: str):
        self.rng = rng
        self.bank = bank
        self.resources = resources
        self.attribute = attribute
        self.numeral_script = numeral_script

        self.words = bank["attributes"][attribute.key]
        self.pool = resources.entities[attribute.pool]
        self.unit = self.words["unit"]
        self.more = self.words["more"]
        self.less = self.words["less"]
        self.yes = bank["answers"]["yes"]
        self.no = bank["answers"]["no"]

        self._grid = list(range(attribute.low, attribute.high + 1, attribute.step))
        self._used_names: list[str] = []

    # -- sampling -----------------------------------------------------------

    def pool_size(self) -> int:
        """How many distinct entity names this split may draw on."""
        return len(self.pool)

    def pick_n(self, options) -> int:
        """Choose an entity count from ``options``, clamped to the pool size.

        Held-out splits get small name pools, so a 5-entity chain is sometimes
        impossible; falling back to the longest feasible chain keeps the split
        populated instead of failing.
        """
        feasible = [n for n in options if n <= self.pool_size()]
        if feasible:
            return self.rng.choice(feasible)
        return max(2, min(self.pool_size(), min(options)))

    def names(self, n: int) -> list[str]:
        """Draw ``n`` distinct entity names from this split's pool.

        Raises:
            SystemExit: If the split's pool is smaller than ``n``.
        """
        if self.pool_size() < n:
            raise SystemExit(
                f"Entity pool '{self.attribute.pool}' for split '{self.resources.split}' has "
                f"{self.pool_size()} names but {n} are needed — add names to the phrase bank."
            )
        self._used_names = self.rng.sample(self.pool, n)
        return list(self._used_names)

    def values(self, n: int, min_gap: int | None = None) -> list[int]:
        """Draw ``n`` distinct values in descending order, well separated.

        A minimum gap keeps comparisons unambiguous once numbers are rounded to
        the attribute's step, and stops "which is larger" from hinging on a
        one-unit difference the model cannot be expected to resolve.
        """
        if n == 1:
            return [self.rng.choice(self._grid)]

        gap = min_gap if min_gap is not None else 2 * self.attribute.step
        for _ in range(80):
            picked = sorted(self.rng.sample(self._grid, n), reverse=True)
            if all(picked[i] - picked[i + 1] >= gap for i in range(n - 1)):
                return picked

        # Deterministic fallback: evenly spaced values across the whole range.
        span = self.attribute.high - self.attribute.low
        stride = max(gap, span // max(1, n - 1))
        base = self.rng.randint(self.attribute.low, max(self.attribute.low,
                                                        self.attribute.high - stride * (n - 1)))
        return sorted((base + i * stride for i in range(n)), reverse=True)

    def shuffled(self, items: list) -> list:
        """Return a shuffled copy, so premise order carries no information."""
        copied = list(items)
        self.rng.shuffle(copied)
        return copied

    def num(self, value: int) -> str:
        """Render a number in the numeral script this language's corpus uses."""
        return format_number(value, self.numeral_script)

    # -- rendering ----------------------------------------------------------

    def _render(self, slot: str, **extra) -> str:
        """Fill a random template of ``slot`` with the attribute words plus extras."""
        template = self.rng.choice(self.resources.templates[slot])
        values = dict(self.words)
        for key in ("v", "vA", "vB", "d"):
            if key in extra and isinstance(extra[key], int):
                extra[key] = self.num(extra[key])
        values.update(extra)
        needed = {name.strip("{}") for name in tasks.SLOTS[slot].allowed}
        return tidy(template.format(**{k: values.get(k, "") for k in needed}))

    def fact(self, name: str, value: int) -> str:
        """Render a premise stating an entity's numeric value."""
        return self._render("fact", E=name, v=value)

    def rel(self, a: str, b: str, cmp: str) -> str:
        """Render a premise stating that ``a``'s attribute is ``cmp`` than ``b``'s."""
        return self._render("rel_fact", A=a, B=b, cmp=cmp)

    def question(self, slot: str, **kwargs) -> str:
        """Render the question sentence for a task."""
        return self._render(slot, **kwargs)

    def step(self, slot: str, **kwargs) -> str:
        """Render one chain-of-thought step."""
        return self._render(slot, **kwargs)


# ---------------------------------------------------------------------------
# Example assembly
# ---------------------------------------------------------------------------

def _normalise_prompt(text: str) -> str:
    """Key used for exact de-duplication across the whole corpus."""
    return _MULTISPACE_RE.sub(" ", text).strip()


def make_example(rng: random.Random, bank: dict, resources: SplitResources,
                 task: tasks.TaskSpec, numeral_script: str,
                 with_reasoning: bool) -> dict:
    """Build one finetuning example for a given task.

    Args:
        rng:            Seeded RNG.
        bank:           The language's phrase bank.
        resources:      The current split's names and templates.
        task:           Which reasoning task to instantiate.
        numeral_script: Numeral script for values.
        with_reasoning: Emit a chain-of-thought target, or a bare answer.

    Returns:
        A record with the rendered prompt, the reasoning trace, the exact
        answer, the assembled training text, and metadata for the stats table.
    """
    attribute = rng.choice(tasks.ATTRIBUTES)
    ctx = TaskContext(rng, bank, resources, attribute, numeral_script)
    premises, question, steps, answer = task.builder(ctx)

    prompt = tidy(" ".join(premises) + " " + question)
    reasoning = tidy(" ".join(steps))

    labels = bank["labels"]
    q_label, r_label, a_label = labels["question"], labels["reasoning"], labels["answer"]

    if with_reasoning:
        prompt_text = f"{q_label}: {prompt}\n{r_label}:"
        target = f" {reasoning}\n{a_label}: {answer}"
        style = "cot"
    else:
        prompt_text = f"{q_label}: {prompt}\n{a_label}:"
        target = f" {answer}"
        style = "direct"

    return {
        "task": task.id,
        "fact_kind": task.fact_kind,
        "hops": task.hops,
        "style": style,
        "attribute": attribute.key,
        "entity_pool": attribute.pool,
        "n_entities": len(ctx._used_names),
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": answer,
        "prompt_text": prompt_text,
        "target": target,
        "answer_label": a_label,
        "text": prompt_text + target,
    }


def _task_sampler(rng: random.Random, task_weights: dict | None):
    """Return a callable drawing tasks according to their configured weights."""
    specs, weights = [], []
    for task in TASKS:
        weight = (task_weights or {}).get(task.id, task.weight)
        if weight and weight > 0:
            specs.append(task)
            weights.append(float(weight))
    if not specs:
        raise SystemExit("Every task weight is zero — nothing to generate.")
    return lambda: rng.choices(specs, weights=weights, k=1)[0]


def generate_split(lang: Language, bank: dict, resources: SplitResources,
                   split: str, target: int, seed: int, numeral_script: str,
                   task_weights: dict | None, direct_answer_fraction: float,
                   seen_prompts: set, tokenizer=None,
                   max_tokens: int | None = None) -> list[dict]:
    """Generate one split's examples, skipping any prompt already produced.

    Args:
        lang:                   The language being generated.
        bank:                   Its phrase bank.
        resources:              This split's allowed names and templates.
        split:                  Split name, used in the example ids.
        target:                 How many examples to produce.
        seed:                   Split-specific RNG seed.
        numeral_script:         Numeral script for values.
        task_weights:           Optional per-task weight overrides.
        direct_answer_fraction: Share of examples with no chain of thought.
        seen_prompts:           Global set of prompt keys already used; mutated.
        tokenizer:              This language's frozen tokenizer, for the length filter.
        max_tokens:             Reject examples longer than this many tokens.

    Returns:
        The generated records, each with an ``id`` assigned.
    """
    rng = random.Random(seed)
    pick_task = _task_sampler(rng, task_weights)

    records: list[dict] = []
    collisions = 0
    oversized = 0
    attempt_budget = target * 60 + 2000

    while len(records) < target and attempt_budget > 0:
        attempt_budget -= 1
        task = pick_task()
        with_reasoning = rng.random() >= direct_answer_fraction
        record = make_example(rng, bank, resources, task, numeral_script, with_reasoning)

        key = _normalise_prompt(record["prompt"])
        if key in seen_prompts:
            collisions += 1
            continue

        # A sequence longer than the pretrained context window would be
        # truncated during finetuning, cutting the answer off the end of its
        # own chain of thought. Drop it rather than train on a broken target.
        if tokenizer is not None and max_tokens:
            n_tokens = len(tokenizer.encode(record["text"]).ids)
            if n_tokens > max_tokens:
                oversized += 1
                continue
            record["n_tokens"] = n_tokens

        seen_prompts.add(key)
        record["id"] = f"{lang.key}-{split}-{len(records):06d}"
        records.append(record)

    if len(records) < target:
        print(f"    [warn] {split}: produced {len(records):,} of {target:,} requested — "
              f"the split's name/template pool cannot express more unique prompts.")
    if collisions or oversized:
        print(f"    {split}: rejected {collisions:,} duplicate prompt(s) and "
              f"{oversized:,} over-length example(s) during sampling")
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def load_tokenizer(lang: Language):
    """Load this language's frozen Phase 1 tokenizer, or None if unavailable.

    Args:
        lang: The language whose tokenizer to load.

    Returns:
        A ``Tokenizer``, or ``None`` if the package or the file is missing.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    if not lang.tokenizer_path.exists():
        return None
    return Tokenizer.from_file(str(lang.tokenizer_path))


def _token_stats(lang: Language, records: list[dict], sample_size: int = 2000) -> dict:
    """Measure finetuning sequence lengths with this language's own tokenizer.

    Confirms the examples fit the pretrained context window and that the
    reasoning text stays inside the Phase 1 vocabulary (no ``<unk>``), which is
    required since the tokenizer must not change during finetuning.
    """
    tokenizer = load_tokenizer(lang)
    if tokenizer is None:
        return {"available": False,
                "reason": f"tokenizers package or {lang.tokenizer_path} unavailable"}

    rng = random.Random(0)
    sample = records if len(records) <= sample_size else rng.sample(records, sample_size)

    lengths, unk_examples, unk_tokens = [], 0, 0
    for record in sample:
        tokens = tokenizer.encode(record["text"]).tokens
        lengths.append(len(tokens))
        count = tokens.count("<unk>")
        if count:
            unk_examples += 1
            unk_tokens += count

    lengths.sort()
    return {
        "available": True,
        "sampled": len(sample),
        "mean_tokens": round(sum(lengths) / len(lengths), 1),
        "median_tokens": lengths[len(lengths) // 2],
        "p95_tokens": lengths[int(0.95 * (len(lengths) - 1))],
        "max_tokens": lengths[-1],
        "examples_with_unk": unk_examples,
        "unk_tokens": unk_tokens,
    }


def _split_stats(records: list[dict]) -> dict:
    """Per-split composition counts for the report table."""
    return {
        "examples": len(records),
        "unique_prompts": len({_normalise_prompt(r["prompt"]) for r in records}),
        "unique_answers": len({r["answer"] for r in records}),
        "by_task": dict(sorted(Counter(r["task"] for r in records).items())),
        "by_fact_kind": dict(sorted(Counter(r["fact_kind"] for r in records).items())),
        "by_hops": {str(k): v for k, v in sorted(Counter(r["hops"] for r in records).items())},
        "by_attribute": dict(sorted(Counter(r["attribute"] for r in records).items())),
        "by_style": dict(sorted(Counter(r["style"] for r in records).items())),
        "by_n_entities": {str(k): v for k, v in
                          sorted(Counter(r["n_entities"] for r in records).items())},
        "mean_prompt_chars": round(sum(len(r["prompt"]) for r in records) / max(1, len(records)), 1),
        "mean_reasoning_chars": round(
            sum(len(r["reasoning"]) for r in records) / max(1, len(records)), 1),
    }


def _leakage_report(resources: dict, datasets: dict) -> dict:
    """Verify — rather than assume — that the held-out axes really are disjoint.

    Returns counts of shared entity names, shared templates and shared prompts
    between every pair of splits.  Every number here must be zero.
    """
    def pairs():
        return (("train", "val"), ("train", "test"), ("val", "test"))

    shared_entities = {}
    for pool in tasks.ENTITY_POOLS:
        for left, right in pairs():
            overlap = set(resources[left].entities[pool]) & set(resources[right].entities[pool])
            if overlap:
                shared_entities[f"{pool}:{left}-{right}"] = sorted(overlap)

    shared_templates = {}
    for slot in tasks.ALL_SLOTS:
        if not slot.splittable:
            continue
        for left, right in pairs():
            overlap = (set(resources[left].templates[slot.name])
                       & set(resources[right].templates[slot.name]))
            if overlap:
                shared_templates[f"{slot.name}:{left}-{right}"] = len(overlap)

    prompt_keys = {split: {_normalise_prompt(r["prompt"]) for r in records}
                   for split, records in datasets.items()}
    shared_prompts = {f"{left}-{right}": len(prompt_keys[left] & prompt_keys[right])
                      for left, right in pairs()}

    return {
        "entity_names_shared_between_splits": shared_entities,
        "held_out_templates_shared_between_splits": shared_templates,
        "prompts_shared_between_splits": shared_prompts,
        "clean": not shared_entities and not shared_templates
                 and all(v == 0 for v in shared_prompts.values()),
    }


def _resource_stats(resources: dict) -> dict:
    """How many names and templates each split was allowed to use."""
    return {
        "entity_pool_sizes": {split: {pool: len(res.entities[pool])
                                      for pool in tasks.ENTITY_POOLS}
                              for split, res in resources.items()},
        "template_pool_sizes": {split: {slot.name: len(res.templates[slot.name])
                                        for slot in tasks.ALL_SLOTS}
                                for split, res in resources.items()},
        "held_out_slots": list(tasks.SPLITTABLE_SLOTS),
        "shared_slots": [s.name for s in tasks.ALL_SLOTS if not s.splittable],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_jsonl(path, records: list[dict]) -> None:
    """Write records as one JSON object per line, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(lang: Language, bank: dict, config: dict,
                  authoring_report: dict | None = None) -> dict:
    """Generate, write and describe one language's reasoning finetuning corpus.

    Args:
        lang:             Which model's language to build for.
        bank:             That language's phrase bank.
        config:           Its ``reasoning_config.yaml`` contents.
        authoring_report: Optional phrase-bank authoring summary, folded into
                          the statistics so the report can cite API usage.

    Returns:
        The full statistics dictionary (also written to
        ``report/<language>/reasoning_dataset_stats.json``).
    """
    seed = int(config.get("seed", 1337))
    numeral_script = config.get("numeral_script", "ascii")
    sizes = config.get("splits", {"train": 30000, "val": 2000, "test": 2000})
    entity_fractions = config.get("entity_split_fractions",
                                  {"train": 0.6, "val": 0.15, "test": 0.25})
    template_fractions = config.get("template_split_fractions",
                                    {"train": 0.6, "val": 0.15, "test": 0.25})
    task_weights = config.get("task_weights")
    direct_fraction = float(config.get("direct_answer_fraction", 0.0))
    max_tokens = config.get("max_sequence_tokens")
    max_tokens = int(max_tokens) if max_tokens else None

    tokenizer = load_tokenizer(lang)
    if max_tokens and tokenizer is None:
        print(f"  [warn] {lang.tokenizer_path} unavailable — cannot enforce the "
              f"{max_tokens}-token limit; sequence lengths will not be filtered.")

    print(f"\n[{lang.display} / {lang.model_label}] building reasoning dataset "
          f"(seed {seed}, numerals: {numeral_script})")

    resources = build_split_resources(bank, entity_fractions, template_fractions, seed)
    for split in SPLITS:
        pools = ", ".join(f"{pool}={len(resources[split].entities[pool])}"
                          for pool in tasks.ENTITY_POOLS)
        print(f"  {split:<5} name pools: {pools}")

    seen_prompts: set = set()
    datasets = {}
    # Test first, then val: the smaller held-out splits claim their prompts
    # before the large train split can accidentally consume them.
    for offset, split in enumerate(("test", "val", "train")):
        records = generate_split(
            lang, bank, resources[split], split, int(sizes.get(split, 0)),
            seed + 7919 * (offset + 1), numeral_script, task_weights,
            direct_fraction, seen_prompts, tokenizer, max_tokens,
        )
        datasets[split] = records
        path = lang.reasoning_split_file(split)
        write_jsonl(path, records)
        print(f"  wrote {len(records):>7,} examples -> {path}")

    stats = {
        "language": lang.key,
        "model": lang.model_label,
        "tier": lang.tier,
        "seed": seed,
        "numeral_script": numeral_script,
        "split_sizes_requested": {k: int(v) for k, v in sizes.items()},
        "direct_answer_fraction": direct_fraction,
        "max_sequence_tokens": max_tokens,
        "tasks": {t.id: {"fact_kind": t.fact_kind, "hops": t.hops,
                         "weight": (task_weights or {}).get(t.id, t.weight),
                         "description": t.description} for t in TASKS},
        "phrasebank": {
            "source": bank.get("source"),
            "llm_model": bank.get("llm_model"),
            "generated_at": bank.get("generated_at"),
            "templates_per_slot": {slot.name: len(bank["templates"][slot.name])
                                   for slot in tasks.ALL_SLOTS},
            "total_templates": sum(len(v) for v in bank["templates"].values()),
            "entity_pool_sizes": {pool: len(bank["entities"][pool])
                                  for pool in tasks.ENTITY_POOLS},
            "attributes": len(bank["attributes"]),
        },
        "authoring": authoring_report or {},
        "resources": _resource_stats(resources),
        "splits": {split: _split_stats(datasets[split]) for split in SPLITS},
        "token_stats": {split: _token_stats(lang, datasets[split]) for split in SPLITS},
        "leakage": _leakage_report(resources, datasets),
    }

    stats_path = lang.report_dir / "reasoning_dataset_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  statistics -> {stats_path}")

    samples_path = lang.reasoning_dir / "samples.txt"
    _write_samples(samples_path, datasets["test"])
    print(f"  qualitative samples -> {samples_path}")

    return stats


def _write_samples(path, records: list[dict], per_task: int = 2) -> None:
    """Write a few readable examples per task for the report's appendix."""
    by_task: dict[str, list[dict]] = {}
    for record in records:
        by_task.setdefault(record["task"], []).append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task_id in tasks.TASK_IDS:
            chosen = by_task.get(task_id, [])[:per_task]
            if not chosen:
                continue
            spec = TASK_BY_ID[task_id]
            f.write(f"{'=' * 78}\n{task_id}  ({spec.fact_kind}, {spec.hops} hop)\n"
                    f"{spec.description}\n{'=' * 78}\n")
            for record in chosen:
                f.write(f"\n{record['text']}\n")
                f.write(f"  [answer={record['answer']!r} attribute={record['attribute']} "
                        f"entities={record['n_entities']}]\n")
            f.write("\n")


def print_summary(stats: dict) -> None:
    """Print the headline dataset numbers as a small aligned table."""
    print(f"\n  {stats['model']} — {stats['language']} reasoning dataset")
    print(f"  {'split':<7}{'examples':>10}{'unique':>10}{'mean tok':>10}{'max tok':>9}{'unk':>6}")
    for split in SPLITS:
        split_stats = stats["splits"][split]
        tokens = stats["token_stats"][split]
        mean_tokens = tokens.get("mean_tokens", "-")
        max_tokens = tokens.get("max_tokens", "-")
        unk = tokens.get("unk_tokens", "-")
        print(f"  {split:<7}{split_stats['examples']:>10,}"
              f"{split_stats['unique_prompts']:>10,}{str(mean_tokens):>10}"
              f"{str(max_tokens):>9}{str(unk):>6}")
    leakage = stats["leakage"]
    verdict = "clean (no shared names, templates or prompts)" if leakage["clean"] else "LEAKAGE DETECTED"
    print(f"  leakage check: {verdict}")
    print(f"  templates in phrase bank: {stats['phrasebank']['total_templates']} "
          f"({stats['phrasebank']['source']})")


def main():
    import argparse
    import os
    import sys
    from src import languages
    from src.data.reasoning import phrasebank

    parser = argparse.ArgumentParser(description="Generate synthetic reasoning datasets.")
    parser.add_argument("--lang", choices=(*languages.CHOICES, "both"), default="both",
                        help="Which model's data to generate (default: both).")
    parser.add_argument("--offline", action="store_true",
                        help="Never call Groq; use the hand-authored seed phrase bank.")
    parser.add_argument("--refresh-phrasebank", action="store_true",
                        help="Re-author phrase bank even if cached exists.")
    parser.add_argument("--phrasebank-only", action="store_true",
                        help="Build phrase bank only, without examples.")
    parser.add_argument("--train", type=int, help="Override train split size.")
    parser.add_argument("--val", type=int, help="Override val split size.")
    parser.add_argument("--test", type=int, help="Override test split size.")
    parser.add_argument("--seed", type=int, help="Override seed.")
    parser.add_argument("--config", help="Path to reasoning_config.yaml override.")
    parser.add_argument("--api-key", help="Groq API key.")
    args = parser.parse_args()

    # Load .env if present
    env_file = languages.PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    has_key = bool(args.api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY"))
    if not args.offline and not has_key:
        print("  No Groq API key found — falling back to offline seed phrase bank.")
        args.offline = True

    keys = tuple(languages.CHOICES) if args.lang == "both" else (args.lang,)
    for key in keys:
        lang = languages.get(key)
        config = lang.load_reasoning_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        sizes = config.setdefault("splits", {})
        for split in ("train", "val", "test"):
            v = getattr(args, split)
            if v is not None:
                sizes[split] = v

        print("\n" + "=" * 62)
        print(f" {lang.display} — {lang.model_label} ({lang.tier})")
        print("=" * 62)

        bank, report = phrasebank.build_phrasebank(
            lang,
            llm_config=config.get("llm"),
            offline=args.offline,
            refresh=args.refresh_phrasebank,
            api_key=args.api_key,
        )

        if not args.phrasebank_only:
            stats = build_dataset(lang, bank, config, authoring_report=report.as_dict())
            print_summary(stats)


if __name__ == "__main__":
    main()

