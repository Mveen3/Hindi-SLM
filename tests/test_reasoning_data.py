#!/usr/bin/env python3
"""
Ground-Truth Verification for the Phase 3 Reasoning Datasets
============================================================

The generator computes each answer from hidden values *before* rendering the
sentences.  This test closes the loop from the other side: it reads only the
emitted text, re-extracts the entity names and numbers from the rendered
prompt, recomputes the expected answer independently, and compares.

That catches the failure mode that matters for a finetuning corpus — a correct
computation paired with the wrong sentence, or a template that drops a premise
— which a check inside the generator could not.

Also asserts the dataset-level properties the report claims: disjoint entity
names and phrasings across splits, no prompt shared between splits, balanced
yes/no tasks, and no ``<unk>`` under the frozen Phase 1 tokenizer.

    python tests/test_reasoning_data.py                 # both languages
    python tests/test_reasoning_data.py --lang hindi
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import languages
from src.data.reasoning import tasks

DEVANAGARI_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


def to_int(text: str) -> int:
    """Parse an integer written in either ASCII or Devanagari digits."""
    return int(text.translate(DEVANAGARI_TO_ASCII))


def load_split(path: Path) -> list:
    """Read one JSONL split."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_pairs(prompt: str, names: list) -> list:
    """Recover (name, value) premises from a rendered prompt.

    Finds each known entity name and takes the first number after it, which is
    exactly how every ``fact`` template is shaped.

    Two subtleties, both of which produced false failures before being handled:
    a name must begin at a word boundary (Devanagari "राम" otherwise matches
    inside "किलोग्राम"), and longer names are matched first with their span
    masked out, so a name that is a prefix of another cannot double-count.

    Args:
        prompt: The rendered prompt text.
        names:  Candidate entity names for this example's pool.

    Returns:
        A list of ``(name, value)`` tuples in order of appearance.
    """
    remaining = list(prompt)
    found = []
    for name in sorted(names, key=len, reverse=True):
        haystack = "".join(remaining)
        # A premise names an entity at the start of a word, never mid-word.
        match = re.search(rf"(?:(?<=^)|(?<=[\s।?,:]))" + re.escape(name), haystack)
        if not match:
            continue
        number = re.search(r"[\d०-९]+", haystack[match.end():])
        if not number:
            continue
        found.append((match.start(), name, to_int(number.group())))
        for index in range(match.start(), match.end()):
            remaining[index] = "\x00"          # mask, so prefixes cannot re-match
    return [(name, value) for _, name, value in sorted(found)]


# ---------------------------------------------------------------------------
# Per-task independent recomputation
# ---------------------------------------------------------------------------

def expected_answer(record: dict, pairs: list, yes: str, no: str):
    """Recompute the answer for a value-fact task from the extracted premises.

    Returns:
        The expected answer string, or ``None`` if this task cannot be checked
        from surface text alone (the relation-fact tasks state no numbers).
    """
    task = record["task"]
    if not pairs:
        return None
    ordered = sorted(pairs, key=lambda p: p[1], reverse=True)

    if task == "compare_two_more":
        return ordered[0][0]
    if task == "compare_two_less":
        return ordered[-1][0]
    if task == "compare_equal":
        return yes if ordered[0][1] == ordered[-1][1] else no
    if task == "numeric_difference":
        return ordered[0][1] - ordered[-1][1]          # compared numerically
    if task == "middle_entity":
        return ordered[len(ordered) // 2][0] if len(ordered) == 3 else None
    if task == "order_ascending":
        return ", ".join(name for name, _ in reversed(ordered))
    if task == "order_descending":
        return ", ".join(name for name, _ in ordered)
    return None


def check_language(key: str) -> int:
    """Verify one language's reasoning corpus.

    Returns:
        The number of failures found.
    """
    lang = languages.get(key)
    print(f"\n{'=' * 62}\n {lang.display} ({lang.model_label})\n{'=' * 62}")

    with open(lang.reasoning_phrasebank, encoding="utf-8") as f:
        bank = json.load(f)
    yes, no = bank["answers"]["yes"], bank["answers"]["no"]
    all_names = {pool: bank["entities"][pool] for pool in tasks.ENTITY_POOLS}

    failures = 0
    splits = {}
    for split in ("train", "val", "test"):
        path = lang.reasoning_split_file(split)
        if not path.exists():
            print(f"  MISSING {path} — run python main.py reasoning-data first")
            return 1
        splits[split] = load_split(path)

    # -- 1. Label correctness, recomputed from the rendered text -------------
    checked = Counter()
    for split, records in splits.items():
        for record in records:
            pairs = extract_pairs(record["prompt"], all_names[record["entity_pool"]])
            expected = expected_answer(record, pairs, yes, no)
            if expected is None:
                continue
            checked[record["task"]] += 1

            actual = record["answer"]
            if isinstance(expected, int):
                numbers = re.findall(r"[\d०-९]+", actual)
                ok = bool(numbers) and to_int(numbers[0]) == expected
            else:
                ok = actual == expected
            if not ok:
                failures += 1
                if failures <= 5:
                    print(f"  LABEL MISMATCH [{split} {record['id']} {record['task']}]")
                    print(f"    prompt   : {record['prompt']}")
                    print(f"    premises : {pairs}")
                    print(f"    expected : {expected!r}  got: {actual!r}")

    print(f"  labels recomputed from text : {sum(checked.values()):,} examples "
          f"across {len(checked)} value-fact tasks -> "
          f"{'all correct' if failures == 0 else f'{failures} MISMATCHES'}")

    # -- 2. The answer must be recoverable from the emitted sequence ---------
    format_errors = 0
    for split, records in splits.items():
        for record in records:
            if record["prompt_text"] + record["target"] != record["text"]:
                format_errors += 1
            elif not record["text"].rstrip().endswith(record["answer"]):
                format_errors += 1
    failures += format_errors
    print(f"  prompt/target/text consistency: "
          f"{'ok' if format_errors == 0 else f'{format_errors} BROKEN'}")

    # -- 3. Relation-fact answers must be an entity named in the prompt ------
    unnamed = 0
    for split, records in splits.items():
        for record in records:
            if record["fact_kind"] != "relation":
                continue
            if record["answer"] not in record["prompt"]:
                unnamed += 1
    failures += unnamed
    print(f"  chained answers appear in their prompt: "
          f"{'ok' if unnamed == 0 else f'{unnamed} BROKEN'}")

    # -- 4. Balance of the yes/no tasks --------------------------------------
    for task in ("compare_equal", "verify_claim"):
        answers = Counter(r["answer"] for records in splits.values()
                          for r in records if r["task"] == task)
        total = sum(answers.values())
        if total:
            share = answers.get(yes, 0) / total
            verdict = "balanced" if 0.35 <= share <= 0.65 else "SKEWED"
            print(f"  {task:<16} yes-share = {share:.2f} over {total:,} examples ({verdict})")
            if verdict == "SKEWED":
                failures += 1

    # -- 5. Split disjointness ----------------------------------------------
    prompts = {split: {r["prompt"] for r in records} for split, records in splits.items()}
    overlap = sum(len(prompts[a] & prompts[b]) for a, b in
                  (("train", "val"), ("train", "test"), ("val", "test")))
    failures += overlap
    print(f"  prompts shared between splits : "
          f"{'0 (clean)' if overlap == 0 else f'{overlap} LEAKED'}")

    stats_path = lang.report_dir / "reasoning_dataset_stats.json"
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        clean = stats["leakage"]["clean"]
        failures += 0 if clean else 1
        print(f"  held-out names and phrasings  : "
              f"{'disjoint' if clean else 'LEAKAGE'}")
        for split in ("train", "val", "test"):
            token_stats = stats["token_stats"][split]
            if token_stats.get("available"):
                unk = token_stats["unk_tokens"]
                failures += 0 if unk == 0 else 1
                print(f"  {split:<5} max {token_stats['max_tokens']:>4} tokens, "
                      f"{unk} <unk> {'(ok)' if unk == 0 else '(BROKEN)'}")

    # -- 6. Task coverage ----------------------------------------------------
    present = {r["task"] for records in splits.values() for r in records}
    missing = set(tasks.TASK_IDS) - present
    failures += len(missing)
    print(f"  task coverage : {len(present)}/{len(tasks.TASK_IDS)}"
          + (f" MISSING {sorted(missing)}" if missing else " (all present)"))

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--lang", choices=(*languages.CHOICES, "both"), default="both")
    args = parser.parse_args()

    keys = languages.CHOICES if args.lang == "both" else (args.lang,)
    total = sum(check_language(key) for key in keys)

    print(f"\n{'=' * 62}")
    if total == 0:
        print(" PASS — every checkable label was reproduced independently.")
    else:
        print(f" FAIL — {total} problem(s) found.")
    print("=" * 62)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
