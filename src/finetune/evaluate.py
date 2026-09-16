"""
Reasoning Evaluation: Pretrained vs Finetuned (Phase 3)
=======================================================

Measures what finetuning actually bought, on the held-out synthetic reasoning
test split: the model is shown only the question, generates its own chain of
thought, and its final answer is compared to the ground truth the generator
computed in Python.

The brief asks for "pretrained vs. finetuned accuracy (or exact match) on the
synthetic test set", so both arms run through *identical* prompting, decoding
and answer-extraction code. The pretrained arm is expected to score near zero —
it has never seen the ``प्रश्न / तर्क / उत्तर`` format — and that is the point
of the comparison, not a bug.

Why exact match. Every answer is a closed-form string the generator knows
exactly: an entity name, a yes/no word, a number with its unit, or a
comma-joined ordering. There is no paraphrase to tolerate, so exact match
after whitespace normalisation is both the strictest and the fairest metric.
Two softer diagnostics are reported alongside it to separate *reasoning*
failure from *format* failure:

    - ``format_ok``     the model emitted the answer label at all,
    - ``answer_in_text`` the right answer appears somewhere in the output,
                        even if the final answer line was wrong.

Batching. Generation is autoregressive, so examples are grouped by exact
prompt length and decoded in batches. Equal lengths mean no padding is needed,
which matters here: this model uses *learned absolute* positional embeddings,
so left-padding a batch would shift every real token onto the wrong position
and silently corrupt the results.

Invoked through ``python main.py evaluate-reasoning``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict

import torch
from tokenizers import Tokenizer

from src.languages import Language
from src.model.transformer import build_model_from_config

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def normalise_answer(text: str) -> str:
    """Collapse whitespace and strip trailing punctuation for comparison.

    Deliberately conservative: it does not lowercase (meaningless for
    Devanagari), does not strip the danda from inside an answer, and does not
    reorder a comma-joined list — an ordering task answered in the wrong order
    is wrong.
    """
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.rstrip("।.!?,: ").strip()


def extract_answer(generated: str, answer_label: str) -> tuple[str, bool]:
    """Pull the final answer out of a generated continuation.

    Args:
        generated:    Text the model produced after the prompt.
        answer_label: In-language answer marker, e.g. ``"उत्तर"``.

    Returns:
        ``(answer, format_ok)``. ``format_ok`` is False when the model never
        emitted the answer label, in which case the first line is used as a
        best-effort answer so the pretrained arm is not scored as vacuously
        empty.
    """
    marker = f"{answer_label}:"
    if marker in generated:
        tail = generated.split(marker, 1)[1]
        # The answer is the remainder of that line.
        return normalise_answer(tail.split("\n", 1)[0]), True
    return normalise_answer(generated.split("\n", 1)[0]), False


# ---------------------------------------------------------------------------
# Batched greedy decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, input_ids: torch.Tensor, max_new_tokens: int,
                   eos_token_id: int, max_seq_len: int) -> list:
    """Greedily decode a batch of equal-length prompts.

    Args:
        model:          The model, in eval mode.
        input_ids:      (B, T) prompt tokens — all rows the same length.
        max_new_tokens: Cap on generated tokens.
        eos_token_id:   Stop token.
        max_seq_len:    Model's positional-embedding limit.

    Returns:
        A list of B token-id lists, each holding only the newly generated
        tokens (the prompt is not repeated back).
    """
    batch_size = input_ids.size(0)
    device = input_ids.device
    generated = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    ids = input_ids
    budget = min(max_new_tokens, max_seq_len - input_ids.size(1))

    for _ in range(max(0, budget)):
        logits, _ = model(ids)                       # (B, T, V)
        next_token = logits[:, -1, :].argmax(dim=-1)  # greedy

        # Once a row has emitted EOS, keep feeding EOS so shapes stay uniform
        # but stop recording its output.
        for i in range(batch_size):
            if not finished[i]:
                token = next_token[i].item()
                if token == eos_token_id:
                    finished[i] = True
                else:
                    generated[i].append(token)

        ids = torch.cat([ids, next_token.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break

    return generated


def load_model(lang: Language, checkpoint_path: str, device):
    """Load a checkpoint into a model built from its own stored config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    step = checkpoint.get("step", "?")
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, config, step


# ---------------------------------------------------------------------------
# Evaluation of one checkpoint
# ---------------------------------------------------------------------------

def evaluate_checkpoint(lang: Language, checkpoint_path: str, records: list,
                        arm: str, batch_size: int = 16,
                        max_new_tokens: int = 160) -> dict:
    """Score one checkpoint on the reasoning test split.

    Args:
        lang:            Language being evaluated.
        checkpoint_path: Checkpoint to score.
        records:         Test examples (dicts from the reasoning JSONL).
        arm:             ``"pretrained"`` or ``"finetuned"``, for logging.
        batch_size:      Prompts decoded at once (within a length group).
        max_new_tokens:  Generation cap per example.

    Returns:
        A results dict with overall and per-task accuracy plus every
        prediction, for the qualitative error analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_model(lang, checkpoint_path, device)
    tokenizer = Tokenizer.from_file(str(lang.tokenizer_path))

    bos_token_id = config.get("bos_token_id", 2)
    eos_token_id = config.get("eos_token_id", 3)
    max_seq_len = config.get("max_seq_len", 512)
    answer_label = records[0].get("answer_label", "उत्तर")

    # Group by exact prompt length so batches need no padding.
    by_length = defaultdict(list)
    encodings = tokenizer.encode_batch([r["prompt_text"] for r in records])
    for record, encoding in zip(records, encodings):
        ids = [bos_token_id] + encoding.ids
        if len(ids) >= max_seq_len:
            continue
        by_length[len(ids)].append((record, ids))

    predictions = []
    start = time.time()
    total = sum(len(v) for v in by_length.values())
    done = 0

    for length in sorted(by_length):
        group = by_length[length]
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            batch_ids = torch.tensor([ids for _, ids in chunk],
                                     dtype=torch.long, device=device)
            outputs = generate_batch(model, batch_ids, max_new_tokens,
                                     eos_token_id, max_seq_len)

            for (record, _), token_ids in zip(chunk, outputs):
                text = tokenizer.decode(token_ids)
                answer, format_ok = extract_answer(text, answer_label)
                gold = normalise_answer(record["answer"])
                predictions.append({
                    "id": record.get("id", ""),
                    "task": record["task"],
                    "hops": record.get("hops", 0),
                    "prompt": record["prompt"],
                    "gold": gold,
                    "predicted": answer,
                    "correct": answer == gold,
                    "format_ok": format_ok,
                    "answer_in_text": gold in text,
                    "generated": text[:400],
                })

            done += len(chunk)
            if done % (batch_size * 20) < batch_size:
                logger.info("  [%s] %d/%d decoded (%.0fs elapsed)",
                            arm, done, total, time.time() - start)

    # ----- Aggregate -----
    n = len(predictions)
    correct = sum(p["correct"] for p in predictions)

    per_task = defaultdict(lambda: {"n": 0, "correct": 0})
    per_hops = defaultdict(lambda: {"n": 0, "correct": 0})
    for p in predictions:
        per_task[p["task"]]["n"] += 1
        per_task[p["task"]]["correct"] += int(p["correct"])
        per_hops[str(p["hops"])]["n"] += 1
        per_hops[str(p["hops"])]["correct"] += int(p["correct"])

    def with_accuracy(table: dict) -> dict:
        return {
            key: {**value,
                  "accuracy": round(value["correct"] / value["n"], 4) if value["n"] else 0.0}
            for key, value in sorted(table.items())
        }

    results = {
        "arm": arm,
        "language": lang.key,
        "model": lang.model_label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "examples": n,
        "exact_match": round(correct / n, 4) if n else 0.0,
        "correct": correct,
        "format_ok_rate": round(sum(p["format_ok"] for p in predictions) / n, 4) if n else 0.0,
        "answer_in_text_rate": round(
            sum(p["answer_in_text"] for p in predictions) / n, 4) if n else 0.0,
        "by_task": with_accuracy(per_task),
        "by_hops": with_accuracy(per_hops),
        "decode_seconds": round(time.time() - start, 1),
        "predictions": predictions,
    }

    logger.info("[%s] %s exact match: %.2f%% (%d/%d) | format ok: %.1f%%",
                lang.display, arm, 100 * results["exact_match"], correct, n,
                100 * results["format_ok_rate"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_qualitative_examples(lang: Language, finetuned: dict, n_each: int = 8) -> None:
    """Write successes and failures for the report's error analysis."""
    path = lang.report_dir / "reasoning_qualitative.txt"
    path.parent.mkdir(parents=True, exist_ok=True)

    predictions = finetuned["predictions"]
    successes = [p for p in predictions if p["correct"]]
    failures = [p for p in predictions if not p["correct"]]

    # Spread the failures across tasks rather than showing eight of the same.
    by_task = defaultdict(list)
    for p in failures:
        by_task[p["task"]].append(p)
    spread_failures = [ps[0] for ps in by_task.values()][:n_each]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{lang.display} ({lang.model_label}) — finetuned reasoning model\n")
        f.write(f"exact match {100 * finetuned['exact_match']:.2f}% "
                f"over {finetuned['examples']} held-out test examples\n")
        f.write("=" * 78 + "\n\n")

        f.write(f"SUCCESSES ({len(successes)} total, showing {min(n_each, len(successes))})\n")
        f.write("-" * 78 + "\n")
        for p in successes[:n_each]:
            f.write(f"\n[{p['task']}, {p['hops']} hop]\n{p['prompt']}\n")
            f.write(f"  model: {p['generated'][:240]}\n")
            f.write(f"  gold={p['gold']!r}  predicted={p['predicted']!r}  OK\n")

        f.write(f"\n\nFAILURES ({len(failures)} total, one per task, "
                f"showing {len(spread_failures)})\n")
        f.write("-" * 78 + "\n")
        for p in spread_failures:
            f.write(f"\n[{p['task']}, {p['hops']} hop]\n{p['prompt']}\n")
            f.write(f"  model: {p['generated'][:240]}\n")
            f.write(f"  gold={p['gold']!r}  predicted={p['predicted']!r}  WRONG"
                    f"{'  (no answer label emitted)' if not p['format_ok'] else ''}\n")


def write_comparison_table(lang: Language, pretrained: dict, finetuned: dict) -> str:
    """Write the pretrained-vs-finetuned markdown table for the report."""
    path = lang.report_dir / "reasoning_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Reasoning results — {lang.display} ({lang.model_label})",
        "",
        f"Held-out synthetic test split: **{finetuned['examples']} examples**, "
        f"entity names and question phrasings disjoint from training.",
        "",
        "## Pretrained vs finetuned",
        "",
        "| Metric | Pretrained | Finetuned | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("Exact match", "exact_match"),
                       ("Emitted answer label", "format_ok_rate"),
                       ("Gold answer appears in output", "answer_in_text_rate")):
        pre, fine = pretrained[key], finetuned[key]
        lines.append(f"| {label} | {100 * pre:.2f}% | {100 * fine:.2f}% "
                     f"| {100 * (fine - pre):+.2f} pp |")

    lines += [
        "",
        f"Pretrained checkpoint: `{pretrained['checkpoint']}` (step {pretrained['checkpoint_step']})  ",
        f"Finetuned checkpoint: `{finetuned['checkpoint']}` (step {finetuned['checkpoint_step']})",
        "",
        "## Accuracy by task",
        "",
        "| Task | Hops | n | Pretrained | Finetuned |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, stats in finetuned["by_task"].items():
        pre = pretrained["by_task"].get(task, {}).get("accuracy", 0.0)
        hops = next((p["hops"] for p in finetuned["predictions"] if p["task"] == task), "")
        lines.append(f"| `{task}` | {hops} | {stats['n']} | "
                     f"{100 * pre:.1f}% | {100 * stats['accuracy']:.1f}% |")

    lines += ["", "## Accuracy by reasoning depth", "",
              "| Hops | n | Pretrained | Finetuned |", "|---:|---:|---:|---:|"]
    for hops, stats in finetuned["by_hops"].items():
        pre = pretrained["by_hops"].get(hops, {}).get("accuracy", 0.0)
        lines.append(f"| {hops} | {stats['n']} | {100 * pre:.1f}% "
                     f"| {100 * stats['accuracy']:.1f}% |")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_reasoning_eval(lang: Language, pretrained_checkpoint: str,
                       finetuned_checkpoint: str, limit: int | None = None,
                       batch_size: int = 16, max_new_tokens: int = 160,
                       skip_pretrained: bool = False) -> dict:
    """Score both arms on the reasoning test split and write the report files.

    Args:
        lang:                  Language to evaluate.
        pretrained_checkpoint: Phase 2 checkpoint (the baseline arm).
        finetuned_checkpoint:  Phase 3 checkpoint.
        limit:                 Optional cap on test examples.
        batch_size:            Prompts decoded at once.
        max_new_tokens:        Generation cap per example.
        skip_pretrained:       Reuse a cached baseline instead of re-decoding it.

    Returns:
        A summary dict, also written to
        ``report/<language>/reasoning_metrics.json``.

    Raises:
        SystemExit: If the reasoning test split has not been generated.
    """
    test_path = lang.reasoning_split_file("test")
    if not test_path.exists():
        raise SystemExit(
            f"{test_path} not found — run "
            f"python main.py reasoning-data --lang {lang.key} first."
        )


    records = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break

    logger.info("Evaluating %s (%s) on %d reasoning test examples",
                lang.display, lang.model_label, len(records))

    metrics_path = lang.report_dir / "reasoning_metrics.json"
    pretrained = None
    if skip_pretrained and metrics_path.exists():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        pretrained = cached.get("pretrained")
        if pretrained:
            logger.info("Reusing cached pretrained-baseline results.")
    if pretrained is None:
        pretrained = evaluate_checkpoint(lang, pretrained_checkpoint, records,
                                         "pretrained", batch_size, max_new_tokens)

    finetuned = evaluate_checkpoint(lang, finetuned_checkpoint, records,
                                    "finetuned", batch_size, max_new_tokens)

    write_qualitative_examples(lang, finetuned)
    table_path = write_comparison_table(lang, pretrained, finetuned)

    summary = {
        "language": lang.key,
        "model": lang.model_label,
        "tier": lang.tier,
        "test_examples": len(records),
        "pretrained": pretrained,
        "finetuned": finetuned,
        "improvement_pp": round(100 * (finetuned["exact_match"] - pretrained["exact_match"]), 2),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Reasoning metrics -> %s", metrics_path)
    logger.info("Comparison table  -> %s", table_path)
    logger.info("Qualitative cases -> %s", lang.report_dir / "reasoning_qualitative.txt")
    return summary


def main():
    import argparse
    from src import languages
    from src.train.utils import find_latest_checkpoint, resolve_checkpoint
    from src.finetune import attention_compare

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate reasoning finetuning and attention.")
    languages.add_language_arg(parser)
    parser.add_argument("--pretrained", default=None,
                        help="Pretrained checkpoint (default: newest in checkpoints/).")
    parser.add_argument("--finetuned", default=None,
                        help="Finetuned checkpoint (default: newest in checkpoints_finetune/).")
    parser.add_argument("--only", nargs="+", choices=["accuracy", "attention"],
                        default=["accuracy", "attention"], help="Run only these stages.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N test examples.")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size",
                        help="Prompts decoded at once (default: %(default)s).")
    parser.add_argument("--max-new-tokens", type=int, default=160, dest="max_new_tokens",
                        help="Generation cap per example (default: %(default)s).")
    parser.add_argument("--skip-pretrained", action="store_true",
                        help="Reuse cached pretrained-baseline scores.")
    parser.add_argument("--prompt", default=None,
                        help="Sentence for attention comparison.")
    args = parser.parse_args()

    lang = languages.get(args.lang)
    pretrained = resolve_checkpoint(lang, args.pretrained)

    finetuned = args.finetuned
    if not finetuned:
        finetuned = find_latest_checkpoint(str(lang.finetune_checkpoint_dir))
        if not finetuned:
            raise SystemExit(
                f"No finetuned checkpoint in {lang.finetune_checkpoint_dir}.\n"
                f"Run: python -m src.finetune.trainer --lang {lang.key}"
            )

    logging.info("Pretrained: %s", pretrained)
    logging.info("Finetuned:  %s", finetuned)

    summary = None
    if "accuracy" in args.only:
        summary = run_reasoning_eval(
            lang, pretrained, finetuned, limit=args.limit,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            skip_pretrained=args.skip_pretrained,
        )

    if "attention" in args.only:
        attention_compare.run_attention_comparison(
            lang, pretrained, finetuned, sentence=args.prompt)

    if summary:
        print("\n" + "=" * 62)
        print(f" Reasoning accuracy — {summary['model']} ({summary['language']})")
        print("=" * 62)
        print(f"  test examples      {summary['test_examples']}")
        print(f"  pretrained         {100 * summary['pretrained']['exact_match']:.2f}% exact match")
        print(f"  finetuned          {100 * summary['finetuned']['exact_match']:.2f}% exact match")
        print(f"  improvement        {summary['improvement_pp']:+.2f} pp")
        print(f"\n  tables -> {lang.report_dir / 'reasoning_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

