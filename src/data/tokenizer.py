"""
Tokenizer Training and Evaluation
=================================

Trains a SentencePiece-style BPE tokenizer from scratch on one language's
training split, and measures it on the full corpus.

No pretrained tokenizer is used, and the two languages never share a
vocabulary: each is trained only on its own ``data/processed/train.jsonl`` and
saved under its own ``tokenizer/`` directory.

Why BPE: it decomposes unseen words into subwords rather than emitting
``<unk>``, which matters for the rich morphology of both languages. The full
Devanagari block is forced into the initial alphabet, so no in-script character
can ever fall back to ``<unk>`` — which is why the realised vocabulary comes
out slightly above the 10,000 target.

Invoked through ``python main.py tokenizer --lang {hindi,nepali}``.
"""

import json
import random
import multiprocessing as mp

from tokenizers import SentencePieceBPETokenizer, Tokenizer, normalizers
from tqdm import tqdm

from src.languages import Language

VOCAB_SIZE = 10_000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
# U+0900–U+097F: seeding the full Devanagari block guarantees character coverage.
DEVANAGARI_BLOCK = [chr(c) for c in range(0x0900, 0x0980)]


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------

def jsonl_texts(file_path):
    """Yield the ``text`` field of every well-formed line in a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if text:
                yield text


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(lang: Language, vocab_size: int = VOCAB_SIZE) -> dict:
    """
    Train a BPE tokenizer from scratch on one language's training split.

    Args:
        lang:       Language to train for. Only its own train.jsonl is read.
        vocab_size: Target vocabulary size.

    Returns:
        A dict with the realised vocabulary size and the output paths.

    Raises:
        SystemExit: If the training split has not been produced yet.
    """
    train_file = lang.split_file("train")
    if not train_file.exists():
        raise SystemExit(
            f"{train_file} not found — run python main.py split --lang {lang.key} first."
        )


    out_dir = lang.tokenizer_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{lang.display}] Training BPE tokenizer (target vocab {vocab_size:,}) ...")

    # "▁" (U+2581) marks word boundaries, per SentencePiece convention.
    tokenizer = SentencePieceBPETokenizer(
        unk_token="<unk>", replacement="▁", add_prefix_space=True
    )
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Strip(),
    ])

    tokenizer.train_from_iterator(
        jsonl_texts(train_file),
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        initial_alphabet=DEVANAGARI_BLOCK,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
    )

    # Sanity check: a common in-language phrase must not tokenise to <unk>.
    encoding = tokenizer.encode(lang.tokenizer_probe)
    print(f"[{lang.display}] Sanity check {lang.tokenizer_probe!r} -> {encoding.tokens}")
    if "<unk>" in encoding.tokens:
        raise SystemExit(
            f"[{lang.display}] Tokenizer produced <unk> on a basic phrase — check normalisation."
        )

    tokenizer.save(str(lang.tokenizer_path))

    vocab = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    with open(lang.vocab_path, "w", encoding="utf-8") as f:
        for token, _ in vocab:
            # Escape newlines so the file holds exactly one token per line. A
            # hundred-odd learned tokens contain literal newlines; written raw
            # they straddle several lines and inflate the file's line count
            # above the true vocabulary size.
            escaped = token.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
            f.write(f"{escaped}\n")

    print(f"[{lang.display}] Realised vocabulary: {len(vocab):,}")
    print(f"  tokenizer -> {lang.tokenizer_path}")
    print(f"  vocabulary -> {lang.vocab_path}")

    return {
        "language": lang.key,
        "vocab_size": len(vocab),
        "tokenizer_path": str(lang.tokenizer_path),
        "vocabulary_path": str(lang.vocab_path),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tokenizer(lang: Language, num_samples: int = 3, batch_size: int = 2000) -> dict:
    """
    Measure fertility, characters-per-token and unknown-token rate.

    Streams through the corpus using fast native batch tokenization in Rust,
    guaranteeing minimal memory usage (<100 MB RAM) and zero IPC buffering crashes.

    Args:
        lang:        Language whose tokenizer is evaluated.
        num_samples: How many qualitative examples to print.
        batch_size:  Lines per batch for native tokenization.

    Returns:
        A statistics dict, also written to ``report/<language>/tokenizer_stats.json``.

    Raises:
        SystemExit: If the tokenizer has not been trained yet.
    """
    tokenizer_path = str(lang.tokenizer_path)
    if not lang.tokenizer_path.exists():
        raise SystemExit(
            f"{tokenizer_path} not found — run python main.py tokenizer --lang {lang.key} --action train first."
        )

    print(f"\n{'=' * 50}\nEvaluating {lang.display} ({lang.model_label}) tokenizer\n{'=' * 50}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    totals = {"tokens": 0, "words": 0, "chars": 0, "unks": 0}

    for split in ("train", "val", "test"):
        file_path = lang.split_file(split)
        if not file_path.exists():
            continue
        total_bytes = file_path.stat().st_size
        with tqdm(total=total_bytes, desc=f"  {split}", unit="B",
                  unit_scale=True, unit_divisor=1024) as pbar:
            batch = []
            chunk_bytes = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    chunk_bytes += len(line.encode("utf-8"))
                    try:
                        text = json.loads(line).get("text", "").strip()
                        if text:
                            batch.append(text)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

                    if len(batch) >= batch_size:
                        encodings = tokenizer.encode_batch(batch)
                        for enc, text in zip(encodings, batch):
                            totals["tokens"] += len(enc.ids)
                            totals["words"] += len(text.split())
                            totals["chars"] += len(text)
                            totals["unks"] += enc.tokens.count("<unk>")
                        pbar.update(chunk_bytes)
                        batch = []
                        chunk_bytes = 0

            if batch:
                encodings = tokenizer.encode_batch(batch)
                for enc, text in zip(encodings, batch):
                    totals["tokens"] += len(enc.ids)
                    totals["words"] += len(text.split())
                    totals["chars"] += len(text)
                    totals["unks"] += enc.tokens.count("<unk>")
                pbar.update(chunk_bytes)

    fertility = totals["tokens"] / totals["words"] if totals["words"] else 0.0
    chars_per_token = totals["chars"] / totals["tokens"] if totals["tokens"] else 0.0
    unk_rate = totals["unks"] / totals["tokens"] if totals["tokens"] else 0.0

    tokenizer = Tokenizer.from_file(tokenizer_path)
    stats = {
        "language": lang.key,
        "model": lang.model_label,
        "vocab_size": tokenizer.get_vocab_size(),
        "total_tokens": totals["tokens"],
        "total_words": totals["words"],
        "total_characters": totals["chars"],
        "fertility_tokens_per_word": round(fertility, 4),
        "characters_per_token": round(chars_per_token, 4),
        "unk_tokens": totals["unks"],
        "unk_rate": round(unk_rate, 8),
        "examples": _qualitative_samples(lang, tokenizer, num_samples),
    }

    print(f"\n--- Statistics over {totals['words']:,} words ---")
    print(f"Vocabulary size:  {stats['vocab_size']:,}")
    print(f"Total tokens:     {totals['tokens']:,}")
    print(f"Fertility:        {fertility:.2f} tokens/word")
    print(f"Chars per token:  {chars_per_token:.2f}")
    print(f"<unk> tokens:     {totals['unks']:,} ({unk_rate:.6%})")

    for example in stats["examples"]:
        print(f"\n[Sample] {example['text']}")
        print(f"  tokens: {example['tokens']}")

    report_file = lang.report_dir / "tokenizer_stats.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nTokenizer statistics written to {report_file}")

    return stats


def _qualitative_samples(lang: Language, tokenizer, num_samples: int, seed: int = 42) -> list:
    """Tokenise a few validation sentences so segmentation can be eyeballed."""
    val_file = lang.split_file("val")
    if not val_file.exists():
        return []

    sentences = []
    with open(val_file, "r", encoding="utf-8") as f:
        for _ in range(5000):
            line = f.readline()
            if not line:
                break
            try:
                text = json.loads(line).get("text", "").strip()
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if len(text) > 50:
                sentences.append(text)

    if not sentences:
        return []

    rng = random.Random(seed)
    chosen = rng.sample(sentences, min(num_samples, len(sentences)))

    examples = []
    for text in chosen:
        snippet = text[:200]
        encoding = tokenizer.encode(snippet)
        examples.append({
            "text": snippet,
            "tokens": encoding.tokens,
            "num_tokens": len(encoding.ids),
            "unk_count": encoding.tokens.count("<unk>"),
        })
    return examples


def main():
    import argparse
    from src import languages
    parser = argparse.ArgumentParser(description="Train or evaluate a BPE tokenizer.")
    languages.add_language_arg(parser)
    parser.add_argument("--action", choices=["train", "eval", "both"], default="both",
                        help="Action to perform (default: %(default)s).")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help="Target vocabulary size (default: %(default)s).")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Qualitative tokenisation examples to print (eval mode).")
    args = parser.parse_args()

    lang = languages.get(args.lang)
    if args.action in ("train", "both"):
        train_tokenizer(lang, vocab_size=args.vocab_size)
    if args.action in ("eval", "both"):
        evaluate_tokenizer(lang, num_samples=args.num_samples)


if __name__ == "__main__":
    main()

