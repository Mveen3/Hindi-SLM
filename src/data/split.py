"""
Train / Validation / Test Splitting
===================================

Partitions one language's cleaned corpus into reproducible 95 / 2.5 / 2.5
splits and records the resulting corpus statistics.

The split is a seeded per-document coin flip rather than a shuffle, so the
whole corpus never has to be held in memory and the assignment is identical on
every re-run for a given seed.

This stage writes ``report/<language>/corpus_stats.json`` — machine-readable
numbers only. The prose that cites them lives in ``report/report.md``
and is never overwritten by a script.

Invoked through ``python main.py split --lang {hindi,nepali}``.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from src.languages import Language

SPLIT_RATIOS = {"train": 0.95, "val": 0.025, "test": 0.025}
DEFAULT_SEED = 42
DEFAULT_NUM_BUCKETS = 32


def split_corpus(lang: Language, seed: int = DEFAULT_SEED, num_buckets: int = DEFAULT_NUM_BUCKETS) -> dict:
    """
    Split the cleaned corpus into pre-shuffled train/val/test and collect statistics.

    Shuffling is performed at partition time via out-of-core bucketed shuffling,
    ensuring train.jsonl, val.jsonl, and test.jsonl are completely shuffled on disk
    and ready to feed directly into the transformer without runtime shuffle buffers.

    Args:
        lang:        Language to split. Reads ``data/interim/<lang>_cleaned.jsonl``
                     and writes ``data/processed/{train,val,test}.jsonl``.
        seed:        RNG seed controlling the split assignment and shuffle (default 42).
        num_buckets: Number of intermediate partition buckets for out-of-core train shuffle.

    Returns:
        A statistics dictionary with per-split document and byte counts.

    Raises:
        SystemExit: If the cleaned corpus has not been produced yet.
    """
    cleaned_file = lang.interim_file
    if not cleaned_file.exists():
        raise SystemExit(
            f"{cleaned_file} not found — run python main.py clean --lang {lang.key} first."
        )

    out_dir = lang.split_file("train").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_shuffle_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in SPLIT_RATIOS}
    byte_counts = {name: 0 for name in SPLIT_RATIOS}

    train_cut = SPLIT_RATIOS["train"]
    val_cut = train_cut + SPLIT_RATIOS["val"]

    print(f"Splitting & shuffling {lang.display} ({lang.model_label}) 95/2.5/2.5 with seed {seed} ...")
    rng = random.Random(seed)

    # 1. Open intermediate bucket files for train and temp files for val/test
    tmp_val_path = tmp_dir / "val_raw.jsonl"
    tmp_test_path = tmp_dir / "test_raw.jsonl"
    val_handle = open(tmp_val_path, "w", encoding="utf-8")
    test_handle = open(tmp_test_path, "w", encoding="utf-8")
    train_handles = [
        open(tmp_dir / f"train_bucket_{i:03d}.jsonl", "w", encoding="utf-8")
        for i in range(num_buckets)
    ]

    try:
        with open(cleaned_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                r = rng.random()
                if r < train_cut:
                    b_idx = rng.randrange(num_buckets)
                    train_handles[b_idx].write(line)
                    counts["train"] += 1
                    byte_counts["train"] += len(line.encode("utf-8"))
                elif r < val_cut:
                    val_handle.write(line)
                    counts["val"] += 1
                    byte_counts["val"] += len(line.encode("utf-8"))
                else:
                    test_handle.write(line)
                    counts["test"] += 1
                    byte_counts["test"] += len(line.encode("utf-8"))
    finally:
        val_handle.close()
        test_handle.close()
        for h in train_handles:
            h.close()

    # 2. Shuffle and write validation split
    if tmp_val_path.exists():
        with open(tmp_val_path, "r", encoding="utf-8") as f:
            val_lines = f.readlines()
        rng.shuffle(val_lines)
        with open(lang.split_file("val"), "w", encoding="utf-8") as f:
            f.writelines(val_lines)
        del val_lines
        tmp_val_path.unlink()

    # 3. Shuffle and write test split
    if tmp_test_path.exists():
        with open(tmp_test_path, "r", encoding="utf-8") as f:
            test_lines = f.readlines()
        rng.shuffle(test_lines)
        with open(lang.split_file("test"), "w", encoding="utf-8") as f:
            f.writelines(test_lines)
        del test_lines
        tmp_test_path.unlink()

    # 4. Shuffle and write training split (bucket-by-bucket out-of-core shuffle)
    bucket_indices = list(range(num_buckets))
    rng.shuffle(bucket_indices)
    with open(lang.split_file("train"), "w", encoding="utf-8") as f_train_out:
        for b_idx in bucket_indices:
            b_path = tmp_dir / f"train_bucket_{b_idx:03d}.jsonl"
            if b_path.exists():
                with open(b_path, "r", encoding="utf-8") as f_b:
                    bucket_lines = f_b.readlines()
                rng.shuffle(bucket_lines)
                f_train_out.writelines(bucket_lines)
                del bucket_lines
                b_path.unlink()

    # Cleanup temp directory
    shutil.rmtree(tmp_dir, ignore_errors=True)

    stats = {
        "language": lang.key,
        "model": lang.model_label,
        "seed": seed,
        "split_ratios": SPLIT_RATIOS,
        "documents": counts,
        "bytes": byte_counts,
        "total_documents": sum(counts.values()),
        "total_bytes": sum(byte_counts.values()),
        "total_mb": round(sum(byte_counts.values()) / (1024 ** 2), 2),
    }

    report_file = lang.report_dir / "corpus_stats.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    for name in SPLIT_RATIOS:
        print(f"  {name:<5} {counts[name]:>10,} docs  "
              f"{byte_counts[name] / (1024 ** 2):>10.2f} MB")
    print(f"Pre-shuffled corpus statistics written to {report_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Split and shuffle cleaned corpus into train/val/test.")
    from src import languages
    languages.add_language_arg(parser)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Split and shuffle seed (default: %(default)s).")
    parser.add_argument("--num-buckets", type=int, default=DEFAULT_NUM_BUCKETS,
                        help="Intermediate partition buckets for out-of-core shuffle (default: %(default)s).")
    args = parser.parse_args()

    split_corpus(languages.get(args.lang), seed=args.seed, num_buckets=args.num_buckets)


if __name__ == "__main__":
    main()
