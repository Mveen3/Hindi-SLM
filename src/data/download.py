"""
Curated Corpus Download
=======================

Streams the curated portion of one language's pretraining corpus from the
HuggingFace Hub into ``<language>/data/raw/curated/``.

Which datasets to pull is not decided here: the list lives in
``<language>/configs/data_sources.yaml`` and is read through
``Language.load_sources()``, so this module contains no language names. Sources
are consumed in priority order and the run stops once the byte target is met.

Downloads are checkpointed per source, so an interrupted run resumes where it
left off rather than starting over.

Invoked through ``python main.py download --lang {hindi,nepali}``.
"""

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.languages import Language, PROJECT_ROOT

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Default token target per language for curated data:
# 900M tokens curated + ~200M crawled tokens ensures total corpus exceeds 1B tokens (> 1.1B total).
DEFAULT_MAX_TOKENS = 900_000_000

# Average bytes per token for Devanagari text (rough estimate).
# Devanagari characters are typically 3 bytes in UTF-8, and a BPE token
# covers ~2-3 characters on average, so ~4-6 characters (~10 bytes per token).
BYTES_PER_TOKEN_ESTIMATE = 10

# Default byte safety threshold derived from 900M tokens: ~9 GB (9 * 1024^3 bytes = ~9.66 GB)
DEFAULT_MAX_BYTES = 9 * 1024 * 1024 * 1024

# Flush interval: write to disk every N records
FLUSH_INTERVAL = 10_000

# Log format
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions & Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string."""
    return max(1, len(text.encode("utf-8")) // BYTES_PER_TOKEN_ESTIMATE)


def format_tokens(n: int) -> str:
    """Format token count for display."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_bytes(n: int) -> str:
    """Format byte count for display."""
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.2f} GB"
    elif n >= 1_048_576:
        return f"{n / 1_048_576:.2f} MB"
    elif n >= 1_024:
        return f"{n / 1_024:.1f} KB"
    return f"{n} B"


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=[
        logging.StreamHandler(sys.stdout),
    ])


def load_hf_token(env_path: str = None) -> str:
    """Load HuggingFace token from the project-root .env file."""
    load_dotenv(env_path or PROJECT_ROOT / ".env")
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        logging.warning("No HF_TOKEN found in .env — some datasets may not be accessible.")
        return None
    logging.info(f"HF_TOKEN loaded (ends with ...{token[-4:]})")
    return token


def sanitize_last_line_if_needed(file_path: Path):
    """Ensure the last line of a file ends with a newline before appending."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return
    with open(file_path, "rb+") as f:
        f.seek(max(0, file_path.stat().st_size - 2))
        tail = f.read()
        if tail and not tail.endswith(b"\n"):
            f.write(b"\n")


def inspect_existing_file(file_path: Path) -> tuple[int, int, int]:
    """
    Inspect an existing JSONL file to count rows, estimated tokens, and bytes.
    Returns (rows, tokens, bytes).
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return 0, 0, 0
    
    rows = 0
    tokens = 0
    bytes_count = file_path.stat().st_size
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                text = data.get("text", "")
                if text:
                    tokens += estimate_tokens(text)
                    rows += 1
            except Exception:
                continue
                
    return rows, tokens, bytes_count


def save_checkpoint(name: str, output_dir: Path, data: dict):
    """Atomically save source download checkpoint to disk."""
    checkpoint_path = output_dir / f".{name}.checkpoint.json"
    temp_path = output_dir / f".{name}.checkpoint.json.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(checkpoint_path)
    except Exception as e:
        logging.debug(f"Failed to save checkpoint for {name}: {e}")


def load_checkpoint(name: str, output_dir: Path, output_file: Path) -> dict:
    """
    Load checkpoint from checkpoint file, manifest, or by inspecting existing output file.
    """
    checkpoint_path = output_dir / f".{name}.checkpoint.json"
    manifest_path = output_dir / "manifest.json"

    # 1. Check sidecar checkpoint file
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            if output_file.exists() and output_file.stat().st_size >= ckpt.get("bytes", 0):
                return ckpt
        except Exception:
            pass

    # 2. Check existing manifest.json
    if manifest_path.exists() and output_file.exists() and output_file.stat().st_size > 0:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for s in manifest.get("sources", []):
                if s.get("name") == name and s.get("status") == "completed":
                    if s.get("bytes", 0) == output_file.stat().st_size:
                        ckpt = {
                            "name": name,
                            "rows": s.get("rows", 0),
                            "raw_records_seen": s.get("rows", 0),
                            "tokens": s.get("tokens", 0),
                            "bytes": s.get("bytes", 0),
                            "status": "completed",
                            "updated_at": datetime.now().isoformat(),
                        }
                        save_checkpoint(name, output_dir, ckpt)
                        return ckpt
        except Exception:
            pass

    # 3. If file exists without metadata, inspect file directly
    if output_file.exists() and output_file.stat().st_size > 0:
        rows, tokens, bytes_count = inspect_existing_file(output_file)
        ckpt = {
            "name": name,
            "rows": rows,
            "raw_records_seen": rows,
            "tokens": tokens,
            "bytes": bytes_count,
            "status": "in_progress",
            "updated_at": datetime.now().isoformat(),
        }
        save_checkpoint(name, output_dir, ckpt)
        return ckpt

    return {
        "name": name,
        "rows": 0,
        "raw_records_seen": 0,
        "tokens": 0,
        "bytes": 0,
        "status": "not_started",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Download Engine
# ──────────────────────────────────────────────────────────────────────────────

def download_source(
    source: dict,
    output_dir: Path,
    max_tokens: int | None,
    current_tokens: int,
    max_bytes: int | None,
    current_bytes: int,
    hf_token: str | None,
    dry_run: bool = False,
    force_redownload: bool = False,
) -> dict:
    """
    Download a single dataset source and save as JSONL with checkpointing.

    Returns a dict with download statistics.
    """
    import itertools
    from datasets import load_dataset

    name = source["name"]
    dataset_id = source["dataset_id"]
    config = source.get("config")
    data_dir = source.get("data_dir")
    split = source.get("split", "train")
    text_field = source.get("text_field", "text")
    requires_token = source.get("requires_token", False)

    target_already_reached = False
    if max_tokens is not None and current_tokens >= max_tokens:
        target_already_reached = True
    elif max_bytes is not None and current_bytes >= max_bytes:
        target_already_reached = True

    if target_already_reached:
        logging.info(f"  ⏭  Skipping {name} — target already reached.")
        return {"name": name, "status": "skipped_target_reached", "tokens": 0, "rows": 0, "bytes": 0}

    output_file = output_dir / f"{name}.jsonl"

    # Check for existing checkpoint / progress
    ckpt = {"rows": 0, "raw_records_seen": 0, "tokens": 0, "bytes": 0, "status": "not_started"}
    if not force_redownload:
        ckpt = load_checkpoint(name, output_dir, output_file)

    existing_rows = ckpt.get("rows", 0)
    existing_raw_seen = ckpt.get("raw_records_seen", existing_rows)
    existing_tokens = ckpt.get("tokens", 0)
    existing_bytes = ckpt.get("bytes", 0)
    is_completed = ckpt.get("status") == "completed"

    if existing_rows > 0:
        logging.info(f"  📦 Checkpoint found for {name}: {existing_rows:,} docs | {format_tokens(existing_tokens)} tokens | {format_bytes(existing_bytes)}")

    # If source is already completed, return stats immediately
    if is_completed and existing_rows > 0:
        logging.info(f"     ✅ Already fully downloaded ({existing_rows:,} docs, {format_tokens(existing_tokens)} tokens). Skipping.")
        return {
            "name": name,
            "dataset_id": dataset_id,
            "config": config,
            "split": split,
            "output_file": str(output_file),
            "status": "completed",
            "tokens": existing_tokens,
            "rows": existing_rows,
            "bytes": existing_bytes,
            "end_time": datetime.now().isoformat(),
        }

    # If existing data satisfies the remaining budget
    satisfies_budget = False
    if existing_tokens > 0:
        if max_tokens is not None and (current_tokens + existing_tokens >= max_tokens):
            satisfies_budget = True
        elif max_bytes is not None and (current_bytes + existing_bytes >= max_bytes):
            satisfies_budget = True

    if satisfies_budget:
        logging.info(
            f"     ✅ Existing data already satisfies target "
            f"({format_tokens(current_tokens + existing_tokens)} tokens / {format_bytes(current_bytes + existing_bytes)} total)."
        )
        save_checkpoint(name, output_dir, {
            "name": name,
            "rows": existing_rows,
            "raw_records_seen": existing_raw_seen,
            "tokens": existing_tokens,
            "bytes": existing_bytes,
            "status": "completed",
            "updated_at": datetime.now().isoformat(),
        })
        return {
            "name": name,
            "dataset_id": dataset_id,
            "config": config,
            "split": split,
            "output_file": str(output_file),
            "status": "completed",
            "tokens": existing_tokens,
            "rows": existing_rows,
            "bytes": existing_bytes,
            "end_time": datetime.now().isoformat(),
        }

    logging.info(f"  📥 Source: {name}")
    logging.info(f"     Dataset: {dataset_id} | Config: {config} | Split: {split}")
    logging.info(f"     Description: {source['description']}")
    if max_tokens is not None:
        rem_tokens = max(0, max_tokens - (current_tokens + existing_tokens))
        logging.info(f"     Remaining token budget: {format_tokens(rem_tokens)}")
    if max_bytes is not None:
        rem_bytes = max(0, max_bytes - (current_bytes + existing_bytes))
        logging.info(f"     Remaining byte budget:  {format_bytes(rem_bytes)}")

    if dry_run:
        logging.info(f"     [DRY RUN] Would download from {dataset_id}")
        return {"name": name, "status": "dry_run", "tokens": existing_tokens, "rows": existing_rows, "bytes": existing_bytes}

    stats = {
        "name": name,
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "output_file": str(output_file),
        "status": "in_progress",
        "tokens": existing_tokens,
        "rows": existing_rows,
        "bytes": existing_bytes,
        "start_time": datetime.now().isoformat(),
    }

    try:
        # Build load_dataset kwargs
        kwargs = {
            "path": dataset_id,
            "split": split,
            "streaming": True,
        }
        if config:
            kwargs["name"] = config
        if data_dir:
            kwargs["data_dir"] = data_dir
        if requires_token and hf_token:
            kwargs["token"] = hf_token
        elif requires_token and not hf_token:
            logging.warning(f"     ⚠ {name} requires HF token but none provided. Skipping.")
            stats["status"] = "skipped_no_token"
            return stats

        logging.info(f"     Loading dataset (streaming)...")
        dataset = load_dataset(**kwargs)
        dataset_iter = iter(dataset)

        # Fast forward if resuming
        if existing_raw_seen > 0:
            logging.info(f"     ⏩ Resuming: fast-forwarding first {existing_raw_seen:,} raw docs in stream...")
            dataset_iter = itertools.islice(dataset_iter, existing_raw_seen, None)
            sanitize_last_line_if_needed(output_file)

        source_tokens = existing_tokens
        source_rows = existing_rows
        raw_records_seen = existing_raw_seen
        source_bytes = existing_bytes
        buffer = []

        file_mode = "a" if (existing_rows > 0 and not force_redownload) else "w"
        if file_mode == "w":
            source_tokens = 0
            source_rows = 0
            raw_records_seen = 0
            source_bytes = 0

        with open(output_file, file_mode, encoding="utf-8") as f:
            pbar = tqdm(
                dataset_iter,
                initial=source_rows,
                desc=f"     {name}",
                unit=" docs",
                dynamic_ncols=True,
                leave=True,
            )

            for record in pbar:
                raw_records_seen += 1

                # Extract text — handle different field names
                text = None
                if text_field in record:
                    text = record[text_field]
                elif "text" in record:
                    text = record["text"]
                elif "content" in record:
                    text = record["content"]
                elif "sentence" in record:
                    text = record["sentence"]

                if not text or not isinstance(text, str) or len(text.strip()) < 10:
                    continue

                text = text.strip()
                text_bytes = len(text.encode("utf-8"))
                tokens = estimate_tokens(text)

                # Write to JSONL
                json_line = json.dumps({"text": text}, ensure_ascii=False)
                buffer.append(json_line)

                source_tokens += tokens
                source_rows += 1
                source_bytes += text_bytes

                # Flush buffer periodically and record checkpoint
                if len(buffer) >= FLUSH_INTERVAL:
                    f.write("\n".join(buffer) + "\n")
                    f.flush()
                    buffer = []
                    save_checkpoint(name, output_dir, {
                        "name": name,
                        "rows": source_rows,
                        "raw_records_seen": raw_records_seen,
                        "tokens": source_tokens,
                        "bytes": source_bytes,
                        "status": "in_progress",
                        "updated_at": datetime.now().isoformat(),
                    })

                # Update progress bar
                if source_rows % 1000 == 0:
                    pbar.set_postfix({
                        "tokens": format_tokens(source_tokens),
                        "size": format_bytes(source_bytes),
                    })

                # Check if we've hit the token budget or byte budget for this language
                hit_tokens = (max_tokens is not None and (current_tokens + source_tokens) >= max_tokens)
                hit_bytes = (max_bytes is not None and (current_bytes + source_bytes) >= max_bytes)
                if hit_tokens or hit_bytes:
                    reason = f"{format_tokens(current_tokens + source_tokens)} tokens" if hit_tokens else f"{format_bytes(current_bytes + source_bytes)}"
                    logging.info(
                        f"     ✅ Target reached! ({reason} total)"
                    )
                    break

            # Flush remaining buffer
            if buffer:
                f.write("\n".join(buffer) + "\n")
                f.flush()

            pbar.close()

        # Safely release streaming iterator and dataset references
        try:
            del dataset_iter
        except Exception:
            pass
        try:
            del dataset
        except Exception:
            pass

        stats["tokens"] = source_tokens
        stats["rows"] = source_rows
        stats["bytes"] = source_bytes
        stats["status"] = "completed"
        stats["end_time"] = datetime.now().isoformat()

        save_checkpoint(name, output_dir, {
            "name": name,
            "rows": source_rows,
            "raw_records_seen": raw_records_seen,
            "tokens": source_tokens,
            "bytes": source_bytes,
            "status": "completed",
            "updated_at": datetime.now().isoformat(),
        })

        logging.info(
            f"     ✅ Done: {source_rows:,} docs | "
            f"{format_tokens(source_tokens)} tokens | "
            f"{format_bytes(source_bytes)}"
        )

    except KeyboardInterrupt:
        logging.warning(f"\n     ⏸ Interrupted! Progress checkpoint saved ({stats['rows']:,} docs, {format_tokens(stats['tokens'])} tokens).")
        stats["status"] = "interrupted"
        save_checkpoint(name, output_dir, {
            "name": name,
            "rows": stats.get("rows", 0),
            "raw_records_seen": raw_records_seen if 'raw_records_seen' in locals() else stats.get("rows", 0),
            "tokens": stats.get("tokens", 0),
            "bytes": stats.get("bytes", 0),
            "status": "interrupted",
            "updated_at": datetime.now().isoformat(),
        })
        return stats
    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        logging.error(f"     ❌ Error downloading {name}: {e}")
        # Only remove file if it's completely 0 bytes
        if output_file.exists() and output_file.stat().st_size == 0:
            output_file.unlink()

    return stats


def download_language(
    language: str,
    sources: list[dict],
    output_dir: Path,
    max_tokens: int | None,
    max_bytes: int | None,
    hf_token: str | None,
    dry_run: bool = False,
    force_redownload: bool = False,
) -> dict:
    """
    Download all sources for a given language.

    Sources are processed in priority order. Stops once the token target is reached.
    """
    lang_dir = output_dir / "curated"
    lang_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = lang_dir / "manifest.json"

    logging.info(f"\n{'='*70}")
    logging.info(f"  DOWNLOADING: {language.upper()}")
    logging.info(f"  Output: {lang_dir}")
    if max_tokens is not None:
        logging.info(f"  Token target: {format_tokens(max_tokens)} (~{max_tokens:,} tokens)")
    if max_bytes is not None:
        logging.info(f"  Size target:  {format_bytes(max_bytes)}")
    logging.info(f"  Sources: {len(sources)}")
    logging.info(f"{'='*70}\n")

    # Sort by priority
    sorted_sources = sorted(sources, key=lambda s: s.get("priority", 99))

    total_tokens = 0
    total_rows = 0
    total_bytes = 0
    source_stats = []

    for i, source in enumerate(sorted_sources, 1):
        if (max_tokens is not None and total_tokens >= max_tokens) or \
           (max_bytes is not None and total_bytes >= max_bytes):
            logging.info(f"\n  🎯 Target reached for {language} ({format_tokens(total_tokens)} tokens)!")
            break

        logging.info(f"\n[{i}/{len(sorted_sources)}] Processing source...")
        stats = download_source(
            source=source,
            output_dir=lang_dir,
            max_tokens=max_tokens,
            current_tokens=total_tokens,
            max_bytes=max_bytes,
            current_bytes=total_bytes,
            hf_token=hf_token,
            dry_run=dry_run,
            force_redownload=force_redownload,
        )
        source_stats.append(stats)

        total_tokens += stats.get("tokens", 0)
        total_rows += stats.get("rows", 0)
        total_bytes += stats.get("bytes", 0)

        # Save intermediate manifest checkpoint after every source
        target_reached = ((max_tokens is not None and total_tokens >= max_tokens) or
                          (max_bytes is not None and total_bytes >= max_bytes))
        interim_summary = {
            "language": language,
            "output_dir": str(lang_dir),
            "total_tokens_estimated": total_tokens,
            "total_rows": total_rows,
            "total_bytes": total_bytes,
            "target_tokens": max_tokens,
            "target_bytes": max_bytes,
            "target_reached": target_reached,
            "shortfall_tokens": max(0, max_tokens - total_tokens) if max_tokens else 0,
            "shortfall_bytes": max(0, max_bytes - total_bytes) if max_bytes else 0,
            "sources": source_stats,
            "timestamp": datetime.now().isoformat(),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(interim_summary, f, indent=2, ensure_ascii=False)

        if stats.get("status") == "interrupted":
            logging.warning(f"  ⏸ Language {language} download paused due to interruption.")
            break

    target_reached = ((max_tokens is not None and total_tokens >= max_tokens) or
                      (max_bytes is not None and total_bytes >= max_bytes))
    # Summary
    summary = {
        "language": language,
        "output_dir": str(lang_dir),
        "total_tokens_estimated": total_tokens,
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "target_tokens": max_tokens,
        "target_bytes": max_bytes,
        "target_reached": target_reached,
        "shortfall_tokens": max(0, max_tokens - total_tokens) if max_tokens else 0,
        "shortfall_bytes": max(0, max_bytes - total_bytes) if max_bytes else 0,
        "sources": source_stats,
        "timestamp": datetime.now().isoformat(),
    }

    target_desc = []
    if max_tokens is not None:
        target_desc.append(f"{format_tokens(max_tokens)} tokens")
    if max_bytes is not None:
        target_desc.append(format_bytes(max_bytes))
    target_str = " / ".join(target_desc)

    logging.info(f"\n{'─'*70}")
    logging.info(f"  {language.upper()} SUMMARY")
    logging.info(f"{'─'*70}")
    logging.info(f"  Total documents: {total_rows:,}")
    logging.info(f"  Total tokens (est.): {format_tokens(total_tokens)}")
    logging.info(f"  Total size: {format_bytes(total_bytes)}")
    logging.info(f"  Target reached: {'✅ YES' if target_reached else '❌ NO'} (Target: {target_str})")
    if not target_reached:
        shortfalls = []
        if max_tokens is not None and total_tokens < max_tokens:
            shortfalls.append(f"{format_tokens(max_tokens - total_tokens)} tokens")
        if max_bytes is not None and total_bytes < max_bytes:
            shortfalls.append(format_bytes(max_bytes - total_bytes))
        logging.info(
            f"  Shortfall: {' / '.join(shortfalls)} "
            f"(need to supplement with manual collection or additional sources)"
        )
    logging.info(f"{'─'*70}\n")

    # Save final manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info(f"  Manifest saved to: {manifest_path}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(lang: Language, max_tokens: int = DEFAULT_MAX_TOKENS, max_bytes: int = None,
        output_dir: str = None, env_file: str = None, dry_run: bool = False,
        force_redownload: bool = False, skip_sources: list = None,
        verbose: bool = False) -> dict:
    """
    Download one language's curated sources.

    Args:
        lang:             Language to download for. Its data_sources.yaml
                          supplies the dataset list; its data/raw/ receives them.
        max_tokens:       Token target; downloading stops once it is reached (default: 900M).
        max_bytes:        Optional byte target override.
        output_dir:       Override for ``<language>/data/raw``.
        env_file:         Path to the .env holding HF_TOKEN.
        dry_run:          Preview only; download nothing.
        force_redownload: Ignore existing checkpoints and start over.
        skip_sources:     Source names to leave out.
        verbose:          Enable debug logging.

    Returns:
        The per-source download summary, also written to download_manifest.json.
    """
    setup_logging(verbose)
    skip_set = set(skip_sources or [])

    sources = [
        s for s in lang.load_sources()["huggingface_sources"]
        if s["name"] not in skip_set
    ]

    output_path = Path(output_dir) if output_dir else lang.raw_dir
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 70)
    logging.info("  LMA — Curated Corpus Download")
    logging.info(f"  Language:     {lang.display} ({lang.model_label})")
    if max_tokens is not None:
        logging.info(f"  Token target: {format_tokens(max_tokens)} (~{max_tokens:,} tokens)")
    if max_bytes is not None:
        logging.info(f"  Size target:  {format_bytes(max_bytes)}")
    logging.info(f"  Output dir:   {output_path}")
    logging.info(f"  Dry run:      {dry_run}")
    logging.info(f"  Resumable:    {'no (force redownload)' if force_redownload else 'yes'}")
    logging.info("=" * 70)

    hf_token = load_hf_token(env_file or str(PROJECT_ROOT / ".env"))

    summary = download_language(
        language=lang.key,
        sources=sources,
        output_dir=output_path,
        max_tokens=max_tokens,
        max_bytes=max_bytes,
        hf_token=hf_token,
        dry_run=dry_run,
        force_redownload=force_redownload,
    )

    logging.info("\n" + "=" * 70)
    logging.info("  SUMMARY")
    logging.info("=" * 70)
    logging.info(
        f"  {lang.key.upper():10s}: "
        f"{summary['total_rows']:>12,} docs | "
        f"{format_tokens(summary['total_tokens_estimated']):>10s} tokens | "
        f"{format_bytes(summary['total_bytes']):>12s} | "
        f"{'target met' if summary['target_reached'] else 'shortfall: ' + format_tokens(summary.get('shortfall_tokens', 0)) + ' tokens'}"
    )
    logging.info("=" * 70)

    manifest_path = output_path / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({lang.key: summary}, f, indent=2, ensure_ascii=False)
    logging.info(f"  Manifest saved to: {manifest_path}")

    # HuggingFace/pyarrow background threads can crash on interpreter exit.
    gc.collect()
    return summary


def main():
    import argparse
    from src import languages
    parser = argparse.ArgumentParser(description="Download one language's curated corpus from HuggingFace Hub.")
    languages.add_language_arg(parser)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Token target for curated data (default: {format_tokens(DEFAULT_MAX_TOKENS)} = 900M tokens).")
    parser.add_argument("--max-bytes", type=int, default=None,
                        help="Optional byte target override.")
    parser.add_argument("--output-dir", default=None,
                        help="Override <language>/data/raw.")
    parser.add_argument("--env-file", default=None, help="Path to .env holding HF_TOKEN.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading.")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Ignore checkpoints and redownload from scratch.")
    parser.add_argument("--skip-sources", nargs="*", default=[],
                        help="Source names to skip.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    run(
        languages.get(args.lang),
        max_tokens=args.max_tokens,
        max_bytes=args.max_bytes,
        output_dir=args.output_dir,
        env_file=args.env_file,
        dry_run=args.dry_run,
        force_redownload=args.force_redownload,
        skip_sources=args.skip_sources,
        verbose=args.verbose,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

