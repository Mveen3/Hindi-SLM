#!/usr/bin/env python3
"""
Pre-Pretraining Pipeline Orchestrator
====================================

Runs the complete data preparation and tokenization pipeline sequentially:
  1. Hindi:   Download -> Clean -> Split -> Tokenizer
  2. Nepali:  Download -> Clean -> Split -> Tokenizer

Invoked directly or via background runner:
  python src/run_pretrain_pipeline.py
"""

import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def run_step(step_name: str, cmd: list[str], check_target: Path = None):
    if check_target and check_target.exists() and check_target.stat().st_size > 0:
        size_mb = check_target.stat().st_size / (1024 * 1024)
        log(f"⏭  SKIPPING: {step_name} ({check_target.name} already exists: {size_mb:.2f} MB)\n")
        return

    log("=" * 60)
    log(f"STARTING STEP: {step_name}")
    log(f"Command: {' '.join(cmd)}")
    log("=" * 60)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"❌ ERROR in {step_name} (exit code: {result.returncode}) after {elapsed:.1f}s")
        sys.exit(result.returncode)
    log(f"✅ COMPLETED: {step_name} in {elapsed:.1f}s ({elapsed / 60:.2f} mins)\n")


def run_tokenizer_step(lang_name: str, step_num: int, python_bin: str):
    tokenizer_json = PROJECT_ROOT / f"{lang_name}/tokenizer/tokenizer.json"
    stats_json = PROJECT_ROOT / f"report/{lang_name}/tokenizer_stats.json"

    if stats_json.exists() and stats_json.stat().st_size > 0:
        log(f"⏭  SKIPPING: {step_num}. Train & Evaluate Tokenizer ({lang_name.capitalize()}) (stats already exist)\n")
        return

    if tokenizer_json.exists() and tokenizer_json.stat().st_size > 0:
        log(f"⚡ Tokenizer already trained for {lang_name.capitalize()}. Running evaluation directly...")
        run_step(f"{step_num}. Evaluate Tokenizer ({lang_name.capitalize()})",
                 [python_bin, "-u", "main.py", "tokenizer", "--lang", lang_name, "--action", "eval"])
    else:
        run_step(f"{step_num}. Train & Evaluate Tokenizer ({lang_name.capitalize()})",
                 [python_bin, "-u", "main.py", "tokenizer", "--lang", lang_name, "--action", "both"])


def main():
    python_bin = sys.executable
    log("🚀 Starting SLM Pre-Pretraining Pipeline...")
    t_start = time.time()

    # 1. Hindi Pipeline
    run_step("1. Download Curated Data (Hindi)", [python_bin, "-u", "main.py", "download", "--lang", "hindi"])
    run_step("2. Clean & Merge Shards (Hindi)", [python_bin, "-u", "main.py", "clean", "--lang", "hindi"],
             check_target=PROJECT_ROOT / "hindi/data/interim/hindi_cleaned.jsonl")
    run_step("3. Split & Shuffle Corpus (Hindi)", [python_bin, "-u", "main.py", "split", "--lang", "hindi"],
             check_target=PROJECT_ROOT / "hindi/data/processed/train.jsonl")
    run_tokenizer_step("hindi", 4, python_bin)

    # 2. Nepali Pipeline
    run_step("5. Download Curated Data (Nepali)", [python_bin, "-u", "main.py", "download", "--lang", "nepali"])
    run_step("6. Clean & Merge Shards (Nepali)", [python_bin, "-u", "main.py", "clean", "--lang", "nepali"],
             check_target=PROJECT_ROOT / "nepali/data/interim/nepali_cleaned.jsonl")
    run_step("7. Split & Shuffle Corpus (Nepali)", [python_bin, "-u", "main.py", "split", "--lang", "nepali"],
             check_target=PROJECT_ROOT / "nepali/data/processed/train.jsonl")
    run_tokenizer_step("nepali", 8, python_bin)

    total_time = time.time() - t_start
    log("=" * 60)
    log(f"🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY in {total_time / 3600:.2f} hours ({total_time / 60:.1f} mins)!")
    log("=" * 60)


if __name__ == "__main__":
    main()
