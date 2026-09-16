#!/usr/bin/env python3
"""
SLM Project Unified CLI
=======================

Single unified entrypoint for all data pipeline, training, inference,
evaluation, and reasoning workflows.

Commands:
    clean               Clean and merge raw shards into interim JSONL.
    split               Split interim JSONL into pre-shuffled train/val/test splits.
    crawl               Crawl seed domains for manual corpus collection.
    download            Download curated corpus from HuggingFace Hub.
    tokenizer           Train or evaluate BPE tokenizer.
    train               Pretrain model locally.
    train-kaggle        Pretrain model on Kaggle with Hub sync.
    evaluate            Run Phase 2 evaluation suite (PPL, generation, attention, causal, plots).
    generate            Generate text continuation from a prompt.
    finetune            Finetune pretrained model on reasoning corpus.
    evaluate-reasoning  Evaluate reasoning accuracy and attention comparison.
    reasoning-data      Generate synthetic reasoning finetuning dataset.
    hub                 Download checkpoints from HuggingFace Hub.
"""

import sys
from src import languages


COMMANDS = {
    "clean": ("src.data.clean", "main"),
    "split": ("src.data.split", "main"),
    "crawl": ("src.data.crawler", "main"),
    "download": ("src.data.download", "main"),
    "tokenizer": ("src.data.tokenizer", "main"),
    "train": ("src.train.trainer", "main"),
    "train-kaggle": ("src.train.kaggle", "main"),
    "evaluate": ("src.eval", "main"),
    "generate": ("src.eval.generation", "main"),
    "finetune": ("src.finetune.trainer", "main"),
    "evaluate-reasoning": ("src.finetune.evaluate", "main"),
    "reasoning-data": ("src.data.reasoning.generator", "main"),
    "hub": ("src.data.hub", "main"),
}


def print_help():
    print(__doc__.strip())
    print("\nUsage:\n    python main.py <command> [options...]\n")
    print(f"Available commands: {', '.join(COMMANDS.keys())}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Error: Unknown command '{cmd}'.")
        print(f"Available commands: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    module_name, func_name = COMMANDS[cmd]
    # Shift sys.argv so sub-command parser sees its own arguments
    sys.argv = [f"main.py {cmd}"] + sys.argv[2:]

    import importlib
    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)
    func()


if __name__ == "__main__":
    main()
