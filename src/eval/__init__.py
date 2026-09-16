"""
Evaluation Suite
================

Comprehensive evaluation suite for language models:
- Perplexity (PPL), Cross-Entropy Loss, and Bits-per-Byte (BPB)
- Text generation metrics (BLEU-4, chrF, ROUGE-L, Diversity)
- Multi-head attention heatmaps, entropy, and mean distance
- Empirical causal mask verification
- Training & validation loss curve generation

Invoked through ``python main.py evaluate --lang {hindi,nepali}`` or ``python -m src.eval --lang hindi``.
"""

import argparse
import logging

from src import languages
from src.eval import lm_metrics, generation, attention, causal_mask, plots
from src.train.utils import resolve_checkpoint

STAGES = ("ppl", "generation", "attention", "causal", "plots")


def run(lang: languages.Language, checkpoint: str = None, only: list = None, split: str = "test",
        num_samples: int = 100, max_batches: int = None):
    """Run specified evaluation stages for the given language."""
    only = only or list(STAGES)
    checkpoint = resolve_checkpoint(lang, checkpoint)
    logging.info("Evaluating %s (%s) from %s", lang.display, lang.model_label, checkpoint)

    if "ppl" in only:
        lm_metrics.evaluate_lm(lang, checkpoint, split, max_batches)
    if "generation" in only:
        generation.run_generation_eval(lang, checkpoint, num_samples=num_samples)
    if "attention" in only:
        attention.run_attention_analysis(lang, checkpoint)
    if "causal" in only:
        causal_mask.verify_causal_mask(lang, checkpoint)
    if "plots" in only:
        plots.plot_from_checkpoint(lang, checkpoint)

    logging.info("Artifacts written to %s", lang.report_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    languages.add_language_arg(parser)
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to evaluate (default: the newest local one).")
    parser.add_argument("--only", nargs="+", choices=STAGES, default=list(STAGES),
                        help="Run only these stages (default: all).")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Split for perplexity/BPB (default: %(default)s).")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Held-out prefixes for generation metrics.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap batches during perplexity evaluation (for quick runs).")
    args = parser.parse_args()

    lang = languages.get(args.lang)
    run(
        lang,
        checkpoint=args.checkpoint,
        only=args.only,
        split=args.split,
        num_samples=args.num_samples,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()

__all__ = [
    "run",
    "main",
    "STAGES",
    "lm_metrics",
    "generation",
    "attention",
    "causal_mask",
    "plots",
]
