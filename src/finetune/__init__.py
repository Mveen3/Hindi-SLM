"""
Reasoning Finetuning Package (Phase 3)
======================================

Finetunes each pretrained model on its own synthetic reasoning corpus, scores
it against its pretrained self, and compares what attention learned before and
after.

Model H and Model L are finetuned independently: separate pretrained
checkpoints, separate frozen tokenizers and vocabularies, separate reasoning
corpora, separate finetuned checkpoints. Nothing is shared.

Modules:
    dataset            Prompt-masked, length-bucketed supervised dataset.
    trainer            The finetuning loop (resume-capable, Phase 2 format).
    evaluate           Pretrained vs finetuned exact match on the test split.
    attention_compare  Pretrained vs finetuned attention (brief §3.2).
"""

from .dataset import ReasoningSFTDataset, create_finetune_dataloader
from .evaluate import run_reasoning_eval
from .attention_compare import run_attention_comparison

__all__ = [
    "ReasoningSFTDataset",
    "create_finetune_dataloader",
    "run_reasoning_eval",
    "run_attention_comparison",
]
