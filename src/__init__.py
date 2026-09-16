"""
LMA — shared implementation for two independent monolingual Transformer LMs.

This package holds the *language-agnostic* code: the decoder-only Transformer,
the streaming dataset, the training loop, the evaluation suite and the data
pipeline. Everything language-specific — corpus, tokenizer, vocabulary,
configuration and weights — lives under ``<project>/hindi/`` and
``<project>/nepali/`` and is reached only through :mod:`src.languages`.

Model H (Hindi) and Model L (Nepali) therefore share code but share no data,
no tokenizer, no vocabulary and no weights.
"""

__version__ = "2.0.0"
