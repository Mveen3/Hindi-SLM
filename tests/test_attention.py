"""
Attention correctness tests
===========================

Three properties the project brief requires us to demonstrate:

  1. Causality — position t must not attend to any position > t.
  2. No future leakage — changing token t+1 must not move the logits at t.
  3. Equivalence — our hand-written scaled-dot-product attention agrees with
     torch's fused SDPA kernel, so the optional `use_fused_attention` speed
     path does not change the model's function.

The model is language-agnostic, so these run once on a small random model
rather than per language; what varies per language is the vocabulary size,
which is exercised by the parametrised config below.

Run:  python -m pytest tests/test_attention.py -v
      python tests/test_attention.py          (no pytest needed)
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.transformer import build_model_from_config  # noqa: E402

ATOL = 1e-4
CONFIG = {
    "vocab_size": 512, "d_model": 64, "n_layers": 2, "n_heads": 4,
    "d_ff": 128, "max_seq_len": 32, "dropout": 0.0, "tie_weights": True,
}


def build(seed: int = 0):
    """Build a small, deterministic, randomly-initialised model."""
    torch.manual_seed(seed)
    return build_model_from_config(CONFIG).eval()


def test_causality():
    """Attention weights must place exactly zero mass on future positions."""
    model = build()
    ids = torch.randint(0, CONFIG["vocab_size"], (2, 16))
    with torch.no_grad():
        _, _, all_attn = model(ids, return_attention=True)

    for layer, attn in enumerate(all_attn):
        future = attn.triu(diagonal=1).abs().max().item()
        assert future == 0.0, f"layer {layer} attends to the future ({future})"
        rows = attn.sum(-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5), \
            f"layer {layer} attention rows do not sum to 1"
    print("causality: OK — zero mass on future positions, rows sum to 1")


def test_future_tokens_do_not_change_logits():
    """Changing token t must leave the logits at every position < t untouched."""
    model = build()
    ids = torch.randint(0, CONFIG["vocab_size"], (1, 16))
    altered = ids.clone()
    altered[0, 10] = (altered[0, 10] + 7) % CONFIG["vocab_size"]

    with torch.no_grad():
        a, _ = model(ids)
        b, _ = model(altered)

    diff_before = (a[0, :10] - b[0, :10]).abs().max().item()
    diff_after = (a[0, 10:] - b[0, 10:]).abs().max().item()
    assert diff_before == 0.0, f"past logits changed ({diff_before})"
    assert diff_after > 0.0, "logits at/after the edit did not change"
    print("no-future-leak: OK — past logits identical, current/later logits change")


def test_fused_matches_manual():
    """The optional fused SDPA path must match our own implementation."""
    manual = build(seed=1)
    fused = build(seed=1)
    for block in fused.blocks:
        block.attn.use_fused_attention = True

    ids = torch.randint(0, CONFIG["vocab_size"], (2, 16))
    with torch.no_grad():
        m, _ = manual(ids)
        f, _ = fused(ids)

    delta = (m - f).abs().max().item()
    assert torch.allclose(m, f, atol=ATOL), f"paths diverge by {delta}"
    print(f"fused-vs-manual: OK — max |diff| = {delta:.2e}")


def test_parameter_count_is_about_25m():
    """Both models target ~25M trainable parameters; check the real configs."""
    from src import languages
    from src.model.transformer import count_parameters

    for key in languages.CHOICES:
        lang = languages.get(key)
        config = lang.load_config()
        model = build_model_from_config(config)
        trainable = count_parameters(model)["trainable"]
        assert 20e6 < trainable < 30e6, \
            f"{key}: {trainable:,} parameters is outside the ~25M target"
        print(f"parameters [{key}]: OK — {trainable:,} trainable")


if __name__ == "__main__":
    test_causality()
    test_future_tokens_do_not_change_logits()
    test_fused_matches_manual()
    test_parameter_count_is_about_25m()
    print("\nAll attention tests passed.")
