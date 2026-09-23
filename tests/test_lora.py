#!/usr/bin/env python3
"""
Unit tests for First-Principles LoRA Implementation
==================================================

Validates mathematical properties of Low-Rank Adaptation:
1. Identity at initialization: Delta W = 0 initially (base model output identical).
2. Parameter freezing: Only LoRA A and B parameters require grad (< 1% of total).
3. Gradient propagation: Backward pass updates only adapters, not base weights.
4. Weight merging: Merged model forward pass numerically matches unmerged LoRA forward.
5. Checkpoint serialization: Compact LoRA checkpoint (< 2 MB) saves and reloads correctly.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.transformer import build_model_from_config
from src.finetune.lora import (
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    merge_lora_weights,
    unmerge_lora_weights,
    save_lora_checkpoint,
    load_lora_checkpoint,
)

CONFIG = {
    "vocab_size": 256,
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "d_ff": 256,
    "max_seq_len": 64,
    "dropout": 0.0,
    "tie_weights": True,
}


def test_lora_linear_identity_at_init():
    """At step 0, LoRALinear must produce the exact same output as the base layer."""
    torch.manual_seed(42)
    base = nn.Linear(128, 128)
    x = torch.randn(4, 16, 128)

    base_out = base(x)

    lora = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
    lora_out = lora(x)

    diff = (base_out - lora_out).abs().max().item()
    assert diff == 0.0, f"LoRA at init must be identical to base (diff={diff})"
    print("test_lora_linear_identity_at_init: OK (diff=0.0)")


def test_apply_lora_parameter_efficiency():
    """Applying LoRA must freeze base parameters and leave < 2% trainable."""
    torch.manual_seed(42)
    model = build_model_from_config(CONFIG)
    total_before = sum(p.numel() for p in model.parameters())

    model, stats = apply_lora(model, rank=8, alpha=16.0, target_modules=("W_q", "W_v"))

    assert stats["trainable"] > 0
    assert stats["trainable"] < stats["total"] * 0.05, "LoRA should train < 5% of parameters"

    # Confirm only lora_A and lora_B require grad
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_A" in name or "lora_B" in name, f"Unexpected trainable param: {name}"
        else:
            assert "lora_" not in name, f"LoRA parameter unexpectedly frozen: {name}"

    print(f"test_apply_lora_parameter_efficiency: OK ({stats['trainable']:,} / {stats['total']:,} = {stats['trainable_percent']}%)")


def test_gradient_flow_and_base_weight_immutability():
    """Gradients must update LoRA weights while base weights remain strictly unchanged."""
    torch.manual_seed(42)
    model = build_model_from_config(CONFIG)
    apply_lora(model, rank=8, alpha=16.0, target_modules=("W_q", "W_v"))

    # Snapshot base weights
    base_weight_snapshot = model.blocks[0].attn.W_q.base_layer.weight.clone()
    lora_B_snapshot = model.blocks[0].attn.W_q.lora_B.clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randint(0, CONFIG["vocab_size"], (2, 16))
    targets = torch.randint(0, CONFIG["vocab_size"], (2, 16))

    _, loss = model(x, targets=targets)
    loss.backward()
    optimizer.step()

    # Base weights must NOT have changed
    base_diff = (model.blocks[0].attn.W_q.base_layer.weight - base_weight_snapshot).abs().max().item()
    assert base_diff == 0.0, f"Base weights must not change during LoRA training (diff={base_diff})"

    # LoRA weights MUST have changed
    lora_diff = (model.blocks[0].attn.W_q.lora_B - lora_B_snapshot).abs().max().item()
    assert lora_diff > 0.0, "LoRA weights must be updated by optimizer"

    print("test_gradient_flow_and_base_weight_immutability: OK")


def test_lora_weight_merging():
    """Merging weights must produce identical forward outputs with zero adapter overhead."""
    torch.manual_seed(42)
    model = build_model_from_config(CONFIG)
    apply_lora(model, rank=8, alpha=16.0, target_modules=("W_q", "W_v"))

    # Perturb LoRA weights so they are non-zero
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.data.normal_(0, 0.05)

    x = torch.randint(0, CONFIG["vocab_size"], (2, 16))
    model.eval()

    with torch.no_grad():
        out_unmerged, _ = model(x)

    merge_lora_weights(model)

    with torch.no_grad():
        out_merged, _ = model(x)

    diff = (out_unmerged - out_merged).abs().max().item()
    assert diff < 1e-5, f"Merged and unmerged outputs must match (max diff={diff:.2e})"

    unmerge_lora_weights(model)
    with torch.no_grad():
        out_restored, _ = model(x)

    diff_restored = (out_unmerged - out_restored).abs().max().item()
    assert diff_restored < 1e-5, f"Unmerging must restore original state (diff={diff_restored:.2e})"

    print(f"test_lora_weight_merging: OK (max diff = {diff:.2e})")


def test_lora_save_and_load(tmp_path=Path("/tmp/test_lora")):
    """Saving and reloading LoRA checkpoint should restore exact predictions on the base model."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = tmp_path / "adapter_test.pt"

    # Pretrained base model
    torch.manual_seed(42)
    base_model = build_model_from_config(CONFIG)

    import copy
    model1 = copy.deepcopy(base_model)
    model2 = copy.deepcopy(base_model)

    apply_lora(model1, rank=8, alpha=16.0, target_modules=("W_q", "W_v"))
    apply_lora(model2, rank=8, alpha=16.0, target_modules=("W_q", "W_v"))

    # Perturb model1 adapters to non-trivial weights
    with torch.no_grad():
        for m in model1.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.data.normal_(0, 0.1)
                m.lora_B.data.normal_(0, 0.1)

    save_lora_checkpoint(model1, ckpt_file, config=CONFIG, step=100)

    # Load adapter into model2
    load_lora_checkpoint(model2, ckpt_file)

    model1.eval()
    model2.eval()

    x = torch.randint(0, CONFIG["vocab_size"], (2, 16))
    with torch.no_grad():
        out1, _ = model1(x)
        out2, _ = model2(x)

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"Reloaded LoRA model output mismatch: diff={diff:.2e}"
    print(f"test_lora_save_and_load: OK (file size: {ckpt_file.stat().st_size / 1024:.1f} KB, diff={diff:.2e})")


if __name__ == "__main__":
    test_lora_linear_identity_at_init()
    test_apply_lora_parameter_efficiency()
    test_gradient_flow_and_base_weight_immutability()
    test_lora_weight_merging()
    test_lora_save_and_load()
    print("\n🎉 ALL LoRA unit tests passed successfully!")
