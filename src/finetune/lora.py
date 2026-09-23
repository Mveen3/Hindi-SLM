"""
First-Principles Low-Rank Adaptation (LoRA) for Parameter-Efficient Fine-Tuning
==============================================================================

Implements LoRA (Hu et al., 2021) directly from mathematical first principles
without external dependencies (like HuggingFace `peft`).

Mathematical Formulation:
-------------------------
For a pretrained linear layer with frozen weights W_0 in R^{d_out x d_in}:
    h = x W_0^T + b

LoRA decomposes the weight update delta_W into two low-rank matrices:
    delta_W = (alpha / r) * (B @ A)
    where:
        A in R^{r x d_in}  ~ N(0, 1/r)  (Gaussian initialization)
        B in R^{d_out x r} = 0          (Zero initialization)
        r << min(d_in, d_out) is the rank (e.g., r = 8 or 16)
        alpha is a constant scaling hyperparameter (e.g., alpha = 16)

Output forward computation:
    h = x W_0^T + (alpha / r) * (dropout(x) @ A^T @ B^T) + b

Properties & Advantages for GenAI / AI-ML:
------------------------------------------
1. Identity at Initialization: Since B = 0, delta_W = 0 at step 0. The model starts
   with the exact representation and predictions of the pretrained base model.
2. Massive Memory Efficiency: In our 24.3M parameter model, adapting W_q and W_v across
   all 6 layers with rank r=8 trains only ~147,456 parameters (~0.6% of total weights).
   Optimizer states (AdamW m and v) drop from ~195 MB to ~1.2 MB.
3. Zero Latency Overhead at Inference: Supports in-place weight merging:
       W_merged = W_0 + (alpha / r) * (B @ A)
   yielding a standard linear layer with zero extra inference computation.
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with low-rank adapter matrices A and B.

    Args:
        base_linear: The original nn.Linear layer (weights will be frozen).
        rank:        Rank r of the low-rank update (r << d_model).
        alpha:       Scaling factor alpha. Scale multiplier is (alpha / rank).
        dropout:     Dropout probability applied to the input of the LoRA branch.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Base linear layer (frozen)
        self.base_layer = base_linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if rank > 0:
            dev = base_linear.weight.device
            dt = base_linear.weight.dtype
            # A initialized from N(0, 1/r), B initialized to 0
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dt))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dt))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes base forward pass plus scaled low-rank update.
        """
        # If merged into base weights, compute standard linear pass
        if self.merged or self.rank <= 0:
            return self.base_layer(x)

        # Base output: x @ W_0^T + b
        base_out = self.base_layer(x)

        # LoRA output: (alpha / r) * ( (dropout(x) @ A^T) @ B^T )
        lora_in = self.lora_dropout(x)
        lora_out = F.linear(F.linear(lora_in, self.lora_A), self.lora_B) * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> None:
        """Merge LoRA delta_W into base_layer.weight in-place for deployment."""
        if self.merged or self.rank <= 0:
            return

        with torch.no_grad():
            # delta_W = (alpha / r) * (B @ A) -> shape (out_features, in_features)
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data.add_(delta_w)
            self.merged = True
            logger.debug("Merged LoRA weights into base layer.")

    def unmerge_weights(self) -> None:
        """Subtract LoRA delta_W from base_layer.weight to resume adaptation."""
        if not self.merged or self.rank <= 0:
            return

        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data.sub_(delta_w)
            self.merged = False
            logger.debug("Unmerged LoRA weights from base layer.")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"merged={self.merged}"
        )


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Sequence[str] = ("W_q", "W_v"),
    dropout: float = 0.05,
) -> tuple[nn.Module, dict[str, int]]:
    """
    Traverse the model, replace target linear layers with LoRALinear modules,
    and freeze all base parameters.

    Args:
        model:          The Transformer model.
        rank:           LoRA rank r.
        alpha:          LoRA scaling alpha.
        target_modules: Layer attribute names to adapt (default: attention queries & values).
        dropout:        Dropout for LoRA branch.

    Returns:
        Tuple of (model_with_lora, parameter_counts_dict).
    """
    # 1. Freeze every parameter in the model first
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0

    # 2. Iterate through transformer blocks and adapt target linear layers
    if hasattr(model, "blocks"):
        for b_idx, block in enumerate(model.blocks):
            if hasattr(block, "attn"):
                attn = block.attn
                for mod_name in target_modules:
                    if hasattr(attn, mod_name):
                        orig_layer = getattr(attn, mod_name)
                        if isinstance(orig_layer, nn.Linear):
                            lora_layer = LoRALinear(
                                orig_layer, rank=rank, alpha=alpha, dropout=dropout
                            )
                            setattr(attn, mod_name, lora_layer)
                            replaced_count += 1

    param_stats = count_lora_parameters(model)
    logger.info(
        "Applied LoRA to %d layers (rank=%d, alpha=%.1f). Trainable: %s / %s (%.2f%%)",
        replaced_count,
        rank,
        alpha,
        f"{param_stats['trainable']:,}",
        f"{param_stats['total']:,}",
        param_stats["trainable_percent"],
    )

    return model, param_stats


def count_lora_parameters(model: nn.Module) -> dict[str, int | float]:
    """Count total, trainable, and frozen parameters in a LoRA-adapted model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    percent = (trainable / total * 100.0) if total > 0 else 0.0

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": round(percent, 3),
    }


def merge_lora_weights(model: nn.Module) -> None:
    """Recursively find all LoRALinear modules and merge weights into base layers."""
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged_count += 1
    logger.info("Merged %d LoRA layers into base model weights.", merged_count)


def unmerge_lora_weights(model: nn.Module) -> None:
    """Recursively find all LoRALinear modules and unmerge weights."""
    unmerged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()
            unmerged_count += 1
    logger.info("Unmerged %d LoRA layers.", unmerged_count)


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the trainable LoRA adapter weights (compact checkpoint)."""
    lora_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            lora_dict[name] = param.data.clone()
    return lora_dict


def is_lora_checkpoint(checkpoint_path: str | Path) -> bool:
    """Check if a checkpoint file is a compact LoRA adapter checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("peft_type") == "lora" or "lora_state_dict" in ckpt
    except Exception:
        return False


def save_lora_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    config: dict,
    step: int,
    optimizer=None,
    scheduler=None,
    scaler=None,
    loss_history=None,
    val_loss_history=None,
) -> None:
    """Save a compact checkpoint containing only adapter parameters and metadata."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "peft_type": "lora",
        "step": step,
        "lora_state_dict": get_lora_state_dict(model),
        "config": config,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss_history": loss_history or [],
        "val_loss_history": val_loss_history or [],
    }

    torch.save(checkpoint, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Saved compact LoRA checkpoint to %s (%.2f MB)", path, size_mb)

    # Update latest_checkpoint.txt
    latest_path = path.parent / "latest_checkpoint.txt"
    with open(latest_path, "w") as f:
        f.write(str(path))


def load_lora_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
) -> dict:
    """Load adapter weights and optional optimizer state from a compact LoRA checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    lora_state = checkpoint["lora_state_dict"]

    model_dict = dict(model.named_parameters())
    loaded_keys = []
    for k, v in lora_state.items():
        if k in model_dict:
            model_dict[k].data.copy_(v)
            loaded_keys.append(k)
        else:
            logger.warning("Key %s in checkpoint not found in LoRA model.", k)

    logger.info("Loaded %d LoRA adapter tensors from %s (step %s)",
                len(loaded_keys), path, checkpoint.get("step", "?"))

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        logger.info("LoRA optimizer state restored and mapped to %s.", device)

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("LoRA scheduler state restored.")

    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("LoRA AMP scaler state restored.")

    return {
        "step": checkpoint.get("step", 0),
        "loss_history": checkpoint.get("loss_history", []),
        "val_loss_history": checkpoint.get("val_loss_history", []),
        "config": checkpoint.get("config", {}),
    }
