"""
Training Utilities
==================

Checkpoint save/load, learning rate scheduling, and loss tracking utilities.
All checkpoint files are self-contained: they store model weights, optimizer
state, scheduler state, AMP scaler state, current training step, and the
full configuration — so that training can be resumed from any checkpoint
without external state.
"""

import os
import re
import math
import json
import torch
import logging
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    config: dict,
    loss_history: list,
    val_loss_history: list,
    checkpoint_dir: str,
):
    """
    Save a full training checkpoint.

    The checkpoint contains everything needed to resume training:
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - scaler_state_dict (AMP)
        - step (current global training step)
        - config (full model + training config)
        - loss_history (list of (step, loss) tuples)
        - val_loss_history (list of (step, val_loss) tuples)

    Args:
        model:            The model.
        optimizer:        The optimizer.
        scheduler:        The LR scheduler.
        scaler:           The GradScaler for AMP.
        step:             Current global step.
        config:           Full configuration dictionary.
        loss_history:     Training loss history.
        val_loss_history: Validation loss history.
        checkpoint_dir:   Directory to save checkpoints.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": config,
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
    }

    ckpt_path = ckpt_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

    # Also save a pointer to the latest checkpoint for easy resume
    latest_path = ckpt_dir / "latest_checkpoint.txt"
    with open(latest_path, "w") as f:
        f.write(str(ckpt_path))


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
):
    """
    Load a training checkpoint and restore all state.

    Args:
        checkpoint_path: Path to the checkpoint .pt file.
        model:           The model to load weights into.
        optimizer:       The optimizer (optional, for resume).
        scheduler:       The LR scheduler (optional).
        scaler:          The GradScaler (optional).
        device:          Device to map tensors to.

    Returns:
        Dictionary with 'step', 'loss_history', 'val_loss_history', 'config'.
    """
    # Load checkpoint to CPU first to prevent CUDA memory fragmentation
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from {checkpoint_path}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # PyTorch optimizer.load_state_dict doesn't guarantee tensors are on device
        # if the param is on device but the state dict is on CPU. We explicitly move it.
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        logger.info("Optimizer state restored and mapped to device.")

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state restored.")

    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("AMP scaler state restored.")

    ret = {
        "step": checkpoint["step"],
        "loss_history": checkpoint.get("loss_history", []),
        "val_loss_history": checkpoint.get("val_loss_history", []),
        "config": checkpoint.get("config", {}),
    }

    del checkpoint
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return ret


def find_latest_checkpoint(checkpoint_dir: str):
    """
    Find the most recent checkpoint in the given directory.

    First checks for a 'latest_checkpoint.txt' pointer file.
    Falls back to finding the highest-step checkpoint file.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints exist.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    # Check for pointer file
    latest_file = ckpt_dir / "latest_checkpoint.txt"
    if latest_file.exists():
        path = latest_file.read_text().strip()
        if Path(path).exists():
            return path

    # Fall back: find highest step number
    pattern = re.compile(r"checkpoint_step_(\d+)\.pt")
    checkpoints = []
    for f in ckpt_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            checkpoints.append((int(match.group(1)), str(f)))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Create a cosine annealing LR scheduler with linear warmup.

    Schedule:
        - Steps 0..warmup_steps:        linear warmup from 0 to base LR.
        - Steps warmup_steps..max_steps: cosine decay to min_lr_ratio × base LR.

    Args:
        optimizer:     The optimizer.
        warmup_steps:  Number of warmup steps.
        max_steps:     Total number of training steps.
        min_lr_ratio:  Minimum LR as a fraction of peak LR (default 0.1).

    Returns:
        LambdaLR scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loss Tracking
# ---------------------------------------------------------------------------

class AverageMeter:
    """
    Running average tracker for loss and other scalar metrics.

    Usage:
        meter = AverageMeter("train_loss")
        for batch in dataloader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=batch_size)
        print(meter)
    """

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Add a new observation.

        Args:
            val: Metric value (e.g., per-sample loss).
            n:   Number of samples this value covers.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Training Log Export
# ---------------------------------------------------------------------------

def save_training_log(loss_history: list, val_loss_history: list, log_dir: str):
    """
    Save training and validation loss histories to JSON for later plotting.

    Args:
        loss_history:     List of (step, loss) tuples.
        val_loss_history: List of (step, val_loss) tuples.
        log_dir:          Directory to save log files.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    with open(log_path / "train_loss.json", "w") as f:
        json.dump(loss_history, f)

    with open(log_path / "val_loss.json", "w") as f:
        json.dump(val_loss_history, f)

    logger.info(f"Training logs saved to {log_path}")


def resolve_checkpoint(lang, explicit: str = None) -> str:
    """
    Resolve which checkpoint to load for a language.

    Args:
        lang:     The Language whose checkpoints/ directory is searched.
        explicit: A caller-supplied path, or None to auto-detect the newest.

    Returns:
        Path to the checkpoint file.

    Raises:
        SystemExit: If no checkpoint is available locally, with the command
                    needed to fetch one.
    """
    if explicit:
        return explicit

    found = find_latest_checkpoint(str(lang.checkpoint_dir))
    if not found:
        raise SystemExit(
            f"No checkpoint in {lang.checkpoint_dir}. "
            f"Run: python main.py hub --lang {lang.key} --stage pretraining"
        )
    return found

