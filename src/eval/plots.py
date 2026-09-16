"""
Training Visualisation
======================

Plot training loss curves, validation loss curves, and learning rate schedule
from saved training logs.

Invoked through ``python main.py evaluate --lang {hindi,nepali} --only plots``.
"""

import json
import logging
from pathlib import Path

from src.languages import Language

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _import_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_from_logs(lang: Language, log_dir: str = None, output_dir: str = None):
    """
    Plot training and validation loss curves from JSON log files.

    Args:
        lang:       Language whose curves are plotted (drives title and output).
        log_dir:    Directory containing train_loss.json and val_loss.json.
                    Defaults to the language's own logs/ directory.
        output_dir: Directory to save plots (default: report/<language>/).
    """
    plt = _import_plotting()

    log_path = Path(log_dir) if log_dir else lang.log_dir
    out_dir = Path(output_dir) if output_dir else lang.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training loss
    train_loss_file = log_path / "train_loss.json"
    val_loss_file = log_path / "val_loss.json"

    if train_loss_file.exists():
        with open(train_loss_file) as f:
            train_loss = json.load(f)
        train_steps, train_losses = zip(*train_loss) if train_loss else ([], [])
    else:
        logger.warning(f"Training loss file not found: {train_loss_file}")
        train_steps, train_losses = [], []

    if val_loss_file.exists():
        with open(val_loss_file) as f:
            val_loss = json.load(f)
        val_steps, val_losses = zip(*val_loss) if val_loss else ([], [])
    else:
        logger.warning(f"Validation loss file not found: {val_loss_file}")
        val_steps, val_losses = [], []

    if train_steps:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_steps, train_losses, linewidth=0.8, alpha=0.7, label="Training Loss")
        if val_steps:
            ax.plot(val_steps, val_losses, linewidth=2, marker="o", markersize=3,
                    label="Validation Loss", color="red")
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax.set_title(lang.title("Training & Validation Loss"), fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = out_dir / "loss_curve.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Loss curve saved: {save_path}")

    logger.info("Done plotting.")


def plot_from_checkpoint(lang: Language, checkpoint_path: str, output_dir: str = None):
    """
    Plot loss curves directly from a checkpoint file's embedded history.

    Every checkpoint carries its own loss history, so a run interrupted before
    the final log flush can still be plotted.

    Args:
        lang:            Language whose curves are plotted.
        checkpoint_path: Path to checkpoint .pt file.
        output_dir:      Directory to save plots.
    """
    import torch
    plt = _import_plotting()

    out_dir = Path(output_dir) if output_dir else lang.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_loss = checkpoint.get("loss_history", [])
    val_loss = checkpoint.get("val_loss_history", [])

    if not train_loss:
        logger.warning("No training loss history in checkpoint.")
        return

    train_steps, train_losses = zip(*train_loss)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_losses, linewidth=0.8, alpha=0.7, label="Training Loss")

    if val_loss:
        val_steps, val_losses = zip(*val_loss)
        ax.plot(val_steps, val_losses, linewidth=2, marker="o", markersize=3,
                label="Validation Loss", color="red")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(lang.title("Training & Validation Loss"), fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = out_dir / "loss_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss curve saved: {save_path}")
