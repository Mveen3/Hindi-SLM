"""
Pretraining Loop
================

End-to-end pretraining of the decoder-only Transformer on one language's
corpus. The language is passed in, so the same loop trains Model H and
Model L from their own configs, corpora and tokenizers into their own
checkpoint directories — no state is shared between the two runs.

Features:
    - Loads model and training config from YAML.
    - Builds model, optimizer (AdamW), cosine LR scheduler with warmup.
    - Streams data from JSONL via the LMDataset.
    - Gradient accumulation for effective larger batch sizes.
    - Mixed-precision training (torch.amp) for GPU memory efficiency.
    - Periodic validation evaluation (cross-entropy loss).
    - Full checkpoint saving and automatic resume from latest checkpoint.
    - CSV/JSON logging of losses for later plotting.

Invoked through ``python main.py train --lang {hindi,nepali}``.
"""

import os
import sys
import time
import json
import torch
import logging
from tqdm import tqdm
from pathlib import Path

from src.languages import Language
from src.model.transformer import count_parameters, build_model_from_config
from src.data.dataset import create_dataloader
from src.train.utils import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    get_lr_scheduler,
    AverageMeter,
    save_training_log,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=50):
    """
    Run validation and compute average cross-entropy loss.

    Args:
        model:       The model (set to eval mode inside).
        dataloader:  Validation DataLoader.
        device:      Device to run on.
        max_batches: Maximum number of batches to evaluate (for speed).

    Returns:
        Average validation loss (float).
    """
    model.eval()
    meter = AverageMeter("val_loss")

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            _, loss = model(input_ids, targets=targets)

        if loss is not None:
            meter.update(loss.item(), n=input_ids.size(0))

    model.train()
    return meter.avg


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(lang: Language, config: dict):
    """
    Main pretraining function.

    Args:
        lang:   The language being trained — supplies every path (corpus,
                tokenizer, checkpoint and log directory) so this loop never
                hard-codes a language.
        config: Merged model + training configuration dictionary.
    """
    # ----- Device setup -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ----- Seed for reproducibility -----
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----- Resolve paths through the language registry -----
    train_data = str(lang.root / config["train_data"])
    val_data = str(lang.root / config["val_data"])
    tokenizer_path = str(lang.tokenizer_path)
    checkpoint_dir = str(lang.root / config.get("checkpoint_dir", "checkpoints"))
    log_dir = str(lang.log_dir)
    logger.info("Training %s (%s)", lang.display, lang.model_label)

    # ----- Build model -----
    model = build_model_from_config(config)
    model = model.to(device)

    param_info = count_parameters(model)
    logger.info(f"Model built with {param_info['trainable']:,} trainable parameters")
    logger.info(f"Parameter breakdown: {json.dumps(param_info['by_module'], indent=2)}")

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config.get("betas", [0.9, 0.95])),
        eps=config.get("eps", 1e-8),
    )

    # ----- LR Scheduler -----
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
    )

    # ----- AMP Scaler -----
    use_amp = config.get("use_amp", True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ----- Resume from checkpoint -----
    global_step = 0
    loss_history = []
    val_loss_history = []

    resume_from = config.get("resume_from", "latest")
    if resume_from == "latest":
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
    elif resume_from and Path(resume_from).exists():
        ckpt_path = resume_from
    else:
        ckpt_path = None

    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt_info = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, device=device
        )
        global_step = ckpt_info["step"]
        loss_history = ckpt_info.get("loss_history", [])
        val_loss_history = ckpt_info.get("val_loss_history", [])
        logger.info(f"Resumed at step {global_step}")
    else:
        logger.info("Starting training from scratch.")

    # ----- Data loaders -----
    max_seq_len = config.get("max_seq_len", 512)
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 2)

    train_loader = create_dataloader(
        data_path=train_data,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        is_eval=False,
        bos_token_id=config.get("bos_token_id", 2),
        eos_token_id=config.get("eos_token_id", 3),
    )

    val_loader = create_dataloader(
        data_path=val_data,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=0,  # Single worker for eval
        is_eval=True,
        pad_token_id=config.get("pad_token_id", 0),
        eos_token_id=config.get("eos_token_id", 3),
    )

    # ----- Training loop -----
    max_steps = config["max_steps"]
    grad_accum_steps = config.get("gradient_accumulation_steps", 4)
    eval_interval = config.get("eval_interval", 1000)
    log_interval = config.get("log_interval", 100)
    save_interval = config.get("save_interval", 5000)
    max_grad_norm = config.get("max_grad_norm", 1.0)

    model.train()
    optimizer.zero_grad()
    loss_meter = AverageMeter("train_loss")
    accum_loss = 0.0
    micro_step = 0
    start_time = time.time()

    logger.info(f"Training for {max_steps} steps (effective batch = {batch_size * grad_accum_steps})")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logger.info(f"Mixed precision: {use_amp}")

    train_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, initial=global_step,
                desc=f"{lang.display} training", dynamic_ncols=True)

    while global_step < max_steps:
        # Get next batch (restart iterator if exhausted = new epoch)
        try:
            batch = next(train_iter)
        except StopIteration:
            logger.info("Data iterator exhausted — starting new epoch")
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(input_ids, targets=targets)
            loss = loss / grad_accum_steps  # Scale for gradient accumulation

        # Backward pass
        scaler.scale(loss).backward()
        accum_loss += loss.item()
        micro_step += 1

        # Optimizer step after accumulation
        if micro_step % grad_accum_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
            pbar.update(1)
            step_loss = accum_loss  # Already accumulated and scaled
            loss_meter.update(step_loss)
            accum_loss = 0.0

            # Logging
            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step}/{max_steps} | "
                    f"Loss: {step_loss:.4f} | "
                    f"Avg Loss: {loss_meter.avg:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.0f}s"
                )
                loss_history.append((global_step, step_loss))

            # Validation
            if global_step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                logger.info(f"Step {global_step} | Validation Loss: {val_loss:.4f}")
                val_loss_history.append((global_step, val_loss))
                model.train()

            # Save checkpoint
            if global_step % save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=global_step,
                    config=config,
                    loss_history=loss_history,
                    val_loss_history=val_loss_history,
                    checkpoint_dir=checkpoint_dir,
                )

            if global_step >= max_steps:
                break

    # ----- Final save -----
    pbar.close()
    logger.info("Training complete!")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=global_step,
        config=config,
        loss_history=loss_history,
        val_loss_history=val_loss_history,
        checkpoint_dir=checkpoint_dir,
    )
    save_training_log(loss_history, val_loss_history, log_dir)

    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time / 3600:.2f} hours")
    logger.info(f"Final training loss: {loss_meter.avg:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(lang: Language, model_config: str = None, train_config: str = None):
    """Load this language's configs and launch pretraining.

    Args:
        lang:         Language to pretrain.
        model_config: Optional override for model_config.yaml.
        train_config: Optional override for train_config.yaml.
    """
    config = lang.load_config(model_config, train_config)
    logger.info(f"Configuration:\n{json.dumps(config, indent=2, default=str)}")
    train(lang, config)


def main():
    import argparse
    from src import languages
    parser = argparse.ArgumentParser(description="Pretrain one model locally.")
    languages.add_language_arg(parser)
    parser.add_argument("--model-config", default=None,
                        help="Override <language>/configs/model_config.yaml.")
    parser.add_argument("--train-config", default=None,
                        help="Override <language>/configs/train_config.yaml.")
    args = parser.parse_args()

    run(languages.get(args.lang), args.model_config, args.train_config)


if __name__ == "__main__":
    main()

