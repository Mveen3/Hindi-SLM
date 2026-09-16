"""
Reasoning Finetuning Loop (Phase 3)
===================================

Finetunes one pretrained model on its own synthetic reasoning corpus. The two
models are finetuned independently: each starts from its own Phase 2
checkpoint, keeps its own frozen Phase 1 tokenizer and vocabulary, reads its
own reasoning data, and writes to its own finetuned-checkpoint directory.
Nothing is shared between Model H and Model L.

Protocol (from the Phase 3 brief):
    - start from that language's own pretrained checkpoint,
    - keep that language's tokenizer and vocabulary fixed,
    - document every hyperparameter (``<language>/configs/finetune_config.yaml``),
    - save checkpoints in the same resume-capable format as pretraining.

Two things differ from ``src/train/trainer.py`` beyond the data:

**Weights are inherited, optimizer state is not.** Finetuning is a new
objective on a new distribution with its own (much smaller) LR schedule, so
Adam moments from a 40k-step pretraining run would actively fight the first
finetuning steps. The pretrained *weights* are loaded; the optimizer,
scheduler and AMP scaler all start fresh.

**Finetuned checkpoints never overwrite pretrained ones.** They go to
``<language>/checkpoints_finetune/``, so the Phase 2 checkpoint stays intact
and available as the "pretrained" arm of the evaluation comparison.

Resume works exactly as in Phase 2: an interrupted run restarts from the
newest checkpoint in the finetune directory, restoring weights, optimizer,
scheduler, scaler, step and config.

Invoked through ``python main.py finetune --lang {hindi,nepali}``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.languages import Language
from src.model.transformer import build_model_from_config, count_parameters
from src.finetune.dataset import create_finetune_dataloader
from src.train.utils import (
    AverageMeter,
    find_latest_checkpoint,
    get_lr_scheduler,
    load_checkpoint,
    save_checkpoint,
    save_training_log,
)

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
def evaluate(model, dataloader, device, use_amp: bool, max_batches: int | None = None) -> float:
    """Average cross-entropy on the reasoning validation split.

    Only answer tokens count: the dataset already masked prompt positions to
    ``pad_token_id``, which the model's loss ignores. So this number is
    directly "how well does it predict the reasoning and the answer", not
    "how well does it model the question text".

    Args:
        model:       The model (restored to train mode before returning).
        dataloader:  Validation loader.
        device:      Device to run on.
        use_amp:     Whether to run under autocast.
        max_batches: Optional cap for a faster mid-training check.

    Returns:
        Mean validation loss.
    """
    model.eval()
    meter = AverageMeter("val_loss")

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(input_ids, targets=targets)

        if loss is not None:
            meter.update(loss.item(), n=input_ids.size(0))

    model.train()
    return meter.avg


# ---------------------------------------------------------------------------
# Pretrained-weight loading
# ---------------------------------------------------------------------------

def load_pretrained_weights(model, checkpoint_path: str, device) -> dict:
    """Load Phase 2 weights into the model, without any optimizer state.

    Args:
        model:           Freshly built model to populate.
        checkpoint_path: The language's pretrained checkpoint.
        device:          Device the model lives on.

    Returns:
        The pretrained checkpoint's config, for a compatibility check.

    Raises:
        SystemExit: If the checkpoint's architecture does not match the
                    model that was just built — silently loading a mismatched
                    vocabulary would produce nonsense rather than an error.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained_config = checkpoint.get("config", {})

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    step = checkpoint.get("step", "?")
    logger.info("Loaded pretrained weights from %s (pretraining step %s)",
                checkpoint_path, step)

    del checkpoint
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pretrained_config


def check_architecture_match(config: dict, pretrained_config: dict) -> None:
    """Fail loudly if the finetuning config contradicts the checkpoint.

    The vocabulary in particular must be identical: the brief requires the
    tokenizer and vocabulary to stay fixed across finetuning, and a mismatch
    would mean the embedding rows no longer correspond to the same tokens.
    """
    critical = ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len")
    mismatches = [
        f"{key}: checkpoint={pretrained_config[key]} config={config.get(key)}"
        for key in critical
        if key in pretrained_config and pretrained_config[key] != config.get(key)
    ]
    if mismatches:
        raise SystemExit(
            "Finetuning config does not match the pretrained checkpoint:\n  - "
            + "\n  - ".join(mismatches)
            + "\nThe tokenizer and architecture must stay fixed across finetuning."
        )


# ---------------------------------------------------------------------------
# Main finetuning loop
# ---------------------------------------------------------------------------

def finetune(lang: Language, config: dict, pretrained_checkpoint: str,
             limit: int | None = None) -> dict:
    """Finetune one model on its own reasoning corpus.

    Args:
        lang:                  Language to finetune — supplies every path.
        config:                Merged model + finetuning configuration.
        pretrained_checkpoint: Phase 2 checkpoint to start from.
        limit:                 Optional cap on training examples (smoke runs).

    Returns:
        A summary dict with the final step, losses and checkpoint directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Finetuning %s (%s) on device %s", lang.display, lang.model_label, device)
    if device.type == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint_dir = str(lang.finetune_checkpoint_dir)
    log_dir = str(lang.finetune_log_dir)

    # ----- Model -----
    model = build_model_from_config(config).to(device)
    param_info = count_parameters(model)
    logger.info("Model: %s trainable parameters", f"{param_info['trainable']:,}")

    # ----- Data (loaded first: the epoch count sets the LR schedule length) -----
    train_loader, train_dataset = create_finetune_dataloader(
        lang, "train", config, shuffle=True, limit=limit)
    val_loader, val_dataset = create_finetune_dataloader(
        lang, "val", config, shuffle=False)

    logger.info("Reasoning data: %s train / %s val examples (%s dropped as over-length)",
                f"{len(train_dataset):,}", f"{len(val_dataset):,}", train_dataset.dropped)

    grad_accum = config.get("gradient_accumulation_steps", 8)
    epochs = config.get("epochs", 3)
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    max_steps = config.get("max_steps") or steps_per_epoch * epochs
    warmup_steps = config.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = max(10, int(0.03 * max_steps))

    # ----- Optimizer / scheduler / scaler (fresh — see module docstring) -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        betas=tuple(config.get("betas", [0.9, 0.95])),
        eps=config.get("eps", 1e-8),
    )
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)
    use_amp = config.get("use_amp", True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ----- Resume, or start from the pretrained weights -----
    global_step = 0
    loss_history: list = []
    val_loss_history: list = []

    resume_path = None
    if config.get("resume_from", "latest") == "latest":
        resume_path = find_latest_checkpoint(checkpoint_dir)
    elif config.get("resume_from") and Path(config["resume_from"]).exists():
        resume_path = config["resume_from"]

    if resume_path:
        logger.info("Resuming interrupted finetuning from %s", resume_path)
        info = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device=device)
        global_step = info["step"]
        loss_history = info.get("loss_history", [])
        val_loss_history = info.get("val_loss_history", [])
        logger.info("Resumed at finetuning step %d", global_step)
    else:
        pretrained_config = load_pretrained_weights(model, pretrained_checkpoint, device)
        check_architecture_match(config, pretrained_config)
        logger.info("Starting finetuning from the pretrained checkpoint "
                    "(fresh optimizer, scheduler and AMP scaler).")

    # Record provenance in the checkpoint config, so a finetuned checkpoint
    # always says which pretrained run it came from.
    config = dict(config)
    config["pretrained_checkpoint"] = str(pretrained_checkpoint)
    config["finetune_max_steps"] = max_steps
    config["finetune_warmup_steps"] = warmup_steps
    config["finetune_epochs"] = epochs

    # ----- Training -----
    eval_interval = config.get("eval_interval", 100)
    log_interval = config.get("log_interval", 25)
    save_interval = config.get("save_interval", 200)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    eval_batches = config.get("eval_max_batches", 50)

    logger.info("Finetuning for %d steps (%d epochs x %d steps), effective batch = %d, "
                "warmup = %d", max_steps, epochs, steps_per_epoch,
                config["batch_size"] * grad_accum, warmup_steps)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_meter = AverageMeter("train_loss")
    accum_loss = 0.0
    micro_step = 0
    start_time = time.time()
    best_val = float("inf")

    pbar = tqdm(total=max_steps, initial=global_step,
                desc=f"{lang.display} finetuning", dynamic_ncols=True)

    epoch = 0
    done = False
    while not done:
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(input_ids, targets=targets)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            pbar.update(1)
            loss_meter.update(accum_loss)
            step_loss, accum_loss = accum_loss, 0.0

            if global_step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info("Step %d/%d | Loss: %.4f | Avg: %.4f | LR: %.2e | %.0fs",
                            global_step, max_steps, step_loss, loss_meter.avg,
                            lr, time.time() - start_time)
                loss_history.append((global_step, step_loss))
                pbar.set_postfix(loss=f"{step_loss:.3f}", avg=f"{loss_meter.avg:.3f}")

            if global_step % eval_interval == 0 or global_step == max_steps:
                val_loss = evaluate(model, val_loader, device, use_amp, eval_batches)
                val_loss_history.append((global_step, val_loss))
                marker = ""
                if val_loss < best_val:
                    best_val, marker = val_loss, "  <- best so far"
                logger.info("Step %d | Validation loss: %.4f%s", global_step, val_loss, marker)

            if global_step % save_interval == 0:
                save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                scaler=scaler, step=global_step, config=config,
                                loss_history=loss_history, val_loss_history=val_loss_history,
                                checkpoint_dir=checkpoint_dir)

            if global_step >= max_steps:
                done = True
                break

        epoch += 1
        if epoch >= epochs and not done:
            done = True

    pbar.close()

    # ----- Final checkpoint and logs -----
    final_val = evaluate(model, val_loader, device, use_amp, max_batches=None)
    val_loss_history.append((global_step, final_val))
    logger.info("Final validation loss (full split): %.4f", final_val)

    save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    step=global_step, config=config, loss_history=loss_history,
                    val_loss_history=val_loss_history, checkpoint_dir=checkpoint_dir)
    save_training_log(loss_history, val_loss_history, log_dir)

    elapsed = time.time() - start_time
    logger.info("Finetuning complete in %.1f min | final train loss %.4f | val loss %.4f",
                elapsed / 60, loss_meter.avg, final_val)
    logger.info("Checkpoints: %s", checkpoint_dir)

    return {
        "language": lang.key,
        "model": lang.model_label,
        "steps": global_step,
        "epochs": epochs,
        "final_train_loss": round(loss_meter.avg, 4),
        "final_val_loss": round(final_val, 4),
        "best_val_loss": round(min(best_val, final_val), 4),
        "minutes": round(elapsed / 60, 2),
        "checkpoint_dir": checkpoint_dir,
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "train_examples": len(train_dataset),
        "effective_batch": config["batch_size"] * grad_accum,
    }


def run(lang: Language, pretrained_checkpoint: str, config_path: str | None = None,
        limit: int | None = None, overrides: dict | None = None) -> dict:
    """Load this language's finetuning config and launch finetuning.

    Args:
        lang:                  Language to finetune.
        pretrained_checkpoint: Phase 2 checkpoint to start from.
        config_path:           Optional override for finetune_config.yaml.
        limit:                 Optional cap on training examples.
        overrides:             CLI overrides merged over the config file.

    Returns:
        The finetuning summary dictionary.
    """
    config = lang.load_finetune_config(config_path)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    logger.info("Finetuning configuration:\n%s", json.dumps(config, indent=2, default=str))
    return finetune(lang, config, pretrained_checkpoint, limit=limit)


def main():
    import argparse
    from src import languages
    from src.train.utils import resolve_checkpoint

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Finetune one pretrained model on reasoning tasks.")
    languages.add_language_arg(parser)
    parser.add_argument("--checkpoint", default=None,
                        help="Pretrained checkpoint to start from (default: newest in <language>/checkpoints/).")
    parser.add_argument("--config", default=None, help="Path to a finetune_config.yaml override.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                        help="Override the micro-batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, dest="learning_rate",
                        help="Override the peak learning rate.")
    parser.add_argument("--max-steps", type=int, default=None, dest="max_steps",
                        help="Hard cap on optimizer steps.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N training examples (smoke runs).")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing finetuning checkpoint and restart.")
    args = parser.parse_args()

    lang = languages.get(args.lang)
    pretrained = resolve_checkpoint(lang, args.checkpoint)

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
    }
    if args.fresh:
        overrides["resume_from"] = None

    summary = run(lang, pretrained, config_path=args.config, limit=args.limit, overrides=overrides)
    print("\n" + "=" * 62)
    print(f" Finetuning summary — {summary['model']} ({summary['language']})")
    print("=" * 62)
    for key in ("steps", "epochs", "train_examples", "effective_batch",
                "final_train_loss", "final_val_loss", "minutes"):
        print(f"  {key:<18} {summary[key]}")
    print(f"  checkpoints        {summary['checkpoint_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

