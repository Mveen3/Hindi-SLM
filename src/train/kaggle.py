"""
Kaggle Pretraining Runner
=========================

The local trainer in :mod:`src.train.trainer` targets a single 4 GB card. This
module runs the same pretraining on Kaggle's 2x T4 (15 GB each), which needs
three things the local loop does not:

  1. ``nn.DataParallel`` across both GPUs, with per-GPU batch sizing that fits
     15 GB once the logits gradients are accounted for.
  2. HuggingFace Hub sync — checkpoints and logs are pushed as they are written,
     so a killed session loses nothing and the next one resumes from the Hub.
  3. A session-time guard that saves and exits before Kaggle's 12-hour cut-off.

Resume order is local checkpoint first, then the Hub. Every checkpoint carries
model weights, optimizer, scheduler, AMP scaler, step and config, so a resumed
run continues exactly where the last one stopped.

Setup on Kaggle:
    1. Add ``HF_TOKEN`` as a Kaggle Secret.
    2. Enable the T4 x2 accelerator.
    3. Run ``notebooks/kaggle_runner.ipynb``.

Invoked through ``python main.py train-kaggle --lang {hindi,nepali}``.
"""

import os
import re
import json
import time
import shutil
import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from huggingface_hub import HfApi, hf_hub_download

from src.languages import Language, PROJECT_ROOT
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

logger = logging.getLogger("kaggle_train")

IS_KAGGLE = os.path.exists("/kaggle/working")
# Kaggle kills sessions at 12h; stop early enough to save and upload.
MAX_SESSION_SECONDS = 11.5 * 3600


def resolve_hf_token() -> str:
    """Read HF_TOKEN from Kaggle secrets when on Kaggle, else from .env."""
    if IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            return os.environ.get("HF_TOKEN", "").strip()

    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    return os.environ.get("HF_TOKEN", "").strip()


# ---------------------------------------------------------------------------
# HuggingFace Hub sync
# ---------------------------------------------------------------------------

class HubSync:
    """
    Pushes and pulls one language's checkpoints and logs to the Hub.

    Every path is namespaced under the language key, so Model H and Model L
    keep separate weights even when they share a repo owner. All methods are
    best-effort: a Hub failure logs a warning and training continues, since
    losing a sync is much cheaper than losing the run.

    Args:
        lang:  Language whose artifacts are synced.
        token: HuggingFace token; sync is disabled when empty.
    """

    def __init__(self, lang: Language, token: str):
        self.lang = lang
        self.repo_id = None
        self.api = None

        if not token:
            logger.warning("No HF_TOKEN found — checkpoints will NOT be synced to the Hub.")
            return

        try:
            from huggingface_hub import login
            login(token=token.strip())
            self.api = HfApi()
            username = self.api.whoami()["name"]
            self.repo_id = f"{username}/src-{lang.key}-checkpoints"
            self.api.create_repo(repo_id=self.repo_id, repo_type="model",
                                 exist_ok=True, private=True)
            logger.info("HuggingFace repo ready: %s", self.repo_id)
        except Exception as exc:
            logger.warning("HuggingFace setup failed (%s) — continuing without sync.", exc)
            self.repo_id = None

    @property
    def enabled(self) -> bool:
        return self.repo_id is not None

    def upload_checkpoint(self, checkpoint_path: str, step: int):
        """Upload one checkpoint plus the latest-checkpoint pointer."""
        if not self.enabled:
            return
        try:
            filename = Path(checkpoint_path).name
            self.api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=f"{self.lang.key}/{filename}",
                repo_id=self.repo_id,
                commit_message=f"{self.lang.display} checkpoint at step {step}",
            )
            logger.info("Uploaded %s to the Hub", filename)

            pointer = self.lang.checkpoint_dir / "latest_checkpoint.txt"
            if pointer.exists():
                self.api.upload_file(
                    path_or_fileobj=str(pointer),
                    path_in_repo=f"{self.lang.key}/latest_checkpoint.txt",
                    repo_id=self.repo_id,
                    commit_message=f"Update latest pointer to step {step}",
                )
        except Exception as exc:
            logger.warning("Checkpoint upload failed: %s", exc)

    def upload_logs(self):
        """Upload the loss histories so curves survive a lost session."""
        if not self.enabled:
            return
        try:
            for name in ("train_loss.json", "val_loss.json"):
                path = self.lang.log_dir / name
                if path.exists():
                    self.api.upload_file(
                        path_or_fileobj=str(path),
                        path_in_repo=f"{self.lang.key}/logs/{name}",
                        repo_id=self.repo_id,
                        commit_message="Update training logs",
                    )
            logger.info("Uploaded training logs to the Hub")
        except Exception as exc:
            logger.warning("Log upload failed: %s", exc)

    def download_latest_checkpoint(self):
        """Fetch the highest-step checkpoint from the Hub, or None."""
        if not self.enabled:
            return None
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id)
        except Exception:
            logger.info("No existing Hub repo contents — starting fresh.")
            return None

        pattern = re.compile(rf"{self.lang.key}/checkpoint_step_(\d+)\.pt")
        steps = [(int(m.group(1)), f) for f in files if (m := pattern.fullmatch(f))]
        if not steps:
            logger.info("No checkpoints on the Hub for %s.", self.lang.key)
            return None

        latest_step, latest_file = max(steps)
        logger.info("Found Hub checkpoint: step %d", latest_step)

        try:
            cached = hf_hub_download(repo_id=self.repo_id, filename=latest_file)
            target = self.lang.checkpoint_dir / Path(latest_file).name
            self.lang.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, target)
            (self.lang.checkpoint_dir / "latest_checkpoint.txt").write_text(str(target))

            for name in ("train_loss.json", "val_loss.json"):
                remote = f"{self.lang.key}/logs/{name}"
                if remote in files:
                    try:
                        log_cached = hf_hub_download(repo_id=self.repo_id, filename=remote)
                        self.lang.log_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(log_cached, self.lang.log_dir / name)
                    except Exception:
                        pass

            logger.info("Downloaded checkpoint to %s", target)
            return str(target)
        except Exception as exc:
            logger.warning("Checkpoint download failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_kaggle_config(lang: Language) -> dict:
    """
    Load the language's configs and apply Kaggle hardware overrides.

    Batch sizing is set from the visible GPU count. On 2x T4, 64 sequences per
    GPU is the largest that fits: 128 per GPU OOMs once the logits gradients
    are materialised.

    Args:
        lang: Language whose configs are loaded.

    Returns:
        The merged and overridden configuration dictionary.
    """
    config = lang.load_config()

    n_gpus = torch.cuda.device_count()
    logger.info("Detected %d GPU(s)", n_gpus)

    if n_gpus >= 2:
        config["batch_size"] = 64                   # 32 per GPU across 2 GPUs (lowered to fix OOM)
        config["gradient_accumulation_steps"] = 4   # effective batch = 256
    elif n_gpus == 1:
        config["batch_size"] = 32
        config["gradient_accumulation_steps"] = 8   # effective batch = 256
    else:
        config["batch_size"] = 8
        config["gradient_accumulation_steps"] = 32

    config["num_workers"] = 4
    config["max_steps"] = 40000
    config["save_interval"] = 2000    # frequent saves, given the session cap
    config["eval_interval"] = 1000
    config["log_interval"] = 50
    config["checkpoint_dir"] = str(lang.checkpoint_dir)
    config["log_dir"] = str(lang.log_dir)

    return config


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, max_batches: int = 50) -> float:
    """Compute mean validation cross-entropy over a bounded number of batches.

    Args:
        model:       The (possibly DataParallel-wrapped) model.
        dataloader:  Validation loader.
        device:      Device to run on.
        max_batches: Cap on batches, to keep mid-training evals cheap.

    Returns:
        Average validation loss.
    """
    was_training = model.training
    model.eval()
    meter = AverageMeter("val_loss")

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            _, loss = model(input_ids, targets=targets)
            # DataParallel returns one loss per GPU.
            if loss is not None and loss.dim() > 0:
                loss = loss.mean()

        if loss is not None:
            meter.update(loss.item(), n=input_ids.size(0))

    if was_training:
        model.train()
    return meter.avg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(lang: Language) -> int:
    """
    Pretrain one model on Kaggle, syncing to the Hub as it goes.

    Args:
        lang: Language to pretrain. Model H and Model L are separate runs
              writing to separate checkpoint directories and Hub repos.

    Returns:
        The global step reached when training stopped.
    """
    hub = HubSync(lang, resolve_hf_token())

    config = get_kaggle_config(lang)
    logger.info("Configuration:\n%s", json.dumps(config, indent=2, default=str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    logger.info("Using device: %s", device)
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info("GPU %d: %s (%.1f GB)", i, props.name, props.total_memory / 1e9)

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lang.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lang.log_dir.mkdir(parents=True, exist_ok=True)

    train_data = str(lang.root / config["train_data"])
    val_data = str(lang.root / config["val_data"])
    tokenizer_path = str(lang.tokenizer_path)

    model = build_model_from_config(config).to(device)
    param_info = count_parameters(model)
    logger.info("Model: %s trainable parameters", f"{param_info['trainable']:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config.get("betas", [0.9, 0.95])),
        eps=config.get("eps", 1e-8),
    )
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
    )
    use_amp = config.get("use_amp", True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ----- Resume: local checkpoint first, then the Hub -----
    global_step = 0
    loss_history = []
    val_loss_history = []

    ckpt_path = find_latest_checkpoint(str(lang.checkpoint_dir))
    if ckpt_path is None and hub.enabled:
        logger.info("No local checkpoint — checking the Hub ...")
        ckpt_path = hub.download_latest_checkpoint()

    if ckpt_path:
        logger.info("Resuming from checkpoint: %s", ckpt_path)
        ckpt_info = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device=device)
        global_step = ckpt_info["step"]
        loss_history = ckpt_info.get("loss_history", [])
        val_loss_history = ckpt_info.get("val_loss_history", [])
        logger.info("Resumed at step %d", global_step)
    else:
        logger.info("Starting training from scratch.")

    if n_gpus > 1:
        logger.info("Wrapping model with DataParallel across %d GPUs", n_gpus)
        model = nn.DataParallel(model)

    def raw_model():
        """The unwrapped module, for checkpointing and gradient clipping."""
        return model.module if isinstance(model, nn.DataParallel) else model

    max_seq_len = config.get("max_seq_len", 512)
    batch_size = config["batch_size"]

    train_loader = create_dataloader(
        data_path=train_data, tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len, batch_size=batch_size,
        num_workers=config.get("num_workers", 4), is_eval=False,
        bos_token_id=config.get("bos_token_id", 2),
        eos_token_id=config.get("eos_token_id", 3),
    )
    val_loader = create_dataloader(
        data_path=val_data, tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len, batch_size=batch_size,
        num_workers=0, is_eval=True,
        pad_token_id=config.get("pad_token_id", 0),
        eos_token_id=config.get("eos_token_id", 3),
    )

    max_steps = config["max_steps"]
    grad_accum_steps = config.get("gradient_accumulation_steps", 2)
    eval_interval = config.get("eval_interval", 1000)
    log_interval = config.get("log_interval", 50)
    save_interval = config.get("save_interval", 2000)
    max_grad_norm = config.get("max_grad_norm", 1.0)

    def persist(step: int):
        """Write a checkpoint and logs locally, then push both to the Hub."""
        save_checkpoint(
            model=raw_model(), optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            step=step, config=config, loss_history=loss_history,
            val_loss_history=val_loss_history, checkpoint_dir=str(lang.checkpoint_dir),
        )
        save_training_log(loss_history, val_loss_history, str(lang.log_dir))
        if hub.enabled:
            hub.upload_checkpoint(str(lang.checkpoint_dir / f"checkpoint_step_{step}.pt"), step)
            hub.upload_logs()

    model.train()
    optimizer.zero_grad()
    loss_meter = AverageMeter("train_loss")
    accum_loss = 0.0
    micro_step = 0
    start_time = session_start = time.time()

    logger.info("Training for %d steps (effective batch = %d)",
                max_steps, batch_size * grad_accum_steps)
    logger.info("Gradient accumulation: %d | AMP: %s", grad_accum_steps, use_amp)
    logger.info("Session limit: %.1f hours", MAX_SESSION_SECONDS / 3600)

    train_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, initial=global_step,
                desc=f"{lang.display} training", dynamic_ncols=True)

    while global_step < max_steps:
        elapsed_session = time.time() - session_start
        if elapsed_session > MAX_SESSION_SECONDS:
            logger.warning("Approaching the session limit (%.1fh) — saving and stopping.",
                           elapsed_session / 3600)
            break

        try:
            batch = next(train_iter)
        except StopIteration:
            logger.info("Data iterator exhausted — starting a new epoch")
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(input_ids, targets=targets)
            if loss.dim() > 0:            # DataParallel: one loss per GPU
                loss = loss.mean()
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step % grad_accum_steps != 0:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model().parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        global_step += 1
        pbar.update(1)
        step_loss = accum_loss
        loss_meter.update(step_loss)
        accum_loss = 0.0

        pbar.set_postfix({
            "loss": f"{step_loss:.4f}",
            "avg": f"{loss_meter.avg:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

        if global_step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = global_step / max(elapsed, 1)
            eta_hours = (max_steps - global_step) / max(steps_per_sec, 1e-3) / 3600
            logger.info(
                "Step %d/%d | Loss: %.4f | Avg: %.4f | LR: %.2e | %.1f steps/s | ETA: %.1fh",
                global_step, max_steps, step_loss, loss_meter.avg,
                scheduler.get_last_lr()[0], steps_per_sec, eta_hours,
            )
            loss_history.append((global_step, step_loss))

        if global_step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            logger.info("Step %d | Validation Loss: %.4f", global_step, val_loss)
            val_loss_history.append((global_step, val_loss))
            model.train()

        if global_step % save_interval == 0:
            persist(global_step)

    pbar.close()
    logger.info("Training complete (or session limit reached).")
    persist(global_step)

    logger.info("Total training time: %.2f hours", (time.time() - start_time) / 3600)
    logger.info("Final step: %d | Final avg loss: %.4f", global_step, loss_meter.avg)
    if hub.enabled:
        logger.info("Checkpoints: https://huggingface.co/%s", hub.repo_id)

    return global_step


def main():
    import argparse
    from src import languages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="Pretrain one model on Kaggle with Hub sync.")
    languages.add_language_arg(parser)
    args = parser.parse_args()

    lang = languages.get(args.lang)
    logging.info("=" * 60)
    logging.info("  KAGGLE TRAINING — %s (%s)", lang.display, lang.model_label)
    logging.info("=" * 60)

    final_step = train(lang)
    logging.info("Done. Trained to step %d.", final_step)


if __name__ == "__main__":
    main()

