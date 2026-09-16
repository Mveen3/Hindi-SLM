"""
Intrinsic Language Model Evaluation
==================================

Compute validation/test cross-entropy loss, perplexity (PPL), and
bits-per-byte (BPB) on held-out data.

Metrics:
    - Cross-entropy loss: average per-token negative log-likelihood.
    - Perplexity: PPL = exp(cross_entropy_loss).
    - Bits-per-byte: total_CE_nats / (total_bytes × ln2).
      BPB normalises for tokeniser differences, making Model H and Model L
      directly comparable.

Invoked through ``python main.py evaluate --lang {hindi,nepali} --only ppl``.
"""

import math
import json
import torch
import logging
from torch.utils.data import DataLoader

from src.languages import Language
from src.model.transformer import build_model_from_config
from src.data.dataset import EvalLMDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Tokens per cross-entropy chunk — bounds the float32 upcast to ~160 MB.
CE_CHUNK = 4096


@torch.no_grad()
def evaluate_lm(lang: Language, checkpoint_path: str,
                data_split: str = "test", max_batches: int = None):
    """
    Evaluate a trained model on its own held-out data.

    Args:
        lang:            Language whose corpus, tokenizer and report directory
                         are used. Each model is only ever scored on its own
                         held-out split.
        checkpoint_path: Path to the model checkpoint.
        data_split:      Which data split to evaluate ("val" or "test").
        max_batches:     Maximum batches to process (None = all).

    Returns:
        Dictionary with CE loss, PPL, BPB, and counts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Build model
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Model loaded from checkpoint.")

    # Resolve this language's own held-out split and tokenizer
    data_file = lang.split_file(data_split)
    tokenizer_path = str(lang.tokenizer_path)

    # Create evaluation dataset
    dataset = EvalLMDataset(
        data_path=str(data_file),
        tokenizer_path=tokenizer_path,
        max_seq_len=config.get("max_seq_len", 512),
        pad_token_id=config.get("pad_token_id", 0),
        eos_token_id=config.get("eos_token_id", 3),
    )

    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True)

    # Accumulators
    total_loss_nats = 0.0       # Sum of per-token CE losses (in nats)
    total_tokens = 0            # Total non-padding tokens
    total_bytes = 0             # Total UTF-8 bytes of original text

    pad_id = config.get("pad_token_id", 0)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        num_tokens_batch = batch["num_tokens"]  # per-sample token counts
        num_bytes_batch = batch["num_bytes"]     # per-sample byte counts

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(input_ids)

        # Compute per-token cross-entropy (no reduction)
        B, T, V = logits.shape
        # Cross-entropy is summed in float32 and in chunks, for two reasons:
        #   1. Under AMP the logits are fp16; a reduction="sum" over ~16k tokens
        #      overflows fp16's 65504 ceiling and silently yields inf.
        #   2. Upcasting the whole (B*T, V) tensor at once needs ~660 MB here,
        #      which does not fit alongside the model on a 4 GB card.
        flat_logits = logits.view(-1, V)
        flat_targets = targets.view(-1)
        ce_sum = 0.0
        for i in range(0, flat_logits.size(0), CE_CHUNK):
            ce_sum += torch.nn.functional.cross_entropy(
                flat_logits[i:i + CE_CHUNK].float(),
                flat_targets[i:i + CE_CHUNK],
                ignore_index=pad_id,
                reduction="sum",
            ).item()

        # Count valid (non-padding) tokens in this batch
        valid_mask = (targets != pad_id)
        n_valid = valid_mask.sum().item()

        total_loss_nats += ce_sum
        total_tokens += n_valid
        total_bytes += num_bytes_batch.sum().item()

        if (batch_idx + 1) % 100 == 0:
            running_ce = total_loss_nats / max(total_tokens, 1)
            logger.info(
                f"Batch {batch_idx + 1} | "
                f"Running CE: {running_ce:.4f} | "
                f"Tokens: {total_tokens:,}"
            )

    # Final metrics
    avg_ce_loss = total_loss_nats / max(total_tokens, 1)
    ppl = math.exp(avg_ce_loss)
    bpb = total_loss_nats / (max(total_bytes, 1) * math.log(2))

    results = {
        "language": lang.key,
        "model": lang.model_label,
        "split": data_split,
        "cross_entropy_loss": round(avg_ce_loss, 4),
        "perplexity": round(ppl, 2),
        "bits_per_byte": round(bpb, 4),
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
    }

    logger.info(f"\n{'='*50}")
    logger.info("  " + lang.title("Evaluation Results"))
    logger.info(f"  Split:              {data_split}")
    logger.info(f"  Cross-Entropy Loss: {avg_ce_loss:.4f}")
    logger.info(f"  Perplexity (PPL):   {ppl:.2f}")
    logger.info(f"  Bits-per-Byte (BPB):{bpb:.4f}")
    logger.info(f"  Total Tokens:       {total_tokens:,}")
    logger.info(f"  Total Bytes:        {total_bytes:,}")
    logger.info(f"{'='*50}\n")

    # Save results
    results_dir = lang.report_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"lm_metrics_{data_split}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    return results
