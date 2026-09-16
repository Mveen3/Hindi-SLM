"""
Causal Mask Verification
========================

Empirically verify that the causal mask works correctly by showing that
changing token t+1 does NOT change the logits at position t.

This satisfies the assignment requirement:
    "verify empirically that the model cannot see the future — for example,
     show that changing token t+1 does not change the logits at position t."

Invoked through ``python main.py evaluate --lang {hindi,nepali} --only causal``.
"""

import json
import torch
import logging

from src.languages import Language
from src.model.transformer import build_model_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def verify_causal_mask(lang: Language, checkpoint_path: str = None):
    """
    Verify that the causal mask prevents information flow from future tokens.

    Method:
        1. Create a random input sequence of length T.
        2. Run a forward pass and record logits at position t.
        3. Modify token at position t+1 (and later positions).
        4. Run forward pass again.
        5. Assert that logits at position t are IDENTICAL.

    If the causal mask is correct, the logits at position t should depend
    only on tokens at positions 0..t, not on any future token.

    Args:
        lang:            Language whose report directory receives the result.
        checkpoint_path: Optional path to a trained checkpoint.
                         If None, tests with a randomly initialised model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = build_model_from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Testing with trained checkpoint.")
    else:
        # Use a small test model
        config = {
            "vocab_size": 1000,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 512,
            "max_seq_len": 64,
            "dropout": 0.0,  # Disable dropout for deterministic comparison
            "tie_weights": True,
            "pad_token_id": 0,
        }
        model = build_model_from_config(config)
        logger.info("Testing with randomly initialised model.")

    model = model.to(device)
    model.eval()

    T = min(32, config.get("max_seq_len", 64))
    vocab_size = config["vocab_size"]

    # Generate a random input sequence
    torch.manual_seed(42)
    original_ids = torch.randint(4, vocab_size, (1, T), device=device)

    # Forward pass with original sequence
    logits_original, _ = model(original_ids)

    # Test positions: check several positions across the sequence
    test_positions = [0, T // 4, T // 2, 3 * T // 4, T - 2]
    test_positions = [p for p in test_positions if p < T - 1]

    all_passed = True
    results = []

    for t in test_positions:
        # Create a modified sequence: change tokens at positions > t
        modified_ids = original_ids.clone()
        for pos in range(t + 1, T):
            modified_ids[0, pos] = (original_ids[0, pos] + 1) % vocab_size

        # Forward pass with modified sequence
        logits_modified, _ = model(modified_ids)

        # Check: logits at position t should be identical
        logits_at_t_original = logits_original[0, t, :]
        logits_at_t_modified = logits_modified[0, t, :]

        max_diff = (logits_at_t_original - logits_at_t_modified).abs().max().item()
        passed = max_diff < 1e-5

        result = {
            "position_t": t,
            "max_logit_diff": max_diff,
            "passed": passed,
        }
        results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            f"  Position t={t}: max logit diff = {max_diff:.2e} → {status}"
        )

        if not passed:
            all_passed = False

    # Also verify that logits at position t DO change when tokens ≤ t change
    logger.info("\n--- Sanity check: logits SHOULD change when past tokens change ---")
    for t in [T // 2]:
        modified_ids = original_ids.clone()
        # Change a token BEFORE position t
        modified_ids[0, max(0, t - 1)] = (original_ids[0, max(0, t - 1)] + 1) % vocab_size

        logits_modified, _ = model(modified_ids)
        max_diff = (logits_original[0, t, :] - logits_modified[0, t, :]).abs().max().item()
        changed = max_diff > 1e-5

        status = "✅ PASS (logits changed)" if changed else "⚠️ WARNING (logits unchanged)"
        logger.info(f"  Position t={t} after changing t-1: max diff = {max_diff:.2e} → {status}")

    # Summary
    logger.info(f"\n{'='*50}")
    if all_passed:
        logger.info("✅ CAUSAL MASK VERIFICATION PASSED")
        logger.info("   Future tokens do NOT affect current position logits.")
    else:
        logger.info("❌ CAUSAL MASK VERIFICATION FAILED")
        logger.info("   Some positions show information leakage from the future!")
    logger.info(f"{'='*50}")

    # Save results
    results_dir = lang.report_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "causal_mask_verification.json"
    with open(results_file, "w") as f:
        json.dump({
            "language": lang.key,
            "model": lang.model_label,
            "all_passed": all_passed,
            "results": results,
        }, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    return all_passed
