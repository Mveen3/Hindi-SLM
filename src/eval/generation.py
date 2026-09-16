"""
Text Generation and Quality Evaluation
=====================================

Generate continuations from held-out prefixes using:
    - Greedy decoding
    - Temperature sampling (0.5, 1.0, 1.5)

Against reference continuations, report:
    - BLEU-4 (corpus-level, n-gram precision up to 4-grams)
    - chrF++ (character-level F-score, robust for morphologically rich languages)
    - ROUGE-L (longest common subsequence based recall)

Also report fluency / diversity diagnostics:
    - Repetition rate (fraction of repeated n-grams)
    - Distinct-1 / Distinct-2 (unique unigrams / bigrams over generated tokens)
    - Qualitative generated samples

Invoked through ``python main.py evaluate`` and ``python main.py generate``.
"""

import json
import math
import torch
import logging
from collections import Counter

from tokenizers import Tokenizer

from src.languages import Language
from src.model.transformer import build_model_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    greedy: bool = False,
    eos_token_id: int = 3,
) -> torch.Tensor:
    """
    Autoregressive text generation.

    Args:
        model:          The Transformer model.
        input_ids:      (1, T) prefix token IDs.
        max_new_tokens: Number of tokens to generate.
        temperature:    Sampling temperature (ignored if greedy=True).
        greedy:         If True, use argmax decoding.
        eos_token_id:   Stop generation upon this token.

    Returns:
        (1, T + generated) full sequence of token IDs.
    """
    model.eval()
    device = input_ids.device
    max_seq_len = model.max_seq_len

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Truncate to max_seq_len if needed
        context = generated[:, -max_seq_len:]

        logits, _ = model(context)
        next_logits = logits[:, -1, :]  # (1, vocab_size)

        if greedy:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            # Temperature scaling
            scaled_logits = next_logits / max(temperature, 1e-8)
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

    return generated


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ngrams(tokens: list, n: int) -> Counter:
    """Compute n-gram counts from a list of tokens."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(references: list, hypotheses: list, max_n: int = 4) -> float:
    """
    Compute corpus-level BLEU score (up to max_n-grams).

    BLEU measures n-gram precision between generated and reference text.
    For open-ended LM generation, BLEU has limitations: it penalises
    valid paraphrases and rewards only exact n-gram matches. However,
    it provides a standardised comparison baseline.

    Args:
        references:  List of reference token lists.
        hypotheses:  List of hypothesis token lists.
        max_n:       Maximum n-gram order (default: BLEU-4).

    Returns:
        BLEU score (0-100).
    """
    precisions = []
    bp_r = 0  # total reference length
    bp_c = 0  # total hypothesis length

    for n in range(1, max_n + 1):
        match_count = 0
        total_count = 0

        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = compute_ngrams(ref, n)
            hyp_ngrams = compute_ngrams(hyp, n)

            # Clipped counts
            for ngram, count in hyp_ngrams.items():
                match_count += min(count, ref_ngrams.get(ngram, 0))
                total_count += count

        precision = match_count / max(total_count, 1)
        precisions.append(precision)

    # Brevity penalty
    for ref, hyp in zip(references, hypotheses):
        bp_r += len(ref)
        bp_c += len(hyp)

    if bp_c == 0:
        return 0.0

    bp = min(1.0, math.exp(1 - bp_r / bp_c)) if bp_c < bp_r else 1.0

    # Geometric mean of precisions
    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / max_n
    bleu = bp * math.exp(log_avg) * 100

    return round(bleu, 2)


def compute_chrf(references: list, hypotheses: list, n: int = 6, beta: float = 2.0) -> float:
    """
    Compute chrF score (character n-gram F-score).

    chrF is more robust for morphologically rich languages like Hindi and
    Nepali because it operates at the character level, capturing partial word
    matches that word-level BLEU would miss entirely.

    Args:
        references:  List of reference strings.
        hypotheses:  List of hypothesis strings.
        n:           Maximum character n-gram order.
        beta:        F-score beta (2.0 = recall-weighted).

    Returns:
        chrF score (0-100).
    """
    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for ref, hyp in zip(references, hypotheses):
        for order in range(1, n + 1):
            ref_ngrams = compute_ngrams(list(ref), order)
            hyp_ngrams = compute_ngrams(list(hyp), order)

            matches = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in hyp_ngrams)

            precision = matches / max(sum(hyp_ngrams.values()), 1)
            recall = matches / max(sum(ref_ngrams.values()), 1)

            total_precision += precision
            total_recall += recall
            count += 1

    avg_p = total_precision / max(count, 1)
    avg_r = total_recall / max(count, 1)

    if avg_p + avg_r == 0:
        return 0.0

    chrf = (1 + beta ** 2) * avg_p * avg_r / (beta ** 2 * avg_p + avg_r)
    return round(chrf * 100, 2)


def compute_rouge_l(references: list, hypotheses: list) -> float:
    """
    Compute ROUGE-L (longest common subsequence based F1).

    ROUGE-L captures sentence-level structural similarity. For LM
    generation, it's informative because it rewards sequences that
    preserve the order of reference tokens, even with insertions.

    Args:
        references:  List of reference token lists.
        hypotheses:  List of hypothesis token lists.

    Returns:
        ROUGE-L F1 score (0-100).
    """
    def lcs_length(x, y):
        """Compute the length of the longest common subsequence."""
        m, n = len(x), len(y)
        # Use 1D DP for memory efficiency
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev = curr
        return prev[n]

    total_f1 = 0.0

    for ref, hyp in zip(references, hypotheses):
        lcs = lcs_length(ref, hyp)
        precision = lcs / max(len(hyp), 1)
        recall = lcs / max(len(ref), 1)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        total_f1 += f1

    rouge_l = total_f1 / max(len(references), 1)
    return round(rouge_l * 100, 2)


def compute_diversity(token_lists: list) -> dict:
    """
    Compute diversity diagnostics for generated text.

    Metrics:
        - Distinct-1: ratio of unique unigrams to total unigrams.
        - Distinct-2: ratio of unique bigrams to total bigrams.
        - Repetition rate: fraction of repeated 4-grams.

    Args:
        token_lists: List of generated token lists.

    Returns:
        Dictionary with distinct_1, distinct_2, repetition_rate.
    """
    all_unigrams = []
    all_bigrams = []
    total_4grams = 0
    repeated_4grams = 0

    for tokens in token_lists:
        all_unigrams.extend(tokens)
        all_bigrams.extend(zip(tokens, tokens[1:]))

        # 4-gram repetition
        fourgrams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
        fourgram_counts = Counter(fourgrams)
        total_4grams += len(fourgrams)
        repeated_4grams += sum(c - 1 for c in fourgram_counts.values() if c > 1)

    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    rep_rate = repeated_4grams / max(total_4grams, 1)

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_rate_4gram": round(rep_rate, 4),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def load_for_inference(lang: Language, checkpoint_path: str, device=None):
    """Load one language's model + tokenizer for generation.

    Args:
        lang:            Language to load.
        checkpoint_path: Path to that language's checkpoint.
        device:          Torch device; defaults to CUDA when available.

    Returns:
        (model, tokenizer, config, device) tuple.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(lang.tokenizer_path))
    return model, tokenizer, config, device


def run_generation_eval(lang: Language, checkpoint_path: str,
                        num_samples: int = 100, prefix_ratio: float = 0.5):
    """
    Run the full generation evaluation pipeline on one model.

    Args:
        lang:            Language being evaluated — supplies the tokenizer, the
                         held-out split and the report directory.
        checkpoint_path: Path to model checkpoint.
        num_samples:     Number of test samples to generate from.
        prefix_ratio:    Fraction of each document to use as prefix.
    """
    model, tokenizer, config, device = load_for_inference(lang, checkpoint_path)

    # Held-out prefixes come from this language's own test split.
    test_file = lang.split_file("test")
    max_seq_len = config.get("max_seq_len", 512)
    eos_id = config.get("eos_token_id", 3)

    # Collect test documents
    test_docs = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if len(text) > 50:  # Need enough text for prefix + reference
                    test_docs.append(text)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if len(test_docs) >= num_samples:
                break

    logger.info(f"Loaded {len(test_docs)} test documents")

    # Prepare prefixes and references
    temperatures = [("greedy", None), ("temp_0.5", 0.5), ("temp_1.0", 1.0), ("temp_1.5", 1.5)]

    all_results = {}
    all_samples = {}

    for temp_name, temp in temperatures:
        logger.info(f"\n--- Generating with {temp_name} ---")

        ref_token_lists = []
        hyp_token_lists = []
        ref_strings = []
        hyp_strings = []

        samples = []

        for idx, text in enumerate(test_docs):
            encoding = tokenizer.encode(text)
            token_ids = encoding.ids

            if len(token_ids) < 10:
                continue

            # Split into prefix and reference
            split_point = max(5, int(len(token_ids) * prefix_ratio))
            split_point = min(split_point, max_seq_len - 10)  # Leave room for generation

            prefix_ids = token_ids[:split_point]
            ref_ids = token_ids[split_point:split_point + 100]  # Max 100 reference tokens

            if len(ref_ids) < 5:
                continue

            # Generate continuation
            prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            is_greedy = (temp is None)
            gen_temp = temp if temp else 1.0

            generated = generate(
                model, prefix_tensor,
                max_new_tokens=len(ref_ids),
                temperature=gen_temp,
                greedy=is_greedy,
                eos_token_id=eos_id,
            )

            gen_ids = generated[0, len(prefix_ids):].tolist()

            # Remove EOS if present
            if eos_id in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos_id)]

            ref_token_lists.append(ref_ids)
            hyp_token_lists.append(gen_ids)

            # Decode for chrF (character-level)
            ref_text = tokenizer.decode(ref_ids)
            hyp_text = tokenizer.decode(gen_ids) if gen_ids else ""
            ref_strings.append(ref_text)
            hyp_strings.append(hyp_text)

            # Save first few samples for qualitative analysis
            if len(samples) < 5:
                prefix_text = tokenizer.decode(prefix_ids)
                samples.append({
                    "prefix": prefix_text,
                    "reference": ref_text,
                    "generated": hyp_text,
                })

            if (idx + 1) % 50 == 0:
                logger.info(f"  Generated {idx + 1}/{len(test_docs)}")

        # Compute metrics
        bleu = compute_bleu(ref_token_lists, hyp_token_lists)
        chrf = compute_chrf(ref_strings, hyp_strings)
        rouge_l = compute_rouge_l(ref_token_lists, hyp_token_lists)
        diversity = compute_diversity(hyp_token_lists)

        metrics = {
            "bleu_4": bleu,
            "chrf": chrf,
            "rouge_l": rouge_l,
            **diversity,
            "num_samples": len(ref_token_lists),
        }

        all_results[temp_name] = metrics
        all_samples[temp_name] = samples

        logger.info(f"  BLEU-4:    {bleu}")
        logger.info(f"  chrF:      {chrf}")
        logger.info(f"  ROUGE-L:   {rouge_l}")
        logger.info(f"  Distinct-1:{diversity['distinct_1']}")
        logger.info(f"  Distinct-2:{diversity['distinct_2']}")
        logger.info(f"  Rep Rate:  {diversity['repetition_rate_4gram']}")

    # Save results
    results_dir = lang.report_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "generation_metrics.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    samples_file = results_dir / "generation_samples.json"
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"Samples saved to {samples_file}")

    return all_results


# ---------------------------------------------------------------------------
# Interactive / CLI Prompt Generation
# ---------------------------------------------------------------------------

def run_prompt(lang: Language, prompt: str, checkpoint: str = None, max_new_tokens: int = 100,
               temperature: float = 1.0, greedy: bool = False) -> str:
    """Generate and print a continuation for the given prompt."""
    from src.train.utils import resolve_checkpoint

    checkpoint = resolve_checkpoint(lang, checkpoint)
    model, tokenizer, config, device = load_for_inference(lang, checkpoint)
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids],
                              dtype=torch.long, device=device)

    output = generate(
        model, prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        greedy=greedy,
        eos_token_id=config.get("eos_token_id", 3),
    )
    result = tokenizer.decode(output[0].tolist())
    print(f"\n--- {lang.display} ({lang.model_label}) ---")
    print(result)
    print()
    return result


def main():
    import argparse
    from src import languages

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Generate text continuation from a prompt.")
    languages.add_language_arg(parser)
    parser.add_argument("--prompt", required=True, help="Input prompt text.")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to generate from (default: newest in checkpoints/).")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum new tokens to generate (default: %(default)s).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: %(default)s).")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling.")
    args = parser.parse_args()

    lang = languages.get(args.lang)
    run_prompt(
        lang,
        prompt=args.prompt,
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        greedy=args.greedy,
    )


if __name__ == "__main__":
    main()

