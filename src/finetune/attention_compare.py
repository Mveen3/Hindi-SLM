"""
Attention Comparison: Pretrained vs Finetuned (Phase 3, §3.2)
=============================================================

Reuses the Phase 2 attention toolkit on the *finetuned* checkpoint and puts it
side by side with the pretrained one, on a comparative-reasoning prompt drawn
from the model's own held-out test split.

The brief asks specifically for "at least one early and one late layer per
model" and for a comment on "whether finetuning changed local vs. long-range
attention or head specialization, especially on comparative-reasoning
prompts". So all three artifacts are keyed to that question:

    - paired heatmap grids (pretrained above, finetuned below) for an early
      and a late layer, on the same sentence and the same token axis, so the
      two are actually comparable rather than merely adjacent;
    - per-head deltas in **mean attention distance** (local vs long-range) and
      **entropy** (focused vs diffuse), which is where a shift in head
      specialization shows up numerically;
    - a machine-readable JSON of every number behind the figures.

Computation and visualisation stay in separate functions, and every figure
carries a title and axis labels, per the project's plotting rules.

Invoked through ``python main.py evaluate-reasoning --only attention``.
"""

from __future__ import annotations

import json
import logging

import torch
from tokenizers import Tokenizer

from src.languages import Language
from src.eval.attention import (
    _import_plotting,
    compute_attention_entropy,
    compute_mean_attention_distance,
    extract_attention_weights,
)
from src.finetune.evaluate import load_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def collect_attention(lang: Language, checkpoint_path: str, text: str, device):
    """Run one sentence through a checkpoint and return attention + tokens.

    Args:
        lang:            Language (supplies the frozen tokenizer).
        checkpoint_path: Checkpoint to inspect.
        text:            Sentence to analyse.
        device:          Device to run on.

    Returns:
        ``(all_attn, tokens, config)`` where ``all_attn`` is one
        ``(1, h, T, T)`` tensor per layer.
    """
    model, config, _ = load_model(lang, checkpoint_path, device)
    tokenizer = Tokenizer.from_file(str(lang.tokenizer_path))

    encoding = tokenizer.encode(text)
    ids = encoding.ids[:config.get("max_seq_len", 512)]
    tokens = encoding.tokens[:len(ids)]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    all_attn = [a.detach().cpu() for a in extract_attention_weights(model, input_ids)]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_attn, tokens, config


def summarise_shift(pre_map: dict, fine_map: dict, n_layers: int, n_heads: int) -> dict:
    """Turn per-head metric maps into a pretrained→finetuned shift summary.

    Args:
        pre_map:  ``{(layer, head): value}`` for the pretrained model.
        fine_map: Same, for the finetuned model.
        n_layers: Layer count.
        n_heads:  Head count.

    Returns:
        Per-layer means, the overall mean, and the heads that moved most in
        each direction — the numeric evidence for the write-up's claims about
        head specialization.
    """
    per_head = {
        f"L{layer}H{head}": {
            "pretrained": pre_map[(layer, head)],
            "finetuned": fine_map[(layer, head)],
            "delta": round(fine_map[(layer, head)] - pre_map[(layer, head)], 4),
        }
        for layer in range(n_layers) for head in range(n_heads)
        if (layer, head) in pre_map and (layer, head) in fine_map
    }

    per_layer = {}
    for layer in range(n_layers):
        pre_values = [pre_map[(layer, h)] for h in range(n_heads) if (layer, h) in pre_map]
        fine_values = [fine_map[(layer, h)] for h in range(n_heads) if (layer, h) in fine_map]
        if pre_values and fine_values:
            per_layer[f"layer_{layer}"] = {
                "pretrained_mean": round(sum(pre_values) / len(pre_values), 4),
                "finetuned_mean": round(sum(fine_values) / len(fine_values), 4),
                "delta": round(sum(fine_values) / len(fine_values)
                               - sum(pre_values) / len(pre_values), 4),
            }

    ranked = sorted(per_head.items(), key=lambda kv: kv[1]["delta"])
    all_deltas = [v["delta"] for v in per_head.values()]
    return {
        "per_head": per_head,
        "per_layer": per_layer,
        "mean_delta": round(sum(all_deltas) / len(all_deltas), 4) if all_deltas else 0.0,
        "largest_decrease": ranked[:3],
        "largest_increase": ranked[-3:][::-1],
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_paired_heatmaps(pre_attn: list, fine_attn: list, tokens: list,
                         layer: int, heads: list, lang: Language,
                         sentence: str, save_path) -> None:
    """Draw pretrained (top row) vs finetuned (bottom row) for one layer.

    Sharing the token axis across both rows is what makes the pair readable:
    the same query/key positions sit in the same place, so a change in where a
    head looks is visible as a change in the pattern, not in the layout.
    """
    plt, sns = _import_plotting()

    n_heads = len(heads)
    fig, axes = plt.subplots(2, n_heads, figsize=(4.2 * n_heads, 8.6), squeeze=False)
    labels = [t.replace("▁", " ") for t in tokens]
    show_ticks = len(tokens) <= 28

    for row, (attn_set, arm) in enumerate(((pre_attn, "Pretrained"),
                                           (fine_attn, "Finetuned"))):
        for col, head in enumerate(heads):
            ax = axes[row][col]
            weights = attn_set[layer][0, head].numpy()
            sns.heatmap(
                weights, ax=ax, cmap="viridis", cbar=(col == n_heads - 1),
                cbar_kws={"label": "attention weight"} if col == n_heads - 1 else None,
                xticklabels=labels if show_ticks else False,
                yticklabels=labels if show_ticks else False,
                vmin=0.0, vmax=1.0, square=False,
            )
            ax.set_title(f"{arm} — layer {layer}, head {head}", fontsize=11)
            ax.set_xlabel("key position (attended to)", fontsize=9)
            ax.set_ylabel("query position (attending from)", fontsize=9)
            if show_ticks:
                ax.tick_params(axis="x", rotation=90, labelsize=7)
                ax.tick_params(axis="y", rotation=0, labelsize=7)

    fig.suptitle(
        f"{lang.title('Attention before vs after reasoning finetuning')}\n"
        f"layer {layer} · \"{sentence[:70]}{'...' if len(sentence) > 70 else ''}\"",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(shift: dict, n_layers: int, n_heads: int, lang: Language,
                       metric_name: str, unit: str, save_path) -> None:
    """Draw the per-head finetuned-minus-pretrained change as a layer x head grid."""
    plt, sns = _import_plotting()

    grid = [[shift["per_head"].get(f"L{layer}H{head}", {}).get("delta", 0.0)
             for head in range(n_heads)] for layer in range(n_layers)]
    limit = max((abs(v) for row in grid for v in row), default=1.0) or 1.0

    fig, ax = plt.subplots(figsize=(1.35 * n_heads + 3, 0.75 * n_layers + 3))
    sns.heatmap(
        grid, ax=ax, cmap="coolwarm", center=0.0, vmin=-limit, vmax=limit,
        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
        cbar_kws={"label": f"change in {metric_name} ({unit})"},
    )
    ax.set_title(lang.title(f"Change in {metric_name} after reasoning finetuning"), fontsize=12)
    ax.set_xlabel("attention head", fontsize=10)
    ax.set_ylabel("transformer layer", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pick_reasoning_prompt(lang: Language, max_tokens: int = 90) -> str:
    """Choose a short comparative-reasoning question from the test split.

    A short chained-inequality question is used on purpose: the heatmaps stay
    legible with token labels on the axes, and it is exactly the prompt type
    the brief wants the comparison made on.

    Args:
        lang:       Language whose reasoning test split to read.
        max_tokens: Rough length cap (measured with the frozen tokenizer).

    Returns:
        The question text, or a fallback sample sentence if the split is missing.
    """
    path = lang.reasoning_split_file("test")
    if not path.exists():
        return lang.attention_samples[0]

    tokenizer = Tokenizer.from_file(str(lang.tokenizer_path))
    preferred = {"transitive_relation", "transitive_largest", "transitive_smallest"}
    fallback = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record["prompt"]
            if len(tokenizer.encode(prompt).ids) > max_tokens:
                continue
            if record["task"] in preferred:
                return prompt
            fallback = fallback or prompt

    return fallback or lang.attention_samples[0]


def run_attention_comparison(lang: Language, pretrained_checkpoint: str,
                             finetuned_checkpoint: str, sentence: str | None = None,
                             n_heads_to_plot: int = 4) -> dict:
    """Compare attention before and after finetuning, and write the artifacts.

    Produces, under ``report/<language>/attention_finetune/``:
        - ``attention_pair_layer<early>.png`` and ``..._layer<late>.png``,
        - ``delta_mean_distance.png`` and ``delta_entropy.png``,
        - ``attention_shift.json`` with every number behind them.

    Args:
        lang:                  Language to analyse.
        pretrained_checkpoint: Phase 2 checkpoint.
        finetuned_checkpoint:  Phase 3 checkpoint.
        sentence:              Optional prompt override.
        n_heads_to_plot:       Heads per row in the paired heatmaps.

    Returns:
        The shift summary dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence = sentence or pick_reasoning_prompt(lang)
    logger.info("Attention comparison prompt: %s", sentence[:90])

    pre_attn, tokens, config = collect_attention(lang, pretrained_checkpoint, sentence, device)
    fine_attn, fine_tokens, _ = collect_attention(lang, finetuned_checkpoint, sentence, device)
    assert tokens == fine_tokens, "tokenizer must be identical across finetuning"

    n_layers = config.get("n_layers", len(pre_attn))
    n_heads = config.get("n_heads", pre_attn[0].shape[1])
    early_layer, late_layer = 0, n_layers - 1
    heads = list(range(min(n_heads_to_plot, n_heads)))

    output_dir = lang.report_dir / "attention_finetune"
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer in sorted({early_layer, late_layer}):
        path = output_dir / f"attention_pair_layer{layer}.png"
        plot_paired_heatmaps(pre_attn, fine_attn, tokens, layer, heads,
                             lang, sentence, path)
        logger.info("  paired heatmap (layer %d) -> %s", layer, path)

    distance_shift = summarise_shift(
        compute_mean_attention_distance(pre_attn),
        compute_mean_attention_distance(fine_attn), n_layers, n_heads)
    entropy_shift = summarise_shift(
        compute_attention_entropy(pre_attn),
        compute_attention_entropy(fine_attn), n_layers, n_heads)

    plot_delta_heatmap(distance_shift, n_layers, n_heads, lang,
                       "mean attention distance", "tokens",
                       output_dir / "delta_mean_distance.png")
    plot_delta_heatmap(entropy_shift, n_layers, n_heads, lang,
                       "attention entropy", "nats",
                       output_dir / "delta_entropy.png")

    summary = {
        "language": lang.key,
        "model": lang.model_label,
        "prompt": sentence,
        "n_tokens": len(tokens),
        "layers_plotted": sorted({early_layer, late_layer}),
        "heads_plotted": heads,
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "finetuned_checkpoint": str(finetuned_checkpoint),
        "mean_attention_distance": distance_shift,
        "attention_entropy": entropy_shift,
    }

    json_path = output_dir / "attention_shift.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("  mean attention distance: %+.3f tokens (finetuned - pretrained)",
                distance_shift["mean_delta"])
    logger.info("  attention entropy:       %+.3f nats  (finetuned - pretrained)",
                entropy_shift["mean_delta"])
    logger.info("  attention shift metrics -> %s", json_path)
    return summary
