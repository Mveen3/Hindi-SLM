"""
Attention Analysis
==================

Visualise and quantify what multi-head attention has learned:
    1. Attention heatmaps (early layer, late layer, multiple heads).
    2. Attention entropy per head/layer (higher = more diffuse).
    3. Mean attention distance per head/layer (local vs long-range).

Invoked through ``python main.py evaluate --lang {hindi,nepali} --only attention``.
"""

import json
import torch
import numpy as np
import logging
from pathlib import Path

from tokenizers import Tokenizer

from src.languages import Language
from src.model.transformer import build_model_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Devanagari-capable families, tried in order. Without one of these the token
# labels on every heatmap render as empty boxes, which would make the figures
# useless for showing attention over real text.
DEVANAGARI_FONTS = ["Noto Sans Devanagari", "Noto Serif Devanagari",
                    "Lohit Devanagari", "FreeSans"]


def _import_plotting():
    """Import matplotlib headlessly and select a Devanagari-capable font."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    usable = [name for name in DEVANAGARI_FONTS if name in available]
    if usable:
        # font.family must be a *list* for matplotlib's per-glyph fallback;
        # setting font.sans-serif instead resolves to one font only, which
        # leaves Latin axis labels as boxes. DejaVu last covers Latin, digits
        # and the SentencePiece word-boundary marker.
        plt.rcParams["font.family"] = usable + ["DejaVu Sans"]
    else:
        logger.warning(
            "No Devanagari font found — token labels will render as boxes. "
            "Install one with: sudo apt install fonts-noto-core"
        )
    return plt, sns


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_weights(model, input_ids: torch.Tensor):
    """
    Run a forward pass and extract attention weights from all layers.

    Args:
        model:     The Transformer model.
        input_ids: (1, T) token IDs.

    Returns:
        List of attention weight tensors, one per layer.
        Each tensor has shape (1, n_heads, T, T).
    """
    model.eval()
    _, _, all_attn = model(input_ids, return_attention=True)
    return all_attn


# ---------------------------------------------------------------------------
# Attention heatmaps
# ---------------------------------------------------------------------------

def plot_attention_grid(
    all_attn: list,
    tokens: list,
    layers: list,
    heads: list,
    title: str,
    save_path: str,
):
    """
    Plot one figure showing query-vs-key attention for several layers and heads.

    The brief asks for heatmaps covering at least one early layer, one late
    layer and multiple heads. Drawing them as a single layer x head grid keeps
    the comparison on one page — reading 12 separate PNGs side by side is what
    the grid replaces.

    Args:
        all_attn:  List of (1, h, T, T) attention tensors, one per layer.
        tokens:    Token strings for the analysed sentence (length T).
        layers:    Layer indices to draw as rows.
        heads:     Head indices to draw as columns.
        title:     Figure title (includes the language and the sentence).
        save_path: Where to write the PNG.
    """
    plt, sns = _import_plotting()

    T = len(tokens)
    display_tokens = [tok[:12] for tok in tokens]
    # Label every token for short sentences, otherwise thin them out so the
    # axis stays readable.
    stride = max(1, T // 24)
    ticks = list(range(0, T, stride))
    tick_labels = [display_tokens[i] for i in ticks]

    fig, axes = plt.subplots(
        len(layers), len(heads),
        figsize=(3.1 * len(heads) + 1.6, 3.0 * len(layers) + 1.2),
        squeeze=False,
    )

    vmax = max(a[0, h].max().item() for a in all_attn for h in heads)

    for r, layer_idx in enumerate(layers):
        attn = all_attn[layer_idx][0]  # (h, T, T)
        for c, head_idx in enumerate(heads):
            ax = axes[r][c]
            mesh = ax.imshow(
                attn[head_idx].cpu().numpy(),
                cmap="viridis", vmin=0.0, vmax=vmax,
                aspect="auto", interpolation="nearest",
            )
            ax.set_title(f"Layer {layer_idx}, Head {head_idx}", fontsize=10)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            if r == len(layers) - 1:
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
                ax.set_xlabel("Key position", fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_yticklabels(tick_labels, fontsize=6)
                ax.set_ylabel("Query position", fontsize=9)
            else:
                ax.set_yticklabels([])

    fig.suptitle(title, fontsize=13)
    fig.colorbar(mesh, ax=axes, shrink=0.6, label="Attention weight")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Attention grid saved: {save_path}")


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------

def compute_attention_entropy(all_attn: list) -> dict:
    """
    Compute attention entropy per head per layer.

    Entropy = -Σ p(i) log p(i) over key positions for each query.
    Higher entropy means more diffuse (spread out) attention.
    Lower entropy means more concentrated (peaked) attention.

    Args:
        all_attn: List of (1, h, T, T) attention tensors.

    Returns:
        Dictionary mapping (layer, head) to average entropy value.
    """
    entropy_map = {}

    for layer_idx, attn in enumerate(all_attn):
        attn = attn[0]  # (h, T, T)
        n_heads = attn.shape[0]

        for head_idx in range(n_heads):
            weights = attn[head_idx]  # (T, T)
            # Clamp to avoid log(0)
            weights = weights.clamp(min=1e-10)
            entropy = -(weights * weights.log()).sum(dim=-1)  # (T,)
            avg_entropy = entropy.mean().item()
            entropy_map[(layer_idx, head_idx)] = round(avg_entropy, 4)

    return entropy_map


# ---------------------------------------------------------------------------
# Mean attention distance
# ---------------------------------------------------------------------------

def compute_mean_attention_distance(all_attn: list) -> dict:
    """
    Compute mean attention distance per head per layer.

    Mean distance = Σ_j attn(i, j) × |i - j| for each query position i,
    then averaged over all positions.

    Heads with low mean distance attend locally (nearby tokens).
    Heads with high mean distance attend to distant tokens.

    Args:
        all_attn: List of (1, h, T, T) attention tensors.

    Returns:
        Dictionary mapping (layer, head) to mean attention distance.
    """
    distance_map = {}

    for layer_idx, attn in enumerate(all_attn):
        attn = attn[0]  # (h, T, T)
        n_heads, T, _ = attn.shape

        # Distance matrix: |i - j| for all (i, j) pairs
        positions = torch.arange(T, device=attn.device, dtype=attn.dtype)
        dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (T, T)

        for head_idx in range(n_heads):
            weights = attn[head_idx]  # (T, T)
            # Weighted average distance per query, then average over queries
            weighted_dist = (weights * dist_matrix).sum(dim=-1)  # (T,)
            avg_dist = weighted_dist.mean().item()
            distance_map[(layer_idx, head_idx)] = round(avg_dist, 2)

    return distance_map


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def plot_entropy_heatmap(lang: Language, entropy_map: dict, n_layers: int,
                         n_heads: int, save_path: str):
    """
    Plot a layer × head heatmap of attention entropy values.

    Args:
        lang:        Language being plotted (drives the title).
        entropy_map: Dictionary mapping (layer, head) to entropy.
        n_layers:    Number of layers.
        n_heads:     Number of heads.
        save_path:   Path to save the plot.
    """
    plt, sns = _import_plotting()

    matrix = np.zeros((n_layers, n_heads))
    for (l, h), val in entropy_map.items():
        matrix[l, h] = val

    fig, ax = plt.subplots(figsize=(max(6, n_heads), max(4, n_layers * 0.6)))
    sns.heatmap(
        matrix,
        annot=True, fmt=".2f",
        xticklabels=[f"H{i}" for i in range(n_heads)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Entropy (nats)"},
    )
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(lang.title("Attention Entropy per Head/Layer"), fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Entropy heatmap saved: {save_path}")


def plot_distance_heatmap(lang: Language, distance_map: dict, n_layers: int,
                          n_heads: int, save_path: str):
    """
    Plot a layer × head heatmap of mean attention distance.

    Args:
        lang:         Language being plotted (drives the title).
        distance_map: Dictionary mapping (layer, head) to mean distance.
        n_layers:     Number of layers.
        n_heads:      Number of heads.
        save_path:    Path to save the plot.
    """
    plt, sns = _import_plotting()

    matrix = np.zeros((n_layers, n_heads))
    for (l, h), val in distance_map.items():
        matrix[l, h] = val

    fig, ax = plt.subplots(figsize=(max(6, n_heads), max(4, n_layers * 0.6)))
    sns.heatmap(
        matrix,
        annot=True, fmt=".1f",
        xticklabels=[f"H{i}" for i in range(n_heads)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cmap="coolwarm",
        ax=ax,
        cbar_kws={"label": "Mean Distance (tokens)"},
    )
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(lang.title("Mean Attention Distance per Head/Layer"), fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Distance heatmap saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_attention_analysis(lang: Language, checkpoint_path: str, sample_text: str = None):
    """
    Run the full attention analysis for one model.

    Produces, under ``report/<language>/attention/``:
        - one heatmap grid per example sentence (early/middle/late layer x heads),
        - an entropy summary heatmap over layer x head,
        - a mean-attention-distance summary heatmap over layer x head,
        - attention_metrics.json with the numeric values behind both summaries.

    Args:
        lang:            Language to analyse — supplies the tokenizer, the
                         example sentences and the report directory.
        checkpoint_path: Path to that language's checkpoint.
        sample_text:     Optional single sentence to analyse instead of the
                         language's default examples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    tokenizer = Tokenizer.from_file(str(lang.tokenizer_path))

    sample_texts = [sample_text] if sample_text else list(lang.attention_samples)

    output_dir = lang.report_dir / "attention"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 8)

    # Early, middle and late layer; first four heads.
    layers_to_plot = sorted({0, n_layers // 2, n_layers - 1})
    heads_to_plot = list(range(min(4, n_heads)))

    all_entropy = {}
    all_distance = {}
    sample_count = 0

    for text_idx, text in enumerate(sample_texts):
        logger.info(f"Analysing sample {text_idx + 1}: {text[:50]}...")

        encoding = tokenizer.encode(text)
        max_len = min(len(encoding.ids), config.get("max_seq_len", 512))
        token_ids = encoding.ids[:max_len]
        tokens = encoding.tokens[:max_len]

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        all_attn = extract_attention_weights(model, input_tensor)

        plot_attention_grid(
            all_attn, tokens, layers_to_plot, heads_to_plot,
            title=lang.title(f"Attention Heatmaps (sample {text_idx + 1})"),
            save_path=str(output_dir / f"heatmaps_sample{text_idx + 1}.png"),
        )

        entropy = compute_attention_entropy(all_attn)
        distance = compute_mean_attention_distance(all_attn)
        for key, val in entropy.items():
            all_entropy[key] = all_entropy.get(key, 0) + val
        for key, val in distance.items():
            all_distance[key] = all_distance.get(key, 0) + val
        sample_count += 1

    avg_entropy = {k: round(v / sample_count, 4) for k, v in all_entropy.items()}
    avg_distance = {k: round(v / sample_count, 2) for k, v in all_distance.items()}

    plot_entropy_heatmap(lang, avg_entropy, n_layers, n_heads,
                         str(output_dir / "entropy_summary.png"))
    plot_distance_heatmap(lang, avg_distance, n_layers, n_heads,
                          str(output_dir / "distance_summary.png"))

    results = {
        "language": lang.key,
        "model": lang.model_label,
        "samples": sample_texts,
        "entropy": {f"layer{k[0]}_head{k[1]}": v for k, v in avg_entropy.items()},
        "mean_distance": {f"layer{k[0]}_head{k[1]}": v for k, v in avg_distance.items()},
    }
    results_file = output_dir / "attention_metrics.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Attention metrics saved to {results_file}")

    logger.info("=" * 50)
    logger.info(lang.title("Attention Analysis Summary"))
    logger.info("=" * 50)
    for layer in range(n_layers):
        for head in range(n_heads):
            e = avg_entropy.get((layer, head), 0)
            d = avg_distance.get((layer, head), 0)
            head_type = "local" if d < 5 else "long-range"
            focus = "focused" if e < 1.5 else "diffuse"
            logger.info(f"  L{layer} H{head}: entropy={e:.3f} ({focus}), dist={d:.1f} ({head_type})")

    return results
