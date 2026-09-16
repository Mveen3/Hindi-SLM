"""
Local Chat UI — Model H (Hindi) and Model L (Nepali)
===================================================

Serves index.html plus a small JSON/SSE API so the trained models can be driven
from the browser with live sampling controls.

Each language is loaded from its own checkpoint and its own tokenizer, so the
two models stay as independent here as they are on disk. Models load lazily on
first use and are then cached, so startup is instant and only the model you
actually chat with occupies memory.

Two variants per language are offered:

    pretrained   the Phase 2 checkpoint from <language>/checkpoints/ —
                 open-ended text continuation
    finetuned    the Phase 3 checkpoint from <language>/checkpoints_finetune/ —
                 comparative / transitive reasoning

Both are exposed rather than the finetuned one simply replacing the other:
flipping between them on the same prompt *is* the Phase 3 comparison, and the
pretrained model remains the Phase 2 deliverable. The finetuned variant is
preselected when it exists.

The finetuned models were trained on a strict prompt format
(``प्रश्न: <question>\\nतर्क:``), so the API can apply that wrapping itself —
see ``reasoning_format`` on /api/generate. Formatting server-side keeps it
byte-identical to what finetuning actually saw.

Usage:
    conda activate ml
    python app.py                 # http://127.0.0.1:8000
    python app.py --port 8080 --device cuda
"""

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Iterator, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from src import languages
from src.model.transformer import build_model_from_config

PROJECT_ROOT = Path(__file__).resolve().parent
LANGUAGES = languages.CHOICES

#: Finetuned first, so the UI (which preselects the first available entry)
#: defaults to the Phase 3 model when one has been trained.
VARIANTS = ("finetuned", "pretrained")

#: Fallback prompt labels, used when a language's phrase bank is absent. These
#: match the labels the reasoning generator writes, and are the same words in
#: both Hindi and Nepali.
DEFAULT_LABELS = {"question": "प्रश्न", "reasoning": "तर्क", "answer": "उत्तर"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chat")

app = FastAPI(title="LMA Chat")

# Loaded models, keyed by (language, variant). Populated on first request.
_CACHE: dict = {}

# Reasoning prompt labels, keyed by language. Read once from the phrase bank.
_LABELS: dict = {}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def language_of(lang: str):
    """Look up a Language, returning HTTP 400 rather than exiting on a typo."""
    if lang not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unknown language '{lang}'")
    return languages.get(lang)


def variant_of(variant: str) -> str:
    """Validate a variant name, returning HTTP 400 rather than exiting."""
    if variant not in VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{variant}' (expected one of {', '.join(VARIANTS)})",
        )
    return variant


def checkpoint_dir_for(language, variant: str) -> Path:
    """The checkpoint directory holding one variant of one language's model."""
    return (language.finetune_checkpoint_dir if variant == "finetuned"
            else language.checkpoint_dir)


def find_checkpoint(lang: str, variant: str = "pretrained") -> Optional[Path]:
    """Newest checkpoint_step_*.pt for this language and variant, or None."""
    ckpt_dir = checkpoint_dir_for(language_of(lang), variant_of(variant))
    if not ckpt_dir.is_dir():
        return None

    best, best_step = None, -1
    for path in ckpt_dir.glob("checkpoint_step_*.pt"):
        try:
            step = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if step > best_step:
            best, best_step = path, step
    return best


def reasoning_labels(language) -> dict:
    """The in-language question/reasoning/answer labels for one language.

    Read from that language's phrase bank so the wrapping is byte-identical to
    what finetuning saw; falls back to the shared defaults if the phrase bank
    has not been generated.
    """
    if language.key in _LABELS:
        return _LABELS[language.key]

    labels = dict(DEFAULT_LABELS)
    try:
        with open(language.reasoning_phrasebank, encoding="utf-8") as f:
            labels.update(json.load(f).get("labels", {}))
    except (OSError, json.JSONDecodeError):
        logger.info("No phrase bank for %s — using default reasoning labels.", language.key)

    _LABELS[language.key] = labels
    return labels


def reasoning_template(language) -> str:
    """The prompt shape the finetuned model was trained on, with a placeholder."""
    labels = reasoning_labels(language)
    return f"{labels['question']}: {{}}\n{labels['reasoning']}:"


def format_reasoning_prompt(language, prompt: str) -> str:
    """Wrap a bare question in the exact format finetuning used.

    Applied server-side so it cannot drift from the training format, and
    skipped if the user already typed the question label themselves.
    """
    labels = reasoning_labels(language)
    if prompt.lstrip().startswith(f"{labels['question']}:"):
        return prompt
    return reasoning_template(language).format(prompt.strip())


def load_language(lang: str, variant: str, device: torch.device) -> dict:
    """Load (and cache) the model + tokenizer for one language and variant."""
    key = (lang, variant)
    if key in _CACHE:
        return _CACHE[key]

    language = language_of(lang)
    variant = variant_of(variant)
    ckpt_path = find_checkpoint(lang, variant)
    if ckpt_path is None:
        directory = checkpoint_dir_for(language, variant).relative_to(PROJECT_ROOT)
        hint = (f"Run: python main.py finetune --lang {lang}"
                if variant == "finetuned"
                else f"Run: python main.py hub --lang {lang}")
        raise HTTPException(
            status_code=404,
            detail=f"No {variant} checkpoint found in {directory}/. {hint}",
        )

    logger.info("Loading %s (%s) from %s ...", lang, variant, ckpt_path.name)
    t0 = time.time()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Each language+variant gets its own model instance, and each language its
    # own tokenizer; nothing in this cache is shared across languages.
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    tokenizer = Tokenizer.from_file(str(language.tokenizer_path))

    entry = {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "device": device,
        "language": language,
        "variant": variant,
        "checkpoint": ckpt_path.name,
        "step": checkpoint.get("step"),
        "max_seq_len": config.get("max_seq_len", 512),
        "eos_token_id": config.get("eos_token_id", 3),
        "vocab_size": config["vocab_size"],
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    _CACHE[key] = entry
    logger.info("Loaded %s (%s) in %.1fs (%.1fM params)",
                lang, variant, time.time() - t0, entry["params"] / 1e6)
    return entry


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def apply_repetition_penalty(logits: torch.Tensor, prev_ids: torch.Tensor, penalty: float):
    """Divide logits of already-seen tokens (CTRL-style repetition penalty)."""
    if penalty == 1.0 or prev_ids.numel() == 0:
        return logits
    unique = torch.unique(prev_ids)
    selected = logits[0, unique]
    # Positive logits are divided, negative ones multiplied, so both move down.
    logits[0, unique] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits


def filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Mask out tokens outside the top-k / nucleus (top-p) candidate set."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        # Exclusive cumulative sum: keep every token up to and including the
        # one that pushes the running mass past top_p.
        exclusive = sorted_probs.cumsum(dim=-1) - sorted_probs
        remove = exclusive > top_p
        remove[..., 0] = False
        logits = logits.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))

    return logits


@torch.no_grad()
def stream_tokens(entry: dict, prompt: str, params: "GenerateRequest") -> Iterator[str]:
    """Yield server-sent events carrying incremental decoded text."""
    model, tokenizer = entry["model"], entry["tokenizer"]
    device, max_seq_len = entry["device"], entry["max_seq_len"]
    eos_id = entry["eos_token_id"]

    if params.seed is not None:
        torch.manual_seed(params.seed)

    ids = tokenizer.encode(prompt).ids
    if not ids:
        yield sse({"error": "Prompt encoded to zero tokens."})
        return

    generated = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = generated.size(1)
    emitted = ""
    t0 = time.time()
    n_new = 0

    for _ in range(params.max_new_tokens):
        context = generated[:, -max_seq_len:]
        logits, _ = model(context)
        next_logits = logits[:, -1, :].float()

        next_logits = apply_repetition_penalty(next_logits, generated[0], params.repetition_penalty)

        if params.greedy:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / max(params.temperature, 1e-8)
            next_logits = filter_top_k_top_p(next_logits, params.top_k, params.top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
        n_new += 1

        if next_token.item() == eos_id:
            break

        # Decode the whole continuation each step and emit only the delta —
        # robust for BPE, where one token is not always one printable chunk.
        text = tokenizer.decode(generated[0, prompt_len:].tolist())
        if len(text) > len(emitted):
            yield sse({"delta": text[len(emitted):]})
            emitted = text

    elapsed = time.time() - t0
    yield sse({
        "done": True,
        "tokens": n_new,
        "elapsed": round(elapsed, 2),
        "tokens_per_sec": round(n_new / elapsed, 1) if elapsed > 0 else None,
    })


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    lang: str = Field(..., description="hindi or nepali")
    variant: str = Field("pretrained", description="pretrained or finetuned")
    prompt: str
    reasoning_format: bool = Field(
        False,
        description="Wrap the prompt as 'प्रश्न: <text>\\nतर्क:', the exact format "
                    "the finetuned models were trained on.",
    )
    max_new_tokens: int = Field(120, ge=1, le=512)
    temperature: float = Field(0.9, ge=0.01, le=2.0)
    top_k: int = Field(40, ge=0, le=5000)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    greedy: bool = False
    seed: Optional[int] = None


@app.get("/")
def index():
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/api/models")
def list_models():
    """Report every language+variant that has a usable checkpoint.

    One entry per (language, variant) pair, finetuned first so a client that
    preselects the first available entry lands on the Phase 3 model.
    """
    out = []
    for lang in LANGUAGES:
        language = languages.get(lang)
        for variant in VARIANTS:
            ckpt = find_checkpoint(lang, variant)
            out.append({
                "lang": lang,
                "variant": variant,
                "label": language.display,
                "model": language.model_label,
                "available": ckpt is not None,
                "checkpoint": ckpt.name if ckpt else None,
                "loaded": (lang, variant) in _CACHE,
                "reasoning_template": reasoning_template(language),
            })
    return {"models": out, "device": str(app.state.device)}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    entry = load_language(req.lang, req.variant, app.state.device)
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty.")

    prompt = req.prompt
    if req.reasoning_format:
        prompt = format_reasoning_prompt(entry["language"], prompt)

    return StreamingResponse(
        stream_tokens(entry, prompt, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    parser = argparse.ArgumentParser(description="Local chat UI for the Hindi and Nepali models")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Default: cuda when available, else cpu")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    app.state.device = device

    import uvicorn
    logger.info("Device: %s", device)
    logger.info("Open http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
