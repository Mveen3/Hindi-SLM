# Language Models and Agents — Bilingual Indic SLM

Two fully independent causal decoder-only Transformer language models, built and trained from scratch in PyTorch:

- **Model H — Hindi**, the higher-resource language
- **Model L — Nepali**, the lower-resource language

The two models share **no data, no tokenizer, no vocabulary and no weights**. They share only the implementation code, which is parameterized by language. The entire codebase operates as a unified, monolithic end-to-end pipeline covering tokenization, pretraining, generative evaluation, synthetic reasoning dataset generation, supervised reasoning fine-tuning, and diagnostic analysis.

---

## Repository Layout

The implementation lives once, in `src/`. Everything that makes a model what it is — its corpus, tokenizer, vocabulary, configuration, checkpoints, and logs — lives in that language's own directory and is never shared.

```
slm/
├── src/                         # ── shared implementation (language-agnostic) ──
│   ├── languages.py             # the ONLY place that knows about hindi/nepali
│   ├── model/transformer.py     # MHA · FFN · TransformerBlock · DecoderOnlyTransformer
│   ├── data/                    # clean · crawl · download · split · tokenizer · hub · dataset
│   │   ├── dataset.py           #   streaming JSONL datasets for train and eval
│   │   ├── clean.py             #   Devanagari normalization & MinHash deduplication
│   │   ├── split.py             #   split-time bucketed out-of-core shuffler
│   │   ├── tokenizer.py         #   BPE tokenizer trainer & fertility evaluator
│   │   ├── crawler.py           #   domain web crawler
│   │   ├── download.py          #   Hugging Face dataset downloader
│   │   ├── hub.py               #   Hub checkpoint downloader
│   │   └── reasoning/           #   synthetic reasoning data generation
│   ├── train/                   # trainer.py (local) · kaggle.py (2× T4) · utils.py
│   ├── finetune/                # reasoning finetuning (dataset, trainer, evaluate, lora)
│   ├── eval/                    # lm_metrics · generation · attention · causal_mask · plots
│   └── kaggle_runner.ipynb      # Kaggle multi-GPU training driver notebook
│
├── hindi/                       # ── Model H artifacts (nothing shared) ──
│   ├── configs/                 # model_config.yaml · train_config.yaml · data_sources.yaml
│   │                              · reasoning_config.yaml · finetune_config.yaml
│   ├── tokenizer/                 tokenizer.json · vocabulary.txt
│   ├── data/                      raw/ interim/ processed/   (gitignored)
│   │                              reasoning/ — phrasebank.json + samples.txt tracked
│   ├── checkpoints/               *.pt  pretrained (step 40000) (gitignored)
│   ├── checkpoints_finetune/      *.pt  best_checkpoint.pt      (gitignored)
│   ├── logs_pretraining/          pretraining loss histories  (gitignored)
│   └── logs_finetune/             finetuning loss histories   (gitignored)
├── nepali/                      # ── Model L artifacts, identical shape ──
│
├── main.py                      # Unified CLI entrypoint for all pipeline & model commands
├── tests/test_attention.py      # causality + fused-vs-manual equivalence + param count
├── tests/test_lora.py           # LoRA mechanics, parameter efficiency, merge correctness
├── tests/test_reasoning_data.py # re-derives every reasoning label from the text
├── report/                      # ALL figures, tables and write-ups
│   ├── report.md                # comprehensive technical report
│   ├── hindi/                   # Model H results (metrics, plots, attention heatmaps)
│   └── nepali/                  # Model L results, same shape
│
├── app.py / index.html          # local inference UI (pretrained + finetuned)
└── docs/                        # assignment brief
```

**Why the code is shared.** Model H and Model L share an identical 24.5M parameter architecture. `src/languages.py` holds the per-language registry; every other module receives a `Language` instance and resolves its paths through it, eliminating hardcoded language names. The models themselves remain completely independent — separate corpora, tokenizers, vocabularies, weights, and training runs.

---

## Architectural & Empirical Results

Full technical report with analysis, plots, and attention heatmaps:
**[report/report.md](report/report.md)**

Both models: 6 layers, $d_{\text{model}} = 512$, 8 heads, 512-token context, **24,534,016** parameters (tied input/output embeddings), 40,000 optimizer steps (~2.62B training tokens each).

### Pretraining Language Modeling Performance

| Metric | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Total Pretraining Documents | 1,748,721 | 1,930,872 |
| Raw Corpus Size | 10.48 GB | 9.51 GB |
| Tokenizer Vocabulary Size | 10,000 | 10,000 |
| Fertility (Tokens per Word) | **1.3141** | **1.4919** |
| Characters per Token | **3.8528** | **4.3432** |
| Out-of-Vocabulary (`<unk>`) Rate | $1.19 \times 10^{-5}$ | $3.21 \times 10^{-6}$ |
| Trainable Parameters | **24,534,016** | **24,534,016** |
| Test Cross-Entropy Loss | **3.3508** | **3.3241** |
| Test Perplexity (PPL) | **28.53** | **27.77** |
| Test Bits-per-Byte (BPB) | **0.4929** | **0.4148** |
| Causal Mask Verification | **100% Passed (0.0 diff)** | **100% Passed (0.0 diff)** |
| Mean Attention Entropy | 0.930 nats | 0.802 nats |
| Mean Attention Distance | 3.50 tokens | 2.93 tokens |
| Best Generation chrF ($T=0.5 / 1.0$) | **27.38** | **26.81** |

**Key Finding:** Model L achieves slightly lower perplexity and significantly lower bits-per-byte (0.4148 vs 0.4929) because its BPE tokenizer compresses more characters per token (4.34 vs 3.85), predicting more efficiently per byte of text.

### Reasoning Finetuning Performance

Evaluated on 2,500 held-out examples per language across 15 comparative and deductive reasoning tasks:

| Metric | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| **Finetuned Exact Match (Best Checkpoint)** | **34.08%** (852 / 2,500) | **32.72%** (818 / 2,500) |
| **Pretrained Baseline Exact Match** | 23.36% (584 / 2,500) | 20.04% (501 / 2,500) |
| **Exact Match Absolute Delta** | **+10.72 pp** | **+12.68 pp** |
| **Format Adherence** | **100.00%** | **100.00%** |
| **1-Hop Accuracy** | **41.7%** | **53.8%** |
| **2-Hop Accuracy** | **32.6%** | **23.2%** |
| **3-Hop Accuracy** | **28.1%** | **23.4%** |
| `compare_equal` Accuracy | 0.0% → **52.8%** | 2.3% → **100.0%** |
| `verify_claim` Accuracy | 0.7% → **48.2%** | 1.4% → **84.7%** |
| `order_ascending` Accuracy | 0.0% → **37.1%** | 0.0% → **5.7%** |
| `order_descending` Accuracy | 0.0% → **31.9%** | 0.0% → **3.9%** |
| Hallucination Error Reduction | **−74.3%** | **−52.5%** |
| Pretraining Knowledge Retention | **91.87%** (Preserved) | **85.45%** (Moderate Drift) |
| Optimal Checkpoint | Step 150 (`best_checkpoint.pt`) | Step 250 (`best_checkpoint.pt`) |

---

## Reproduction & Pipeline Execution

```bash
conda activate ml
pip install -r requirements.txt
```

Set `HF_TOKEN=hf_...` in your `.env` file for HuggingFace Hub sync. For synthetic phrasing generation, set `GROK_API_KEY=gsk_...` (optional — `--offline` uses cached and hand-authored phrase banks).

```bash
# 1. Data Pipeline
python main.py download --lang hindi                  # download curated corpora from HF Hub
python main.py crawl --lang hindi                     # web crawl for domain corpus (resumable)
python main.py clean --lang hindi                     # Devanagari normalization & LSH deduplication
python main.py split --lang hindi                     # out-of-core bucketed shuffling -> train/val/test
python main.py tokenizer --lang hindi --action both   # train 10k BPE tokenizer & evaluate fertility

# 2. Pretraining (resumes automatically if interrupted)
python main.py train-kaggle --lang hindi              # on Kaggle 2x T4 GPUs (with HF Hub sync)
python main.py train --lang hindi                     # or locally on consumer GPU/CPU

# 3. Pretraining Evaluation
python main.py evaluate --lang hindi                  # CE, PPL, BPB, generation, attention heatmaps
pytest tests/test_attention.py                        # attention causality and fused-vs-manual tests
pytest tests/test_lora.py                             # LoRA mechanics and gradient flow tests

# 4. Synthetic Reasoning Data Generation
python main.py reasoning-data --lang both --offline   # deterministic generation using phrasebank
python tests/test_reasoning_data.py --lang both       # verify every reasoning label independently

# 5. Supervised Reasoning Fine-Tuning (SFT)
python main.py finetune --lang hindi --fresh          # full parameter SFT with validation checkpointing
python main.py finetune --lang nepali --fresh         # runs ~15-20 min on an RTX 3050

# 6. Reasoning Evaluation & Attention Shift Analysis
python main.py evaluate-reasoning --lang hindi        # evaluates best checkpoint against pretrained
python main.py evaluate-reasoning --lang nepali

# 7. Interactive Serving UI
python app.py                                         # launches local chat UI at http://127.0.0.1:8000
```

---

## Synthetic Reasoning Data Generation

The reasoning generation engine (`src/data/reasoning/generator.py`) generates comparative and transitive reasoning datasets without data leakage:

| Stage | Responsibility | Mechanism |
|---|---|---|
| **1. Phrasing** | Grammatical variety | Authoring ~200 templates per language cached in `phrasebank.json` (or offline seeds) |
| **2. Instantiation** | Ground truth calculation | Python derives values, comparisons, chains, and answers deterministically |

**Key Properties:**
- **Zero Hallucinated Labels**: Ground-truth answers are computed mathematically by the generator, not generated by an LLM.
- **Strict Leakage Partitioning**: Entity pools (names, cities, objects) and phrasing templates are strictly disjoint between train, val, and test splits.
- **Vocabulary-Conscious Digits**: Hindi uses ASCII numerals (`0-9`) and Nepali uses Devanagari numerals (`०-९`), matching tokenizer frequencies and ensuring **zero `<unk>` tokens**.
- **Independent Verification**: `python tests/test_reasoning_data.py --lang both` re-parses generated texts, extracts entities, and independently re-derives all answers (100% passed).

---

## Supervised Reasoning Finetuning (SFT)

Implemented in `src/finetune/trainer.py` and evaluated in `src/finetune/evaluate.py`:

| Parameter | Configuration | Rationale |
|---|---|---|
| **PEFT Mode** | `peft_type: "full"` | Full parameter finetuning; LoRA adapters with 24M parameters caused bottlenecking |
| **Framing** | Direct Question-Answer (`\nउत्तर: <ans>`) | Eliminates autoregressive repetition loops observed with multi-sentence CoT |
| **Loss Masking** | Target-only (`-100` on prompt) | Model is only penalized on answer emission, not on prompt tokens |
| **Checkpointing** | Validation loss selection (`best_checkpoint.pt`) | Overfitting after 200 steps tripled entity hallucinations; best checkpoint preserves generalization |
| **Regularization** | `weight_decay: 0.05`, `label_smoothing: 0.05` | Prevents entity memorization and stabilizes token probabilities |
| **Decoding** | `repetition_penalty = 1.15` (Greedy) | Solved repetition loops in multi-entity ordering tasks (`0% → 37.1%`) |

---

## Interactive Serving & Chat UI

A lightweight FastAPI application (`app.py` + `index.html`) serves all four model variants:
- Hindi Pretrained (`hindi/checkpoints/checkpoint_step_40000.pt`)
- Hindi Finetuned (`hindi/checkpoints_finetune/best_checkpoint.pt`)
- Nepali Pretrained (`nepali/checkpoints/checkpoint_step_40000.pt`)
- Nepali Finetuned (`nepali/checkpoints_finetune/best_checkpoint.pt`)

Includes real-time greedy decoding, temperature, top-$p$ nucleus sampling, and side-by-side comparative reasoning inspection.

---

## Checkpoints and Datasets

All models and datasets are hosted externally on the Hugging Face Hub:

- **Hugging Face Model Repository:** [mveen3/hindi_slm](https://huggingface.co/mveen3/hindi_slm)
  - Pretrained checkpoints & training logs: `hindi/pretraining/`, `nepali/pretraining/`
  - Finetuned reasoning checkpoints & logs: `hindi/finetuning/`, `nepali/finetuning/`
- **Hugging Face Dataset Repository:** [mveen3/hindi_slm](https://huggingface.co/datasets/mveen3/hindi_slm)
  - Pretrained processed corpora (`train.jsonl`, `val.jsonl`, `test.jsonl`): `hindi/pretraining/`, `nepali/pretraining/`
  - Synthetic reasoning datasets & phrasebanks: `hindi/finetuning/`, `nepali/finetuning/`

```bash
# Pull pretrained checkpoints from Hugging Face:
python main.py hub --lang hindi --stage pretraining
python main.py hub --lang nepali --stage pretraining

# Pull finetuned reasoning checkpoints:
python main.py hub --lang hindi --stage finetuning
python main.py hub --lang nepali --stage finetuning

# Upload processed pretraining & reasoning datasets to Hugging Face:
python scripts/upload_datasets.py

# Upload finetuned reasoning models & logs to Hugging Face:
python scripts/upload_finetuned_models.py
```


