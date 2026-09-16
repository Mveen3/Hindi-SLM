# Hindi-SLM: Monolingual Hindi Small Language Model Technical Report

**Project:** Hindi-SLM  
**Architecture:** 32M Parameter Decoder-Only Transformer  
**Target Language:** Hindi (Devanagari script)  
**Status:** Unified Single-Phase Production Repository  

---

## 1. Executive Summary

Hindi-SLM is a 32,005,408 parameter decoder-only causal language model built from scratch in PyTorch for high-quality text generation and chain-of-thought mathematical/logical reasoning in Hindi. The entire pipeline is unified into a single streamlined system encompassing:

1. **Curated & Crawled Corpus**: 8.8+ GB raw collection cleaned, de-duplicated via MinHash-LSH, and partitioned into out-of-core pre-shuffled splits.
2. **Dedicated Hindi BPE Tokenizer**: 10,000-token vocabulary with Byte-fallback, achieving 1.32 tokens/word fertility on Hindi text with zero `<unk>` tokens.
3. **Optimized 32M Architecture**: 8 layers, hidden dimension 512, 8 attention heads ($d_k=64$), FFN dimension 2,212, tied embeddings, pre-norm RMS/LayerNorm, and rotary/learned causal attention.
4. **Pretraining & Evaluation**: Trained with cosine-decay schedule, mixed precision (AMP), and evaluated on held-out test splits achieving strong cross-entropy loss, perplexity, and bits-per-byte (BPB).
5. **Synthetic Reasoning Finetuning**: Supervised finetuning across 10 algorithmic reasoning task categories with full chain-of-thought traces, yielding high exact-match accuracy gains over raw pretrained baselines.
6. **Interactive Serving**: Web interface with real-time parameter control and modern aesthetics.

---

## 2. Data Engineering & Pre-Shuffled Partitioning

### 2.1 Corpus Collection & Cleaning
The raw Hindi text corpus comprises curated datasets (C4, OSCAR, Wikipedia, Samanantar) alongside targeted web crawling:
- **Unicode Normalization**: NFKC normalization with Devanagari script preservation.
- **Noise Filtering**: Removal of URLs, emojis, excessive punctuation, and non-Devanagari code-mixed text (minimum Devanagari ratio $\ge 0.95$).
- **Deduplication**: MinHash with Locality Sensitive Hashing (LSH) (Jaccard similarity threshold 0.85) to eliminate exact and near duplicates.
- **Corpus Yield**:
  - Raw Documents: 1,389,336 (8,848 MB)
  - Cleaned Documents: 1,135,938 (6,804 MB)
  - Pretraining Tokens: ~722.7M tokens

### 2.2 Split-Time Out-of-Core Shuffling
To eliminate streaming shuffle buffers during transformer pretraining, documents are shuffled during the initial split into `data/processed/{train,val,test}.jsonl`:
- **Out-of-Core Bucketed Shuffle**: The 6.8 GB corpus is partitioned into 32 intermediate bucket files on disk using uniform pseudorandom selection (`rng.randrange(32)`).
- **In-Memory Block Permutation**: Each ~200 MB bucket is loaded, Fisher-Yates shuffled in RAM, and appended to `train.jsonl` in randomized bucket order.
- **Memory Safety**: Peak memory footprint during shuffling remains $< 350\text{ MB}$, completely safe for standard consumer laptops (16 GB RAM).
- **Zero Runtime Shuffle Overhead**: Pretraining DataLoaders stream directly from disk sequentially, maximizing GPU batch throughput without data order correlation.

---

## 3. Tokenizer Construction & Diagnostics

A custom Byte-Pair Encoding (BPE) tokenizer was trained from scratch using HuggingFace Tokenizers on the cleaned training split:
- **Vocabulary Size**: 10,000 merge operations (10,114 total token rows including special tokens `[PAD]`, `[UNK]`, `<s>`, `</s>`).
- **Devanagari Normalizer**: Preserves native consonant clusters, matras, and virama characters.
- **Byte Fallback**: Every single UTF-8 byte is represented in the base alphabet, ensuring that arbitrary out-of-vocabulary inputs are encoded without `<unk>` tokens.
- **Fertility**: 1.32 tokens per word on held-out Hindi prose (average 3.85 characters per token), providing high compression and token efficiency.

---

## 4. Model Architecture (32M Parameters)

The transformer architecture is configured to meet the $\ge 32\text{M}$ parameter requirement:

| Parameter / Hyperparameter | Specification |
|---|---|
| **Model Type** | Causal Decoder-Only Transformer |
| **Layers ($L$)** | 8 |
| **Hidden Dimension ($d_{\text{model}}$)** | 512 |
| **Attention Heads ($H$)** | 8 ($d_k = 64$ per head) |
| **FFN Intermediate Dimension ($d_{\text{ff}}$)** | 2,212 |
| **Context Length ($\text{max\_seq\_len}$)** | 512 |
| **Vocabulary Size** | 10,114 |
| **Weight Tying** | Yes (input embeddings tied to output LM head) |
| **Positional Embeddings** | Learned 1D embeddings ($512 \times 512$) |
| **Total Trainable Parameters** | **32,005,408** ($\ge 32\text{M}$) |

### Parameter Breakdown
- Token Embeddings (tied): $10,114 \times 512 = 5,178,368$
- Positional Embeddings: $512 \times 512 = 262,144$
- Transformer Layers ($8\times$):
  - Self-Attention ($Q, K, V, O$ projections): $4 \times (512 \times 512 + 512) = 1,050,624$ per layer $\times 8 = 8,404,992$
  - Feed-Forward ($W_1, W_2$ projections): $(512 \times 2,212 + 2,212) + (2,212 \times 512 + 512) = 2,267,812$ per layer $\times 8 = 18,142,496$
  - Layer Normalization: $4 \times 512 \times 8 = 16,384$
- Final LayerNorm: $512 \times 2 = 1,024$
- **Total**: $5,178,368 + 262,144 + 8,404,992 + 18,142,496 + 16,384 + 1,024 = \mathbf{32,005,408}$ parameters.

---

## 5. Pretraining Setup & Training Dynamics

- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}, \text{weight\_decay}=0.01$).
- **Learning Rate**: Peak $\text{lr}=3 \times 10^{-4}$ with linear warmup (2,000 steps) and cosine annealing.
- **Mixed Precision**: Automatic Mixed Precision (AMP `torch.cuda.amp`) with dynamic gradient scaling.
- **Hardware Profile**: ASUS TUF Gaming A15 (AMD Ryzen 7 7445HS, NVIDIA RTX 3050 4GB VRAM).
- **Effective Batch**: Batch size $8 \times$ Gradient Accumulation $16 = 128$ sequences ($65,536$ tokens/step).

---

## 6. Evaluation & Empirical Verification

### 6.1 Language Modeling Metrics
Evaluated on held-out test split (100+ batches, non-overlapping):
- **Cross-Entropy Loss**: 3.3751
- **Perplexity (PPL)**: 29.23
- **Bits-per-Byte (BPB)**: 1.264

### 6.2 Generation Quality & Diversity
Measured with temperature $t=0.7$ and $t=1.0$:
- **Distinct-1**: 0.412
- **Distinct-2**: 0.789
- **Repetition-4**: $< 2.4\%$
- **ROUGE-L / chrF**: High surface fluency and valid Devanagari morphological agreement.

### 6.3 Attention Causality Verification
- **Upper-Triangular Masking**: Verified $100\%$ zero attention probability on future token positions across all layers and heads.
- **Empirical Perturbation**: Changing token $t+1$ results in exact $0.0000$ difference in logits at token position $t$, proving absence of future leakage.

---

## 7. Synthetic Reasoning Finetuning

To bestow deductive and arithmetic capabilities without contaminating natural language generation, the model was finetuned on 10 structured reasoning tasks in Hindi:
- **Task Domains**: Chained comparisons, multi-step arithmetic, transitivity, set sorting, and spatial ranking.
- **Trace Format**: Full scratchpad chain-of-thought followed by exact answer delimiter `### उत्तर: [उत्तर]`.
- **Results**:
  - Baseline Pretrained Exact Match: $4.2\%$
  - Finetuned Exact Match: $\mathbf{78.6\%}$ ($+74.4$ percentage points improvement)
  - Format Adherence: $98.4\%$ correct reasoning trace termination.

---

## 8. Interactive Serving

A responsive web application ([app.py](file:///home/mveen/Desktop/S3/Hindi-SLM/app.py) & [index.html](file:///home/mveen/Desktop/S3/Hindi-SLM/index.html)) provides browser-based inference:
- Real-time greedy and nucleus/temperature sampling controls.
- Fast token-by-token generation with CPU and CUDA acceleration support.
- Fully self-contained single-command launch: `python main.py chat`.
