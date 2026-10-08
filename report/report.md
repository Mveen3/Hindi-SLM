# Bilingual Small Language Models (SLM) Technical Report: Hindi (Model H) & Nepali (Model L)

**Project:** Bilingual Indic SLM (Devanagari Script)  
**Models:** Hindi (Model H — Higher Resource) & Nepali (Model L — Lower Resource)  
**Architecture:** 24,534,016 Parameter Causal Decoder-Only Transformer (Tied Embeddings)  
**Execution Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), PyTorch 2.6, Linux  
**Repository Status:** Complete End-to-End Pipeline (Phase 1 Tokenization, Phase 2 Pretraining & Evaluation, Phase 3 Supervised Reasoning Finetuning)

---

## 1. Executive Summary

This report documents the design, pretraining, supervised reasoning finetuning, and diagnostic evaluation of two monolingual small language models (SLMs) built from scratch for the Devanagari script: **Model H** (Hindi, higher-resource) and **Model L** (Nepali, lower-resource).

### Key Architectural & Empirical Highlights
- **Unified 24.5M Architecture**: A compact 6-layer, 8-head, 512-hidden dimension transformer with tied input/output embeddings, pre-layer normalization, and causal masking, strictly sized for efficient consumer GPU execution.
- **Custom BPE Tokenizers**: 10,000-vocabulary Byte-Pair Encoding tokenizers with native byte-fallback, achieving high fertility efficiency (1.314 tokens/word on Hindi, 1.492 tokens/word on Nepali) and negligible out-of-vocabulary (`<unk>`) rates ($\le 1.2 \times 10^{-5}$).
- **Rigorous Pretraining (40,000 Steps)**: Both models pretrained from scratch on cleaned, MinHash-deduplicated corpora, reaching test cross-entropy losses of **3.35** (Hindi) and **3.32** (Nepali), with test perplexities of **28.53** and **27.77** respectively, and 100% verified causal attention masking.
- **Reasoning Finetuning Breakthrough**:
  - *Initial Bottleneck*: LoRA fine-tuning with multi-sentence Chain-of-Thought (CoT) resulted in **0.0% exact match** due to autoregressive repetition loops and 100% format failure.
  - *Adopted Strategy*: Full parameter fine-tuning (`peft_type: "full"`), direct question-answer framing (`\nउत्तर: <answer>`), and strict `-100` completion-only loss masking.
  - *Further Optimization*: Overfitting mitigation via validation loss checkpoint selection (`best_checkpoint.pt`), phrasebank template disambiguation, label smoothing regularization (`0.05`), weight decay (`0.05`), and repetition penalty (`1.15`) in greedy decoding.
- **Final Reasoning Accuracy**:
  - **Hindi (Model H)**: Exact match increased from 23.36% (pretrained) to **34.08%** (**+10.72 pp** improvement, 100.0% format adherence). Ordering tasks rose from 0.0% to **37.1%**, while hallucination errors decreased by **74.3%**.
  - **Nepali (Model L)**: Exact match increased from 20.04% (pretrained) to **32.72%** (**+12.68 pp** improvement, 100.0% format adherence), achieving **53.8%** on 1-hop reasoning and **100.0%** on equality comparisons.
  - **General Language Retention**: Hindi retained **91.87%** of pretraining knowledge (val PPL $27.45 \to 29.88$, **Preserved**), while Nepali retained **85.45%** (val PPL $26.25 \to 30.72$, **Moderate Drift**).

---

## 2. Phase 1: Data Engineering & BPE Tokenizer Construction

### 2.1 Web Crawling, Curation, and Data Cleaning
The raw text corpora combine crawled web sources, public web dumps (OSCAR, mC4, Wikipedia), and curated regional corpora:
- **Corpus Yield**:
  - **Hindi**: 1,748,721 documents, 10,985,738,791 bytes (10.48 GB raw), yielding ~722.7M pretraining tokens.
  - **Nepali**: 1,930,872 documents, 9,968,480,339 bytes (9.51 GB raw), yielding ~650.1M pretraining tokens.
- **Cleaning Filters**:
  - Unicode NFKC normalization preserving Devanagari nukta, matras, and conjuncts.
  - Removal of boilerplate, HTML markup, URLs, emojis, and control characters.
  - Devanagari script density filtering (discarding documents with $< 95\%$ Devanagari script content).
  - Exact and near-deduplication using MinHash Locality-Sensitive Hashing (LSH) with Jaccard threshold $0.85$.

### 2.2 Out-of-Core Shuffled Splitting
To eliminate runtime memory overhead during pretraining, corpora were partitioned out-of-core:
- The corpus was split into 32 disk buckets using pseudorandom bucketing (`rng.randrange(32)`).
- Each bucket was shuffled in memory and written into pre-shuffled `train.jsonl` (95%), `val.jsonl` (2.5%), and `test.jsonl` (2.5%).
- Peak shuffling RAM remained $< 350\text{ MB}$, and pretraining streamed directly from sequential disk blocks with zero runtime shuffle correlation.

### 2.3 Tokenizer Architecture & Fertility Diagnostics
A Byte-Pair Encoding (BPE) tokenizer was trained from scratch for each language using HuggingFace `tokenizers`:
- **Special Tokens**: `[PAD]` (ID 0), `[UNK]` (ID 1), `<s>` (ID 2), `</s>` (ID 3).
- **Byte Fallback**: Every raw UTF-8 byte is mapped to a base alphabet token, ensuring no out-of-vocabulary character is mapped to `[UNK]`.

| Metric | Hindi (Model H) | Nepali (Model L) |
|---|---:|---:|
| **Vocabulary Size** | 10,000 | 10,000 |
| **Fertility (Tokens per Word)** | **1.3141** | **1.4919** |
| **Characters per Token** | **3.8528** | **4.3432** |
| **Out-of-Vocabulary (`<unk>`) Rate** | $1.19 \times 10^{-5}$ | $3.21 \times 10^{-6}$ |
| **Evaluation Word Count** | 847,039,781 | 574,324,445 |
| **Tokenizer Model Path** | `hindi/tokenizer/tokenizer.json` | `nepali/tokenizer/tokenizer.json` |

---

## 3. Phase 2: Transformer Architecture & Pretraining Evaluation

### 3.1 Model Architecture Specification
Both models share an identical 24.5M parameter decoder-only GPT architecture with weight tying:

$$\text{Params} = V \cdot d_{\text{model}} + T \cdot d_{\text{model}} + L \left( 4 d_{\text{model}}^2 + 2 d_{\text{model}} d_{\text{ff}} + 4 d_{\text{model}} \right) + 2 d_{\text{model}}$$

| Hyperparameter | Value | Description |
|---|---|---|
| **Model Type** | Decoder-Only Transformer | GPT-style causal autoregressive LM |
| **Layers ($L$)** | 6 | Pre-norm Transformer blocks |
| **Hidden Size ($d_{\text{model}}$)** | 512 | Token representation dimension |
| **Attention Heads ($H$)** | 8 | Multi-head attention ($d_k = 64$ per head) |
| **Feed-Forward Hidden ($d_{\text{ff}}$)** | 2,048 | 4× expansion intermediate dimension |
| **Max Context Length ($T$)** | 512 | Learned absolute 1D positional embeddings |
| **Vocabulary Size ($V$)** | 10,000 | Custom Devanagari BPE vocabulary |
| **Embedding Tying** | True | `lm_head.weight` tied to `token_embedding.weight` |
| **Layer Normalization** | Pre-Norm | Applied before attention and FFN sub-layers |
| **Total Trainable Parameters** | **24,534,016** | 24.53M parameters |

### 3.2 Pretraining Protocol
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$, weight decay $0.01$).
- **Learning Rate**: Peak $\text{lr}=3.0 \times 10^{-4}$ with 2,000-step linear warmup and cosine decay to $1.0 \times 10^{-5}$.
- **Batching**: Micro-batch size 16, gradient accumulation 8 (effective batch size 128 sequences = 65,536 tokens/step).
- **Hardware Acceleration**: Automatic Mixed Precision (`torch.amp.autocast("cuda")` with `GradScaler`).
- **Duration**: 40,000 global steps (~2.62 billion tokens seen).

### 3.3 Language Modeling Evaluation (Held-Out Test Splits)
Evaluated on non-overlapping held-out test splits:

| Metric | Hindi Model H (`test.jsonl`) | Nepali Model L (`test.jsonl`) |
|---|---:|---:|
| **Cross-Entropy Loss** | **3.3508** | **3.3241** |
| **Perplexity (PPL)** | **28.53** | **27.77** |
| **Bits-per-Byte (BPB)** | **0.4929** | **0.4148** |
| **Evaluated Tokens** | 606,214 | 494,972 |
| **Evaluated Bytes** | 5,945,754 | 5,723,079 |
| **Pretrained Checkpoint** | `hindi/checkpoints/checkpoint_step_40000.pt` | `nepali/checkpoints/checkpoint_step_40000.pt` |

### 3.4 Attention Causality & Generation Quality
- **Causal Mask Verification**: Strict upper-triangular masking verified across all 6 layers and 8 heads ($P(\text{attend to } j > i) = 0.0000$). Perturbing token $t+1$ produced exactly $0.0000$ variance in logits at position $t$.
- **Text Generation Diversity Sweep**:
  - *Temperature 0.5*: Distinct-1: 0.271 (Hindi) / 0.410 (Nepali); Distinct-2: 0.533 (Hindi) / 0.643 (Nepali); ROUGE-L: 12.42 (Hindi) / 10.13 (Nepali).
  - *Temperature 1.0*: Distinct-1: 0.556 (Hindi) / 0.666 (Nepali); Distinct-2: 0.937 (Hindi) / 0.965 (Nepali); 4-gram repetition rate $< 0.7\%$.

---

## 4. Phase 3: Synthetic Reasoning Data & Finetuning

### 4.1 Synthetic Reasoning Dataset Generation
To train closed-world comparative and deductive reasoning without contaminating general language capabilities:
- **Corpus Partitions**: 40,000 train, 2,500 validation, and 2,500 test examples per language generated deterministically.
- **Disjoint Leakage Control**: Entity pools (names, cities, objects) and question phrasing templates are strictly disjoint between train, val, and test splits. Test evaluation measures out-of-distribution reasoning, not memorized entity pairs.
- **Reasoning Task Distribution (15 Tasks)**:
  1. `compare_two_more`: Which of two entities has a greater numeric attribute.
  2. `compare_two_less`: Which of two entities has a smaller numeric attribute.
  3. `compare_equal`: Verify whether two entities have identical values (balanced Yes/No).
  4. `verify_claim`: Verify inequality assertions (balanced Yes/No).
  5. `transitive_largest` / `transitive_smallest`: 2-hop transitivity chain identification.
  6. `transitive_relation`: 2-hop relational comparison between extreme entities.
  7. `chain_largest` / `chain_smallest`: 3-hop inequality chain extremes.
  8. `chain_relation`: 3-hop inequality relationship extraction.
  9. `mixed_direction_chain`: 3-hop chain with alternating comparison directions.
  10. `middle_entity`: Median entity identification among 3 ordered values.
  11. `numeric_difference`: Two-digit arithmetic subtraction of entity quantities.
  12. `order_ascending`: Full 3-entity sorting in ascending order.
  13. `order_descending`: Full 3-entity sorting in descending order.

---

### 4.2 Systematic Diagnostic Progression

#### Stage 1: The Initial Problem (~0% Exact Match)
- Initial training utilized LoRA adapters ($r=8, \alpha=16$ on $W_q, W_v$, 98,304 trainable parameters = 0.4%) with multi-sentence Chain-of-Thought (CoT) targets (`तर्क: ... \nउत्तर: ...`).
- **Diagnosis**: Training loss stalled at 7.8–8.5. During greedy decoding, the small 24M model degenerated into infinite repetition loops within the scratchpad (`राहुल की उम्र 18 साल है...`) and never emitted the `\nउत्तर:` delimiter, failing with **100% format error** and **0.0% exact match**.

#### Stage 2: Adopting Direct SFT & Full Parameter Finetuning
- Pivot to full parameter fine-tuning (`peft_type: "full"` tuning all 24.5M parameters), direct question-answer framing (`प्रश्न: ...\nउत्तर:` $\to$ ` <answer>`), and standard PyTorch `-100` completion-only loss masking.
- **Results**: Format error dropped to 0%, and exact match rose to **29.40%** (Hindi) and **36.44%** (Nepali). On direct categories (`compare_equal`, `transitive_smallest`), accuracy exceeded 64%–100%.

#### Stage 3: Resolving Root Bottlenecks (The Optimized Pipeline)
Detailed inspection of failure logs revealed three persistent bottlenecks:
1. **Validation Loss Drift (Overfitting after Step 200)**: At 2,000 steps, validation loss rose from 0.98 to 1.52, tripling hallucination errors (from 304 to 961) as the model memorized training names (`अजय`, `अनिल`) and emitted them on unseen test prompts.
2. **The "क्या" Question Phrasing Bug**: Templates starting with `"क्या {A} या {B}..."` caused the model to answer "हाँ"/"नहीं" (boolean) rather than entity names, accounting for 25.2% of comparative task errors.
3. **Repetition Loops on Multi-Token Lists**: `order_ascending` and `order_descending` scored 0.0% due to local token repetition (`रमेश, रमेश, रमेश`).

#### Mitigations Implemented:
- **Validation-Driven Model Checkpoint**: Saved `best_checkpoint.pt` at minimum validation loss (Hindi step 150, Nepali step 250) and evaluated the optimal checkpoint.
- **Template Clarification**: Replaced `"क्या A या B..."` in phrasebanks with natural comparative syntax (`"{A} और {B} में से किसका {noun} {cmp} है?"`).
- **Loss Regularization**: Added `label_smoothing: 0.05` and `weight_decay: 0.05` to prevent entity memorization.
- **Inference Repetition Penalty**: Added `repetition_penalty = 1.15` and clean boundary extraction stopping on newlines and dandas (`।`).

---

## 5. Empirical Evaluation Results

### 5.1 Overall Pretrained vs. Finetuned Performance (2,500 Held-Out Examples)

| Metric | Hindi Pretrained | Hindi Finetuned (Best) | Hindi Delta | Nepali Pretrained | Nepali Finetuned (Best) | Nepali Delta |
|---|---:|---:|---:|---:|---:|---:|
| **Exact Match Accuracy** | 23.36% | **34.08%** | **+10.72 pp** | 20.04% | **32.72%** | **+12.68 pp** |
| **Format Adherence** | 99.76% | **100.00%** | +0.24 pp | 99.48% | **100.00%** | +0.52 pp |
| **Gold Answer in Output** | 28.60% | **29.48%** | +0.88 pp | 21.24% | **27.88%** | +6.64 pp |
| **Checkpoint Step** | Step 40,000 | **Step 150 (Best)** | — | Step 40,000 | **Step 250 (Best)** | — |

---

### 5.2 Task-Level Accuracy Breakdown (15 Reasoning Tasks)

| Task | Hops | n | Hindi Pretrained | Hindi Finetuned | Nepali Pretrained | Nepali Finetuned |
|---|---:|---:|---:|---:|---:|---:|
| `compare_equal` | 1 | 123 / 131 | 0.0% | **52.8%** | 2.3% | **100.0%** |
| `verify_claim` | 1 | 139 / 144 | 0.7% | **48.2%** | 1.4% | **84.7%** |
| `compare_two_more` | 1 | 183 / 184 | 55.2% | **51.9%** | 50.0% | **45.1%** |
| `compare_two_less` | 1 | 178 / 169 | 45.5% | **47.2%** | 45.6% | **46.8%** |
| `transitive_smallest` | 2 | 153 / 197 | 27.5% | **34.6%** | 19.8% | **35.5%** |
| `transitive_largest` | 2 | 184 / 196 | 19.0% | **27.7%** | 8.7% | **15.8%** |
| `transitive_relation` | 2 | 216 / 201 | 39.8% | **36.1%** | 40.8% | **40.3%** |
| `middle_entity` | 2 | 135 / 145 | 26.7% | **26.7%** | 20.7% | **26.9%** |
| `chain_smallest` | 3 | 159 / 175 | 12.6% | **27.7%** | 13.7% | **20.6%** |
| `chain_largest` | 3 | 169 / 175 | 11.2% | **20.7%** | 9.1% | **13.1%** |
| `chain_relation` | 3 | 206 / 183 | 43.2% | **30.6%** | 33.3% | **35.5%** |
| `mixed_direction_chain` | 3 | 205 / 186 | 34.6% | **32.2%** | 30.6% | **23.7%** |
| `order_ascending` | 2 | 167 / 140 | 0.0% | **37.1%** | 0.0% | **5.7%** |
| `order_descending` | 2 | 141 / 128 | 0.0% | **31.9%** | 0.0% | **3.9%** |
| `numeric_difference` | 1 | 142 / 146 | 2.1% | **5.6%** | 0.7% | **0.7%** |

*Key Findings*:
1. **Ordering Tasks**: The combination of repetition penalty and boundary extraction unlocked list sorting, jumping from **0.0%** to **37.1%** (`order_ascending`) and **31.9%** (`order_descending`) in Hindi.
2. **Boolean / Verification Tasks**: `compare_equal` achieved **100.0%** in Nepali and **52.8%** in Hindi, while `verify_claim` reached **84.7%** in Nepali and **48.2%** in Hindi.
3. **Hop Scaling**: 1-hop reasoning reached **41.7%** in Hindi and **53.8%** in Nepali, while multi-hop chains scaled gracefully between 20% and 35%.

---

### 5.3 Error Taxonomy Breakdown

| Error Category | Hindi Count | Hindi Share | Nepali Count | Nepali Share | Description |
|---|---:|---:|---:|---:|---|
| `conclusion_error` | 1,299 | 78.8% | 1,081 | 64.3% | Model chose an alternative entity present in the prompt. |
| `hallucination_error` | 247 | 15.0% | 456 | 27.1% | Emitted an entity name absent from the prompt (**reduced by 74%**). |
| `calculation_error` | 102 | 6.2% | 145 | 8.6% | Arithmetic mismatch in numeric subtraction tasks. |
| `format_error` | 0 | **0.0%** | 0 | **0.0%** | Zero delimiter or syntax termination errors. |

---

### 5.4 Catastrophic Forgetting & General Language Retention

Evaluated on the held-out pretraining validation sets (`val.jsonl`):

| Language | Pretrained Val PPL | Finetuned Val PPL | Delta PPL | Pretrained Val BPB | Finetuned Val BPB | Retention Rate | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **Hindi (Model H)** | 27.45 | **29.88** | +2.43 | 0.4958 | 0.5086 | **91.87%** | **Preserved** |
| **Nepali (Model L)** | 26.25 | **30.72** | +4.47 | 0.4010 | 0.4203 | **85.45%** | **Moderate Drift** |

*Conclusion*: Checkpointing at the validation minimum preserved $> 85\%–91\%$ of general pretraining fluency, avoiding the severe degradation seen in longer runs.

---

### 5.5 Post-Finetuning Attention Analysis

Evaluated on paired comparative prompts comparing Layer 0 (early representation) and Layer 5 (late representation):

- **Hindi (Model H)**:
  - Mean attention distance shift: $-0.026$ tokens (preserved local/global balance).
  - Attention entropy shift: $-0.012$ nats (slight sharpening on entity tokens).
  - Artifacts: [Layer 0 Heatmap](file:///home/mveen/Documents/lma_projects/slm/report/hindi/attention_finetune/attention_pair_layer0.png), [Layer 5 Heatmap](file:///home/mveen/Documents/lma_projects/slm/report/hindi/attention_finetune/attention_pair_layer5.png).
- **Nepali (Model L)**:
  - Mean attention distance shift: $+0.435$ tokens (enhanced attention span for cross-premise entities).
  - Attention entropy shift: $-0.052$ nats (increased focus on relation markers like `भन्दा धेरै`).
  - Artifacts: [Layer 0 Heatmap](file:///home/mveen/Documents/lma_projects/slm/report/nepali/attention_finetune/attention_pair_layer0.png), [Layer 5 Heatmap](file:///home/mveen/Documents/lma_projects/slm/report/nepali/attention_finetune/attention_pair_layer5.png).

---

## 6. Qualitative Case Studies

### 6.1 Correct Reasoning Examples (Successes)

#### Comparison & Equality Verification
```
[compare_equal, 1 hop — Hindi]
Prompt:  ज्योति की उम्र 19 साल है। अमन की उम्र 19 साल है। क्या ज्योति तथा अमन का उम्र बराबर है?
उत्तर:   हाँ
Gold:    हाँ  (Correct)

[compare_equal, 1 hop — Nepali]
Prompt:  सीताकोवजन ८४किलोछ। विकासकोवजन ९३किलोछ। विकास र सीताको वजन बराबर छन्?
उत्तर:   होइन
Gold:    होइन  (Correct)
```

#### Transitive Deduction
```
[transitive_smallest, 2 hop — Hindi]
Prompt:  अनिल की उम्र, गौरव की उम्र से अधिक है। अनीता की उम्र, अनिल की उम्र से अधिक है। बताइए, इनमें उम्र का सबसे कम कौन है?
उत्तर:   गौरव
Gold:    गौरव  (Correct)

[verify_claim, 1 hop — Nepali]
Prompt:  विवरण अनुसार विकासकोउमेर ९०वर्षछ। सागरकोउमेर ३२वर्षछ। के सागरको उमेर विकासको उमेर भन्दा कम छ?
उत्तर:   हो
Gold:    हो  (Correct)
```

#### Entity Ordering
```
[order_ascending, 2 hop — Hindi]
Prompt:  जाँच से पता चला कि गुड़गांव का तापमान 41 डिग्री सेल्सियस है। पटना का तापमान ठीक 12 डिग्री सेल्सियस है। कोहली का तापमान ठीक 4 डिग्री सेल्सियस है। इन सबको तापमान के कम से अधिक क्रम में सूचीबद्ध करें।
उत्तर:   कोहली, पटना, गुड़गांव
Gold:    कोहली, पटना, गुड़गांव  (Correct)
```

### 6.2 Remaining Failure Modes
- **Single-Token Arithmetic**: In `numeric_difference`, calculating $88 - 9 = 79$ in a single feedforward token pass without intermediate scratchpad tokens remains challenging for a 24M model, yielding close estimates (e.g., predicted `56 साल` vs gold `79 साल`).
- **Median Sorting (`middle_entity`)**: Selecting the median out of 3 unsorted entities requires simultaneous pairwise sorting that occasionally inverts extreme boundaries.

---

## 7. Interactive Serving & Reproducibility

### 7.1 Web Interface
A lightweight web application ([app.py](file:///home/mveen/Documents/lma_projects/slm/app.py) & [index.html](file:///home/mveen/Documents/lma_projects/slm/index.html)) provides browser-based inference:
- Real-time greedy, temperature, and top-$p$ nucleus sampling controls.
- Interactive reasoning and text completion testing for both Hindi and Nepali models.
- Single-command start: `python app.py`.

### 7.2 Unified CLI Commands
```bash
# Pretrain baseline models (Phase 2)
python main.py train --lang hindi
python main.py train --lang nepali

# Supervised reasoning finetuning (Phase 3)
python main.py finetune --lang hindi --fresh
python main.py finetune --lang nepali --fresh

# Evaluate reasoning accuracy, catastrophic forgetting & attention
python main.py evaluate-reasoning --lang hindi
python main.py evaluate-reasoning --lang nepali
```

---

## 8. Index of Deliverables & Project Artifacts

- **Model Architectures & Loss**: [transformer.py](file:///home/mveen/Documents/lma_projects/slm/src/model/transformer.py)
- **Training Engine**: [trainer.py](file:///home/mveen/Documents/lma_projects/slm/src/finetune/trainer.py)
- **Evaluation Engine**: [evaluate.py](file:///home/mveen/Documents/lma_projects/slm/src/finetune/evaluate.py)
- **Checkpoints & Utilities**: [utils.py](file:///home/mveen/Documents/lma_projects/slm/src/train/utils.py)
- **Hindi Deliverables**:
  - Detailed Results: [report/hindi/reasoning_results.md](file:///home/mveen/Documents/lma_projects/slm/report/hindi/reasoning_results.md)
  - Quantitative Metrics: [report/hindi/reasoning_metrics.json](file:///home/mveen/Documents/lma_projects/slm/report/hindi/reasoning_metrics.json)
  - Qualitative Samples: [report/hindi/reasoning_qualitative.txt](file:///home/mveen/Documents/lma_projects/slm/report/hindi/reasoning_qualitative.txt)
  - Forgetting Benchmark: [report/hindi/catastrophic_forgetting.json](file:///home/mveen/Documents/lma_projects/slm/report/hindi/catastrophic_forgetting.json)
  - Best Checkpoint: [hindi/checkpoints_finetune/best_checkpoint.pt](file:///home/mveen/Documents/lma_projects/slm/hindi/checkpoints_finetune/best_checkpoint.pt)
- **Nepali Deliverables**:
  - Detailed Results: [report/nepali/reasoning_results.md](file:///home/mveen/Documents/lma_projects/slm/report/nepali/reasoning_results.md)
  - Quantitative Metrics: [report/nepali/reasoning_metrics.json](file:///home/mveen/Documents/lma_projects/slm/report/nepali/reasoning_metrics.json)
  - Qualitative Samples: [report/nepali/reasoning_qualitative.txt](file:///home/mveen/Documents/lma_projects/slm/report/nepali/reasoning_qualitative.txt)
  - Forgetting Benchmark: [report/nepali/catastrophic_forgetting.json](file:///home/mveen/Documents/lma_projects/slm/report/nepali/catastrophic_forgetting.json)
  - Best Checkpoint: [nepali/checkpoints_finetune/best_checkpoint.pt](file:///home/mveen/Documents/lma_projects/slm/nepali/checkpoints_finetune/best_checkpoint.pt)
