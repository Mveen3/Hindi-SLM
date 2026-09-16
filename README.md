# Language Models and Agents — Individual Project

Two fully independent decoder-only Transformer language models, built and
trained from scratch in PyTorch:

- **Model H — Hindi**, the higher-resource language
- **Model L — Nepali**, the lower-resource language

The two models share **no data, no tokenizer, no vocabulary and no weights**.
They share only the implementation code, which is parameterised by language.

---

## Repository layout

The implementation lives once, in `src/`. Everything that makes a model what it
is — its corpus, tokenizer, vocabulary, configuration, checkpoints and logs —
lives in that language's own directory and is never shared.

```
project/
├── src/                         # ── shared implementation (language-agnostic) ──
│   ├── languages.py             # the ONLY place that knows about hindi/nepali
│   ├── model/transformer.py     # MHA · FFN · TransformerBlock · DecoderOnlyTransformer
│   ├── data/                    # clean · crawl · download · split · tokenizer · hub · dataset
│   │   ├── dataset.py           #   streaming JSONL datasets for train and eval
│   │   ├── clean.py             #   Devanagari normalization & deduplication
│   │   ├── split.py             #   split-time bucketed out-of-core shuffler
│   │   ├── tokenizer.py         #   SentencePiece BPE tokenizer trainer & evaluator
│   │   ├── crawler.py           #   domain web crawler
│   │   ├── download.py          #   Hugging Face dataset downloader
│   │   ├── hub.py               #   Hub checkpoint downloader
│   │   └── reasoning/           #   synthetic reasoning data generation
│   ├── train/                   # trainer.py (local) · kaggle.py (2× T4) · utils.py
│   ├── finetune/                # reasoning finetuning (dataset, trainer, evaluate, attention)
│   ├── eval/                    # lm_metrics · generation · attention · causal_mask · plots
│   └── kaggle_runner.ipynb      # Kaggle multi-GPU training driver notebook
│
├── hindi/                       # ── Model H artifacts (nothing shared) ──
│   ├── configs/                 # model_config.yaml · train_config.yaml · data_sources.yaml
│   │                              · reasoning_config.yaml · finetune_config.yaml
│   ├── tokenizer/                 tokenizer.json · vocabulary.txt
│   ├── data/                      raw/ interim/ processed/   (gitignored)
│   │                              reasoning/ — phrasebank.json + samples.txt tracked
│   ├── checkpoints/               *.pt  pretrained            (gitignored)
│   ├── checkpoints_finetune/      *.pt  finetuned             (gitignored)
│   ├── logs_pretraining/          pretraining loss histories  (gitignored)
│   └── logs_finetune/             finetuning loss histories   (gitignored)
├── nepali/                      # ── Model L artifacts, identical shape ──
│
├── main.py                      # Unified CLI entrypoint for all pipeline & model commands
├── tests/test_attention.py      # causality + fused-vs-manual equivalence
├── tests/test_reasoning_data.py # re-derives every reasoning label from the text
├── report/                      # ALL figures, tables and write-ups
│   ├── report.md                # comprehensive technical report
│   ├── hindi/                   # Model H results (metrics, plots, attention heatmaps)
│   └── nepali/                  # Model L results, same shape
│
├── app.py / index.html          # local inference UI (pretrained + finetuned)
└── docs/                        # assignment brief
```

**Why the code is shared.** Model H and Model L are the same architecture at
different vocabulary sizes. `src/languages.py` holds the per-language registry;
every other module receives a `Language` and resolves its paths through it, so
nothing else contains a language name. The models themselves remain completely
independent — separate corpora, tokenizers, vocabularies, weights, training runs.

---

## Results

Full write-up with analysis, plots and heatmaps:
**[report/report.md](report/report.md)**

Both models: 6 layers, d_model 512, 8 heads, 512-token context, ~24.36M
parameters, 25,000 optimizer steps (~3.28B training tokens each).

| Metric | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Corpus tokens | 722,757,422 | 749,977,230 |
| Tokenizer vocabulary | 10,000 | 10,000 |
| Trainable parameters | 24,355,840 | 24,359,424 |
| Test cross-entropy | **3.4729** | 3.6550 |
| Test perplexity | **32.23** | 38.67 |
| Test bits-per-byte | 0.5163 | **0.4431** |
| Best chrF (T = 1.0 / 1.5) | **26.44** | 25.72 |
| BLEU-4 (greedy) | **1.73** | 1.09 |
| Mean attention entropy | 0.930 | 0.802 |
| Mean attention distance | 3.50 | 2.93 |

**Headline finding:** Model H wins on perplexity, Model L wins on
bits-per-byte — the Nepali tokenizer packs more bytes per token (11.90 vs
9.70), which inflates its perplexity without meaning it predicts worse per
byte. See the comprehensive analysis in [report/report.md](report/report.md).

### Reasoning finetuning results

| Metric | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Finetuned exact match | **16.12%** (403/2500) | 15.00% (375/2500) |
| Format adherence | **88.20%** | 79.24% |
| Pretrained baseline EM | 0.00% | 0.00% |
| 1-hop accuracy | 30.5% | 28.5% |
| 2-hop accuracy | 11.3% | 10.5% |
| 3-hop accuracy | 7.7% | 6.7% |
| Δ attention distance (FT) | +0.516 tokens | +0.978 tokens |
| Δ attention entropy (FT) | −0.170 nats | −0.165 nats |

Finetuning taught both models the question/reasoning/answer format (0% → 80–88%
format adherence) and genuine comparative reasoning (16% EM on held-out entities
and phrasings). Accuracy drops with reasoning depth (30% → 8% from 1-hop to
3-hop), and attention becomes more focused and longer-range after finetuning.
Full analysis in the [consolidated technical report](report/report.md).

Causal masking is verified empirically for both models — altering a future
token changes the logits at the current position by exactly 0.0
(`report/<language>/causal_mask_verification.json`, plus `tests/test_attention.py`).

---

## Reproduction

```bash
conda activate ml
pip install -r requirements.txt
```

Add `HF_TOKEN=hf_...` to a project-root `.env` (needed for gated corpora and
checkpoint transfer), and `GROK_API_KEY=gsk_...` for the Phase 3 phrasing step
(optional — `--offline` uses the hand-authored phrase banks instead).

```bash
# 1. Data Pipeline (manual execution per language: --lang hindi or --lang nepali)
python main.py download --lang hindi                  # download curated corpora from HF Hub
python main.py crawl --lang hindi                     # web crawl for domain corpus (resumable)
python main.py clean --lang hindi                     # Devanagari normalization & LSH deduplication
python main.py split --lang hindi                     # out-of-core bucketed shuffling -> train/val/test
python main.py tokenizer --lang hindi --action both   # train 10k BPE tokenizer and compute statistics

# 2. Pretraining (resumes automatically if interrupted)
python main.py train-kaggle --lang hindi              # on Kaggle 2x T4 GPUs (with HF Hub sync)
python main.py train --lang hindi                     # or locally on consumer GPU/CPU

# 3. Pretraining Evaluation (writes all metrics & heatmaps to report/<language>/)
python main.py evaluate --lang hindi
python tests/test_attention.py                        # attention causality and equivalence tests

# 4. Synthetic Reasoning Finetuning Data
python main.py reasoning-data --lang both             # generate for both languages
python main.py reasoning-data --lang both --offline   # offline mode using cached/authored phrase banks
python tests/test_reasoning_data.py                   # verify every label independently

# 5. Reasoning Fine-Tuning (SFT)
python main.py finetune --lang hindi                 # ~10-20 min on an RTX 3050 (starts from pretrained)
python main.py finetune --lang nepali                # resumes automatically if interrupted

# 6. Reasoning Evaluation & Attention Shift Analysis
python main.py evaluate-reasoning --lang hindi
python main.py evaluate-reasoning --lang nepali

# 7. Quick Generation & Interactive Chat UI
python main.py generate --lang hindi --prompt "भारत एक"
python app.py                                         # launches local chat UI at http://127.0.0.1:8000
```

The chat UI serves **four** models — pretrained and finetuned for each language
— and preselects the finetuned one when it exists. Picking a finetuned model
auto-enables greedy decoding and the reasoning prompt format
(`प्रश्न: <question>\nतर्क:`, applied server-side so it stays byte-identical to
what finetuning saw). Flipping between the two variants on the same question is
the quickest way to see what Phase 3 changed:

| Variant | Same question, greedy |
|---|---|
| Pretrained | invents numbers, loops, never emits an answer label |
| Finetuned | states the premises, reasons, then `उत्तर: हरि` and stops |

Every checkpoint stores model weights, optimizer state, scheduler state, AMP
scaler state, the training step and the full config, so training resumes from
exactly where it stopped. Training on Kaggle additionally syncs checkpoints and
logs to the HuggingFace Hub, so a killed session loses at most `save_interval`
steps — the driver is `src/kaggle_runner.ipynb`.

Attention heatmaps label tokens in Devanagari and need a Devanagari font — on
Ubuntu: `sudo apt install fonts-noto-core`.

---

## Phase 3 — synthetic reasoning finetuning data

`python main.py reasoning-data` (implemented in `src/data/reasoning_data.py`) builds one comparative / transitive
reasoning corpus per model. Run it with no arguments for a menu (Hindi, Nepali,
or both); the two corpora are generated independently and share no entity names,
no templates and no examples.

**Ground truth is never produced by an LLM.** Generation is two separate stages:

| Stage | Who does it | What it decides |
|---|---|---|
| 1. Phrasing | `openai/gpt-oss-120b` on Groq | wording only — entity names, attribute vocabulary, and many grammatical variants of each sentence template |
| 2. Instantiation | Python (`generator.py`) | the entities, the hidden numeric values, and **every answer**, derived by comparison / sorting |

The model therefore never sees a value and never emits a label. A hallucination
can at worst yield an awkward sentence; it cannot produce a wrong answer. This
also fits the free Groq tier (30 req/min, 8K tokens/min, 200K tokens/day):
authoring ~200 reusable phrasings costs 8 requests and ~28K tokens per language
(roughly 5–10 minutes, since the tokens/min ceiling binds before the request
count) and is cached in `<language>/data/reasoning/phrasebank.json`, after which the
combinatorics of phrasing × entity × attribute × value × task yield an
arbitrarily large corpus offline. `--offline` skips the API entirely and uses the
hand-authored seed phrase banks in `src/data/reasoning/seeds.py`, so the corpus
is reproducible without our key.

**Tasks (15 types, 7 question groups).** Weighted towards multi-hop chaining:

- *Value premises* (numbers stated): which is greater / smaller, equality,
  numeric difference, claim verification, middle element, ascending / descending
  ordering of 3–4 entities.
- *Relation premises* (only pairwise inequalities stated, so the answer must be
  chained): `A > B, B > C ⇒ A vs C`; largest / smallest from a chain; 4–5 entity
  multi-hop chains; and chains whose premises mix "more" and "less" phrasing, so
  direction has to be normalised before chaining. Queried pairs are always
  **non-adjacent** in the chain, so a single premise can never answer them.

Nine attributes (age, height, weight, price, quantity, distance, population,
speed, temperature) over five entity pools (people, cities, objects, animals,
fruits). Each example is emitted as a chain-of-thought sequence:

```
प्रश्न: <premises in shuffled order> <question>
तर्क:  <step-by-step derivation>
उत्तर: <exact answer>
```

**Leakage control.** Two generalisation axes are partitioned *disjointly*
across train / val / test, and the partition is then verified rather than
assumed (`report/<language>/reasoning_dataset_stats.json` → `leakage`):

- **entity names** — a test question names entities the model never trained on;
- **premise and question phrasings** — the `fact`, `rel_fact` and `q_*` template
  pools are partitioned too, so the test set asks in sentence patterns never
  seen during finetuning.

Chain-of-thought *step* phrasings are shared deliberately: they are the output
style being taught, not a generalisation axis, and accuracy is scored on the
final answer. Numeric ranges are shared for the same reason — holding them out
would measure extrapolation, not reasoning. Prompts are exactly de-duplicated
globally, with the held-out splits claiming their prompts first.

**Two per-language decisions.** Numerals follow each pretraining corpus rather
than a single convention: Hindi uses ASCII digits and Nepali uses Devanagari
digits, because the Phase 1 tokenizers show exactly that skew (174 vs 30
Hindi vocabulary entries contain `0-9`/`०-९`; 24 vs 185 for Nepali). Keeping
numbers inside the frozen vocabulary avoids fragmenting every value into rare
single characters — the generated corpora contain **zero `<unk>` tokens** under
their own tokenizers. Comparisons are anchored on the *attribute* ("X's height
is more than Y's height") rather than the person ("X is taller"), since the
latter would require the entity's gender, which a name alone does not give; the
attribute's own possessive, oblique possessive, interrogative and copula are
carried as placeholders so agreement is always correct.

**Validation.** Every LLM-authored template must satisfy its placeholder
contract, contain no Latin script, and survive three semantic checks that the
contract alone misses: it may not state an attribute noun or unit literally
(one template serves all nine attributes), may not hard-code a comparative word
alongside a `{cmp}` placeholder (which would contradict itself when `{cmp}` is
filled with "less"), and may not place a copula next to `{copula}` (a doubled
verb). The hand-authored seeds are held to the same rules. Anything that fails
is dropped and counted in the statistics.

**Verification.** `python tests/test_reasoning_data.py` re-reads the emitted
text, re-extracts the entities and numbers from each rendered prompt,
recomputes the expected answer independently of the generator, and compares —
catching a correct computation paired with the wrong sentence. It also asserts
split disjointness, yes/no balance, task coverage, and the absence of `<unk>`.

**Rate limits — pacing, resuming, and the two ways a 429 can happen.** All
authoring traffic is paced *locally* before it is ever sent (20 req/min, 6500
tok/min against Groq's real 30/8000 ceiling, plus a fixed minimum gap between
requests), so a normal run never actually reaches Groq's limit. If Groq still
returns 429 — a shared key, clock drift between its accounting window and
ours — the response is read to tell the two cases apart, since only one of
them is worth waiting out:

- **Per-minute limit** — Groq reports exactly how many seconds are left in its
  window; that is slept, then the request retries automatically. This is
  fully recoverable and does not count against ordinary error retries (capped
  separately at 10 consecutive waits, so a genuine outage still gives up
  instead of looping forever).
- **Daily quota** (200K tokens/day on the free tier) — cannot be waited out in
  one process. Authoring **checkpoints to disk after every stage** — entity
  pools, attribute vocabulary, each batch of templates — and stops cleanly
  with an unmissable banner explaining what happened and what to do next. The
  run is not lost: it falls back to the seed phrase bank plus whatever was
  authored so far and still produces a complete dataset.

Because progress is checkpointed rather than the whole phrase bank being
redone, **re-running the exact same command later resumes** — already-authored
entities, attributes and template batches are detected and skipped, so no
quota already spent is ever paid for twice:

```bash
python main.py reasoning-data --lang hindi   # hits the daily cap partway through
# ... (hours or a day later) ...
python main.py reasoning-data --lang hindi   # picks up exactly where it stopped
```

Every run — including every resume — appends timestamped events to
`<language>/data/reasoning/authoring_log.txt` (entities/attributes/each
template batch authored, every rate-limit wait, every quota stop) rather than
overwriting it, so the full history across as many re-runs as the free tier
forces stays in one place. `phrasebank.json` itself works the same way: it is
never replaced from scratch on a resume, only added to. `--refresh-phrasebank`
is the one way to discard a cached bank and re-author everything from the seed
(re-spending the quota an earlier run already paid for). The generated
`train.jsonl` / `val.jsonl` / `test.jsonl` splits are the one thing that *is*
regenerated fresh on every run — that step is instant, offline and
deterministic, so there is nothing to preserve, and always reflects the
current, most complete phrase bank.

---

## Phase 3 — reasoning finetuning

`python main.py finetune` (implemented in `src/training/finetune.py`) finetunes each pretrained model on its own synthetic
reasoning corpus. The two runs are fully independent: each starts from its own
Phase 2 checkpoint, keeps its own frozen Phase 1 tokenizer and vocabulary,
reads its own reasoning splits, and writes to its own
`<language>/checkpoints_finetune/` — the pretrained checkpoint is never
overwritten, because it is the baseline arm of the evaluation.

```bash
python main.py finetune --lang hindi
python main.py finetune --lang nepali
python main.py evaluate-reasoning --lang hindi     # accuracy + attention
```

**Protocol** (brief §3.1), all of it recorded in
`<language>/configs/finetune_config.yaml` and copied into every checkpoint:

| Setting | Value | Why |
|---|---|---|
| Starting weights | that language's own Phase 2 checkpoint | required by the brief |
| Tokenizer / vocabulary | frozen, unchanged from Phase 1 | required by the brief |
| Optimizer | AdamW, fresh state | Adam moments from a 40k-step pretraining run would fight the first finetuning steps |
| Learning rate | 1.5e-5, cosine + 3% warmup | ~20× below pretraining's 3e-4; the model already speaks the language |
| Epochs | 3 over 40,000 examples | ~940 optimizer steps |
| Batch | 16 × 8 accumulation = 128 sequences | measured 2.6 GB peak on a 4 GB RTX 3050 |
| Precision | AMP fp16 | |

Model H and Model L use **identical** finetuning hyperparameters on purpose, so
the cross-tier comparison is not confounded by a difference in finetuning effort.

**Loss is computed on the answer only.** Each example is
`<s> prompt target </s>`, and prompt positions are set to `pad_token_id` in the
label tensor — which the Phase 2 model's own
`cross_entropy(..., ignore_index=pad_token_id)` already skips, so no model
change was needed. Training on the prompt would teach the model to generate
questions instead of answering them.

**Padding.** Batches are formed from examples of similar length and padded only
to the batch maximum, cutting wasted compute from ~69% (fixed 448-token
padding) to ~5%. Padding is on the **right**: causal masking already stops a
real token from attending to the pads after it, whereas left-padding would
shift every token onto the wrong learned positional embedding.

**Resume is mandatory and tested.** Checkpoints use the identical format as
pretraining (weights, optimizer, scheduler, AMP scaler, step, full config).
Re-running the same command after an interruption picks up from the newest
finetuning checkpoint; `--fresh` restarts from the pretrained weights instead.

### Evaluation

`python main.py evaluate-reasoning` (implemented in `src/eval/evaluate_reasoning.py`) runs both stages and writes everything graded
into `report/<language>/`:

| Artifact | Contents |
|---|---|
| `reasoning_results.md` | pretrained vs finetuned exact match, by task and by reasoning depth |
| `reasoning_metrics.json` | every number, both arms, plus all predictions |
| `reasoning_qualitative.txt` | successes and failures for the error analysis |
| `attention_finetune/attention_pair_layer{0,5}.png` | pretrained (top) vs finetuned (bottom), early and late layer |
| `attention_finetune/delta_{mean_distance,entropy}.png` | per-head shift after finetuning |
| `attention_finetune/attention_shift.json` | the numbers behind those figures |

Both arms go through identical prompting, greedy decoding and answer
extraction. The model sees only the question, generates its own chain of
thought, and its final answer is string-matched against the ground truth the
generator computed in Python. Two softer diagnostics separate *reasoning*
failure from *format* failure: whether the answer label was emitted at all, and
whether the correct answer appears anywhere in the output.

Exact match is the right metric here because every answer is closed-form — an
entity name, a yes/no word, a number with its unit, or a comma-joined ordering
— so there is no paraphrase to tolerate. The pretrained arm is expected to
score near zero: it has never seen the `प्रश्न / तर्क / उत्तर` format, which is
exactly what the comparison is meant to show.

Generation is batched by *exact* prompt length so no padding is needed —
necessary because learned absolute positional embeddings make left-padded
batches silently wrong.

---

## Large artifact links

Datasets, tokenized corpora and checkpoints are too large for git and are
hosted externally, per the project brief.

**Phase 1 — data and tokenizers (both languages):**
<https://drive.google.com/drive/folders/18OsopS1Teyavu2xgwPuL6AHPqFJ9vwwu?usp=drive_link>

**Phase 2 — pretrained checkpoints:**  
<https://drive.google.com/drive/folders/18OsopS1Teyavu2xgwPuL6AHPqFJ9vwwu?usp=drive_link>

| Model | File | Size |
|---|---|---|
| Model H (Hindi) | `checkpoint_step_25000.pt` | 281 MB |
| Model L (Nepali) | `checkpoint_step_25000.pt` | 281 MB |

**Phase 3 — finetuned reasoning checkpoints:**  
<https://drive.google.com/drive/folders/18OsopS1Teyavu2xgwPuL6AHPqFJ9vwwu?usp=drive_link>

| Model | File | Produced by |
|---|---|---|
| Model H (Hindi) | `hindi/checkpoints_finetune/checkpoint_step_*.pt` | `python main.py finetune --lang hindi` |
| Model L (Nepali) | `nepali/checkpoints_finetune/checkpoint_step_*.pt` | `python main.py finetune --lang nepali` |

The reasoning corpora themselves (`{train,val,test}.jsonl`, ~170 MB per
language) are in the same Drive folder; the phrase banks, qualitative samples
and dataset statistics that make them reproducible are committed in the repo.


Also mirrored on the HuggingFace Hub, which the resume logic uses directly:

```bash
python main.py hub --lang hindi     # mveen3/src-hindi-checkpoints
python main.py hub --lang nepali    # mveen3/src-nepali-checkpoints
```

Training logs and loss curves are committed in the repo, not external:
`report/hindi/loss_curve.png`, `report/nepali/loss_curve.png`, plus the
per-step loss histories embedded in every checkpoint.
