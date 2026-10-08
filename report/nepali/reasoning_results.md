# Reasoning results — Nepali (Model L)

Held-out synthetic test split: **2500 examples**, entity names and question phrasings disjoint from training.

## Pretrained vs finetuned

| Metric | Pretrained | Finetuned | Delta |
|---|---:|---:|---:|
| Exact match | 20.04% | 32.72% | +12.68 pp |
| Emitted answer label | 99.48% | 100.00% | +0.52 pp |
| Gold answer appears in output | 21.24% | 27.88% | +6.64 pp |

Pretrained checkpoint: `/home/mveen/Documents/lma_projects/slm/nepali/checkpoints/checkpoint_step_40000.pt` (step 40000)  
Finetuned checkpoint: `/home/mveen/Documents/lma_projects/slm/nepali/checkpoints_finetune/best_checkpoint.pt` (step 250)

## Accuracy by task

| Task | Hops | n | Pretrained | Finetuned |
|---|---:|---:|---:|---:|
| `chain_largest` | 3 | 175 | 9.1% | 13.1% |
| `chain_relation` | 3 | 183 | 33.3% | 35.5% |
| `chain_smallest` | 3 | 175 | 13.7% | 20.6% |
| `compare_equal` | 1 | 131 | 2.3% | 100.0% |
| `compare_two_less` | 1 | 169 | 45.6% | 46.8% |
| `compare_two_more` | 1 | 184 | 50.0% | 45.1% |
| `middle_entity` | 2 | 145 | 20.7% | 26.9% |
| `mixed_direction_chain` | 3 | 186 | 30.6% | 23.7% |
| `numeric_difference` | 1 | 146 | 0.7% | 0.7% |
| `order_ascending` | 2 | 140 | 0.0% | 5.7% |
| `order_descending` | 2 | 128 | 0.0% | 3.9% |
| `transitive_largest` | 2 | 196 | 8.7% | 15.8% |
| `transitive_relation` | 2 | 201 | 40.8% | 40.3% |
| `transitive_smallest` | 2 | 197 | 19.8% | 35.5% |
| `verify_claim` | 1 | 144 | 1.4% | 84.7% |

## Accuracy by reasoning depth

| Hops | n | Pretrained | Finetuned |
|---:|---:|---:|---:|
| 1 | 774 | 22.6% | 53.8% |
| 2 | 1007 | 16.7% | 23.2% |
| 3 | 719 | 22.0% | 23.4% |

## Error Taxonomy (Failure Mode Distribution)

| Error Category | Failure Count | Share of Errors |
|---|---:|---:|
| `conclusion_error` | 1081 | 64.3% |
| `hallucination_error` | 456 | 27.1% |
| `calculation_error` | 145 | 8.6% |

## General Language Retention (Catastrophic Forgetting Benchmark)

Evaluated on held-out general language pretraining validation set (`val.jsonl`):

| Metric | Pretrained Model | Finetuned Model | Delta / Retention |
|---|---:|---:|---:|
| Perplexity (PPL) | 26.25 | 30.72 | +4.47 PPL |
| Bits-Per-Byte (BPB) | 0.401 | 0.4203 | +0.0193 BPB |
| Knowledge Retention Rate | 100.0% | 85.45% | **Moderate Drift** |
