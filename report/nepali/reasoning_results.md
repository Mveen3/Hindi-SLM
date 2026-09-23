# Reasoning results — Nepali (Model L)

Held-out synthetic test split: **2500 examples**, entity names and question phrasings disjoint from training.

## Pretrained vs finetuned

| Metric | Pretrained | Finetuned | Delta |
|---|---:|---:|---:|
| Exact match | 0.00% | 5.08% | +5.08 pp |
| Emitted answer label | 0.00% | 43.24% | +43.24 pp |
| Gold answer appears in output | 22.60% | 29.88% | +7.28 pp |

Pretrained checkpoint: `/home/mveen/Documents/lma_projects/slm/nepali/checkpoints/checkpoint_step_40000.pt` (step 40000)  
Finetuned checkpoint: `/home/mveen/Documents/lma_projects/slm/nepali/checkpoints_finetune/checkpoint_step_1560.pt` (step 1560)

## Accuracy by task

| Task | Hops | n | Pretrained | Finetuned |
|---|---:|---:|---:|---:|
| `chain_largest` | 3 | 175 | 0.0% | 2.9% |
| `chain_relation` | 3 | 183 | 0.0% | 4.9% |
| `chain_smallest` | 3 | 175 | 0.0% | 4.0% |
| `compare_equal` | 1 | 131 | 0.0% | 3.0% |
| `compare_two_less` | 1 | 169 | 0.0% | 6.5% |
| `compare_two_more` | 1 | 184 | 0.0% | 5.4% |
| `middle_entity` | 2 | 145 | 0.0% | 6.2% |
| `mixed_direction_chain` | 3 | 186 | 0.0% | 3.2% |
| `numeric_difference` | 1 | 146 | 0.0% | 0.0% |
| `order_ascending` | 2 | 140 | 0.0% | 0.0% |
| `order_descending` | 2 | 128 | 0.0% | 0.0% |
| `transitive_largest` | 2 | 196 | 0.0% | 3.1% |
| `transitive_relation` | 2 | 201 | 0.0% | 12.9% |
| `transitive_smallest` | 2 | 197 | 0.0% | 11.2% |
| `verify_claim` | 1 | 144 | 0.0% | 8.3% |

## Accuracy by reasoning depth

| Hops | n | Pretrained | Finetuned |
|---:|---:|---:|---:|
| 1 | 774 | 0.0% | 4.8% |
| 2 | 1007 | 0.0% | 6.3% |
| 3 | 719 | 0.0% | 3.8% |

## Error Taxonomy (Failure Mode Distribution)

| Error Category | Failure Count | Share of Errors |
|---|---:|---:|
| `format_error` | 1418 | 59.8% |
| `reasoning_step_error` | 547 | 23.1% |
| `hallucination_error` | 352 | 14.8% |
| `conclusion_error` | 50 | 2.1% |
| `calculation_error` | 6 | 0.3% |

## General Language Retention (Catastrophic Forgetting Benchmark)

Evaluated on held-out general language pretraining validation set (`val.jsonl`):

| Metric | Pretrained Model | Finetuned Model | Delta / Retention |
|---|---:|---:|---:|
| Perplexity (PPL) | 26.25 | 50.0 | +23.75 PPL |
| Bits-Per-Byte (BPB) | 0.401 | 0.4801 | +0.0791 BPB |
| Knowledge Retention Rate | 100.0% | 52.5% | **Catastrophic Forgetting** |
