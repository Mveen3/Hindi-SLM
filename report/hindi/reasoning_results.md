# Reasoning results — Hindi (Model H)

Held-out synthetic test split: **2500 examples**, entity names and question phrasings disjoint from training.

## Pretrained vs finetuned

| Metric | Pretrained | Finetuned | Delta |
|---|---:|---:|---:|
| Exact match | 23.36% | 34.08% | +10.72 pp |
| Emitted answer label | 99.76% | 100.00% | +0.24 pp |
| Gold answer appears in output | 28.60% | 29.48% | +0.88 pp |

Pretrained checkpoint: `/home/mveen/Documents/lma_projects/slm/hindi/checkpoints/checkpoint_step_40000.pt` (step 40000)  
Finetuned checkpoint: `/home/mveen/Documents/lma_projects/slm/hindi/checkpoints_finetune/best_checkpoint.pt` (step 150)

## Accuracy by task

| Task | Hops | n | Pretrained | Finetuned |
|---|---:|---:|---:|---:|
| `chain_largest` | 3 | 169 | 11.2% | 20.7% |
| `chain_relation` | 3 | 206 | 43.2% | 30.6% |
| `chain_smallest` | 3 | 159 | 12.6% | 27.7% |
| `compare_equal` | 1 | 123 | 0.0% | 52.8% |
| `compare_two_less` | 1 | 178 | 45.5% | 47.2% |
| `compare_two_more` | 1 | 183 | 55.2% | 51.9% |
| `middle_entity` | 2 | 135 | 26.7% | 26.7% |
| `mixed_direction_chain` | 3 | 205 | 34.6% | 32.2% |
| `numeric_difference` | 1 | 142 | 2.1% | 5.6% |
| `order_ascending` | 2 | 167 | 0.0% | 37.1% |
| `order_descending` | 2 | 141 | 0.0% | 31.9% |
| `transitive_largest` | 2 | 184 | 19.0% | 27.7% |
| `transitive_relation` | 2 | 216 | 39.8% | 36.1% |
| `transitive_smallest` | 2 | 153 | 27.5% | 34.6% |
| `verify_claim` | 1 | 139 | 0.7% | 48.2% |

## Accuracy by reasoning depth

| Hops | n | Pretrained | Finetuned |
|---:|---:|---:|---:|
| 1 | 765 | 24.3% | 41.7% |
| 2 | 996 | 20.0% | 32.6% |
| 3 | 739 | 26.9% | 28.1% |

## Error Taxonomy (Failure Mode Distribution)

| Error Category | Failure Count | Share of Errors |
|---|---:|---:|
| `conclusion_error` | 1299 | 78.8% |
| `hallucination_error` | 247 | 15.0% |
| `calculation_error` | 102 | 6.2% |

## General Language Retention (Catastrophic Forgetting Benchmark)

Evaluated on held-out general language pretraining validation set (`val.jsonl`):

| Metric | Pretrained Model | Finetuned Model | Delta / Retention |
|---|---:|---:|---:|
| Perplexity (PPL) | 27.45 | 29.88 | +2.43 PPL |
| Bits-Per-Byte (BPB) | 0.4958 | 0.5086 | +0.0128 BPB |
| Knowledge Retention Rate | 100.0% | 91.87% | **Preserved** |
