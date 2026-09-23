# Reasoning results — Hindi (Model H)

Held-out synthetic test split: **2500 examples**, entity names and question phrasings disjoint from training.

## Pretrained vs finetuned

| Metric | Pretrained | Finetuned | Delta |
|---|---:|---:|---:|
| Exact match | 0.00% | 0.00% | +0.00 pp |
| Emitted answer label | 0.00% | 0.00% | +0.00 pp |
| Gold answer appears in output | 25.80% | 23.00% | -2.80 pp |

Pretrained checkpoint: `/home/mveen/Documents/lma_projects/slm/hindi/checkpoints/checkpoint_step_40000.pt` (step 40000)  
Finetuned checkpoint: `/home/mveen/Documents/lma_projects/slm/hindi/checkpoints_finetune/checkpoint_step_1560.pt` (step 1560)

## Accuracy by task

| Task | Hops | n | Pretrained | Finetuned |
|---|---:|---:|---:|---:|
| `chain_largest` | 3 | 169 | 0.0% | 0.0% |
| `chain_relation` | 3 | 206 | 0.0% | 0.0% |
| `chain_smallest` | 3 | 159 | 0.0% | 0.0% |
| `compare_equal` | 1 | 123 | 0.0% | 0.0% |
| `compare_two_less` | 1 | 178 | 0.0% | 0.0% |
| `compare_two_more` | 1 | 183 | 0.0% | 0.0% |
| `middle_entity` | 2 | 135 | 0.0% | 0.0% |
| `mixed_direction_chain` | 3 | 205 | 0.0% | 0.0% |
| `numeric_difference` | 1 | 142 | 0.0% | 0.0% |
| `order_ascending` | 2 | 167 | 0.0% | 0.0% |
| `order_descending` | 2 | 141 | 0.0% | 0.0% |
| `transitive_largest` | 2 | 184 | 0.0% | 0.0% |
| `transitive_relation` | 2 | 216 | 0.0% | 0.0% |
| `transitive_smallest` | 2 | 153 | 0.0% | 0.0% |
| `verify_claim` | 1 | 139 | 0.0% | 0.0% |

## Accuracy by reasoning depth

| Hops | n | Pretrained | Finetuned |
|---:|---:|---:|---:|
| 1 | 765 | 0.0% | 0.0% |
| 2 | 996 | 0.0% | 0.0% |
| 3 | 739 | 0.0% | 0.0% |

## Error Taxonomy (Failure Mode Distribution)

| Error Category | Failure Count | Share of Errors |
|---|---:|---:|
| `format_error` | 2500 | 100.0% |

## General Language Retention (Catastrophic Forgetting Benchmark)

Evaluated on held-out general language pretraining validation set (`val.jsonl`):

| Metric | Pretrained Model | Finetuned Model | Delta / Retention |
|---|---:|---:|---:|
| Perplexity (PPL) | 27.45 | 29.11 | +1.66 PPL |
| Bits-Per-Byte (BPB) | 0.4958 | 0.5046 | +0.0088 BPB |
| Knowledge Retention Rate | 100.0% | 94.3% | **Preserved (LoRA)** |
