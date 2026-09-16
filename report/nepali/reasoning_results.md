# Reasoning results — Nepali (Model L)

Held-out synthetic test split: **2500 examples**, entity names and question phrasings disjoint from training.

## Pretrained vs finetuned

| Metric | Pretrained | Finetuned | Delta |
|---|---:|---:|---:|
| Exact match | 0.00% | 15.00% | +15.00 pp |
| Emitted answer label | 0.00% | 79.24% | +79.24 pp |
| Gold answer appears in output | 20.88% | 49.36% | +28.48 pp |

Pretrained checkpoint: `/home/mveen/Desktop/S3/lma/project/nepali/checkpoints/checkpoint_step_40000.pt` (step 40000)  
Finetuned checkpoint: `/home/mveen/Desktop/S3/lma/project/nepali/checkpoints_finetune/checkpoint_step_936.pt` (step 936)

## Accuracy by task

| Task | Hops | n | Pretrained | Finetuned |
|---|---:|---:|---:|---:|
| `chain_largest` | 3 | 175 | 0.0% | 4.0% |
| `chain_relation` | 3 | 183 | 0.0% | 8.7% |
| `chain_smallest` | 3 | 175 | 0.0% | 4.0% |
| `compare_equal` | 1 | 131 | 0.0% | 57.2% |
| `compare_two_less` | 1 | 169 | 0.0% | 23.1% |
| `compare_two_more` | 1 | 184 | 0.0% | 18.5% |
| `middle_entity` | 2 | 145 | 0.0% | 5.5% |
| `mixed_direction_chain` | 3 | 186 | 0.0% | 9.7% |
| `numeric_difference` | 1 | 146 | 0.0% | 0.7% |
| `order_ascending` | 2 | 140 | 0.0% | 0.0% |
| `order_descending` | 2 | 128 | 0.0% | 0.0% |
| `transitive_largest` | 2 | 196 | 0.0% | 13.8% |
| `transitive_relation` | 2 | 201 | 0.0% | 16.4% |
| `transitive_smallest` | 2 | 197 | 0.0% | 19.3% |
| `verify_claim` | 1 | 144 | 0.0% | 50.0% |

## Accuracy by reasoning depth

| Hops | n | Pretrained | Finetuned |
|---:|---:|---:|---:|
| 1 | 774 | 0.0% | 28.5% |
| 2 | 1007 | 0.0% | 10.5% |
| 3 | 719 | 0.0% | 6.7% |
