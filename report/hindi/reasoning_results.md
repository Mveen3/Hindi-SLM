# Reasoning results — Hindi (Model H)

Held-out synthetic test split: **2500 examples**, entity names and question phrasings disjoint from training.

## Pretrained vs finetuned

| Metric | Pretrained | Finetuned | Delta |
|---|---:|---:|---:|
| Exact match | 0.00% | 16.12% | +16.12 pp |
| Emitted answer label | 0.00% | 88.20% | +88.20 pp |
| Gold answer appears in output | 21.52% | 49.96% | +28.44 pp |

Pretrained checkpoint: `/home/mveen/Desktop/S3/lma/project/hindi/checkpoints/checkpoint_step_40000.pt` (step 40000)  
Finetuned checkpoint: `/home/mveen/Desktop/S3/lma/project/hindi/checkpoints_finetune/checkpoint_step_936.pt` (step 936)

## Accuracy by task

| Task | Hops | n | Pretrained | Finetuned |
|---|---:|---:|---:|---:|
| `chain_largest` | 3 | 169 | 0.0% | 1.8% |
| `chain_relation` | 3 | 206 | 0.0% | 15.0% |
| `chain_smallest` | 3 | 159 | 0.0% | 2.5% |
| `compare_equal` | 1 | 123 | 0.0% | 52.0% |
| `compare_two_less` | 1 | 178 | 0.0% | 25.3% |
| `compare_two_more` | 1 | 183 | 0.0% | 32.2% |
| `middle_entity` | 2 | 135 | 0.0% | 16.3% |
| `mixed_direction_chain` | 3 | 205 | 0.0% | 9.3% |
| `numeric_difference` | 1 | 142 | 0.0% | 1.4% |
| `order_ascending` | 2 | 167 | 0.0% | 0.0% |
| `order_descending` | 2 | 141 | 0.0% | 0.0% |
| `transitive_largest` | 2 | 184 | 0.0% | 14.7% |
| `transitive_relation` | 2 | 216 | 0.0% | 18.1% |
| `transitive_smallest` | 2 | 153 | 0.0% | 16.3% |
| `verify_claim` | 1 | 139 | 0.0% | 45.3% |

## Accuracy by reasoning depth

| Hops | n | Pretrained | Finetuned |
|---:|---:|---:|---:|
| 1 | 765 | 0.0% | 30.5% |
| 2 | 996 | 0.0% | 11.3% |
| 3 | 739 | 0.0% | 7.7% |
