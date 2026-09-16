# Reasoning results — Hindi (Model H)

Held-out synthetic test split: **2 examples**, entity names and question phrasings disjoint from training.

## Pretrained vs finetuned

| Metric | Pretrained | Finetuned | Delta |
|---|---:|---:|---:|
| Exact match | 0.00% | 0.00% | +0.00 pp |
| Emitted answer label | 0.00% | 100.00% | +100.00 pp |
| Gold answer appears in output | 100.00% | 100.00% | +0.00 pp |

Pretrained checkpoint: `/home/mveen/Desktop/S3/slm/hindi/checkpoints/checkpoint_step_40000.pt` (step 40000)  
Finetuned checkpoint: `/home/mveen/Desktop/S3/lma/project/hindi/checkpoints_finetune/checkpoint_step_936.pt` (step 936)

## Accuracy by task

| Task | Hops | n | Pretrained | Finetuned |
|---|---:|---:|---:|---:|
| `compare_two_less` | 1 | 1 | 0.0% | 0.0% |
| `compare_two_more` | 1 | 1 | 0.0% | 0.0% |

## Accuracy by reasoning depth

| Hops | n | Pretrained | Finetuned |
|---:|---:|---:|---:|
| 1 | 2 | 0.0% | 0.0% |
