| Model | Reverse (Step 1) | No-b words | b-words skip-b exact | b-leak | Step-2 task (random words) |
|---|---|---|---|---|---|
| Pretrained (no label smoothing) | 100.0% | 100.0% | 0.0% | 100.0% | 85.5% |
| Pretrained (label smoothing 0.1) | 100.0% | 100.0% | 0.0% | 100.0% | 85.5% |
| GRPO - naive reward | 25.9% | 30.3% | 8.7% | 0.0% | 27.3% |
| GRPO - similarity reward | 83.2% | 97.3% | 19.9% | 0.0% | 86.1% |
| GRPO - shaped reward, from no-LS model | 100.0% | 100.0% | 0.0% | 100.0% | 85.5% |
| GRPO - shaped reward, 600 steps | 81.8% | 95.7% | 33.0% | 0.0% | 86.3% |
| GRPO - shaped + entropy 0.02 | 82.8% | 96.9% | 31.4% | 0.0% | 87.0% |
| GRPO - shaped + entropy 0.05 | 84.1% | 98.4% | 31.7% | 0.1% | 88.3% |
| GRPO - shaped, G=32, 2000 steps | 82.4% | 96.4% | 27.3% | 0.0% | 86.1% |
| GRPO - shaped, G=16, 2000 steps | 83.7% | 97.9% | 38.7% | 0.0% | 90.2% |
| GRPO - shaped, 4000 steps (FINAL) | 82.5% | 96.5% | 39.0% | 0.0% | 88.1% |

Per-length exact match on b-words (skip-b task):

| Model | len 2 | len 3 | len 4 | len 5 | len 6 |
|---|---|---|---|---|---|
| Pretrained (no label smoothing) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Pretrained (label smoothing 0.1) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| GRPO - naive reward | 37.5% | 17.0% | 5.5% | 4.5% | 2.0% |
| GRPO - similarity reward | 45.0% | 26.0% | 23.5% | 11.5% | 13.5% |
| GRPO - shaped reward, from no-LS model | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| GRPO - shaped reward, 600 steps | 97.5% | 55.5% | 28.0% | 20.5% | 15.0% |
| GRPO - shaped + entropy 0.02 | 97.5% | 51.0% | 29.0% | 16.0% | 16.5% |
| GRPO - shaped + entropy 0.05 | 97.5% | 51.5% | 28.5% | 17.5% | 16.0% |
| GRPO - shaped, G=32, 2000 steps | 45.0% | 41.5% | 29.0% | 18.5% | 16.5% |
| GRPO - shaped, G=16, 2000 steps | 97.5% | 56.5% | 35.5% | 31.5% | 19.5% |
| GRPO - shaped, 4000 steps (FINAL) | 97.5% | 56.5% | 38.5% | 27.5% | 22.0% |
