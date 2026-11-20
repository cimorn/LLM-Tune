# Calculator results across three models

[中文](README.md) · English

[Local split](../../../Datasets/calculator/manifest.json) with 256 training, 64 validation, and 256 held-out test items. GRPO, PPO, and DPO start from each model's SFT adapter.

## Held-out test

| Model | Base | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [10/256](Qwen3-0.6B/LLM/eval_final/metrics.json) | [235/256](Qwen3-0.6B/SFT/eval_final/metrics.json) | [235/256](Qwen3-0.6B/GRPO/eval_final/metrics.json) | [234/256](Qwen3-0.6B/PPO/eval_final/metrics.json) | [233/256](Qwen3-0.6B/DPO/eval_final/metrics.json) |
| DeepSeek-R1-1.5B | [0/256](DeepSeek-R1-1.5B/LLM/eval_final/metrics.json) | [235/256](DeepSeek-R1-1.5B/SFT/eval_final/metrics.json) | [234/256](DeepSeek-R1-1.5B/GRPO/eval_final/metrics.json) | [235/256](DeepSeek-R1-1.5B/PPO/eval_final/metrics.json) | [234/256](DeepSeek-R1-1.5B/DPO/eval_final/metrics.json) |
| Gemma-3-1B-IT | [77/256](Gemma-3-1B-IT/LLM/eval_final/metrics.json) | [185/256](Gemma-3-1B-IT/SFT/eval_final/metrics.json) | [187/256](Gemma-3-1B-IT/GRPO/eval_final/metrics.json) | [185/256](Gemma-3-1B-IT/PPO/eval_final/metrics.json) | [189/256](Gemma-3-1B-IT/DPO/eval_final/metrics.json) |

## Validation

| Model | Base | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [0/64](Qwen3-0.6B/LLM/eval/metrics.json) | [63/64](Qwen3-0.6B/SFT/eval/metrics.json) | [63/64](Qwen3-0.6B/GRPO/eval/metrics.json) | [63/64](Qwen3-0.6B/PPO/eval/metrics.json) | [63/64](Qwen3-0.6B/DPO/eval/metrics.json) |
| DeepSeek-R1-1.5B | [0/64](DeepSeek-R1-1.5B/LLM/eval/metrics.json) | [54/64](DeepSeek-R1-1.5B/SFT/eval/metrics.json) | [55/64](DeepSeek-R1-1.5B/GRPO/eval/metrics.json) | [56/64](DeepSeek-R1-1.5B/PPO/eval/metrics.json) | [56/64](DeepSeek-R1-1.5B/DPO/eval/metrics.json) |
| Gemma-3-1B-IT | [16/64](Gemma-3-1B-IT/LLM/eval/metrics.json) | [59/64](Gemma-3-1B-IT/SFT/eval/metrics.json) | [60/64](Gemma-3-1B-IT/GRPO/eval/metrics.json) | [60/64](Gemma-3-1B-IT/PPO/eval/metrics.json) | [59/64](Gemma-3-1B-IT/DPO/eval/metrics.json) |
