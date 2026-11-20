# LLM-Tune 计算器三模型评估

中文 · [English](README.en.md)

[自建数据划分](../../../Datasets/calculator/manifest.json)为训练 256 题、验证 64 题、留出测试 256 题。GRPO、PPO、DPO 从各模型的 SFT 权重接续。

## 留出测试

| 模型 | 基座 | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [10/256](Qwen3-0.6B/LLM/eval_final/metrics.json) | [235/256](Qwen3-0.6B/SFT/eval_final/metrics.json) | [235/256](Qwen3-0.6B/GRPO/eval_final/metrics.json) | [234/256](Qwen3-0.6B/PPO/eval_final/metrics.json) | [233/256](Qwen3-0.6B/DPO/eval_final/metrics.json) |
| DeepSeek-R1-1.5B | [0/256](DeepSeek-R1-1.5B/LLM/eval_final/metrics.json) | [235/256](DeepSeek-R1-1.5B/SFT/eval_final/metrics.json) | [234/256](DeepSeek-R1-1.5B/GRPO/eval_final/metrics.json) | [235/256](DeepSeek-R1-1.5B/PPO/eval_final/metrics.json) | [234/256](DeepSeek-R1-1.5B/DPO/eval_final/metrics.json) |
| Gemma-3-1B-IT | [77/256](Gemma-3-1B-IT/LLM/eval_final/metrics.json) | [185/256](Gemma-3-1B-IT/SFT/eval_final/metrics.json) | [187/256](Gemma-3-1B-IT/GRPO/eval_final/metrics.json) | [185/256](Gemma-3-1B-IT/PPO/eval_final/metrics.json) | [189/256](Gemma-3-1B-IT/DPO/eval_final/metrics.json) |

## 验证集

| 模型 | 基座 | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [0/64](Qwen3-0.6B/LLM/eval/metrics.json) | [63/64](Qwen3-0.6B/SFT/eval/metrics.json) | [63/64](Qwen3-0.6B/GRPO/eval/metrics.json) | [63/64](Qwen3-0.6B/PPO/eval/metrics.json) | [63/64](Qwen3-0.6B/DPO/eval/metrics.json) |
| DeepSeek-R1-1.5B | [0/64](DeepSeek-R1-1.5B/LLM/eval/metrics.json) | [54/64](DeepSeek-R1-1.5B/SFT/eval/metrics.json) | [55/64](DeepSeek-R1-1.5B/GRPO/eval/metrics.json) | [56/64](DeepSeek-R1-1.5B/PPO/eval/metrics.json) | [56/64](DeepSeek-R1-1.5B/DPO/eval/metrics.json) |
| Gemma-3-1B-IT | [16/64](Gemma-3-1B-IT/LLM/eval/metrics.json) | [59/64](Gemma-3-1B-IT/SFT/eval/metrics.json) | [60/64](Gemma-3-1B-IT/GRPO/eval/metrics.json) | [60/64](Gemma-3-1B-IT/PPO/eval/metrics.json) | [59/64](Gemma-3-1B-IT/DPO/eval/metrics.json) |
