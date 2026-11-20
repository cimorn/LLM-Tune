# Human preference results across three models

[中文](README.md) · English

Predict whether people prefer answer A, answer B, or a tie (C). [Kaggle data](https://www.kaggle.com/competitions/llm-classification-finetuning) · [Split manifest](../../../Datasets/human_preference/manifest.json) · [Diagnostics](diagnostics.json)

1,536 training items, 256 labeled validation items, and 3 unlabeled submission examples. SFT ran for 64 optimizer steps. Each post-training branch started from the same SFT checkpoint and ran 24 steps. GRPO and PPO each sampled 768 label actions with a binary match reward. DPO used 96 exposures to pairs of correct and sampled incorrect labels with frozen SFT as its reference.

## Validation

Cells show **Log Loss / accuracy**. Log Loss is the mean negative log probability of the human label.

| Model | Base | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [1.4986 / 34.8%](Qwen3-0.6B/LLM/eval/metrics.json) | [1.1057 / 34.8%](Qwen3-0.6B/SFT/eval/metrics.json) | [1.1379 / 27.0%](Qwen3-0.6B/GRPO/eval/metrics.json) | [1.1457 / 28.9%](Qwen3-0.6B/PPO/eval/metrics.json) | [1.1125 / 32.0%](Qwen3-0.6B/DPO/eval/metrics.json) |
| DeepSeek-R1-1.5B | [1.2453 / 32.0%](DeepSeek-R1-1.5B/LLM/eval/metrics.json) | [1.1326 / 37.1%](DeepSeek-R1-1.5B/SFT/eval/metrics.json) | [1.1961 / 28.9%](DeepSeek-R1-1.5B/GRPO/eval/metrics.json) | [1.1962 / 32.4%](DeepSeek-R1-1.5B/PPO/eval/metrics.json) | [1.1422 / 35.5%](DeepSeek-R1-1.5B/DPO/eval/metrics.json) |
| Gemma-3-1B-IT | [4.2456 / 37.1%](Gemma-3-1B-IT/LLM/eval/metrics.json) | [1.2087 / 38.7%](Gemma-3-1B-IT/SFT/eval/metrics.json) | [1.2320 / 28.5%](Gemma-3-1B-IT/GRPO/eval/metrics.json) | [1.2567 / 28.5%](Gemma-3-1B-IT/PPO/eval/metrics.json) | [1.2001 / 38.7%](Gemma-3-1B-IT/DPO/eval/metrics.json) |

Qwen SFT has the lowest Log Loss at **1.1057**. Uniform probabilities give 1.0986, and training label frequencies give 1.0943. Validation contains 89 A, 98 B, and 69 C labels. Gemma SFT and DPO each get **99/256**. Always choosing B, the validation majority label, gets 98.

## Paired comparison with SFT

Method columns show **incorrect to correct / correct to incorrect** against SFT. The loss delta is DPO minus SFT.

| Model | GRPO | PPO | DPO | DPO Δ Log Loss | 95% interval |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 66 / 86 | 59 / 74 | 30 / 37 | +0.0068 | [-0.0001, +0.0137] |
| DeepSeek-R1-1.5B | 53 / 74 | 27 / 39 | 8 / 12 | +0.0096 | [-0.0002, +0.0193] |
| Gemma-3-1B-IT | 48 / 74 | 60 / 86 | 4 / 4 | -0.0085 | [-0.0217, +0.0045] |

The paired 95% intervals for DPO accuracy and Log Loss differences contain zero for all three models. Intervals use 10,000 paired bootstrap samples grouped by normalized question.

## Label distribution

Each cell counts **A / B / C** among 256 predictions.

| Model | Base | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 254 / 1 / 1 | 34 / 199 / 23 | 0 / 0 / 256 | 0 / 26 / 230 | 4 / 137 / 115 |
| DeepSeek-R1-1.5B | 169 / 33 / 54 | 0 / 236 / 20 | 0 / 45 / 211 | 0 / 131 / 125 | 1 / 205 / 50 |
| Gemma-3-1B-IT | 131 / 116 / 9 | 0 / 255 / 1 | 0 / 60 / 196 | 0 / 30 / 226 | 0 / 239 / 17 |

Gemma SFT chose B on 255 items. DPO raised tie recall from 1/69 to 5/69, with A recall still at zero.

### Two Qwen SFT → DPO examples

Probabilities are in A / B / C order. Ties for the top probability go to the earlier label.

| Question summary and ID | Gold | SFT | DPO |
| :--- | ---: | ---: | ---: |
| Do cats like dogs? · 1645990150 | C | B · 28.03% / 35.99% / 35.99% | C · 29.26% / 33.16% / 37.58% |
| What is Stable Diffusion? · 716611868 | B | B · 30.62% / 34.69% / 34.69% | C · 29.26% / 33.16% / 37.58% |

## Answer order and truncation

Swap the answers and map the A/B labels back. Cells show **argmax agreement / mean total variation**.

| Model | Base | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 5.1% / 0.431 | 13.3% / 0.064 | 100.0% / 0.048 | 85.2% / 0.120 | 28.1% / 0.061 |
| DeepSeek-R1-1.5B | 24.2% / 0.262 | 6.6% / 0.207 | 75.4% / 0.174 | 38.7% / 0.232 | 16.0% / 0.199 |
| Gemma-3-1B-IT | 32.0% / 0.672 | 0.4% / 0.379 | 70.3% / 0.225 | 83.2% / 0.222 | 5.1% / 0.340 |

Qwen GRPO always chooses C, giving 100% swap agreement and 69/256 correct. The 512-token limit truncates the same 160 items (62.5%) in every run. [Diagnostics JSON](diagnostics.json) contains confusion matrices, macro F1, truncation splits, and training time and memory.

[Qwen SFT predictions](Qwen3-0.6B/SFT/eval/predictions.jsonl) · [Qwen DPO predictions](Qwen3-0.6B/DPO/eval/predictions.jsonl)

## Separate Qwen holdout run

This Qwen run trained on 5,000 examples and fit temperatures on another 512. The 2,048 labeled holdout examples use separate question groups.

| Stage | Accuracy | Raw Log Loss | Calibrated Log Loss | Tie recall |
| :--- | ---: | ---: | ---: | ---: |
| SFT | 44.63% | 1.0471 | 1.0491 | 9.28% |
| GRPO | 44.58% | 1.3723 | 1.0427 | 18.24% |
| PPO | 44.82% | 1.4734 | 1.0460 | 11.84% |

Temperature scaling preserves the highest-probability label. Calibrated GRPO minus raw SFT gives a Log Loss difference of −0.0044, with a paired 95% interval of [−0.0128, +0.0040]. [Holdout report](related/qwen-calibration-holdout.json)
