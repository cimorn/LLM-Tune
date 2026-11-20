# 人类偏好三模型评估

中文 · [English](README.en.md)

预测人工偏好 A、B 或平局 C。[Kaggle 数据](https://www.kaggle.com/competitions/llm-classification-finetuning) · [固定划分](../../../Datasets/human_preference/manifest.json) · [完整诊断](diagnostics.json)

1,536 条训练样本、256 条带标签验证样本、3 条无标签提交样例。SFT 更新 64 步。各后训练分支从同一份 SFT 权重更新 24 步。GRPO 和 PPO 各采样 768 个标签，用人工标签命中情况给 0/1 奖励。DPO 用正确标签和抽样错误标签配对，96 次配对曝光，参考策略为初始 SFT。



## 验证集

表内为 **Log Loss / 准确率**。Log Loss 取人工标签负对数概率的均值。

| 模型 | 基座 | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | [1.4986 / 34.8%](Qwen3-0.6B/LLM/eval/metrics.json) | [1.1057 / 34.8%](Qwen3-0.6B/SFT/eval/metrics.json) | [1.1379 / 27.0%](Qwen3-0.6B/GRPO/eval/metrics.json) | [1.1457 / 28.9%](Qwen3-0.6B/PPO/eval/metrics.json) | [1.1125 / 32.0%](Qwen3-0.6B/DPO/eval/metrics.json) |
| DeepSeek-R1-1.5B | [1.2453 / 32.0%](DeepSeek-R1-1.5B/LLM/eval/metrics.json) | [1.1326 / 37.1%](DeepSeek-R1-1.5B/SFT/eval/metrics.json) | [1.1961 / 28.9%](DeepSeek-R1-1.5B/GRPO/eval/metrics.json) | [1.1962 / 32.4%](DeepSeek-R1-1.5B/PPO/eval/metrics.json) | [1.1422 / 35.5%](DeepSeek-R1-1.5B/DPO/eval/metrics.json) |
| Gemma-3-1B-IT | [4.2456 / 37.1%](Gemma-3-1B-IT/LLM/eval/metrics.json) | [1.2087 / 38.7%](Gemma-3-1B-IT/SFT/eval/metrics.json) | [1.2320 / 28.5%](Gemma-3-1B-IT/GRPO/eval/metrics.json) | [1.2567 / 28.5%](Gemma-3-1B-IT/PPO/eval/metrics.json) | [1.2001 / 38.7%](Gemma-3-1B-IT/DPO/eval/metrics.json) |

最低 Log Loss 为千问 SFT 的 **1.1057**。均匀概率为 1.0986，按训练集标签比例分配概率为 1.0943。验证集 A/B/C 分布为 89/98/69。Gemma SFT 和 DPO 各答对 **99/256**，固定猜验证集多数类 B 可答对 98 题。



## 后训练与 SFT 的配对比较

与 SFT 逐题比较，各算法列为**改对 / 改错**。DPO 差值为 DPO 减 SFT。

| 模型 | GRPO | PPO | DPO | DPO Δ Log Loss | 95% 区间 |
| :--- | ---: | :--: | :--: | :--: | :--: |
| Qwen3-0.6B | 66 / 86 | 59 / 74 | 30 / 37 | +0.0068 | [-0.0001, +0.0137] |
| DeepSeek-R1-1.5B | 53 / 74 | 27 / 39 | 8 / 12 | +0.0096 | [-0.0002, +0.0193] |
| Gemma-3-1B-IT | 48 / 74 | 60 / 86 | 4 / 4 | -0.0085 | [-0.0217, +0.0045] |

三个模型的 DPO 准确率差与 Log Loss 差的 95% 区间均跨 0。按规范化问题分组进行 10,000 次配对重采样。



## 预测标签分布

每格为 256 次预测中的 **A / B / C** 次数。

| 模型 | 基座 | SFT | GRPO | PPO | DPO |
| :--- | :--: | :--: | :--: | :--: | :--: |
| Qwen3-0.6B | 254 / 1 / 1 | 34 / 199 / 23 | 0 / 0 / 256 | 0 / 26 / 230 | 4 / 137 / 115 |
| DeepSeek-R1-1.5B | 169 / 33 / 54 | 0 / 236 / 20 | 0 / 45 / 211 | 0 / 131 / 125 | 1 / 205 / 50 |
| Gemma-3-1B-IT | 131 / 116 / 9 | 0 / 255 / 1 | 0 / 60 / 196 | 0 / 30 / 226 | 0 / 239 / 17 |

Gemma SFT 选了 255 次 B。DPO 的平局召回从 1/69 升到 5/69，A 类召回仍为 0。



### 千问 SFT 与 DPO 的两个例子

概率按 A / B / C 排列。最高概率并列时取字母靠前的标签。

| 问题摘要与 ID | 人工标签 | SFT | DPO |
| :--- | :--: | :--: | :--: |
| 猫喜不喜欢狗 · 1645990150 | C | B · 28.03% / 35.99% / 35.99% | C · 29.26% / 33.16% / 37.58% |
| 什么是 Stable Diffusion · 716611868 | B | B · 30.62% / 34.69% / 34.69% | C · 29.26% / 33.16% / 37.58% |



## 回答顺序与输入截断

交换两份回答并还原 A/B 标签后，统计**首选标签一致率 / 平均总变差**。

| 模型 | 基座 | SFT | GRPO | PPO | DPO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 5.1% / 0.431 | 13.3% / 0.064 | 100.0% / 0.048 | 85.2% / 0.120 | 28.1% / 0.061 |
| DeepSeek-R1-1.5B | 24.2% / 0.262 | 6.6% / 0.207 | 75.4% / 0.174 | 38.7% / 0.232 | 16.0% / 0.199 |
| Gemma-3-1B-IT | 32.0% / 0.672 | 0.4% / 0.379 | 70.3% / 0.225 | 83.2% / 0.222 | 5.1% / 0.340 |

千问 GRPO 全部预测为 C，交换一致率 100%，答对 69/256。512 token 上限使每组都截断同样的 160 条输入（62.5%）。[诊断 JSON](diagnostics.json)含混淆矩阵、宏平均 F1、截断分组统计和各阶段训练耗时及显存。

[千问 SFT 逐题预测](Qwen3-0.6B/SFT/eval/predictions.jsonl) · [千问 DPO 逐题预测](Qwen3-0.6B/DPO/eval/predictions.jsonl)



## 千问独立留出测试

千问用 5,000 条样本训练，在另外 512 条样本上拟合温度。留出集有 2,048 条带标签样本，按问题组与训练集、验证集分离。

| 阶段 | 准确率 | 原始 Log Loss | 校准后 Log Loss | 平局召回率 |
| :--- | :--: | :--: | :--: | :--: |
| SFT | 44.63% | 1.0471 | 1.0491 | 9.28% |
| GRPO | 44.58% | 1.3723 | 1.0427 | 18.24% |
| PPO | 44.82% | 1.4734 | 1.0460 | 11.84% |

温度校准保持首选标签不变。校准后 GRPO 相对原始 SFT 的 Log Loss 差为 −0.0044，配对 95% 区间为 [−0.0128, +0.0040]。[留出测试报告](related/qwen-calibration-holdout.json)
