# LLM-Tune

中文 · [English](README.en.md)

LLM-Tune 是一个面向小型语言模型的微调实验项目，支持在不同模型和数据集上开展不同算法微调(如强化学习，监督学习等)，并对比微调效果。

<br>

## 实验入口

### 模型

| 模型 | 简介 | 完成日期 | 来源 |
| :--- | :--- | :--- | :--- |
| [Qwen3-0.6B](step1_LLM/Models/Qwen3-0.6B/model.json) | 千问 0.6B 模型，本项目关闭思考模式以直接生成工具调用 | 261115 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| [DeepSeek-R1-1.5B](step1_LLM/Models/DeepSeek-R1-1.5B/model.json) | DeepSeek-R1 的 1.5B 推理蒸馏模型，底座是 Qwen2.5-Math-1.5B | 261115 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| [Gemma-3-1B-IT](step1_LLM/Models/Gemma-3-1B-IT/model.json) | Google Gemma 3 的 1B 文本指令模型 | 待跑 | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) |

<br>

### 数据集

| 数据集 | 训练任务 | 实验结果 | 来源 |
| :---: | :--- | :--- | :--- |
| [calculator](Datasets/calculator/manifest.json) | 为算式和表格题生成可执行的计算器调用。<br>训练 [256 题](Datasets/calculator/train.jsonl) · 验证 [64 题](Datasets/calculator/val.jsonl) · 最终测试 [256 题](Datasets/calculator/test_final.jsonl) | [中文](Outputs/calculator/README.md) · [English](Outputs/calculator/README.en.md) | 项目生成 |
| [ToolACE 子集](Datasets/ToolACE/manifest.json) | 从候选工具中选择一个，并按用户请求填写参数。<br>训练 [36 题](Datasets/ToolACE/train.jsonl) · 验证 [11 题](Datasets/ToolACE/val.jsonl) · 最终测试 [12 题](Datasets/ToolACE/test_final.jsonl) | 待跑实验 | [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) |

<br>

### 微调算法

| 方法 | 类型 | 参数配置 |
| :--- | :---: | :---: |
| [SFT](step2_SFT/train.py) | 监督学习 | [Configs](step2_SFT/Configs/) |
| [GRPO](step3_Tune/RL/GRPO/train.py) | 强化学习 | [Configs](step3_Tune/RL/GRPO/Configs/) |
| [PPO](step3_Tune/RL/PPO/train.py) | 强化学习 | [Configs](step3_Tune/RL/PPO/Configs/) |

<br>

## 运行

在项目根目录使用 Python 3.11 和支持 CUDA BF16 的 PyTorch 环境。

```bash
python -m pip install -r requirements.txt
python main.py --dry-run
python main.py
```

在 [main.py](main.py) 里改 `MODEL`、`DATASET`、`ALGORITHM` 和 `STEP`。默认用 Qwen3-0.6B 跑计算器任务，依次执行基座评估、SFT、GRPO 和 PPO。也可在命令行指定参数，例如 `python main.py --model DeepSeek-R1-1.5B --dataset calculator --step compare` 会在本地实验跑完后，用训练输出重建图表和结果页。

`STEP` 取 `1`、`2`、`3`、`all` 或 `compare`。单独运行训练阶段时，会检查并复用已完成的前置结果。`ALGORITHM` 可选 `GRPO`、`PPO` 或 `both`。

<br>

## 配置与结果

- `step1_LLM/Models/<模型>/model.json` 保存模型来源、版本和加载参数。
- `step2_SFT/Configs/<模型>/<数据集>.json` 保存 SFT 参数。
- `step3_Tune/RL/<算法>/Configs/<模型>/<数据集>.json` 保存各 RL 算法的参数。
- `Outputs/<数据集>/<模型>/` 保存各阶段的评估、日志、权重和图表，`Outputs/<数据集>/README.md` 汇总该数据集的实验结果。

<br>

## 许可证

第三方模型和数据集遵循各自的许可条款。项目自有代码采用 [MIT 许可证](LICENSE)。
