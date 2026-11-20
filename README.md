# LLM-Tune

中文 · [English](README.en.md)

小模型 LoRA 实验，支持 SFT、GRPO、PPO 和 DPO。

<br>

## 文件目录

```text
LLM-Tune/
├─ README.md
├─ README.en.md
├─ main.py                 # 运行入口
├─ compare.py              # 生成结果汇总与图表
├─ requirements.txt        # Python 依赖
├─ Datasets/               # 数据与判分规则
│  ├─ calculator/
│  ├─ ToolACE/
│  └─ human_preference/
├─ step1_LLM/              # 模型准备与评估
│  ├─ Models/
│  ├─ prepare.py
│  ├─ evaluate.py
│  └─ judge_eval.py
├─ step2_SFT/              # 监督微调
│  ├─ Configs/
│  └─ train.py
├─ step3_Tune/             # 微调算法
│  └─ RL/
│     ├─ GRPO/
│     │  ├─ Configs/
│     │  ├─ run/
│     │  └─ train.py
│     ├─ PPO/
│     │  ├─ Configs/
│     │  ├─ run/
│     │  └─ train.py
│     └─ DPO/
│        ├─ Configs/
│        ├─ run/
│        │  ├─ core.py
│        │  ├─ pairs.py
│        │  └─ scoring.py
│        └─ train.py
└─ Outputs/                # 按数据集和模型保存结果
   ├─ calculator/
   │  ├─ README.md
   │  ├─ README.en.md
   │  ├─ DeepSeek-R1-1.5B/
   │  ├─ Qwen3-0.6B/
   │  └─ matrix-v1/
   ├─ ToolACE-expanded/
   │  ├─ README.md
   │  ├─ README.en.md
   │  └─ report.html
   └─ human_preference/
      └─ matrix-v1/
```

<br>

## 实验入口

### 模型

| 模型 | 简介 | 完成日期 | 来源 |
| :--- | :--- | :--- | :--- |
| [Qwen3-0.6B](step1_LLM/Models/Qwen3-0.6B/model.json) | 千问 0.6B 模型，本项目关闭思考模式以直接生成工具调用 | 261115 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| [DeepSeek-R1-1.5B](step1_LLM/Models/DeepSeek-R1-1.5B/model.json) | DeepSeek-R1 的 1.5B 推理蒸馏模型，底座是 Qwen2.5-Math-1.5B | 261115 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| [Gemma-3-1B-IT](step1_LLM/Models/Gemma-3-1B-IT/model.json) | Google Gemma 3 的 1B 文本指令模型 | 261118 | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) |

<br>

### 数据集

| 数据集 | 训练任务 | 实验结果 | 完成日期 | 来源 |
| :---: | :--- | :--- | :---: | :--- |
| [Calculator](Datasets/calculator/manifest.json) | 为算式和表格题生成计算器调用。<br>训练 [256 题](Datasets/calculator/train.jsonl) · 验证 [64 题](Datasets/calculator/val.jsonl) · 最终测试 [256 题](Datasets/calculator/test_final.jsonl) | [中文](Outputs/calculator/matrix-v1/README.md) · [English](Outputs/calculator/matrix-v1/README.en.md) | 261115 | LLM-Tune 自建 |
| [人类偏好](Datasets/human_preference/manifest.json) | 预测人工更偏好回答 A、回答 B，还是平局。<br>训练 [1,536 条](Datasets/human_preference/train.jsonl) · 验证 [256 条](Datasets/human_preference/val.jsonl) | [中文](Outputs/human_preference/matrix-v1/README.md) · [English](Outputs/human_preference/matrix-v1/README.en.md) | 261117 | [Kaggle LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning) |
| [ToolACE](Datasets/ToolACE/manifest.json) | 选择工具并填写参数。<br>训练 [694 题](Datasets/ToolACE/train.jsonl) · 验证 [100 题](Datasets/ToolACE/val.jsonl) · 最终测试 [250 题](Datasets/ToolACE/test_final.jsonl) | [中文](Outputs/ToolACE-expanded/README.md) · [English](Outputs/ToolACE-expanded/README.en.md) | 261118 | [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) |

<br>

### 微调算法

| 方法 | 类型 | 参数配置 |
| :--- | :---: | :---: |
| [SFT](step2_SFT/train.py) | 监督学习 | [Configs](step2_SFT/Configs/) |
| [GRPO](step3_Tune/RL/GRPO/train.py) | 强化学习 | [Configs](step3_Tune/RL/GRPO/Configs/) |
| [PPO](step3_Tune/RL/PPO/train.py) | 强化学习 | [Configs](step3_Tune/RL/PPO/Configs/) |
| [DPO](step3_Tune/RL/DPO/train.py) | 强化学习 | [Configs](step3_Tune/RL/DPO/Configs/) |

<br>

## 运行

使用 Python 3.11 和支持 CUDA BF16 的 PyTorch。

```bash
python -m pip install -r requirements.txt
python main.py --dataset calculator --algorithm DPO
python main.py --dataset calculator --step test --algorithm DPO
```

已有 SFT 时用 `--step 3 --algorithm DPO` 接训。默认 `--algorithm both` 跑 GRPO 和 PPO，`--algorithm all` 再加入 DPO。

Kaggle 的 `train.csv` 和 `test.csv` 放在 `Datasets/human_preference/raw/`，也可用 `--input-dir` 指定目录。

```bash
python main.py --dataset human_preference --algorithm DPO --train-limit 1536 --val-limit 256
python main.py --dataset ToolACE --model Qwen3-0.6B --algorithm all
python main.py --dataset ToolACE --model Qwen3-0.6B --step test --algorithm all
```

ToolACE 使用已归档的 694/100/250 划分。用 `--model` 选择模型，结果写到 `Outputs/ToolACE/`。



## 配置与结果

- `step1_LLM/Models/<模型>/model.json` 保存模型来源、版本和加载参数。
- `step2_SFT/Configs/<模型>/<数据集>.json` 保存 SFT 参数。
- `step3_Tune/RL/<算法>/Configs/<模型>/<数据集>.json` 保存各 RL 算法的参数。
- `Outputs/<数据集>/<模型>/` 保存各阶段的评估、日志、权重和图表，`Outputs/<数据集>/README.md` 汇总该数据集的实验结果。
