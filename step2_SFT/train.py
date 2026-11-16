"""步骤 3：用工具调用示范做 completion-only LoRA SFT。

系统提示、用户输入和 padding 的标签为 -100；只有 JSON 答案及 EOS 参与损失。
模型内部负责 causal shift，数据不能再手动 shift 一次。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments

from step1_LLM.prepare import (
    dataset_task,
    ROOT, load_model_and_tokenizer, read_config, render_prompt,
    seed_everything, write_json,
)


def project_path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def prepare_output(path: Path) -> None:
    """每次实验用新目录，防止混入旧 adapter、日志和结果。"""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录非空，请指定新的 --output：{path}")
    path.mkdir(parents=True, exist_ok=True)


def load_training_records(path: Path, limit: int | None = None, config: dict | None = None) -> list[dict]:
    records = dataset_task(config or {}).load_records(path)
    if not records or any(row.get("split") != "train" for row in records):
        raise ValueError("训练只接受非空 train split；不能使用验证集或测试集")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须为正整数")
        records = records[:limit]
    return records


def snapshot_trainable(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().float().cpu().clone()
            for name, value in model.named_parameters() if value.requires_grad}


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure_update(model, before: dict[str, torch.Tensor]) -> dict:
    after = snapshot_trainable(model)
    if set(before) != set(after):
        raise RuntimeError("训练前后的可训练参数集合变化")
    finite = all(bool(torch.isfinite(t).all()) for t in after.values())
    changed = sum(not torch.equal(before[n], after[n]) for n in before)
    squares = sum(float((after[n] - before[n]).double().square().sum()) for n in before)
    return {
        "finite_parameters": finite,
        "changed_tensors": changed,
        "total_trainable_tensors": len(before),
        "trainable_parameters": sum(t.numel() for t in before.values()),
        "parameter_delta_l2": math.sqrt(squares),
        "before_sha256": state_fingerprint(before),
        "after_sha256": state_fingerprint(after),
        "nonzero_update": finite and changed > 0,
    }


def checkpoint_settings(cfg: dict) -> dict:
    """保留全部指定检查点；save_total_limit仅在显式配置时启用。"""
    save_at = cfg.get("save_at_steps", [])
    if not isinstance(save_at, list) or any(type(x) is not int or x <= 0 for x in save_at):
        raise ValueError("save_at_steps 必须是正整数列表")
    if len(save_at) != len(set(save_at)) or any(x > int(cfg["max_steps"]) for x in save_at):
        raise ValueError("save_at_steps 不能重复或超过max_steps")
    interval = cfg.get("save_steps", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("save_steps 必须是非负整数")
    if save_at and interval:
        raise ValueError("save_at_steps与save_steps只选一种")
    limit = cfg.get("save_total_limit")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("save_total_limit 必须为正整数或null")
    if save_at and limit is not None and limit < len(save_at):
        raise ValueError("save_total_limit会删除指定检查点，请保留全部save_at_steps")
    result = {"save_strategy": "steps" if interval else "no", "save_total_limit": limit,
              "save_only_model": False}
    if interval:
        result["save_steps"] = interval
    return result


class CheckpointMetadataCallback(TrainerCallback):
    """给每个adapter检查点记录自己的step/hash，而不是引用最终权重的hash。"""
    def __init__(self, stage: str, config: dict, before: dict, data_path: Path):
        self.stage = stage
        self.config = config
        self.before = before
        self.data_path = data_path
        self.save_at = set(config[stage].get("save_at_steps", []))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_at:
            control.should_save = True
        return control

    def on_save(self, args, state, control, model=None, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not (checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(f"检查点adapter未保存：{checkpoint}")
        if model is None:
            raise RuntimeError("检查点回调缺少model，无法保存参数指纹")
        write_json(checkpoint / "run_summary.json", {
            "stage": f"{self.stage}_checkpoint", "status": "checkpoint",
            "config": self.config, "optimizer_steps": state.global_step,
            "parent_output": str(Path(args.output_dir).resolve()),
            "data_path": str(self.data_path),
            "data_sha256": hashlib.sha256(self.data_path.read_bytes()).hexdigest(),
            "update": measure_update(model, self.before),
            "checkpoint_contents": "adapter, tokenizer, trainer/optimizer/scheduler/RNG state",
        })
        return control


class FiniteLogCallback(TrainerCallback):
    """尽早报告数值故障，避免把 NaN loss 包装成完成。"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key, value in (logs or {}).items():
            if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"step={state.global_step}: {key}={value}")


class CompletionDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int, config: dict | None = None):
        self.examples = []
        self.audit = []
        for row in records:
            prompt = render_prompt(tokenizer, row, config)
            answer = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(prompt + answer + tokenizer.eos_token,
                                        add_special_tokens=False)
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(f"{row['id']}: prompt/answer 边界分词不一致，拒绝错误 mask")
            if len(full_ids) > max_length:
                raise ValueError(f"{row['id']}: {len(full_ids)} tokens 超过 max_length={max_length}；提高上限")
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            if not any(value != -100 for value in labels):
                raise ValueError(f"{row['id']}: 没有监督 token")
            self.examples.append({"input_ids": full_ids, "labels": labels})
            self.audit.append({"id": row["id"], "prompt_tokens": len(prompt_ids),
                               "completion_tokens": len(full_ids)-len(prompt_ids),
                               "total_tokens": len(full_ids)})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        length = max(len(row["input_ids"]) for row in features)
        ids, labels, attention = [], [], []
        for row in features:
            padding = length - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            attention.append([1] * len(row["input_ids"]) + [0] * padding)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long)}


def run(args) -> dict:
    config = read_config(project_path(args.config))
    cfg = dict(config["sft"])
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if int(cfg["max_steps"]) <= 0:
        raise ValueError("max_steps 必须为正整数")
    saving = checkpoint_settings(cfg)
    # CLI覆盖后的步数是本次真实配置，也用于检查点元数据。
    config["sft"] = cfg
    records = load_training_records(project_path(args.data), args.limit, config)
    output = project_path(args.output)
    prepare_output(output)
    seed_everything(int(config.get("seed", 42)))
    summary = {"stage": "sft", "status": "running", "config": config,
               "actual_sft": cfg, "data_path": str(project_path(args.data)),
               "data_sha256": hashlib.sha256(project_path(args.data).read_bytes()).hexdigest(),
               "training_rows": len(records), "output": str(output)}
    write_json(output / "run_summary.json", summary)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("该配置需要 CUDA；先运行环境探针")
        model, tokenizer = load_model_and_tokenizer(config)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        model.enable_input_require_grads()
        dataset = CompletionDataset(records, tokenizer, int(cfg["max_length"]), config)
        write_json(output / "token_mask_audit.json", dataset.audit)
        before = snapshot_trainable(model)
        bf16 = torch.cuda.is_bf16_supported()
        train_args = TrainingArguments(
            output_dir=str(output), max_steps=int(cfg["max_steps"]),
            learning_rate=float(cfg["learning_rate"]),
            per_device_train_batch_size=int(cfg["batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=bf16, fp16=not bf16, optim="adamw_torch", lr_scheduler_type="constant",
            warmup_steps=0, weight_decay=0.0, max_grad_norm=1.0,
            logging_steps=1, logging_nan_inf_filter=False,
            **saving, eval_strategy="no", report_to="none",
            dataloader_num_workers=0, dataloader_pin_memory=False,
            remove_unused_columns=False, label_names=["labels"],
            seed=int(config.get("seed", 42)), data_seed=int(config.get("seed", 42)),
            push_to_hub=False, disable_tqdm=True,
        )
        trainer = Trainer(model=model, args=train_args, train_dataset=dataset,
                          data_collator=CompletionCollator(tokenizer.pad_token_id),
                          processing_class=tokenizer, callbacks=[
                              FiniteLogCallback(),
                              CheckpointMetadataCallback("sft", config, before, project_path(args.data)),
                          ])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        runtime = time.perf_counter() - started
        update = measure_update(model, before)
        if not update["finite_parameters"]:
            raise FloatingPointError("SFT adapter 参数出现非有限数")
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        trainer.save_state()
        log_dir = getattr(args, "log_dir", None)
        log_path = project_path(log_dir) / "train_log.json" if log_dir else output / "train_log.json"
        write_json(log_path, trainer.state.log_history)
        summary.update(status="completed" if update["nonzero_update"] else "completed_no_update",
                       training_seconds=runtime,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                       bf16=bf16, optimizer_steps=trainer.state.global_step,
                       metrics=result.metrics, update=update,
                       loss_protocol="completion-only labels; causal shift handled by model")
        write_json(output / "run_summary.json", summary)
        return summary
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run_summary.json", summary)
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
