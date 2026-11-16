"""从 SFT adapter 继续 GRPO，由 TRL 执行采样和策略更新。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
from datasets import Dataset
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

from step1_LLM.prepare import (
    dataset_task, load_model_and_tokenizer, render_prompt, seed_everything, write_json,
)
from step2_SFT.train import (
    CheckpointMetadataCallback, FiniteLogCallback, checkpoint_settings,
    load_training_records, measure_update, prepare_output, snapshot_trainable, state_fingerprint,
)
from .callbacks import RewardRecorder, check_sampling_protocol


def train_grpo(config: dict, *, data_path: Path, output: Path, adapter: Path,
               max_steps: int | None = None, limit: int | None = None,
               log_dir: Path | None = None) -> dict:
    cfg = dict(config["grpo"])
    if max_steps is not None:
        cfg["max_steps"] = max_steps
    if int(cfg["max_steps"]) <= 0:
        raise ValueError("max_steps 必须为正整数")
    saving = checkpoint_settings(cfg)
    config["grpo"] = cfg
    if float(cfg.get("beta", 0.0)) != 0.0:
        raise ValueError("本项目明确采用beta=0；更改参考模型方案需单独设计")
    group_size = int(cfg["num_generations"])
    effective_batch = int(cfg["batch_size"]) * int(cfg["gradient_accumulation_steps"])
    if group_size < 2 or effective_batch % group_size:
        raise ValueError("单卡 batch_size * gradient_accumulation_steps 必须被G整除，且G>=2")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"缺少SFT adapter：{adapter}")
    records = load_training_records(data_path, limit, config)
    prepare_output(output)
    logs = log_dir or output
    logs.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config.get("seed", 42)))
    summary = {"stage": "grpo", "status": "running", "config": config,
               "actual_grpo": cfg, "sft_adapter": str(adapter),
               "data_path": str(data_path),
               "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
               "training_rows": len(records), "output": str(output),
               "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                            "repetition_penalty": 1.0, "num_generations": group_size},
               "loss": {"loss_type": cfg["loss_type"], "scale_rewards": cfg["scale_rewards"],
                        "beta": 0.0, "num_iterations": 1, "epsilon": 0.2}}
    recorder = RewardRecorder(logs / "rollouts.jsonl", group_size, dataset_task(config).verify)
    write_json(output / "run_summary.json", summary)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("该配置需要 CUDA；先运行环境探针")
        model, tokenizer = load_model_and_tokenizer(config, adapter=adapter, trainable=True)
        recorder.eos_token_id = tokenizer.eos_token_id
        recorder.max_completion_length = int(cfg["max_completion_length"])
        if not isinstance(model, PeftModel):
            raise TypeError("GRPO必须加载训练过的SFT PeftModel")
        tokenizer.padding_side = "left"
        model.enable_input_require_grads()
        # prompt 已经应用所选模型模板。字符串模式阻止 TRL 二次添加 chat template。
        prepared = [{"prompt": render_prompt(tokenizer, row, config), "record": row} for row in records]
        max_prompt_length = int(cfg.get("max_prompt_length", 768))
        lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in prepared]
        if max(lengths) > max_prompt_length:
            raise ValueError(f"最长prompt={max(lengths)}超过max_prompt_length={max_prompt_length}；拒绝静默截断")
        before = snapshot_trainable(model)
        if not before:
            raise ValueError("SFT adapter 没有可训练参数，请检查 is_trainable=True")
        source_summary = adapter / "run_summary.json"
        if source_summary.exists():
            source = json.loads(source_summary.read_text(encoding="utf-8"))
            source_hash = source.get("update", {}).get("after_sha256")
            summary["initial_state_matches_sft"] = (
                state_fingerprint(before) == source_hash if source_hash else None)
            summary["sft_source"] = {"path": str(adapter), "stage": source.get("stage"),
                                     "optimizer_steps": source.get("optimizer_steps"),
                                     "expected_adapter_sha256": source_hash}
            if summary["initial_state_matches_sft"] is False:
                raise ValueError("实际加载adapter指纹与指定SFT/checkpoint记录不一致")
        bf16 = torch.cuda.is_bf16_supported()
        train_args = GRPOConfig(
            output_dir=str(output), max_steps=int(cfg["max_steps"]),
            learning_rate=float(cfg["learning_rate"]),
            per_device_train_batch_size=int(cfg["batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            num_generations=group_size, max_completion_length=int(cfg["max_completion_length"]),
            max_prompt_length=max_prompt_length, beta=0.0, num_iterations=1, epsilon=0.2,
            loss_type=cfg["loss_type"], scale_rewards=cfg["scale_rewards"],
            temperature=1.0, top_p=1.0, top_k=0, min_p=None, repetition_penalty=1.0,
            use_vllm=False, use_transformers_paged=False, disable_dropout=True,
            mask_truncated_completions=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=bf16, fp16=not bf16, optim="adamw_torch", lr_scheduler_type="constant",
            warmup_steps=0, weight_decay=0.0, max_grad_norm=1.0,
            logging_steps=1, logging_nan_inf_filter=False,
            **saving, eval_strategy="no", report_to="none",
            dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False,
            seed=int(config.get("seed", 42)), data_seed=int(config.get("seed", 42)),
            push_to_hub=False, disable_tqdm=True,
        )
        trainer = GRPOTrainer(model=model, args=train_args, reward_funcs=recorder,
                              train_dataset=Dataset.from_list(prepared),
                              processing_class=tokenizer, callbacks=[
                                  FiniteLogCallback(),
                                  CheckpointMetadataCallback("grpo", config, before, data_path),
                              ])
        # 不传 peft_config：保留已有 SFT adapter，避免重新创建随机 LoRA。
        summary["effective_sampling"] = check_sampling_protocol(model, trainer.generation_config)
        write_json(output / "run_summary.json", summary)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        runtime = time.perf_counter() - started
        update = measure_update(model, before)
        if not update["finite_parameters"]:
            raise FloatingPointError("GRPO adapter 参数出现非有限数")
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        trainer.save_state()
        write_json(logs / ("train_log.json" if log_dir else "training_log.json"), trainer.state.log_history)
        summary.update(status="completed" if update["nonzero_update"] else "completed_no_update",
                       training_seconds=runtime,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                       bf16=bf16, optimizer_steps=trainer.state.global_step,
                       metrics=result.metrics, update=update, reward=recorder.summary(),
                       diagnostic="参数已更新，仍需独立评测" if update["nonzero_update"] else
                                  "没有有效参数更新；检查零优势比例、任务难度、采样和SFT起点")
        write_json(output / "run_summary.json", summary)
        return summary
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}", reward=recorder.summary())
        write_json(output / "run_summary.json", summary)
        (logs / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
