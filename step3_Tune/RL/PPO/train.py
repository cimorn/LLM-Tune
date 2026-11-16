"""在单卡上用程序执行奖励训练语言模型的 PPO-Clip。

策略从 SFT 的 LoRA 权重出发，价值头读取相同的前缀隐藏状态。每批生成后
固定旧策略的 token log probability 和价值预测，再做 GAE 与多轮裁剪更新。
公式见 https://arxiv.org/abs/1707.06347 及 https://arxiv.org/abs/1506.02438。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
import traceback

import torch
from peft import PeftModel
from safetensors.torch import save_file
from torch import nn
from transformers import GenerationConfig

from step1_LLM.prepare import (
    dataset_task, load_model_and_tokenizer, read_config, render_prompt, seed_everything, write_json,
)
from step1_LLM.evaluate import generation_call_kwargs, sha256_file
from step2_SFT.train import (
    load_training_records, measure_update, prepare_output,
    project_path, snapshot_trainable, state_fingerprint,
)
from .run.rollout import collect_rollout
from .run.update import text_hidden_size, train_rollout


@dataclass
class PPOSettings:
    max_steps: int = 24
    learning_rate: float = 3e-6
    value_learning_rate: float = 1e-3
    batch_size: int = 2
    prompts_per_rollout: int = 2
    num_generations: int = 4
    max_prompt_length: int = 768
    max_completion_length: int = 128
    ppo_epochs: int = 2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    cliprange: float = 0.2
    value_cliprange: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    save_at_steps: list[int] = field(default_factory=list)

    def validate(self) -> None:
        positive_ints = (
            "max_steps", "batch_size", "prompts_per_rollout", "num_generations",
            "max_prompt_length", "max_completion_length", "ppo_epochs",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"ppo.{name} 必须是正整数")
        for name in ("learning_rate", "value_learning_rate", "value_coef", "max_grad_norm", "value_cliprange"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"ppo.{name} 必须是有限正数")
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and 0 < self.cliprange < 1):
            raise ValueError("gamma/gae_lambda 必须在 [0,1]，cliprange 必须在 (0,1)")
        if not isinstance(self.save_at_steps, list) or any(
            type(step) is not int or not 1 <= step <= self.max_steps for step in self.save_at_steps
        ):
            raise ValueError("ppo.save_at_steps 必须是本次 rollout 范围内的整数列表")


def save_policy(path: Path, model: PeftModel, tokenizer, value_head: nn.Linear) -> None:
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    save_file({name: value.detach().cpu() for name, value in value_head.state_dict().items()},
              path / "value_head.safetensors")


def run(args: argparse.Namespace) -> dict:
    config = read_config(args.config)
    cfg = PPOSettings(**config["ppo"])
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    cfg.validate()
    config["ppo"] = asdict(cfg)
    data_path, output, adapter = map(project_path, (args.data, args.output, args.sft_adapter))
    records = load_training_records(data_path, args.limit, config)
    if cfg.prompts_per_rollout > len(records):
        raise ValueError("每轮提示数量超过可用训练样本")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"缺少 SFT adapter: {adapter}")
    prepare_output(output)
    log_dir = getattr(args, "log_dir", None)
    logs = project_path(log_dir) if log_dir is not None else output
    logs.mkdir(parents=True, exist_ok=True)
    verify = dataset_task(config).verify
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    summary = {
        "stage": "ppo", "status": "running", "config": config, "actual_ppo": asdict(cfg),
        "data_path": str(data_path), "data_sha256": sha256_file(data_path),
        "training_rows": len(records), "sft_adapter": str(adapter), "output": str(output),
        "loss_protocol": "PPO-Clip with masked GAE and a learned value head; normalized over valid completion tokens",
        "value_protocol": "Linear head on detached actor hidden states; initial weights zero; clipped squared error",
        "reward_protocol": "Dataset verifier supplies a strict 0/1 terminal reward; EOS or length limit ends the episode",
        "reference_kl": False,
    }
    write_json(output / "run_summary.json", summary)
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("此 PPO 配置需要支持 BF16 的 CUDA GPU")
        model, tokenizer = load_model_and_tokenizer(config, adapter=adapter, trainable=True)
        if not isinstance(model, PeftModel):
            raise TypeError("PPO 必须加载可训练的 SFT LoRA adapter")
        model.eval()  # 关闭 dropout，仍保留策略梯度。
        tokenizer.padding_side = "left"
        device = next(model.parameters()).device
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("PPO 需要明确的 EOS 与 PAD token")
        prompts = [render_prompt(tokenizer, row, config) for row in records]
        for row, prompt in zip(records, prompts):
            length = len(tokenizer.encode(prompt, add_special_tokens=False))
            if not 1 <= length <= cfg.max_prompt_length:
                raise ValueError(f"{row['id']}: 提示长度 {length} 超出 1–{cfg.max_prompt_length}，不执行截断")
        before = snapshot_trainable(model)
        if not before:
            raise ValueError("SFT adapter 没有可训练参数")
        checkpoint_summary = adapter / "run_summary.json"
        if checkpoint_summary.is_file():
            old_summary = json.loads(checkpoint_summary.read_text(encoding="utf-8"))
            expected = old_summary.get("update", {}).get("after_sha256")
            summary["initial_state_matches_sft"] = state_fingerprint(before) == expected if expected else None
            summary["sft_source"] = {
                "path": str(adapter), "stage": old_summary.get("stage"),
                "optimizer_steps": old_summary.get("optimizer_steps"), "expected_adapter_sha256": expected,
            }
            if summary["initial_state_matches_sft"] is False:
                raise ValueError("加载的 adapter 指纹与 SFT 保存记录不一致")
        value_head = nn.Linear(text_hidden_size(model), 1, device=device, dtype=torch.float32)
        nn.init.zeros_(value_head.weight)
        nn.init.zeros_(value_head.bias)
        before_value = snapshot_trainable(value_head)
        actor_params = [param for param in model.parameters() if param.requires_grad]
        value_params = list(value_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": actor_params, "lr": cfg.learning_rate},
            {"params": value_params, "lr": cfg.value_learning_rate},
        ], weight_decay=0.0)
        generation = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        generation_kwargs = generation_call_kwargs(cfg.max_completion_length)
        summary["sampling"] = {**generation_kwargs, "num_generations": cfg.num_generations,
                               "prompts_per_rollout": cfg.prompts_per_rollout}
        data_generator = torch.Generator().manual_seed(seed)
        selected: list[int] = []
        errors: Counter[str] = Counter()
        sampled_tokens = ended_count = truncated_count = zero_groups = 0
        history = []
        timing = dict(generation_seconds=0.0, reward_and_log_seconds=0.0,
                      old_policy_seconds=0.0, optimization_seconds=0.0)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        training_started = time.perf_counter()
        with (logs / "rollouts.jsonl").open("w", encoding="utf-8") as stream:
            for step in range(1, cfg.max_steps + 1):
                if len(selected) < cfg.prompts_per_rollout:
                    selected = torch.randperm(len(records), generator=data_generator).tolist()
                indices = selected[:cfg.prompts_per_rollout]
                selected = selected[cfg.prompts_per_rollout:]
                indices = [i for i in indices for _ in range(cfg.num_generations)]
                rollout = collect_rollout(
                    model, tokenizer, records, prompts, indices, generation, generation_kwargs,
                    cfg, step, verify, stream, device,
                )
                sampled_tokens += sum(rollout.lengths)
                ended_count += sum(rollout.terminated)
                truncated_count += rollout.truncated_count
                zero_groups += rollout.groups_equal
                errors.update(rollout.errors)
                values, seconds_old, seconds_opt = train_rollout(
                    model, value_head, optimizer, actor_params + value_params, rollout.ids,
                    rollout.attention, rollout.valid,
                    torch.tensor(rollout.rewards, device=device, dtype=torch.float32), cfg,
                )
                step_timing = dict(generation_seconds=rollout.generation_seconds,
                                   reward_and_log_seconds=rollout.reward_and_log_seconds,
                                   old_policy_seconds=seconds_old, optimization_seconds=seconds_opt)
                for phase, elapsed in step_timing.items():
                    timing[phase] += elapsed
                event = {
                    "step": step, "optimizer_steps": step * cfg.ppo_epochs, "reward": sum(rollout.rewards) / len(rollout.rewards),
                    "frac_reward_zero_std": rollout.groups_equal / cfg.prompts_per_rollout,
                    "sampled_completion_tokens": sum(rollout.lengths), **values, **step_timing,
                }
                history.append(event)
                print(json.dumps(event, ensure_ascii=False), flush=True)
                if step in cfg.save_at_steps:
                    checkpoint = output / f"checkpoint-{step}"
                    save_policy(checkpoint, model, tokenizer, value_head)
                    write_json(checkpoint / "run_summary.json", {
                        "stage": "ppo", "status": "checkpoint", "rollout_steps": step,
                        "optimizer_steps": step * cfg.ppo_epochs, "update": measure_update(model, before),
                        "value_update": measure_update(value_head, before_value),
                    })
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - training_started
        update = measure_update(model, before)
        value_update = measure_update(value_head, before_value)
        if not update["finite_parameters"] or not value_update["finite_parameters"]:
            raise FloatingPointError("PPO 训练后参数存在非有限值")
        save_policy(output, model, tokenizer, value_head)
        write_json(logs / ("train_log.json" if log_dir is not None else "training_log.json"), history)
        samples = sum(errors.values())
        groups = cfg.max_steps * cfg.prompts_per_rollout
        summary.update(
            status="completed" if update["nonzero_update"] else "completed_no_update",
            rollout_steps=cfg.max_steps, optimizer_steps=cfg.max_steps * cfg.ppo_epochs,
            training_seconds=elapsed, peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30, bf16=True,
            phase_timing=timing, update=update, value_update=value_update,
            reward={
                "generation_calls": cfg.max_steps, "sampled_completions": samples,
                "sampled_completion_tokens": sampled_tokens, "token_observed_samples": samples,
                "groups": groups, "zero_variance_groups": zero_groups,
                "zero_variance_fraction": zero_groups / groups, "mean_sample_reward": errors["ok"] / samples,
                "error_counts": dict(errors), "terminated_samples": ended_count,
                "truncated_at_limit": truncated_count, "truncated_fraction": truncated_count / samples,
            },
        )
        write_json(output / "run_summary.json", summary)
        return summary
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run_summary.json", summary)
        (logs / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
