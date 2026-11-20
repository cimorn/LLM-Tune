"""PPO-Clip for the one-step A/B/C human preference judging task."""

from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path

import torch
from peft import PeftModel
from torch import nn

from Datasets.human_preference.task import LABELS
from step1_LLM.evaluate import adapter_hashes, sha256_file
from step1_LLM.judge_eval import PreferenceEncoder
from step1_LLM.prepare import load_model_and_tokenizer, seed_everything, write_json
from step2_SFT.train import (
    load_training_records, measure_update, prepare_output,
    snapshot_trainable, state_fingerprint,
)
from step3_Tune.RL.GRPO.run.grpo_judge import _batch, _prompt_batches
from step3_Tune.RL.PPO.run.update import clipped_policy_loss, text_hidden_size
from step3_Tune.RL.PPO.train import save_policy


def _settings(config: dict) -> dict:
    cfg = config["preference_ppo"]
    for name in ("max_steps", "batch_size", "num_generations", "update_epochs"):
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f"preference_ppo.{name} 必须是正整数")
    if cfg["num_generations"] < 2:
        raise ValueError("偏好 PPO 每题至少采样两次")
    for name in ("learning_rate", "value_learning_rate", "value_cliprange", "value_coef", "max_grad_norm"):
        if not math.isfinite(float(cfg[name])) or float(cfg[name]) <= 0:
            raise ValueError(f"preference_ppo.{name} 必须是有限正数")
    if not 0 < float(cfg["epsilon"]) < 1 or cfg["beta"] != 0:
        raise ValueError("偏好 PPO 要求 0 < epsilon < 1 且 beta=0；当前未加入参考策略 KL 奖励")
    return cfg


def _score(model, value_head, inputs, mask, positions, label_ids):
    base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    parameters = inspect.signature(base.forward).parameters
    keep = {}
    if "logits_to_keep" in parameters:
        keep["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        keep["num_logits_to_keep"] = 1
    result = model(
        input_ids=inputs, attention_mask=mask, position_ids=positions,
        output_hidden_states=True, use_cache=False, **keep,
    )
    logits = result.logits[:, -1].index_select(1, label_ids).float()
    # The critic trains on fixed features within each forward pass. Its loss does
    # not update the policy backbone; only the PPO actor loss changes the LoRA.
    features = result.hidden_states[-1][:, -1].detach().float()
    return logits.log_softmax(dim=-1), value_head(features)


def _losses(policy, values, actions, old_action_log_probs, old_values, rewards, advantages, cfg):
    log_ratio = policy.gather(1, actions) - old_action_log_probs
    ratio = log_ratio.exp()
    actor_loss = clipped_policy_loss(ratio, advantages, float(cfg["epsilon"])).mean()
    clipped_values = old_values + (values - old_values).clamp(
        -float(cfg["value_cliprange"]), float(cfg["value_cliprange"]),
    )
    value_loss = 0.5 * torch.maximum((values - rewards).square(), (clipped_values - rewards).square()).mean()
    return actor_loss, value_loss, (ratio - 1 - log_ratio).mean()


def train_preference_ppo(
    config: dict, *, data_path: Path, output: Path, adapter: Path, log_dir: Path | None = None,
) -> dict:
    cfg = _settings(config)
    if not torch.cuda.is_available():
        raise RuntimeError("偏好 PPO 需要 CUDA")
    records = load_training_records(data_path, config=config)
    source_hashes = adapter_hashes(adapter)
    if "adapter_model.safetensors" not in source_hashes:
        raise FileNotFoundError(f"缺少 SFT adapter 权重：{adapter}")
    if source_hashes != config.get("sft_adapter_files_sha256", source_hashes):
        raise ValueError("SFT adapter 文件与实验配置不一致")
    data_hash = sha256_file(data_path)
    if data_hash != config.get("data_sha256", {}).get("train", data_hash):
        raise ValueError("偏好 PPO 训练集与实验配置不一致")

    prepare_output(output)
    summary_path = output / "run_summary.json"
    summary = {
        "stage": "preference_ppo", "status": "running", "config": config,
        "data_path": str(data_path), "data_sha256": data_hash,
        "training_rows": len(records), "sft_adapter": str(adapter),
        "sft_adapter_files_sha256": source_hashes,
        "action_space": list(LABELS),
        "policy_protocol": "softmax restricted to the A/B/C label tokens",
        "reward_protocol": "1 for matching the human label, including tie C, otherwise 0",
        "advantage_protocol": "one-step terminal return minus rollout value; normalize across sampled batch with population std floor 1e-6",
        "value_protocol": "zero-initialized linear critic on detached final prompt hidden state; clipped value loss",
        "reference_kl_penalty": False,
        "step_unit": "one optimizer update; update_epochs reuses fixed rollout actions, log probabilities and values",
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    try:
        seed = int(config.get("seed", 42))
        seed_everything(seed)
        model, tokenizer = load_model_and_tokenizer(config, adapter=adapter, trainable=True)
        if not isinstance(model, PeftModel):
            raise TypeError("偏好 PPO 必须加载 SFT LoRA 权重")
        model.eval()
        encoder = PreferenceEncoder(tokenizer, config)
        before = snapshot_trainable(model)
        if not before:
            raise ValueError("SFT adapter 没有可训练参数")
        source_summary = adapter / "run_summary.json"
        if source_summary.is_file():
            source_update = json.loads(source_summary.read_text(encoding="utf-8")).get("update", {})
            expected = source_update.get("after_sha256")
            if expected is not None and state_fingerprint(before) != expected:
                raise ValueError("实际加载的 LoRA 参数与原 SFT 实验记录不符")
            summary["initial_state_matches_sft"] = True if expected is not None else None

        device = next(model.parameters()).device
        label_ids = torch.tensor(encoder.label_ids, device=device)
        # Critic initialization must not consume the actor's sampling RNG stream.
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            value_head = nn.Linear(text_hidden_size(model), 1, device=device, dtype=torch.float32)
            nn.init.zeros_(value_head.weight)
            nn.init.zeros_(value_head.bias)
        before_value = snapshot_trainable(value_head)
        actor_parameters = [p for p in model.parameters() if p.requires_grad]
        parameters = actor_parameters + list(value_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": actor_parameters, "lr": float(cfg["learning_rate"])},
            {"params": list(value_head.parameters()), "lr": float(cfg["value_learning_rate"])},
        ], weight_decay=0)
        batches = _prompt_batches(records, cfg["batch_size"], seed)
        history = []
        prompt_count = correct_count = rollout_count = 0
        group_size = cfg["num_generations"]
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        while len(history) < cfg["max_steps"]:
            rows = next(batches)
            inputs, mask, positions, targets = _batch(encoder, rows, device)
            with torch.no_grad():
                old_policy, old_values = _score(model, value_head, inputs, mask, positions, label_ids)
                actions = torch.multinomial(old_policy.exp(), group_size, replacement=True)
                old_action_log_probs = old_policy.gather(1, actions)
                rewards = (actions == targets[:, None]).float()
                advantages = rewards - old_values
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
                batch_correct = int(rewards.sum().item())
            prompt_count += len(rows)
            correct_count += batch_correct
            rollout_count += 1
            for _ in range(min(cfg["update_epochs"], cfg["max_steps"] - len(history))):
                optimizer.zero_grad(set_to_none=True)
                policy, values = _score(model, value_head, inputs, mask, positions, label_ids)
                actor_loss, value_loss, approximate_kl = _losses(
                    policy, values, actions, old_action_log_probs, old_values, rewards, advantages, cfg,
                )
                loss = actor_loss + float(cfg["value_coef"]) * value_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("偏好 PPO 产生非有限损失")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(cfg["max_grad_norm"]), error_if_nonfinite=True)
                optimizer.step()
                step = len(history) + 1
                history.append({
                    "step": step, "rollout": rollout_count,
                    "loss": loss.item(), "actor_loss": actor_loss.item(), "value_loss": value_loss.item(),
                    "grad_norm": grad_norm.item(), "reward": batch_correct / (len(rows) * group_size),
                    "approx_kl_to_rollout_policy": approximate_kl.item(),
                })
                if step == 1 or step % 16 == 0 or step == cfg["max_steps"]:
                    print(f"[Preference PPO] {step}/{cfg['max_steps']}, "
                          f"reward={history[-1]['reward']:.3f}, actor={actor_loss.item():.4f}, "
                          f"critic={value_loss.item():.4f}", flush=True)

        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        update = measure_update(model, before)
        value_update = measure_update(value_head, before_value)
        if not update["finite_parameters"] or not value_update["finite_parameters"]:
            raise FloatingPointError("偏好 PPO actor 或 critic 出现非有限参数")
        save_policy(output, model, tokenizer, value_head)
        settings = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
        settings.update(base_model_name_or_path=config["model"]["name"], revision=config["model"]["revision"])
        write_json(output / "adapter_config.json", settings)
        write_json((log_dir or output) / "train_log.json", history)
        summary.update(
            status="completed" if update["nonzero_update"] else "completed_no_update",
            optimizer_steps=len(history), rollout_count=rollout_count, training_seconds=seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
            update=update, value_update=value_update,
            reward={"sampled_prompts": prompt_count, "sampled_actions": prompt_count * group_size,
                    "mean": correct_count / (prompt_count * group_size)},
        )
        write_json(summary_path, summary)
        return summary
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(summary_path, summary)
        raise
