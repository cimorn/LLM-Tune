"""GRPO for the three single-token actions in human preference prediction."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from peft import PeftModel

from Datasets.human_preference.task import LABELS
from step1_LLM.evaluate import adapter_hashes, sha256_file
from step1_LLM.judge_eval import PreferenceEncoder
from step1_LLM.prepare import load_model_and_tokenizer, seed_everything, write_json
from step2_SFT.train import (
    load_training_records,
    measure_update,
    prepare_output,
    snapshot_trainable,
    state_fingerprint,
)


def group_advantage(rewards: torch.Tensor) -> torch.Tensor:
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / (rewards.std(dim=1, unbiased=False, keepdim=True) + 1e-4)


def clipped_loss(log_probs, old_log_probs, advantages, epsilon):
    ratio = (log_probs - old_log_probs).exp()
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    return -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()


def _policy(model, inputs, mask, positions, label_ids):
    # The final token of each left-padded sequence is the prediction position.
    result = model(
        input_ids=inputs, attention_mask=mask, position_ids=positions,
        use_cache=False, logits_to_keep=1,
    )
    logits = result.logits[:, -1].index_select(1, label_ids).float()
    return logits.log_softmax(dim=-1)


def _batch(encoder, rows, device):
    input_ids = [encoder.encode_prompt(row)[0] for row in rows]
    width = max(map(len, input_ids))
    padding = [width - len(ids) for ids in input_ids]
    inputs = torch.tensor(
        [[encoder.tokenizer.pad_token_id] * pad + ids for pad, ids in zip(padding, input_ids)],
        device=device,
    )
    mask = torch.tensor(
        [[0] * pad + [1] * len(ids) for pad, ids in zip(padding, input_ids)], device=device,
    )
    positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
    targets = torch.tensor([LABELS.index(row["target"]) for row in rows], device=device)
    return inputs, mask, positions, targets


def _prompt_batches(records, size, seed):
    rng = random.Random(seed)
    indices = list(range(len(records)))
    while True:
        rng.shuffle(indices)
        for start in range(0, len(indices), size):
            yield [records[i] for i in indices[start:start + size]]


def train_preference_grpo(
    config: dict, *, data_path: Path, output: Path, adapter: Path, log_dir: Path | None = None,
) -> dict:
    cfg = config["preference_grpo"]
    max_steps, batch_size, group_size, update_epochs = (
        cfg[key] for key in ("max_steps", "batch_size", "num_generations", "update_epochs")
    )
    if (any(type(value) is not int or value <= 0 for value in (max_steps, batch_size, group_size, update_epochs))
        or group_size < 2):
        raise ValueError("偏好 GRPO 的步数和批量须为正整数，每题至少采样两次")
    epsilon = float(cfg["epsilon"])
    lr = float(cfg["learning_rate"])
    if not 0 < epsilon < 1 or not math.isfinite(lr) or lr <= 0 or cfg["beta"] != 0:
        raise ValueError("偏好 GRPO 要求 0 < epsilon < 1、正的有限学习率和 beta=0")
    if not torch.cuda.is_available():
        raise RuntimeError("偏好 GRPO 需要 CUDA")
    records = load_training_records(data_path, config=config)
    source_hashes = adapter_hashes(adapter)
    if "adapter_model.safetensors" not in source_hashes:
        raise FileNotFoundError(f"缺少 SFT adapter 权重：{adapter}")
    if source_hashes != config.get("sft_adapter_files_sha256", source_hashes):
        raise ValueError("SFT adapter 文件与实验配置不一致")
    data_hash = sha256_file(data_path)
    if data_hash != config.get("data_sha256", {}).get("train", data_hash):
        raise ValueError("偏好 GRPO 训练集与实验配置不一致")

    prepare_output(output)
    summary_path = output / "run_summary.json"
    summary = {
        "stage": "preference_grpo",
        "status": "running",
        "config": config,
        "data_path": str(data_path),
        "data_sha256": data_hash,
        "training_rows": len(records),
        "sft_adapter": str(adapter),
        "sft_adapter_files_sha256": source_hashes,
        "action_space": list(LABELS),
        "reward_protocol": "1 for matching the human label, otherwise 0",
        "advantage_protocol": "within-prompt centering and population standard deviation (epsilon=1e-4)",
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    try:
        seed = int(config.get("seed", 42))
        seed_everything(seed)
        model, tokenizer = load_model_and_tokenizer(config, adapter=adapter, trainable=True)
        if not isinstance(model, PeftModel):
            raise TypeError("偏好 GRPO 必须加载 SFT LoRA 权重")
        # Deterministic forward passes keep rollout and update probabilities comparable.
        # Evaluation mode disables dropout while autograd still updates the LoRA weights.
        model.eval()
        encoder = PreferenceEncoder(tokenizer, config)
        before = snapshot_trainable(model)
        if not before:
            raise ValueError("SFT adapter 没有可训练参数")
        sft_summary = adapter / "run_summary.json"
        if sft_summary.is_file():
            source_update = json.loads(sft_summary.read_text(encoding="utf-8")).get("update", {})
            expected = source_update.get("after_sha256")
            if expected is not None and state_fingerprint(before) != expected:
                raise ValueError("实际加载的 LoRA 参数与原 SFT 实验记录不符")
            summary["initial_state_matches_sft"] = True if expected is not None else None

        device = next(model.parameters()).device
        label_ids = torch.tensor(encoder.label_ids, device=device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0)
        batches = _prompt_batches(records, batch_size, seed)
        history = []
        prompt_count = correct_count = zero_group_count = 0
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        while len(history) < max_steps:
            rows = next(batches)
            inputs, mask, positions, targets = _batch(encoder, rows, device)
            with torch.no_grad():
                old_log_probs = _policy(model, inputs, mask, positions, label_ids)
                actions = torch.multinomial(old_log_probs.exp(), group_size, replacement=True)
                old_action_log_probs = old_log_probs.gather(1, actions)
                rewards = (actions == targets.unsqueeze(1)).float()
                advantages = group_advantage(rewards)
                batch_correct = int(rewards.sum().item())
                batch_zero = int((advantages == 0).all(dim=1).sum().item())
            prompt_count += len(rows)
            correct_count += batch_correct
            zero_group_count += batch_zero

            for _ in range(min(update_epochs, max_steps - len(history))):
                optimizer.zero_grad(set_to_none=True)
                new_log_probs = _policy(model, inputs, mask, positions, label_ids)
                action_log_probs = new_log_probs.gather(1, actions)
                loss = clipped_loss(action_log_probs, old_action_log_probs, advantages, epsilon)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("偏好 GRPO 产生非有限损失")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                step = len(history) + 1
                history.append({
                    "step": step,
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "reward": batch_correct / (len(rows) * group_size),
                    "zero_advantage_fraction": batch_zero / len(rows),
                })
                if step == 1 or step % 16 == 0 or step == max_steps:
                    print(f"[Preference GRPO] {step}/{max_steps}, "
                          f"reward={history[-1]['reward']:.3f}, "
                          f"zero_advantage_fraction={history[-1]['zero_advantage_fraction']:.3f}", flush=True)

        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        update = measure_update(model, before)
        if not update["finite_parameters"]:
            raise FloatingPointError("偏好 GRPO adapter 出现非有限参数")
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        adapter_config = output / "adapter_config.json"
        settings = json.loads(adapter_config.read_text(encoding="utf-8"))
        settings.update(base_model_name_or_path=config["model"]["name"], revision=config["model"]["revision"])
        write_json(adapter_config, settings)
        write_json((log_dir or output) / "train_log.json", history)
        summary.update(
            status="completed" if update["nonzero_update"] else "completed_no_update",
            optimizer_steps=len(history),
            training_seconds=seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
            update=update,
            reward={
                "sampled_prompts": prompt_count,
                "sampled_actions": prompt_count * group_size,
                "mean": correct_count / (prompt_count * group_size),
                "zero_advantage_group_fraction": zero_group_count / prompt_count,
            },
        )
        write_json(summary_path, summary)
        return summary
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(summary_path, summary)
        raise
