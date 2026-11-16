"""生成 PPO 样本、用所选数据集判分并记录回答。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import time

import torch
from torch import Tensor

from step1_LLM.evaluate import generated_tokens_through_eos


@dataclass
class Rollout:
    ids: Tensor
    attention: Tensor
    valid: Tensor
    rewards: list[float]
    lengths: list[int]
    terminated: list[bool]
    errors: Counter[str]
    truncated_count: int
    groups_equal: int
    generation_seconds: float
    reward_and_log_seconds: float


def collect_rollout(
    model, tokenizer, records: list[dict], prompts: list[str], indices: list[int],
    generation, generation_kwargs: dict, cfg, step: int, verify, stream, device,
) -> Rollout:
    """为同一批问题采样回答，返回 PPO 更新所需的有效 token 与奖励。"""
    inputs = tokenizer([prompts[i] for i in indices], padding=True, return_tensors="pt",
                       add_special_tokens=False).to(device)
    prompt_width = inputs.input_ids.shape[1]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(**inputs, generation_config=generation, **generation_kwargs)
    torch.cuda.synchronize(device)
    seconds_generate = time.perf_counter() - started
    started = time.perf_counter()
    completion_ids = generated[:, prompt_width:].tolist()
    rewards, lengths, terminated = [], [], []
    errors = Counter()
    truncated_count = 0
    for position, (tokens, idx) in enumerate(zip(completion_ids, indices)):
        tokens, ended = generated_tokens_through_eos(tokens, tokenizer.eos_token_id)
        truncated = not ended and len(tokens) >= cfg.max_completion_length
        completion = tokenizer.decode(tokens, skip_special_tokens=True)
        row = records[idx]
        checked = verify(completion, row)
        reward = float(checked["reward"])
        if reward not in (0.0, 1.0):
            raise ValueError("执行奖励必须为 0 或 1")
        stream.write(json.dumps({
            "step": step, "call": step - 1,
            "group": (step - 1) * cfg.prompts_per_rollout + position // cfg.num_generations,
            "sample_in_group": position % cfg.num_generations,
            "id": row["id"], "prompt": prompts[idx], "completion": completion,
            "completion_tokens": len(tokens), "ended_with_eos": ended,
            "truncated_at_limit": truncated, "verification": checked,
        }, ensure_ascii=False) + "\n")
        rewards.append(reward)
        lengths.append(len(tokens))
        terminated.append(ended)
        errors[checked["error"]] += 1
        truncated_count += int(truncated)
    stream.flush()
    if not lengths or min(lengths) < 1 or len(lengths) != len(indices):
        raise ValueError("生成结果缺失，无法计算 PPO rollout")
    width = len(completion_ids[0])
    valid = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    attention = torch.cat([inputs.attention_mask, valid.long()], dim=1)

    groups_equal = sum(len(set(rewards[i:i + cfg.num_generations])) == 1
                       for i in range(0, len(rewards), cfg.num_generations))
    return Rollout(
        ids=generated, attention=attention, valid=valid, rewards=rewards,
        lengths=lengths, terminated=terminated, errors=errors,
        truncated_count=truncated_count, groups_equal=groups_equal,
        generation_seconds=seconds_generate,
        reward_and_log_seconds=time.perf_counter() - started,
    )
