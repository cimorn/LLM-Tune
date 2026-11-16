"""GRPO 的生成协议检查、奖励记录和组内统计。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time


def check_sampling_protocol(model, generation_config) -> dict:
    """检查 Transformers 合并模型默认值后的实际分布，防止温度/top-p被覆盖。"""
    effective, _ = model._prepare_generation_config(generation_config)
    expected = {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                "min_p": None, "repetition_penalty": 1.0, "do_sample": True,
                "num_beams": 1, "no_repeat_ngram_size": 0,
                "bad_words_ids": None, "suppress_tokens": None,
                "begin_suppress_tokens": None, "forced_bos_token_id": None,
                "forced_eos_token_id": None}
    actual = {key: getattr(effective, key) for key in expected}
    mismatch = {key: actual[key] for key in expected if actual[key] != expected[key]}
    if mismatch:
        raise ValueError(f"实际生成分布与GRPO logprob假设不一致：{mismatch}")
    return {**actual, "max_new_tokens": effective.max_new_tokens,
            "eos_token_id": effective.eos_token_id,
            "probe": "Transformers._prepare_generation_config after model-default merge"}


class RewardRecorder:
    """把真实采样、判分和组统计写入 JSONL；日志本身不改变奖励。"""
    __name__ = "strict_tool_reward"

    def __init__(self, path: Path, group_size: int, verify, eos_token_id=None, max_completion_length=None):
        self.path = path
        self.verify = verify
        self.group_size = group_size
        self.calls = 0
        self.samples = 0
        self.groups = 0
        self.zero_variance_groups = 0
        self.reward_sum = 0.0
        self.errors = Counter()
        self.eos_token_id = eos_token_id
        self.max_completion_length = max_completion_length
        self.token_observed_samples = 0
        self.sampled_completion_tokens = 0
        self.terminated_samples = 0
        self.truncated_at_limit = 0
        self.reward_and_log_seconds = 0.0

    def __call__(self, prompts, completions, record, trainer_state=None,
                 completion_ids=None, **kwargs):
        started = time.perf_counter()
        if not (len(prompts) == len(completions) == len(record)):
            raise ValueError("reward 输入长度不一致")
        if len(record) % self.group_size:
            raise ValueError("采样数量没有构成完整G组")
        if completion_ids is not None and len(completion_ids) != len(record):
            raise ValueError("completion_ids数量与样本不一致")
        rewards = []
        with self.path.open("a", encoding="utf-8") as stream:
            for index, (prompt, completion, row) in enumerate(zip(prompts, completions, record)):
                if not isinstance(completion, str):
                    raise TypeError("本项目使用预渲染字符串prompt，completion必须为字符串")
                checked = self.verify(completion, row)
                reward = float(checked["reward"])
                if reward not in (0.0, 1.0):
                    raise ValueError("执行器必须返回严格的0/1奖励")
                rewards.append(reward)
                self.reward_sum += reward
                self.errors[checked.get("error", "")] += 1
                tokens = completion_ids[index] if completion_ids is not None else None
                terminated = bool(tokens and tokens[-1] == self.eos_token_id) if self.eos_token_id is not None and tokens is not None else None
                truncated = bool(not terminated and len(tokens) >= self.max_completion_length) if terminated is not None and self.max_completion_length else None
                if tokens is not None:
                    self.token_observed_samples += 1
                    self.sampled_completion_tokens += len(tokens)
                    self.terminated_samples += int(terminated is True)
                    self.truncated_at_limit += int(truncated is True)
                stream.write(json.dumps({
                    "call": self.calls, "step": getattr(trainer_state, "global_step", None),
                    "group": self.groups + index // self.group_size,
                    "sample_in_group": index % self.group_size,
                    "id": row["id"], "prompt": prompt, "completion": completion,
                    "completion_tokens": len(tokens) if tokens is not None else None,
                    "ended_with_eos": terminated, "truncated_at_limit": truncated,
                    "verification": checked,
                }, ensure_ascii=False) + "\n")
        for start in range(0, len(record), self.group_size):
            group = record[start:start+self.group_size]
            if len({row["id"] for row in group}) != 1:
                raise ValueError("GRPO组内混入不同问题，拒绝继续更新")
            values = rewards[start:start+self.group_size]
            self.zero_variance_groups += int(len(set(values)) == 1)
        self.calls += 1
        self.samples += len(record)
        self.groups += len(record) // self.group_size
        self.reward_and_log_seconds += time.perf_counter() - started
        return rewards

    def summary(self):
        return {"generation_calls": self.calls, "sampled_completions": self.samples,
                "groups": self.groups, "zero_variance_groups": self.zero_variance_groups,
                "zero_variance_fraction": self.zero_variance_groups/self.groups if self.groups else None,
                "mean_sample_reward": self.reward_sum/self.samples if self.samples else None,
                "error_counts": dict(self.errors),
                "sampled_completion_tokens": self.sampled_completion_tokens if self.token_observed_samples else None,
                "token_observed_samples": self.token_observed_samples,
                "token_count_scope": "actual sampled completion IDs, including EOS; excludes post-EOS padding",
                "terminated_samples": self.terminated_samples if self.eos_token_id is not None else None,
                "truncated_at_limit": self.truncated_at_limit if self.eos_token_id is not None and self.max_completion_length else None,
                "truncated_fraction": self.truncated_at_limit / self.token_observed_samples if self.token_observed_samples and self.eos_token_id is not None and self.max_completion_length else None,
                "reward_and_log_seconds": self.reward_and_log_seconds}
