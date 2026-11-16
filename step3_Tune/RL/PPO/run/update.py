"""PPO 的 token 评分、GAE、策略与价值损失。"""
from __future__ import annotations

import inspect
import math
import time
from typing import TYPE_CHECKING

import torch
from peft import PeftModel
from torch import Tensor, nn

if TYPE_CHECKING:
    from ..train import PPOSettings


def masked_gae(
    values: Tensor, rewards: Tensor, valid: Tensor, gamma: float, gae_lambda: float
) -> tuple[Tensor, Tensor]:
    """只在有效 token 上回传末步奖励。EOS 和长度上限都结束本题。"""
    next_valid = torch.nn.functional.pad(valid[:, 1:], (0, 1), value=False)
    next_value = torch.nn.functional.pad(values[:, 1:], (0, 1)).masked_fill(~next_valid, 0)
    terminal = valid & ~next_valid
    delta = (terminal * rewards[:, None] + gamma * next_value - values).masked_fill(~valid, 0)
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(rewards)
    for token in range(values.shape[1] - 1, -1, -1):
        carry = (delta[:, token] + gamma * gae_lambda * carry).masked_fill(~valid[:, token], 0)
        advantages[:, token] = carry
    targets = (advantages + values).masked_fill(~valid, 0)
    return advantages, targets


def clipped_policy_loss(ratio: Tensor, advantage: Tensor, cliprange: float) -> Tensor:
    clipped = ratio.clamp(1 - cliprange, 1 + cliprange)
    return -torch.minimum(ratio * advantage, clipped * advantage)


def score_tokens(
    model: PeftModel, value_head: nn.Linear, ids: Tensor, attention: Tensor, completion_width: int
) -> tuple[Tensor, Tensor]:
    """取下一 token 的 log probability 和对应前缀的价值。

    使用模型的 LM head 前先切到回答相关位置，避免对整个提示构造词表 logits。
    价值头训练时截断隐藏状态梯度，策略的 LoRA 更新由 PPO 策略损失决定。
    """
    position = attention.cumsum(dim=-1) - 1
    position.masked_fill_(attention == 0, 1)
    # PEFT 的 forward 接收 **kwargs；检查实际基础模型，避免传入不支持的参数。
    base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    parameters = inspect.signature(base.forward).parameters
    keep = {}
    if "logits_to_keep" in parameters:
        keep["logits_to_keep"] = completion_width
    elif "num_logits_to_keep" in parameters:
        keep["num_logits_to_keep"] = completion_width
    result = model(
        input_ids=ids[:, :-1],
        attention_mask=attention[:, :-1],
        position_ids=position[:, :-1],
        **keep,
        output_hidden_states=True,
        use_cache=False,
    )
    logits = result.logits[:, -completion_width:].float()
    selected = logits.gather(-1, ids[:, -completion_width:].unsqueeze(-1)).squeeze(-1)
    log_prob = selected - logits.logsumexp(dim=-1)
    features = result.hidden_states[-1][:, -completion_width:].detach().float()
    values = value_head(features).squeeze(-1)
    return log_prob, values


def text_hidden_size(model) -> int:
    """从文本配置或输入嵌入取得价值头宽度。"""
    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else getattr(config, "text_config", None)
    for candidate in (text_config, config):
        width = candidate.get("hidden_size") if isinstance(candidate, dict) else getattr(candidate, "hidden_size", None)
        if type(width) is int and width > 0:
            return width
    embeddings = model.get_input_embeddings()
    width = getattr(embeddings, "embedding_dim", None)
    if width is None and getattr(embeddings, "weight", None) is not None:
        width = embeddings.weight.shape[-1]
    if type(width) is not int or width <= 0:
        raise ValueError("无法从文本配置或输入嵌入确定 PPO 价值头宽度")
    return width


def train_rollout(
    model: PeftModel,
    value_head: nn.Linear,
    optimizer: torch.optim.Optimizer,
    params: list[Tensor],
    ids: Tensor,
    attention: Tensor,
    valid: Tensor,
    rewards: Tensor,
    cfg: PPOSettings,
) -> tuple[dict, float, float]:
    """以同一份 rollout 进行多轮更新，loss 按本轮全部有效 token 归一化。"""
    width = valid.shape[1]
    microbatches = list(torch.arange(len(ids), device=ids.device).split(cfg.batch_size))
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        scores = [score_tokens(model, value_head, ids[b], attention[b], width) for b in microbatches]
        old_log_prob = torch.cat([logp for logp, _ in scores])
        old_values = torch.cat([v for _, v in scores])
        advantages, returns = masked_gae(old_values, rewards, valid, cfg.gamma, cfg.gae_lambda)
        active = advantages[valid]
        advantages = (advantages - active.mean()) / active.std(unbiased=False).clamp_min(1e-6)
        token_count = active.numel()
    torch.cuda.synchronize()
    old_policy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics = []
    for _ in range(cfg.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        sums = torch.zeros(4, device=ids.device)
        for indices in torch.randperm(len(ids), device=ids.device).split(cfg.batch_size):
            mask = valid[indices]
            logp, value = score_tokens(model, value_head, ids[indices], attention[indices], width)
            log_ratio = logp[mask] - old_log_prob[indices][mask]
            ratio = log_ratio.exp()
            policy_loss = clipped_policy_loss(ratio, advantages[indices][mask], cfg.cliprange).sum()
            previous = old_values[indices][mask]
            target = returns[indices][mask]
            value = value[mask]
            clipped_value = previous + (value - previous).clamp(-cfg.value_cliprange, cfg.value_cliprange)
            value_loss = 0.5 * torch.maximum((value - target).square(), (clipped_value - target).square()).sum()
            ((policy_loss + cfg.value_coef * value_loss) / token_count).backward()
            with torch.no_grad():
                sums += torch.stack([
                    policy_loss, value_loss, (ratio - 1 - log_ratio).sum(),
                    ((ratio - 1).abs() > cfg.cliprange).sum(),
                ])
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        metrics.append((sums / token_count).cpu().tolist() + [float(grad_norm)])
    torch.cuda.synchronize()
    optimize_seconds = time.perf_counter() - started
    names = ("policy_loss", "value_loss", "approx_kl", "clip_fraction", "grad_norm")
    mean = {key: sum(row[i] for row in metrics) / len(metrics) for i, key in enumerate(names)}
    mean["loss"] = mean["policy_loss"] + cfg.value_coef * mean["value_loss"]
    mean["last_epoch_approx_kl"] = metrics[-1][2]
    if not all(math.isfinite(value) for value in mean.values()):
        raise FloatingPointError("PPO 出现非有限训练指标")
    return mean, old_policy_seconds, optimize_seconds
