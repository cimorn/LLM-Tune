"""Sigmoid DPO with immutable SFT reference log probabilities and pair-mean loss."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

import torch
from torch.nn import functional as F


@dataclass
class DPOSettings:
    max_steps: int = 64
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    beta: float = 0.1
    max_prompt_length: int = 768
    max_completion_length: int = 128
    max_grad_norm: float = 1.0
    save_at_steps: list[int] = field(default_factory=list)

    def validate(self):
        for name in ("max_steps", "batch_size", "gradient_accumulation_steps", "max_prompt_length", "max_completion_length"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"dpo.{name} must be a positive integer")
        for name in ("learning_rate", "beta", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"dpo.{name} must be finite and positive")
        if (not isinstance(self.save_at_steps, list)
            or any(type(step) is not int or not 1 <= step <= self.max_steps for step in self.save_at_steps)
            or len(set(self.save_at_steps)) != len(self.save_at_steps)):
            raise ValueError("dpo.save_at_steps must be distinct optimizer steps within this run")


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    """Return unreduced pair losses and beta-scaled reference-relative margins."""
    margin = beta * ((chosen - rejected) - (reference_chosen.detach() - reference_rejected.detach()))
    return -F.logsigmoid(margin), margin


def masked_sequence_logps(logits, token_ids, completion_mask):
    """Sum completion log probabilities, including EOS and excluding prompt/padding."""
    logits = logits.float()
    logps = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(dim=-1)
    return (logps * completion_mask).sum(dim=-1)


def training_schedule(count, steps, effective_batch, seed):
    if min(count, steps, effective_batch) <= 0:
        raise ValueError("DPO schedule sizes must be positive")
    rng = random.Random(seed)
    pool, cursor = [], 0
    schedule = []
    for _ in range(steps):
        selected = []
        while len(selected) < effective_batch:
            if cursor == len(pool):
                pool = list(range(count))
                rng.shuffle(pool)
                cursor = 0
            take = min(effective_batch - len(selected), len(pool) - cursor)
            selected.extend(pool[cursor:cursor + take])
            cursor += take
        schedule.append(selected)
    return schedule


def fit_pairs(model, pairs, score, cfg: DPOSettings, *, seed, on_reference=None, on_step=None):
    """Precompute the untouched SFT scores for every scheduled pair, then update.

    score(batch) returns differentiable (chosen_logps, rejected_logps). The initial
    reference is never recomputed after the first optimizer update. Only scheduled
    rows are cached, avoiding a full-corpus prefill for short experiments.
    """
    cfg.validate()
    device = next(model.parameters()).device
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    model.eval()  # Deterministic forward passes; eval mode does not disable autograd.
    schedule = training_schedule(len(pairs), cfg.max_steps, cfg.batch_size * cfg.gradient_accumulation_steps, seed)
    used = sorted({index for indices in schedule for index in indices})
    reference = torch.full((len(pairs), 2), float("nan"), dtype=torch.float32)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("DPO requires trainable policy parameters")
    sync()
    started = time.perf_counter()
    reference_rows = []
    with torch.no_grad():
        for start in range(0, len(used), cfg.batch_size):
            indices = used[start:start + cfg.batch_size]
            chosen, rejected = score([pairs[index] for index in indices])
            values = torch.stack((chosen, rejected), dim=-1).detach().float().cpu()
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("Non-finite SFT reference log probability")
            reference[indices] = values
            reference_rows.extend({"index": index, "id": pairs[index]["id"],
                                   "chosen_logp": float(value[0]), "rejected_logp": float(value[1])}
                                  for index, value in zip(indices, values))
    sync()
    reference_seconds = time.perf_counter() - started
    if on_reference is not None:
        on_reference(reference_rows, schedule)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=0)
    history = []
    sync()
    started = time.perf_counter()
    for step, indices in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(4, device=device)
        for start in range(0, len(indices), cfg.batch_size):
            batch_indices = indices[start:start + cfg.batch_size]
            chosen, rejected = score([pairs[index] for index in batch_indices])
            ref = reference[batch_indices].to(device)
            losses, margin = dpo_loss(chosen, rejected, ref[:, 0], ref[:, 1], cfg.beta)
            if not bool(torch.isfinite(losses).all()):
                raise FloatingPointError("Non-finite DPO loss")
            (losses.sum() / len(indices)).backward()
            with torch.no_grad():
                totals += torch.stack((losses.sum(), margin.sum(), (margin > 0).float().sum(),
                                       (chosen - rejected).sum()))
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        means = (totals / len(indices)).cpu().tolist()
        event = dict(step=step, optimizer_steps=step, loss=means[0], reward_margin=means[1],
                     reference_relative_pair_accuracy=means[2], policy_logp_margin=means[3],
                     grad_norm=float(grad_norm), training_pairs=len(indices))
        history.append(event)
        if on_step is not None:
            on_step(event, optimizer)
    sync()
    return dict(optimizer_steps=len(history), history=history, reference_logps=reference_rows,
                reference_rows=len(used), reference_seconds=reference_seconds,
                optimization_seconds=time.perf_counter() - started,
                training_pair_exposures=sum(map(len, schedule)))
