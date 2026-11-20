"""DPO probability scoring shares the existing task prompts and judge encoder."""
from __future__ import annotations

import inspect

import torch

from Datasets.human_preference.task import LABELS
from step1_LLM.prepare import render_prompt
from step3_Tune.RL.GRPO.run.grpo_judge import _batch, _policy
from .core import masked_sequence_logps


class ToolPairScorer:
    """Causal completion logp sums, with explicit prompt/EOS/padding boundaries."""
    def __init__(self, model, tokenizer, config, cfg):
        self.model, self.tokenizer, self.config, self.cfg = model, tokenizer, config, cfg
        self.device = next(model.parameters()).device
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Tool DPO requires EOS and PAD token IDs")

    def __call__(self, pairs):
        prompts, answers = [], []
        for key in ("chosen", "rejected"):
            for pair in pairs:
                prompt = render_prompt(self.tokenizer, pair["record"], self.config)
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                full = self.tokenizer.encode(prompt + pair[key] + self.tokenizer.eos_token, add_special_tokens=False)
                if not prompt_ids or full[:len(prompt_ids)] != prompt_ids:
                    raise ValueError(f"{pair['id']}: prompt/completion tokenizer boundary changed")
                answer = full[len(prompt_ids):]
                if not answer or answer[-1] != self.tokenizer.eos_token_id:
                    raise ValueError(f"{pair['id']}: completion must end in EOS")
                if len(prompt_ids) > self.cfg.max_prompt_length or len(answer) > self.cfg.max_completion_length:
                    raise ValueError(f"{pair['id']}: DPO prompt/completion too long; truncation is disabled")
                prompts.append(prompt_ids)
                answers.append(answer)
        prompt_width, answer_width = max(map(len, prompts)), max(map(len, answers))
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (prompt_width - len(p)) + p + a + [pad] * (answer_width - len(a))
                            for p, a in zip(prompts, answers)], device=self.device)
        attention = torch.tensor([[0] * (prompt_width - len(p)) + [1] * (len(p) + len(a))
                                  + [0] * (answer_width - len(a)) for p, a in zip(prompts, answers)], device=self.device)
        valid = torch.tensor([[1] * len(a) + [0] * (answer_width - len(a)) for a in answers], device=self.device)
        positions = (attention.cumsum(dim=-1) - 1).clamp_min(0)
        base = self.model.get_base_model() if callable(getattr(self.model, "get_base_model", None)) else self.model
        parameters = inspect.signature(base.forward).parameters
        keep = ({"logits_to_keep": answer_width} if "logits_to_keep" in parameters else
                {"num_logits_to_keep": answer_width} if "num_logits_to_keep" in parameters else {})
        result = self.model(input_ids=ids[:, :-1], attention_mask=attention[:, :-1],
                            position_ids=positions[:, :-1], use_cache=False, **keep)
        scores = masked_sequence_logps(result.logits[:, -answer_width:], ids[:, -answer_width:], valid)
        return scores[:len(pairs)], scores[len(pairs):]


class PreferencePairScorer:
    """Score the existing three-label judge action; do not optimize response prose."""
    def __init__(self, model, encoder):
        self.model, self.encoder = model, encoder
        self.device = next(model.parameters()).device
        self.label_ids = torch.tensor(encoder.label_ids, device=self.device)

    def __call__(self, pairs):
        inputs, mask, positions, _ = _batch(self.encoder, [pair["record"] for pair in pairs], self.device)
        logps = _policy(self.model, inputs, mask, positions, self.label_ids)
        rows = torch.arange(len(pairs), device=self.device)
        chosen = torch.tensor([LABELS.index(pair["chosen"]) for pair in pairs], device=self.device)
        rejected = torch.tensor([LABELS.index(pair["rejected"]) for pair in pairs], device=self.device)
        return logps[rows, chosen], logps[rows, rejected]
