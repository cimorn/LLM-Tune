"""Shared input encoding and probability evaluation for the Arena task."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from Datasets.human_preference.task import (
    LABELS, SUBMISSION_COLUMNS, load_records, load_test_csv, render_pair,
)
from step1_LLM.evaluate import adapter_hashes, json_hash, model_identity, sha256_file
from step1_LLM.prepare import load_model_and_tokenizer, render_prompt


SWAPPED_LABELS = {"A": "B", "B": "A", "C": "C"}


def swap_responses(row: dict) -> dict:
    swapped = {**row, "response_a": row["response_b"], "response_b": row["response_a"]}
    swapped["prompt"] = render_pair(swapped["question"], swapped["response_a"], swapped["response_b"])
    if "target" in row:
        swapped["target"] = SWAPPED_LABELS[row["target"]]
    return swapped


class PreferenceEncoder:
    """Reserve one label and EOS and use the same input for training and scoring."""

    def __init__(self, tokenizer, config: dict):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = int(config.get("preference", {}).get("max_length", 768))
        if self.max_length != config.get("sft", {}).get("max_length", self.max_length):
            raise ValueError("preference.max_length 与 sft.max_length 必须一致")
        if tokenizer.eos_token_id is None:
            raise ValueError("偏好任务需要 EOS token")

        self.prompt_budget = self.max_length - 2
        empty = self._render("", "", "")
        empty_ids = self._tokenize(empty)
        self.content_budget = self.prompt_budget - len(empty_ids)
        if self.content_budget < 6:
            raise ValueError("preference.max_length 过小，无法容纳聊天模板与问题/回答")

        label_ids = []
        for label in LABELS:
            full_ids = self._tokenize(empty + label + tokenizer.eos_token)
            suffix = full_ids[len(empty_ids):]
            if full_ids[:len(empty_ids)] != empty_ids or len(suffix) != 2 or suffix[1] != tokenizer.eos_token_id:
                raise ValueError(f"标签 {label!r} 必须在聊天模板后编码为单 token，且与 EOS 边界稳定")
            label_ids.append(suffix[0])
        if len(set(label_ids)) != len(LABELS):
            raise ValueError("三个标签的 token ID 必须不同")
        self.label_ids = tuple(label_ids)

    def _render(self, question: str, response_a: str, response_b: str) -> str:
        return render_prompt(self.tokenizer, {"prompt": render_pair(question, response_a, response_b)}, self.config)

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False, verbose=False)

    def encode_prompt(self, row: dict) -> tuple[list[int], bool]:
        fields = (row["question"], row["response_a"], row["response_b"])
        original = self._tokenize(self._render(*fields))
        if len(original) <= self.prompt_budget:
            return original, False

        field_ids = [self._tokenize(field) for field in fields]
        budget = self.content_budget
        while budget >= 6:
            # Both answers get the same cap, so changing their order does not
            # preferentially preserve one of the answers.
            question_cap = min(len(field_ids[0]), budget // 3)
            answer_cap = (budget - question_cap) // 2
            clipped = []
            for field, ids, cap in zip(fields, field_ids, (question_cap, answer_cap, answer_cap)):
                if len(ids) <= cap:
                    clipped.append(field)
                else:
                    clipped.append(self.tokenizer.decode(ids[:cap], clean_up_tokenization_spaces=False))
            result = self._tokenize(self._render(*clipped))
            excess = len(result) - self.prompt_budget
            if excess <= 0:
                return result, True
            # Retokenization of the clipped fields can change boundary merges.
            budget -= max(excess, 2)
        raise ValueError(f"{row['id']}: 无法在 max_length={self.max_length} 内构造偏好输入")


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False,
        ) as handle:
            partial = Path(handle.name)
            write(handle)
        partial.replace(path)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def evaluate_preferences(
    config: dict, data_path: str | Path, output: str | Path, *,
    adapter: str | Path | None = None, batch_size: int = 4, limit: int | None = None,
    submission: str | Path | None = None,
) -> dict:
    """Score A/B/C at the answer boundary and export probabilities for every test ID."""
    import torch

    data_path, output = Path(data_path), Path(output)
    adapter = Path(adapter) if adapter is not None else None
    submission = Path(submission) if submission is not None else None
    if batch_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("batch_size 和 limit 必须为正整数")
    if data_path.suffix.lower() == ".csv":
        records = load_test_csv(data_path)
    else:
        records = load_records(data_path)
    if not records:
        raise ValueError("偏好评测数据为空")
    splits = {row["split"] for row in records}
    if splits not in ({"val"}, {"test"}):
        raise ValueError("偏好评测仅支持单一 val 或 test 数据集")
    is_val = splits == {"val"}
    if submission is not None and (is_val or (limit is not None and limit < len(records))):
        raise ValueError("生成 Kaggle 提交文件需要完整 test 数据")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("评测数据中的 ID 必须唯一")
    records = records[:limit]
    destinations = [output / "predictions.jsonl", output / "metrics.json"]
    if submission is not None:
        destinations.append(submission)
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("评测和提交输出必须使用不同文件")
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"偏好评测输出已存在：{path}，请使用新目录")

    model, tokenizer = load_model_and_tokenizer(config, adapter=adapter, trainable=False)
    model.eval()
    encoder = PreferenceEncoder(tokenizer, config)
    device = next(model.parameters()).device
    label_ids = torch.tensor(encoder.label_ids, device=device)

    def predict(batch: list[dict]) -> tuple[list[list[float]], list[list[float]], list[bool]]:
        encoded = [encoder.encode_prompt(row) for row in batch]
        width = max(len(ids) for ids, _ in encoded)
        inputs = torch.tensor(
            [[tokenizer.pad_token_id] * (width - len(ids)) + ids for ids, _ in encoded], device=device,
        )
        mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids, _ in encoded], device=device)
        positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
        with torch.inference_mode():
            # Left padding keeps the final position on the actual prompt. Qwen3
            # can compute just those logits without materializing vocabulary × length.
            results = model(input_ids=inputs, attention_mask=mask, position_ids=positions,
                            use_cache=False, logits_to_keep=1)
            scores = results.logits[:, -1].index_select(dim=1, index=label_ids).float()
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("偏好标签 logits 包含非有限值")
            log_probs = scores.log_softmax(dim=-1)
        return log_probs.exp().tolist(), log_probs.tolist(), [truncated for _, truncated in encoded]

    predictions, losses, correct, swap_distances, swap_matches = [], [], [], [], []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        probabilities, log_probs, truncated = predict(batch)
        swapped_probs = predict([swap_responses(row) for row in batch])[0] if is_val else []
        for index, row in enumerate(batch):
            probs = probabilities[index]
            prediction = max(range(len(LABELS)), key=probs.__getitem__)
            entry = {
                "id": row["id"], "prediction": LABELS[prediction],
                "probabilities": dict(zip(SUBMISSION_COLUMNS[1:], probs)), "truncated": truncated[index],
            }
            if is_val:
                other_a, other_b, other_tie = swapped_probs[index]
                restored = [other_b, other_a, other_tie]
                tv = 0.5 * sum(abs(a - b) for a, b in zip(probs, restored))
                swap_distances.append(tv)
                swap_matches.append(prediction == max(range(len(LABELS)), key=restored.__getitem__))
                gold = LABELS.index(row["target"])
                loss = -log_probs[index][gold]
                losses.append(loss)
                correct.append(gold == prediction)
                entry.update(target=row["target"], log_loss=loss, swap_total_variation=tv)
            predictions.append(entry)

    count = len(predictions)
    metrics = {
        "count": count, "split": "val" if is_val else "test",
        "log_loss": math.fsum(losses) / count if is_val else None,
        "accuracy": sum(correct) / count if is_val else None,
        "swap_mean_total_variation": math.fsum(swap_distances) / count if is_val else None,
        "swap_consistency_rate": sum(swap_matches) / count if is_val else None,
        "truncation_rate": sum(row["truncated"] for row in predictions) / count,
        "max_length": encoder.max_length,
        "probability_protocol": "softmax over three single-token label logits at the answer position",
        "label_token_ids": dict(zip(LABELS, encoder.label_ids)), "batch_size": batch_size,
        "model": model_identity(model, config), "adapter_files_sha256": adapter_hashes(adapter),
        "data_sha256": sha256_file(data_path), "config_sha256": json_hash(config),
    }

    def write_predictions(handle):
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    _atomic_write(destinations[0], write_predictions)
    if submission is not None:
        def write_submission(handle):
            writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS)
            writer.writeheader()
            writer.writerows({"id": row["id"], **row["probabilities"]} for row in predictions)

        _atomic_write(submission, write_submission)
    # Write metrics last so it can serve as the completion marker for the run.
    _atomic_write(destinations[1], lambda handle: json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False))
    return metrics
