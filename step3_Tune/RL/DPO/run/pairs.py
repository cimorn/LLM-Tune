"""Deterministic, verifier-checked training pairs; no model-generated preferences."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random

from step1_LLM.prepare import dataset_task


def _different(value, rng):
    """Change one JSON value, retaining its type whenever a change is possible."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "_incorrect"
    if isinstance(value, (int, float)):
        return value - 1 if value >= 1_000_000 else value + 1
    if isinstance(value, list):
        changed = deepcopy(value)
        if changed:
            index = rng.randrange(len(changed))
            changed[index] = _different(changed[index], rng)
        else:
            changed.append("incorrect")
        return changed
    if isinstance(value, dict):
        changed = deepcopy(value)
        if changed:
            key = rng.choice(sorted(changed))
            changed[key] = _different(changed[key], rng)
        else:
            changed["incorrect"] = True
        return changed
    if value is None:
        return "incorrect"
    raise ValueError(f"Unsupported JSON value: {type(value).__name__}")


def build_pairs(records: list[dict], config: dict, seed: int) -> list[dict]:
    """Keep correct judge labels (including C); corrupt tool answers deterministically."""
    task_name = config.get("dataset", "calculator")
    if task_name not in ("calculator", "ToolACE", "human_preference"):
        raise ValueError(f"DPO has no preference construction rule for {task_name!r}")
    if not records or any(row.get("split") != "train" for row in records):
        raise ValueError("DPO pairs require a nonempty train split")
    if len({str(row["id"]) for row in records}) != len(records):
        raise ValueError("DPO training IDs must be unique")
    task = dataset_task(config)
    pairs = []
    for row in records:
        identity = json.dumps([seed, row["id"]], ensure_ascii=False, separators=(",", ":"))
        rng = random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big"))
        if task_name == "human_preference":
            chosen = task.format_target(row)
            rejected = rng.choice([label for label in task.LABELS if label != chosen])
            protocol = "correct_judge_label_vs_seeded_wrong_label; ties retained as C"
        else:
            target = deepcopy(row["target"])
            wrong = deepcopy(target)
            if task_name == "calculator":
                if rng.randrange(2):
                    wrong["op"] = rng.choice([op for op in ("add", "subtract", "multiply") if op != target["op"]])
                else:
                    index = rng.randrange(2)
                    wrong["args"][index] = _different(wrong["args"][index], rng)
            elif wrong["arguments"]:
                name = rng.choice(sorted(wrong["arguments"]))
                wrong["arguments"][name] = _different(wrong["arguments"][name], rng)
            else:
                # The task verifier's call schema allows any tool name; no external tool is executed.
                wrong["tool"] += "_incorrect"
            chosen = json.dumps(target, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            rejected = json.dumps(wrong, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            correct, incorrect = task.verify(chosen, row), task.verify(rejected, row)
            if correct["reward"] != 1 or incorrect["reward"] != 0:
                raise ValueError(f"{row['id']}: preference pair failed verifier checks")
            if incorrect["error"] in ("invalid_json", "invalid_schema", "invalid_text"):
                raise ValueError(f"{row['id']}: corrupted response violates the task call schema")
            protocol = "verified_reference_vs_deterministic_wrong_call"
        pairs.append({"id": str(row["id"]), "record": row, "chosen": chosen,
                      "rejected": rejected, "pair_protocol": protocol})
    return pairs
