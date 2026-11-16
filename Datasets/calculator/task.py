"""计算器数据、严格 JSON 工具调用与执行奖励。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = 'You select a calculator tool call. Return exactly one JSON object and no other text. The schema is {"tool":"calculate","op":"add|subtract|multiply","args":[integer,integer]}. Choose one of add, subtract, multiply. Preserve the operand order requested in the task. Use the original operands, not the computed answer. Do not add keys or use code fences.'


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL。记录包含训练用标签；模型输入只使用 SYSTEM_PROMPT 与 prompt。"""
    records = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: invalid JSON') from exc
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{number}: record must be an object')
            records.append(record)
    return records


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f'Non-finite JSON constant: {value}')


def _valid_call(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and set(call) == {'tool', 'op', 'args'}
        and call['tool'] == 'calculate'
        and call['op'] in ('add', 'subtract', 'multiply')
        and isinstance(call['args'], list)
        and len(call['args']) == 2
        and all(type(x) is int and abs(x) <= 1_000_000 for x in call['args'])
    )


def _execute(call: dict[str, Any]) -> int:
    a, b = call['args']
    if call['op'] == 'add':
        return a + b
    if call['op'] == 'subtract':
        return a - b
    return a * b


def verify(completion: str, record: dict[str, Any]) -> dict[str, Any]:
    """合法 schema、正确动作/参数及执行答案全对才奖励 1；所有文本作为数据解析。"""
    outcome = {'reward': 0.0, 'valid_json': False, 'error': '', 'result': None}
    if not isinstance(completion, str) or len(completion) > 4096:
        return {**outcome, 'error': 'invalid_text'}
    try:
        call = json.loads(
            completion,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError, RecursionError):
        return {**outcome, 'error': 'invalid_json'}
    outcome['valid_json'] = True
    if not _valid_call(call):
        return {**outcome, 'error': 'invalid_schema'}
    target = record.get('target')
    if not _valid_call(target) or type(record.get('answer')) is not int:
        raise ValueError('Reference record has an invalid target or answer.')
    if _execute(target) != record['answer']:
        raise ValueError('Reference target and answer disagree.')
    outcome['result'] = _execute(call)
    if call['op'] != target['op']:
        return {**outcome, 'error': 'wrong_operation'}
    if call['args'] != target['args']:
        return {**outcome, 'error': 'wrong_arguments'}
    if outcome['result'] != record['answer']:
        return {**outcome, 'error': 'wrong_result'}
    return {**outcome, 'reward': 1.0, 'error': 'ok'}


def prepare_data(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    prompts, identifiers = set(), set()
    for split in ("train", "val", "test_final"):
        expected = manifest["splits"][split]
        path = directory / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"题目文件哈希不一致：{path}")
        rows = load_records(path)
        if len(rows) != expected["count"] or any(row["split"] != split for row in rows):
            raise ValueError(f"题目数量或 split 不一致：{path}")
        for row in rows:
            if row["id"] in identifiers or row["prompt"] in prompts:
                raise ValueError(f"题目重复：{row['id']}")
            identifiers.add(row["id"])
            prompts.add(row["prompt"])
            if verify(json.dumps(row["target"]), row)["reward"] != 1:
                raise ValueError(f"标准调用或答案错误：{row['id']}")
    return manifest
