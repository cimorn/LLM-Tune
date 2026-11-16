"""ToolACE 单轮工具选择与参数填充，按参考调用离线判分。"""

import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    'Choose one of the available tools for the user request. Return exactly one '
    'JSON object: {"tool":"tool name","arguments":{"parameter":"value"}}. '
    'Use the exact tool name and parameter names. Include the arguments requested by the '
    'user. If no argument is needed, use an empty object. Do not add prose or code fences.'
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复的 JSON 键：{key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"不支持的 JSON 常量：{value}")


def _decode(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = _decode(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number} 必须是 JSON 对象")
                rows.append(value)
    return rows


def verify(completion: str, record: dict[str, Any]) -> dict[str, Any]:
    """工具名及参数的 JSON 值、类型均一致时奖励 1。"""
    result = {"reward": 0.0, "valid_json": False, "result": None}
    if not isinstance(completion, str) or len(completion) > 4096:
        return {**result, "error": "invalid_text"}
    try:
        call = _decode(completion)
        canonical = _canonical(call)
    except (ValueError, TypeError, RecursionError):
        return {**result, "error": "invalid_json"}
    result["valid_json"] = True
    if (
        not isinstance(call, dict)
        or set(call) != {"tool", "arguments"}
        or not isinstance(call["tool"], str)
        or not isinstance(call["arguments"], dict)
    ):
        return {**result, "error": "invalid_schema"}
    expected = record["target"]
    if call["tool"] != expected["tool"]:
        return {**result, "error": "wrong_tool"}
    if canonical != _canonical(expected):
        return {**result, "error": "wrong_arguments"}
    return {**result, "reward": 1.0, "error": "ok", "result": call}


def prepare_data(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest = _decode((directory / "manifest.json").read_text(encoding="utf-8"))
    ids, requests, reference_tools = set(), set(), set()
    for split in ("train", "val", "test_final"):
        expected = manifest["splits"][split]
        path = directory / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"数据哈希不一致：{path}")
        rows = load_records(path)
        if len(rows) != expected["count"]:
            raise ValueError(f"数据数量不一致：{path}")
        for row in rows:
            request = " ".join(row["request"].casefold().split())
            tool = row["target"]["tool"].casefold()
            if row["split"] != split or row["id"] in ids or request in requests or tool in reference_tools:
                raise ValueError(f"数据划分或去重检查失败：{row['id']}")
            ids.add(row["id"])
            requests.add(request)
            reference_tools.add(tool)
            if verify(row["answer"], row)["reward"] != 1:
                raise ValueError(f"参考调用校验失败：{row['id']}")
    return manifest
