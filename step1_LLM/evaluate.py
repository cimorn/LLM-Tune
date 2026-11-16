"""基座、SFT、GRPO、PPO 共用的评估程序。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数；小样本也返回可解释的实测值。"""
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list")
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def example_seed(seed: int, example_id: str) -> int:
    value = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()
    return int.from_bytes(value[:4], "big") % (2**31 - 1)


def batch_seed(seed: int, example_ids: list[str]) -> int:
    """批采样种子绑定有序题目列表；避免不同分批悄悄共享同一协议。"""
    payload = json.dumps([seed, example_ids], ensure_ascii=False, separators=(",", ":"))
    value = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(value[:4], "big") % (2**31 - 1)


def generated_tokens_through_eos(tokens: list[int], eos_token_id: int | list[int] | None) -> tuple[list[int], bool]:
    """仅处理已切掉完整padding输入宽度的生成区，保留首个EOS并排除其后padding。

    PAD等于EOS时首个EOS仍是真实生成的终止符；不能先strip掉它。
    没有EOS则保留全部生成token，包括可能采样出的PAD特殊token。
    """
    eos = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    for index, token in enumerate(tokens):
        if token in eos:
            return tokens[:index+1], True
    return tokens, False


def generation_call_kwargs(max_new_tokens: int) -> dict:
    """禁止 Transformers 用模型默认值覆盖全支持采样协议。

    4.57.1 会把 GenerationConfig 中等于全局默认的 T=1/top_p=1
    当作未指定，回填 Qwen 自带的 T=0.6/top_p=0.95。
    use_model_defaults=False 加显式 kwargs 同时锁住实际生成行为。
    """
    return {
        "use_model_defaults": False,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": max_new_tokens,
    }


def schema_valid_from_verdict(verdict: dict) -> bool:
    """JSON语法合法与工具schema合法分别统计；错动作仍可满足schema。"""
    return bool(verdict["valid_json"] and verdict.get("error") != "invalid_schema")


def model_identity(model: Any, config: dict) -> dict:
    """优先使用下载时锁定的 revision，同时保存本地模型配置指纹。"""
    spec = config.get("model", {})
    model_config = model.config.to_dict()
    revision = spec.get("revision") or getattr(model.config, "_commit_hash", None)
    return {
        "name": spec.get("name", "Qwen/Qwen3-0.6B"),
        "revision": revision,
        "local_path": str(spec.get("local_path", "")),
        "architecture_config_sha256": json_hash(model_config),
    }


def adapter_hashes(adapter: Path | None) -> dict[str, str]:
    if adapter is None:
        return {}
    names = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    return {name: sha256_file(adapter / name) for name in names if (adapter / name).is_file()}


def prompt_protocol(config: dict) -> dict:
    """记录传给聊天模板的实际设置；未提供 thinking 开关不代表关闭推理。"""
    spec = config.get("model", {})
    template_kwargs = dict(spec.get("chat_template_kwargs", {"enable_thinking": False}))
    return {
        "prompt_mode": spec.get("prompt_mode", "system"),
        "chat_template_kwargs": template_kwargs,
        "enable_thinking": template_kwargs.get("enable_thinking"),
    }


def evaluate(args: argparse.Namespace) -> dict:
    # 延迟导入，比较报告和纯CPU单元测试无需加载CUDA库。
    import torch
    from transformers import GenerationConfig

    from step1_LLM.prepare import (
        ROOT,
        dataset_task,
        load_model_and_tokenizer,
        read_config,
        render_prompt,
        seed_everything,
        write_json,
    )

    def resolve(path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else ROOT / value

    config_path = resolve(args.config)
    data_path = resolve(args.data)
    output = resolve(args.output)
    adapter = resolve(args.adapter) if args.adapter else None
    config = read_config(config_path)
    task = dataset_task(config)
    sealed = {"test_final"}
    records = task.load_records(data_path)
    if (data_path.stem in sealed or any(row.get("split") in sealed for row in records)) and not args.allow_final_test:
        raise ValueError("最终评估题需要显式传入 --allow-final-test；调参请使用 val。")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]
    if not records:
        raise ValueError("Evaluation data is empty")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Evaluation IDs must be unique")
    output.mkdir(parents=True, exist_ok=True)
    protected = [output / name for name in ("predictions.jsonl", "metrics.json")]
    if not args.overwrite and any(path.exists() for path in protected):
        raise FileExistsError("评测输出已存在；请使用新目录或显式 --overwrite。")
    if args.overwrite:
        for path in protected:
            path.unlink(missing_ok=True)

    batch_size = int(getattr(args, "batch_size", 1))
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    seed_everything(seed)
    model, tokenizer = load_model_and_tokenizer(config, adapter=adapter, trainable=False)
    model.eval()
    tokenizer.padding_side = "left"
    max_new_tokens = int(config.get("eval", {}).get("max_new_tokens", 128))
    if max_new_tokens <= 0:
        raise ValueError("eval.max_new_tokens must be positive")
    device = next(model.parameters()).device
    cuda = device.type == "cuda"

    def synchronize() -> None:
        if cuda:
            torch.cuda.synchronize(device)

    # 用显式生成配置覆盖checkpoint中可能带入的top-p/top-k/repetition penalty。
    generation = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        use_cache=True,
    )
    identity = model_identity(model, config)
    protocol = {
        "sampling": "one_completion_per_example",
        "seed": seed,
        "seed_strategy": ("sha256(seed:id)_first4bytes_mod_2^31-1" if batch_size == 1 else
                          "sha256(json([seed,ordered_batch_ids]))_first4bytes_mod_2^31-1"),
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        **prompt_protocol(config),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": generation.pad_token_id,
        "model": identity,
        "torch_dtype": str(next(model.parameters()).dtype),
    }
    if batch_size > 1:
        protocol["ordered_example_ids_sha256"] = json_hash([row["id"] for row in records])
        protocol["batching"] = "contiguous file-order chunks; final partial batch retained; left padding"
    config_hash = json_hash(config)
    data_hash = sha256_file(data_path)
    adapter_files = adapter_hashes(adapter)
    adapter_hash = json_hash(adapter_files) if adapter_files else None
    label = args.label or (adapter.name if adapter else "baseline")

    def generate(batch: list[dict], batch_id: int, warmup: bool = False) -> tuple[list[dict], float]:
        rendered = [render_prompt(tokenizer, record, config) for record in batch]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        sample_seed = (example_seed(seed, batch[0]["id"]) if batch_size == 1 else
                       batch_seed(seed, [record["id"] for record in batch]))
        seed_everything(sample_seed)
        synchronize()
        start = time.perf_counter()
        kwargs = generation_call_kwargs(min(8, max_new_tokens) if warmup else max_new_tokens)
        with torch.inference_mode():
            outputs = model.generate(**inputs, generation_config=generation, **kwargs)
        synchronize()
        elapsed = time.perf_counter() - start
        # 所有生成结果都从padding后的输入宽度切片，而非单题的非padding长度。
        input_width = inputs["input_ids"].shape[-1]
        generated_batch = outputs[:, input_width:].detach().cpu().tolist()
        lengths = inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()
        result = []
        for index, record in enumerate(batch):
            tokens, ended = generated_tokens_through_eos(generated_batch[index], tokenizer.eos_token_id)
            completion = tokenizer.decode(tokens, skip_special_tokens=True)
            verdict = task.verify(completion, record)
            result.append({
            "id": record["id"],
            "split": record.get("split"),
            "family": record.get("family"),
            "prompt": record["prompt"],
            "rendered_prompt_sha256": hashlib.sha256(rendered[index].encode("utf-8")).hexdigest(),
            "completion": completion,
            "expected_call": record["target"],
            "expected_answer": record["answer"],
            "reward": float(verdict["reward"]),
            "correct": bool(verdict["reward"] == 1),
            "valid_json": bool(verdict["valid_json"]),
            "schema_valid": schema_valid_from_verdict(verdict),
            "error": verdict.get("error", ""),
            "result": verdict.get("result"),
            "generated_tokens": len(tokens),
            "input_tokens": int(lengths[index]),
            "padded_input_tokens": int(input_width),
            "batch_id": batch_id,
            "batch_count": len(batch),
            "batch_latency_s": elapsed,
            "amortized_latency_s": elapsed / len(batch),
            "latency_s": elapsed / len(batch),
            "latency_kind": "single_request" if batch_size == 1 else "amortized_batch_time_per_example",
            "sample_seed": sample_seed,
            "ended_with_eos": ended,
            "config_sha256": config_hash,
            "data_sha256": data_hash,
            "adapter_sha256": adapter_hash,
            "model_revision": identity["revision"],
            })
        return result, elapsed

    _, warmup_time = generate(records[:batch_size], batch_id=-1, warmup=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    predictions: list[dict] = []
    batch_times: list[float] = []
    predictions_temp = output / "predictions.jsonl.partial"
    started = time.perf_counter()
    with predictions_temp.open("w", encoding="utf-8") as handle:
        for batch_id, start in enumerate(range(0, len(records), batch_size)):
            current = records[start:start+batch_size]
            generated_rows, batch_time = generate(current, batch_id=batch_id)
            batch_times.append(batch_time)
            predictions.extend(generated_rows)
            for prediction in generated_rows:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            correct = sum(row["correct"] for row in generated_rows)
            print(f"[{label}] {start+len(current)}/{len(records)} batch={batch_id} correct={correct}/{len(current)} batch_time={batch_time:.2f}s", flush=True)
    predictions_temp.replace(output / "predictions.jsonl")
    elapsed = time.perf_counter() - started
    times = [row["latency_s"] for row in predictions]
    generated = sum(row["generated_tokens"] for row in predictions)
    metrics = {
        "label": label,
        "count": len(predictions),
        "success_rate": statistics.mean(row["correct"] for row in predictions),
        "format_rate": statistics.mean(row["valid_json"] for row in predictions),
        "schema_valid_rate": statistics.mean(row["schema_valid"] for row in predictions),
        "error_counts": dict(Counter(row["error"] or "correct" for row in predictions)),
        "generated_tokens": generated,
        "mean_generated_tokens": generated / len(predictions),
        "generation_time_s": sum(batch_times),
        "wall_time_s": elapsed,
        "generation_tokens_per_s": generated / sum(batch_times),
        "examples_per_generation_s": len(predictions) / sum(batch_times),
        "latency_p50_s": percentile(times, 0.5),
        "latency_p90_s": percentile(times, 0.9),
        "batch_count": len(batch_times),
        "batch_latency_p50_s": percentile(batch_times, 0.5),
        "batch_latency_p90_s": percentile(batch_times, 0.9),
        "latency_kind": "single_request" if batch_size == 1 else "amortized_batch_time_per_example",
        "warmup_time_s_excluded": warmup_time,
        "latency_scope": ("CUDA-synchronized generate only; separate 8-token warmup excluded; tokenization/verifier excluded"
                          if batch_size == 1 else
                          "CUDA-synchronized batch generate; latency_p50/p90 are per-example amortized batch time, NOT single-request latency; batch_latency_p50/p90 measure full batches; warmup/tokenization/verifier excluded"),
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
        "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if cuda else None,
        "hardware": {"device": str(device), "gpu": torch.cuda.get_device_name(device) if cuda else None, "torch": torch.__version__, "cuda": torch.version.cuda},
        "data_path": str(data_path),
        "data_sha256": data_hash,
        "config_sha256": config_hash,
        "model": identity,
        "adapter_path": str(adapter) if adapter else None,
        "adapter_files_sha256": adapter_files,
        "inference_protocol": protocol,
        "final_test_unsealed": bool(data_path.stem in sealed),
        "example_ids": [row["id"] for row in predictions],
    }
    run_record = getattr(args, "run_record", None)
    if run_record is None:
        metrics["config"] = config
    else:
        metrics["run"] = {
            **run_record, "status": "completed", "eval_count": metrics["count"],
            "success_rate": metrics["success_rate"],
        }
    metrics_temp = output / "metrics.json.partial"
    write_json(metrics_temp, metrics)
    metrics_temp.replace(output / "metrics.json")
    print(json.dumps({key: metrics[key] for key in ("label", "count", "success_rate", "format_rate", "latency_p50_s", "peak_memory_allocated_mb")}, ensure_ascii=False), flush=True)
    return metrics
