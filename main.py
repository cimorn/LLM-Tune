"""LLM-Tune 的模型、数据和训练入口。"""
from argparse import ArgumentParser, Namespace
import gc
import hashlib
from importlib import import_module
import json
from pathlib import Path
import re
import sys

from step1_LLM import prepare
from step1_LLM.evaluate import json_hash


MODEL = "Qwen3-0.6B"  # 也可选 DeepSeek-R1-1.5B、Gemma-3-1B-IT
DATASET = "calculator"  # 对应 Datasets/ 下已准备好的数据集目录
ALGORITHM = "both"  # "GRPO"、"PPO" 或 "both"
STEP = "all"  # "1"、"2"、"3"、"all" 或 "compare"
EVAL_SEED = 42
EVAL_MAX_NEW_TOKENS = 128

ROOT = Path(__file__).resolve().parent
SFT_SOURCE = ROOT / "step2_SFT"
RL_SOURCE = ROOT / "step3_Tune/RL"
TRAINERS = {
    "GRPO": "step3_Tune.RL.GRPO.train",
    "PPO": "step3_Tune.RL.PPO.train",
}
MODEL_ALIASES = {"Qwen3": "Qwen3-0.6B", "DeepSeek": "DeepSeek-R1-1.5B"}
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"应为 JSON 对象：{path}")
    return value


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adapter_hash(directory: Path) -> str:
    weights = directory / "adapter" / "adapter_model.safetensors"
    return file_hash(weights)


def get_paths(args: Namespace) -> tuple[Path, Path, dict[str, Path]]:
    for value in (args.model, args.dataset):
        if not FILE_NAME.fullmatch(value) or value.endswith("."):
            raise ValueError(f"名称格式无效：{value!r}")
    data = ROOT / "Datasets" / args.dataset
    model = ROOT / "step1_LLM/Models" / args.model / "model.json"
    if not data.is_dir() or not (data / "task.py").is_file():
        raise ValueError(f"数据集尚未接入：{args.dataset}")
    if not model.is_file():
        raise ValueError(f"没有此模型配置：{args.model}")
    model_output = ROOT / "Outputs" / args.dataset / args.model
    paths = {name: model_output / name for name in ("LLM", "SFT", *TRAINERS, "Comparison")}
    return data, model, paths


def training_config(method: str, args: Namespace, base: dict) -> dict:
    directory = SFT_SOURCE if method == "SFT" else RL_SOURCE / method
    path = directory / "Configs" / args.model / f"{args.dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少这组训练参数：{path}")
    params = read_json(path)
    config = {
        **base,
        "method": method,
        "seed": params.pop("seed", EVAL_SEED),
        "source_config": path.relative_to(ROOT).as_posix(),
    }
    if method == "SFT":
        config["lora"] = params.pop("lora")
    config[method.lower()] = params
    return config


def completed_run(path: Path) -> tuple[dict, dict]:
    metrics_path = path / "eval" / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"找不到已完成的结果：{path}")
    metrics = read_json(metrics_path)
    record = metrics.get("run")
    legacy_marker = path / "run.json"
    if record is None and legacy_marker.is_file():
        record = read_json(legacy_marker)
    if not isinstance(record, dict):
        raise ValueError(f"缺少运行记录：{path}")
    if record.get("status") != "completed":
        raise ValueError(f"该步骤未完成：{path}")
    if not (path / "eval" / "predictions.jsonl").is_file():
        raise ValueError(f"已记录为完成，但评估文件缺失：{path}")
    if record.get("method") != "LLM":
        if record.get("adapter_sha256") != adapter_hash(path):
            raise ValueError(f"权重与运行记录不一致：{path}")
        if not (path / "train_log.json").is_file():
            raise ValueError(f"训练日志缺失：{path}")
    return record, metrics


def require_run(path: Path, config: dict) -> dict:
    record, metrics = completed_run(path)
    config = prepare.resolve_model_config(config)
    digest = metrics.get("config_sha256")
    if json_hash(config) == digest:
        return record

    # 旧目录布局下的配置指纹仍有效。只为复用检查还原旧路径。
    method = config["method"]
    model = config["selected_model"]
    dataset = config["dataset"]
    variants = [config]
    if method == "SFT" or method in TRAINERS:
        old_source = "tune/SFT" if method == "SFT" else f"tune/RL/{method}"
        variants.append({**config, "source_config": f"{old_source}/Configs/{model}/{dataset}.json"})
    if method in TRAINERS:
        legacy_adapter = f"Outputs/{dataset}/tune/SFT/{model}/adapter"
        variants += [{**item, "sft_adapter": legacy_adapter} for item in variants]
    if not any(json_hash(item) == digest for item in variants[1:]):
        raise ValueError(f"已有结果的配置与本次不同，请先把已有结果移到备份目录：{path}")
    return record


def release_cuda() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_stage(
    method: str, directory: Path, config: dict, data: Path,
    sft_adapter: Path | None = None, environment: dict | None = None,
) -> None:
    from step1_LLM.evaluate import evaluate

    if directory.exists():
        require_run(directory, config)
        print(f"[{method}] 复用 {directory.relative_to(ROOT)}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=False)
    run_path = directory.relative_to(ROOT / "Outputs")
    runtime = ROOT / ".cache" / "runs" / run_path / json_hash(prepare.resolve_model_config(config))
    snapshot = runtime / "config.json"
    prepare.write_json(snapshot, config)
    marker = runtime / "run.json"
    record = {"method": method, "model": config["selected_model"],
              "dataset": config["dataset"], "status": "running"}
    if sft_adapter is not None:
        record["sft_adapter_sha256"] = config["sft_adapter_sha256"]
    prepare.write_json(marker, record)
    print(f"[{method}] 开始 {directory.relative_to(ROOT)}", flush=True)
    adapter = None
    try:
        if method == "LLM":
            if environment:
                prepare.write_json(runtime / "environment.json", environment)
        else:
            module = "step2_SFT.train" if method == "SFT" else TRAINERS[method]
            train = import_module(module).run
            adapter = directory / "adapter"
            result = train(Namespace(
                config=snapshot, data=data / "train.jsonl", output=adapter,
                log_dir=directory, sft_adapter=sft_adapter, max_steps=None, limit=None,
            ))
            record["training_status"] = result["status"]
            record["optimizer_steps"] = result["optimizer_steps"]
            record["adapter_sha256"] = adapter_hash(directory)
            if not (directory / "train_log.json").is_file():
                raise FileNotFoundError(f"训练日志没有写入：{directory}")
            release_cuda()
        settings = config["evaluation"]
        metrics = evaluate(Namespace(
            config=snapshot, data=data / "val.jsonl", output=directory / "eval",
            adapter=adapter, label="baseline" if method == "LLM" else method.lower(),
            seed=settings["seed"], limit=settings["limit"], batch_size=1,
            allow_final_test=False, overwrite=False, run_record=record,
        ))
        record = metrics["run"]
        print(f"[{method}] 验证集 {metrics['count']} 题，成功率 {metrics['success_rate']:.2%}", flush=True)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if adapter is not None:
            # PEFT 和 TRL 保存 adapter 与检查点时会自动生成模型卡。
            for card in adapter.rglob("README.md"):
                card.unlink(missing_ok=True)
        prepare.write_json(marker, record)
        release_cuda()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--algorithm", choices=(*TRAINERS, "both"), default=ALGORITHM)
    parser.add_argument("--step", choices=("1", "2", "3", "all", "compare"), default=STEP)
    parser.add_argument("--eval-limit", type=int, help="限制验证题数。省略时评估完整验证集。")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.eval_limit is not None and args.eval_limit < 1:
        parser.error("--eval-limit 必须大于 0")
    args.model = MODEL_ALIASES.get(args.model, args.model)
    data, model_file, paths = get_paths(args)
    algorithms = list(TRAINERS) if args.algorithm == "both" else [args.algorithm]
    steps = {"1": ["LLM"], "2": ["SFT"], "3": algorithms,
             "all": ["LLM", "SFT", *algorithms], "compare": []}[args.step]
    dataset_output = ROOT / "Outputs" / args.dataset
    if args.step == "compare" and not args.dry_run:
        from compare import compare_dataset
        compare_dataset(dataset_output, args.model)
        print(f"[Comparison] {paths['Comparison']}")
        return

    base = {
        "selected_model": args.model,
        "model": read_json(model_file),
        "dataset": args.dataset,
        "seed": EVAL_SEED,
        "eval": {"max_new_tokens": EVAL_MAX_NEW_TOKENS},
        "evaluation": {"split": "val", "seed": EVAL_SEED, "limit": args.eval_limit},
    }
    for method in steps:
        if method != "LLM":
            training_config(method, args, base)
    if args.dry_run:
        print(json.dumps({
            "model": base["model"]["name"], "dataset": args.dataset,
            "steps": {step: str(paths[step]) for step in steps},
            "comparison": str(paths["Comparison"]),
        }, ensure_ascii=False, indent=2))
        return

    prepare.dataset_task(base).prepare_data(data)
    base["data_sha256"] = {name: file_hash(data / f"{name}.jsonl") for name in ("train", "val")}
    prepare.resolve_model_revision(base)
    baseline_config = {**base, "method": "LLM"}
    if args.step in ("2", "3"):
        require_run(paths["LLM"], baseline_config)
    sft_config = training_config("SFT", args, base) if args.step in ("2", "3", "all") else None
    if args.step == "3":
        require_run(paths["SFT"], sft_config)

    environment = prepare.check_environment()
    print(f"设备 {environment['gpu']}，CUDA {environment['torch_cuda']}", flush=True)
    prepare.prepare_model(base)
    for method in steps:
        if method == "LLM":
            run_stage(method, paths[method], baseline_config, data, environment=environment)
        elif method == "SFT":
            run_stage(method, paths[method], sft_config, data)
        else:
            sft = require_run(paths["SFT"], sft_config)
            config = training_config(method, args, base)
            adapter = paths["SFT"] / "adapter"
            config["sft_adapter"] = adapter.relative_to(ROOT).as_posix()
            config["sft_adapter_sha256"] = sft["adapter_sha256"]
            run_stage(method, paths[method], config, data, sft_adapter=adapter)

    from compare import compare_dataset
    compare_dataset(dataset_output, args.model)
    print(f"[Comparison] {paths['Comparison']}", flush=True)


if __name__ == "__main__":
    main()
