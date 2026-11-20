"""LLM-Tune 的模型、数据和训练入口。"""
from argparse import ArgumentParser, Namespace
import gc
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory

from step1_LLM import prepare
from step1_LLM.evaluate import adapter_hashes, json_hash


MODEL = "Qwen3-0.6B"  # 也可选 DeepSeek-R1-1.5B、Gemma-3-1B-IT
DATASET = "calculator"  # 对应 Datasets/ 下已准备好的数据集目录
ALGORITHM = "both"  # GRPO、PPO、DPO；both 保留原 GRPO/PPO 组合。
STEP = "all"  # "1"、"2"、"3"、"all" 或 "compare"
EVAL_SEED = 42
EVAL_MAX_NEW_TOKENS = 128

ROOT = Path(__file__).resolve().parent
SFT_SOURCE = ROOT / "step2_SFT"
RL_SOURCE = ROOT / "step3_Tune/RL"
TRAINERS = {
    "GRPO": "step3_Tune.RL.GRPO.train",
    "PPO": "step3_Tune.RL.PPO.train",
    "DPO": "step3_Tune.RL.DPO.train",
}
MODEL_ALIASES = {"Qwen3": "Qwen3-0.6B", "DeepSeek": "DeepSeek-R1-1.5B", "Gemma": "Gemma-3-1B-IT"}
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


PREFERENCE_MODELS = ("Qwen3-0.6B", "DeepSeek-R1-1.5B", "Gemma-3-1B-IT")
PREFERENCE_TRAINERS = {
    "GRPO": ("step3_Tune.RL.GRPO.run.grpo_judge", "train_preference_grpo", "preference_grpo"),
    "PPO": ("step3_Tune.RL.PPO.run.ppo_judge", "train_preference_ppo", "preference_ppo"),
    "DPO": ("step3_Tune.RL.DPO.train", "train_dpo", "preference_dpo"),
}


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


def evaluate_final(data: Path, paths: dict[str, Path], methods: list[str]) -> None:
    from step1_LLM.evaluate import evaluate

    questions = data / "test_final.jsonl"
    dataset_digest = file_hash(questions)
    stages = [(method, *completed_run(paths[method])) for method in ("LLM", "SFT", *methods)]
    for method, record, validation in stages:
        directory = paths[method]
        output = directory / "eval_final"
        label = "baseline" if method == "LLM" else method.lower()
        adapter = None if method == "LLM" else directory / "adapter"
        config_digest = validation["config_sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", config_digest):
            raise ValueError(f"验证配置指纹无效：{directory}")
        snapshot = ROOT / ".cache" / "runs" / directory.relative_to(ROOT / "Outputs") / config_digest / "config.json"
        config = prepare.read_config(snapshot)
        if json_hash(config) != config_digest:
            raise ValueError(f"验证使用的配置已改变：{snapshot}")
        report = output / "metrics.json"
        if report.is_file():
            metrics = read_json(report)
            if (metrics.get("data_sha256") != dataset_digest or metrics.get("config_sha256") != config_digest
                    or metrics.get("run", {}).get("adapter_sha256") != record.get("adapter_sha256")
                    or not (output / "predictions.jsonl").is_file()):
                raise ValueError(f"已有最终评测与数据、配置或权重不符：{output}")
            continue
        try:
            evaluate(Namespace(config=snapshot, data=questions, output=output, adapter=adapter,
                               label=label, seed=EVAL_SEED, limit=None, batch_size=16,
                               allow_final_test=True, overwrite=False, run_record=record))
        finally:
            release_cuda()


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
            seed=settings["seed"], limit=settings["limit"], batch_size=settings.get("batch_size", 1),
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


def selected_algorithms(value: str | None, *, default: str | None = None) -> list[str]:
    if value == "all":
        return list(PREFERENCE_TRAINERS)
    if value == "both":
        return ["GRPO", "PPO"]
    selected = value or default
    if selected is None:
        return []
    if selected not in (*PREFERENCE_TRAINERS, "SFT"):
        raise ValueError(f"偏好任务不支持算法：{selected}")
    return [selected]


def submission_destination(destination: Path | None, stage: str, *, multiple: bool) -> Path | None:
    if destination is None or not multiple:
        return destination
    return destination.with_name(f"{destination.stem}-{stage}{destination.suffix}")


def validate_selected_model(config: dict, selected_model: str) -> None:
    """显式指定的实验目录不能悄悄切换到其中保存的另一模型。"""
    if config.get("dataset") != "human_preference" or config.get("selected_model") != selected_model:
        raise ValueError(f"实验目录绑定的任务/模型与 --model {selected_model} 不一致")


def prepare_preference_model(config: dict) -> None:
    """下载后仅用tokenizer检查共享模板和A/B/C边界，再允许进入GPU训练。"""
    from transformers import AutoTokenizer
    from step1_LLM.judge_eval import PreferenceEncoder

    path = prepare.prepare_model(config)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    prepare.validate_tokenizer(tokenizer)
    # Gemma和DeepSeek的system说明通过model.prompt_mode=user折叠进user消息；
    # PreferenceEncoder统一调用prepare.render_prompt，不复制Qwen专用模板。
    encoder = PreferenceEncoder(tokenizer, config)
    print(f"[Tokenizer] {config['selected_model']}，prompt_mode={config['model'].get('prompt_mode', 'system')}，"
          f"标签token={encoder.label_ids}，max_length={encoder.max_length}", flush=True)


def experiment_config(model: str, max_steps: int | None) -> dict:
    source = root / "step2_SFT" / "Configs" / model / "human_preference.json"
    params = read_json(source)
    if max_steps is not None:
        params["max_steps"] = max_steps
    config = {
        "dataset": "human_preference",
        "selected_model": model,
        "model": read_json(root / "step1_LLM" / "Models" / model / "model.json"),
        "seed": params.pop("seed"),
        "lora": params.pop("lora"),
        "preference": {"max_length": params["max_length"], "eval_batch_size": params.pop("eval_batch_size")},
        "sft": params,
    }
    return prepare.resolve_model_config(config)


def evaluate_preference(config: dict, data: Path, output: Path, *, adapter=None, limit=None) -> dict:
    metrics_file = output / "metrics.json"
    if metrics_file.exists() and (output / "predictions.jsonl").is_file():
        metrics = read_json(metrics_file)
        expected_count = len(task_records(data)[:limit])
        if (metrics["data_sha256"] != file_hash(data)
            or metrics["adapter_files_sha256"] != adapter_hashes(adapter)
            or metrics["config_sha256"] != json_hash(prepare.resolve_model_config(config))
            or metrics["count"] != expected_count):
            raise ValueError(f"现有评测使用了不同的数据、权重或配置，请指定新的 --output-dir：{output}")
        return metrics
    try:
        return evaluate_to_directory(config, data, output, adapter=adapter, limit=limit)
    finally:
        release_cuda()


def evaluate_to_directory(config, data, output, *, adapter=None, limit=None, submission=False) -> dict:
    from step1_LLM.judge_eval import evaluate_preferences

    if output.exists():
        raise FileExistsError(f"评测目录不完整，请指定新的 --output-dir：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".eval-", dir=output.parent) as scratch:
        staged = Path(scratch) / "result"
        metrics = evaluate_preferences(config, data, staged, adapter=adapter,
                                       batch_size=config["preference"]["eval_batch_size"], limit=limit,
                                       submission=staged / "submission.csv" if submission else None)
        staged.rename(output)
    return metrics


def task_records(data: Path) -> list[dict]:
    from Datasets.human_preference.task import load_records

    return load_records(data)


def trained_config(adapter: Path, stage: str = "SFT") -> dict:
    summary = adapter / "run_summary.json"
    if not summary.is_file() or not (adapter / "adapter_model.safetensors").is_file():
        step = 2 if stage == "SFT" else 3
        raise FileNotFoundError(f"找不到已训练的偏好 {stage} LoRA：{adapter}。请先运行 --step {step}。")
    record = read_json(summary)
    if record.get("status") != "completed" or not record.get("update", {}).get("nonzero_update"):
        raise ValueError(f"偏好 {stage} 未完成参数更新：{adapter}")
    return record["config"]


def train_preference_sft(config: dict, data: Path, run_dir: Path, adapter: Path) -> None:
    from step2_SFT.train import run

    if adapter.exists():
        if trained_config(adapter) != prepare.resolve_model_config(config):
            raise ValueError(f"已有偏好 SFT 配置不同，请指定新的 --output-dir：{run_dir}")
        return
    snapshot = run_dir / "sft_config.json"
    prepare.write_json(snapshot, config)
    print(f"[SFT] 开始训练，优化器步数 {config['sft']['max_steps']}", flush=True)
    try:
        run(Namespace(config=snapshot, data=data, output=adapter, log_dir=adapter.parent,
                      max_steps=None, limit=None))
    finally:
        release_cuda()
    trained_config(adapter)
    # Store the Hub identity so the adapter can also be loaded from a Kaggle input.
    adapter_config = adapter / "adapter_config.json"
    settings = read_json(adapter_config)
    settings["base_model_name_or_path"] = config["model"]["name"]
    settings["revision"] = config["model"]["revision"]
    prepare.write_json(adapter_config, settings)


def export_preference(config: dict, adapter: Path, test: Path, run_dir: Path, destination: Path | None) -> Path:
    data_hash = file_hash(test)
    adapter_files = adapter_hashes(adapter)
    key = hashlib.sha256(f"{data_hash}:{adapter_files['adapter_model.safetensors']}".encode()).hexdigest()[:16]
    output = run_dir / "Submissions" / key
    submission = output / "submission.csv"
    if submission.is_file() and (output / "predictions.jsonl").is_file():
        metrics = read_json(output / "metrics.json")
        if (metrics["data_sha256"] != data_hash or metrics["adapter_files_sha256"] != adapter_files
            or metrics["config_sha256"] != json_hash(config)):
            raise ValueError(f"现有提交记录与数据、模型配置或权重不一致：{output}")
    else:
        evaluate_to_directory(config, test, output, adapter=adapter, submission=True)
    if destination is not None and destination != submission:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with TemporaryDirectory(prefix=".submission-", dir=destination.parent) as scratch:
                staged = Path(scratch) / "submission.csv"
                shutil.copyfile(submission, staged)
                try:
                    os.link(staged, destination)
                except FileExistsError:
                    pass
        if file_hash(destination) != file_hash(submission):
            raise FileExistsError(f"提交输出已有不同内容，请指定新的 --submission-file：{destination}")
        return destination
    return submission


def run_tuning(result: Path, sft_adapter: Path, *, algorithm: str, max_steps=None,
               eval_limit=None, selected_model=None) -> tuple[dict, Path]:
    if algorithm not in PREFERENCE_TRAINERS:
        raise ValueError(f"偏好任务不支持算法：{algorithm}")
    module, function, config_key = PREFERENCE_TRAINERS[algorithm]
    sft = trained_config(sft_adapter)
    if selected_model is not None:
        validate_selected_model(sft, selected_model)
    data = {key: result / "data" / f"{key}.jsonl" for key in ("train", "val")}
    if {name: file_hash(path) for name, path in data.items()} != sft["data_sha256"]:
        raise ValueError("当前训练或验证划分与 SFT 记录不一致")
    source = RL_SOURCE / algorithm / "Configs" / sft["selected_model"] / "human_preference.json"
    params = read_json(source)
    if max_steps is not None:
        params["max_steps"] = max_steps
    config = {**sft, config_key: params, "sft_adapter_files_sha256": adapter_hashes(sft_adapter)}
    sft_metrics = evaluate_preference(sft, data["val"], result / "SFT/eval", adapter=sft_adapter, limit=eval_limit)
    stage = result / algorithm
    adapter = stage / "adapter"
    if stage.exists():
        if trained_config(adapter, algorithm) != config:
            raise ValueError(f"已有偏好 {algorithm} 的配置或 SFT 权重与本次不同：{stage}")
    else:
        environment = prepare.check_environment()
        print(f"[{algorithm}] {environment['gpu']}，从 SFT 继续训练 {params['max_steps']} 步", flush=True)
        trainer = getattr(import_module(module), function)
        try:
            with TemporaryDirectory(prefix=f".{algorithm.lower()}-", dir=result) as scratch:
                staged = Path(scratch) / algorithm
                trainer(config, data_path=data["train"], output=staged / "adapter",
                        adapter=sft_adapter, log_dir=staged)
                trained_config(staged / "adapter", algorithm)
                staged.rename(stage)
        finally:
            release_cuda()
    metrics = evaluate_preference(config, data["val"], stage / "eval", adapter=adapter, limit=eval_limit)
    print(f"[{algorithm}] n={metrics['count']} Log Loss={metrics['log_loss']:.6f} "
          f"(SFT {sft_metrics['log_loss']:.6f})", flush=True)
    return config, adapter


def run_preference(args: Namespace, parser: ArgumentParser) -> None:
    if args.model not in PREFERENCE_MODELS:
        parser.error("human_preference 支持：" + "、".join(PREFERENCE_MODELS))
    if args.step not in ("prepare", "1", "2", "3", "all", "submit"):
        parser.error("human_preference 的 --step 支持 prepare、1、2、3、all、submit")
    if args.algorithm not in (None, *PREFERENCE_TRAINERS, "all", "both") or (args.algorithm and args.step not in ("3", "all", "submit")):
        parser.error("偏好任务的 --algorithm GRPO/PPO/DPO/all/both 支持 --step 3、all、submit")
    for key in ("max_steps", "train_limit", "val_limit"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} 必须大于 0")
    if (args.test_csv is not None or args.submission_file is not None) and args.step not in ("3", "all", "submit"):
        parser.error("--test-csv 和 --submission-file 需要 --step 3、all 或 submit")
    if args.step == "3" and any(value is not None for value in (args.train_limit, args.val_limit)):
        parser.error("--step 3 使用目标实验目录中已有的 SFT 数据划分。用 --output-dir 选择该目录。")

    raw = (args.input_dir or ROOT / "Datasets" / args.dataset / "raw").resolve()
    result = (args.output_dir or ROOT / "Outputs" / args.dataset / args.model).resolve()
    data = result / "data"
    adapter = result / "SFT" / "adapter"
    test_csv = (args.test_csv or raw / "test.csv").resolve()
    destination = args.submission_file.resolve() if args.submission_file is not None else None
    algorithms = selected_algorithms(args.algorithm, default="GRPO" if args.step == "3" else None)
    if args.dry_run:
        print(json.dumps({"dataset": args.dataset, "model": args.model, "stage": args.step,
                          "algorithm": args.algorithm or ("GRPO" if args.step == "3" else "SFT"),
                          "post_training_sequence": algorithms,
                          "post_training_source": str(adapter),
                          "train_csv": str(raw / "train.csv"), "test_csv": str(test_csv),
                          "output": str(result)}, ensure_ascii=False, indent=2))
        return

    if args.step == "submit":
        if not test_csv.is_file():
            parser.error(f"找不到测试 CSV：{test_csv}。请在 --test-csv 指定 Kaggle 测试文件。")
        stages = algorithms or ["SFT"]
        for stage in stages:
            adapter = result / stage / "adapter"
            config = trained_config(adapter, stage)
            validate_selected_model(config, args.model)
            target = submission_destination(destination, stage, multiple=len(stages) > 1)
            try:
                submission = export_preference(config, adapter, test_csv, result, target)
            finally:
                release_cuda()
            print(f"[Submission {stage}] {submission}", flush=True)
        return

    if args.step == "3":
        if (args.test_csv is not None or destination is not None) and not test_csv.is_file():
            parser.error(f"找不到测试 CSV：{test_csv}")
        sft_adapter = adapter
        for algorithm in algorithms:
            config, tuned_adapter = run_tuning(result, sft_adapter, algorithm=algorithm,
                                               max_steps=args.max_steps, eval_limit=args.eval_limit,
                                               selected_model=args.model)
            if args.test_csv is not None or destination is not None:
                target = submission_destination(destination, algorithm, multiple=len(algorithms) > 1)
                try:
                    submission = export_preference(config, tuned_adapter, test_csv, result, target)
                finally:
                    release_cuda()
                print(f"[Submission {algorithm}] {submission}", flush=True)
        return

    from Datasets.human_preference.task import prepare_data

    config = experiment_config(args.model, args.max_steps)
    for path in (raw / "train.csv", test_csv):
        if not path.is_file():
            parser.error(f"找不到 {path}。请解压 Kaggle 数据后指定 --input-dir。")
    manifest = prepare_data(data, train_csv=raw / "train.csv", test_csv=test_csv, seed=config["seed"],
                            train_limit=args.train_limit, val_limit=args.val_limit)
    print("[Data] " + ", ".join(f"{key}={value['count']}" for key, value in manifest["splits"].items()), flush=True)
    if args.step == "prepare":
        return
    config["data_sha256"] = {key: manifest["splits"][key]["sha256"] for key in ("train", "val")}
    prepare_preference_model(config)
    val = data / "val.jsonl"
    if args.step in ("1", "all"):
        baseline = evaluate_preference(config, val, result / "LLM" / "eval", limit=args.eval_limit)
        print(f"[Base] n={baseline['count']} Log Loss={baseline['log_loss']:.6f}", flush=True)
    if args.step in ("2", "all"):
        if not adapter.exists():
            environment = prepare.check_environment()
            print(f"[SFT] {environment['gpu']}, CUDA {environment['torch_cuda']}", flush=True)
        train_preference_sft(config, data / "train.jsonl", result, adapter)
        metrics = evaluate_preference(config, val, result / "SFT" / "eval", adapter=adapter, limit=args.eval_limit)
        print(f"[SFT] n={metrics['count']} Log Loss={metrics['log_loss']:.6f}", flush=True)
    if args.step == "all":
        sft_adapter = adapter
        for algorithm in algorithms or ["SFT"]:
            if algorithm != "SFT":
                config, adapter = run_tuning(result, sft_adapter, algorithm=algorithm,
                                             eval_limit=args.eval_limit, selected_model=args.model)
            target = submission_destination(destination, algorithm, multiple=len(algorithms) > 1)
            try:
                submission = export_preference(config, adapter, data / "test.jsonl", result, target)
            finally:
                release_cuda()
            print(f"[Submission {algorithm}] {submission}", flush=True)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--algorithm", choices=(*TRAINERS, "both", "all"))
    parser.add_argument("--step", choices=("1", "2", "3", "all", "compare", "test", "prepare", "submit"), default=STEP)
    parser.add_argument("--eval-limit", type=int, help="限制验证题数。省略时评估完整验证集。")
    parser.add_argument("--dry-run", action="store_true")
    preference = parser.add_argument_group("human_preference 任务")
    preference.add_argument("--input-dir", type=Path, help="包含 Kaggle train.csv 和 test.csv 的目录。")
    preference.add_argument("--output-dir", type=Path, help="偏好任务的实验目录。")
    preference.add_argument("--train-limit", type=int, help="限制偏好训练样本数。")
    preference.add_argument("--val-limit", type=int, help="限制偏好验证样本数。")
    preference.add_argument("--max-steps", type=int, help="覆盖偏好 SFT 或单独运行 step 3 时的优化器步数。")
    preference.add_argument("--test-csv", type=Path, help="为 submit 阶段指定实际测试 CSV。")
    preference.add_argument("--submission-file", type=Path, help="指定提交文件的输出位置。")
    args = parser.parse_args()
    if args.eval_limit is not None and args.eval_limit < 1:
        parser.error("--eval-limit 必须大于 0")
    args.model = MODEL_ALIASES.get(args.model, args.model)
    if args.dataset == "human_preference":
        run_preference(args, parser)
        return
    args.algorithm = args.algorithm or ALGORITHM
    if args.step in ("prepare", "submit") or any(
        getattr(args, key) is not None for key in
        ("input_dir", "output_dir", "train_limit", "val_limit", "max_steps", "test_csv", "submission_file")
    ):
        parser.error("偏好任务参数仅支持 --dataset human_preference")
    data, model_file, paths = get_paths(args)
    algorithms = {"all": list(TRAINERS), "both": ["GRPO", "PPO"]}.get(args.algorithm, [args.algorithm])
    steps = {"1": ["LLM"], "2": ["SFT"], "3": algorithms,
             "all": ["LLM", "SFT", *algorithms], "compare": [], "test": []}[args.step]
    dataset_output = ROOT / "Outputs" / args.dataset
    if args.step == "test":
        if args.eval_limit is not None:
            parser.error("最终测试要求评估完整测试集")
        if args.dry_run:
            print(json.dumps({method: str(paths[method] / "eval_final")
                              for method in ("LLM", "SFT", *algorithms)}, ensure_ascii=False, indent=2))
            return
        prepare.dataset_task({"dataset": args.dataset}).prepare_data(data)
        evaluate_final(data, paths, algorithms)
        return
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
    if args.dataset == "ToolACE":
        base["evaluation"]["batch_size"] = 16
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
