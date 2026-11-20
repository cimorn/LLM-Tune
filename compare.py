"""比较同一数据集上、同一基座模型的已完成实验。"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory


STAGES = {
    "baseline": Path("LLM"),
    "sft": Path("SFT"),
    "grpo": Path("GRPO"),
    "ppo": Path("PPO"),
    "dpo": Path("DPO"),
}
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
END_GENERATED_RESULTS = "<!-- END GENERATED RESULTS -->"
README_TEXT = {
    "README.md": {
        "title": "{dataset} 评估结果",
        "switch": "中文 · [English](README.en.md)",
        "results_columns": "| 模型 | 方法 | 数据划分 | 正确 / 总数 | 成功率 |",
        "validation_summary": "验证集，共 {count} 题",
        "validation_columns": "| 模型 | 方法 | 正确 / 总数 | 成功率 |",
        "comparison_title": "与 SFT 的配对比较",
        "comparison_columns": "| 模型 | 方法 | 成功率变化（百分点） | 95% 置信区间（百分点） |",
        "charts_title": "成功率与训练曲线",
        "charts_columns": "| 模型 | 验证集成功率 | 训练曲线 |",
        "captions": ("{model} 的验证集成功率", "{model} 的训练曲线"),
    },
    "README.en.md": {
        "title": "{dataset} evaluation results",
        "switch": "[中文](README.md) · English",
        "results_columns": "| Model | Method | Split | Correct / Total | Success rate |",
        "validation_summary": "Validation set, {count} questions",
        "validation_columns": "| Model | Method | Correct / Total | Success rate |",
        "comparison_title": "Paired comparison with SFT",
        "comparison_columns": "| Model | Method | Change in success rate (pp) | 95% confidence interval (pp) |",
        "charts_title": "Success rates and training curves",
        "charts_columns": "| Model | Validation success rate | Training curves |",
        "captions": ("{model} validation success rate", "{model} training curves"),
    },
}
LABELS = {"baseline": "Base model", "sft": "SFT", "grpo": "SFT + GRPO", "ppo": "SFT + PPO", "dpo": "SFT + DPO"}
COLORS = {"baseline": "#8797A4", "sft": "#087F83", "grpo": "#7255A4", "ppo": "#C17835", "dpo": "#95704D"}
METRICS = (
    "count", "success_rate", "format_rate", "schema_valid_rate", "mean_generated_tokens",
    "latency_p50_s", "latency_p90_s", "peak_memory_allocated_mb", "error_counts",
)
PAIRED_FIELDS = ("prompt", "rendered_prompt_sha256", "expected_call", "expected_answer", "sample_seed")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(stage_dir: Path, label: str, *, eval_subdir: str = "eval") -> dict | None:
    eval_dir = stage_dir / eval_subdir
    metrics_path = eval_dir / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.is_file() else None
    marker = stage_dir / "run.json"
    metadata = metrics.get("run") if metrics is not None else None
    if metadata is None and marker.is_file():
        metadata = load_json(marker)
    if metadata is None or metadata.get("status") != "completed":
        return None
    if metrics is None:
        raise FileNotFoundError(f"缺少已完成阶段的评估指标：{metrics_path}")
    with (eval_dir / "predictions.jsonl").open(encoding="utf-8") as predictions_file:
        predictions = [json.loads(line) for line in predictions_file if line.strip()]
    by_id = {row["id"]: row for row in predictions}
    if not predictions or len(predictions) != metrics.get("count") or len(by_id) != len(predictions):
        raise ValueError(f"评估样本数量不符或题目 ID 重复：{eval_dir}")
    if not metrics.get("data_sha256") or not metrics.get("inference_protocol"):
        raise ValueError(f"缺少数据指纹或评估协议：{eval_dir}")
    for row in predictions:
        if type(row.get("correct")) is not bool or not all(field in row for field in PAIRED_FIELDS):
            raise ValueError(f"逐题结果缺少判分或配对字段：{eval_dir} / {row['id']}")
        if row.get("data_sha256", metrics["data_sha256"]) != metrics["data_sha256"]:
            raise ValueError(f"逐题结果的数据指纹不一致：{eval_dir} / {row['id']}")
    if not math.isclose(statistics.mean(row["correct"] for row in predictions), metrics["success_rate"], abs_tol=1e-10):
        raise ValueError(f"成功率与逐题判分不一致：{eval_dir}")
    return {"label": label, "directory": stage_dir, "metrics": metrics, "by_id": by_id, "metadata": metadata}


def load_model_runs(model_dir: Path) -> dict[str, dict]:
    runs = {}
    for label, stage in STAGES.items():
        run = load_run(model_dir / stage, label)
        if run is not None:
            runs[label] = run
    return runs


def validate_comparable(runs: dict[str, dict]) -> None:
    if "baseline" not in runs:
        raise ValueError("没有已完成的基座评估。")
    reference = runs["baseline"]
    for label, run in runs.items():
        if run["metrics"]["data_sha256"] != reference["metrics"]["data_sha256"]:
            raise ValueError(f"{label} 的评估数据与基座评估不同。")
        if run["metrics"]["inference_protocol"] != reference["metrics"]["inference_protocol"]:
            raise ValueError(f"{label} 的模型版本或评估协议与基座评估不同。")
        if run["by_id"].keys() != reference["by_id"].keys():
            raise ValueError(f"{label} 的逐题评估范围与基座评估不同。")
        for item_id, expected in reference["by_id"].items():
            for field in PAIRED_FIELDS:
                if run["by_id"][item_id][field] != expected[field]:
                    raise ValueError(f"{label} 评估题目 {item_id} 的 {field} 与基座评估不同。")

    for label in ("grpo", "ppo", "dpo"):
        if label not in runs:
            continue
        sft = runs.get("sft")
        adapter = sft["metadata"].get("adapter_sha256") if sft else None
        if not isinstance(adapter, str) or not adapter or runs[label]["metadata"].get("sft_adapter_sha256") != adapter:
            raise ValueError(f"{label} 的初始权重与当前已完成的 SFT 权重不一致。")


def percentile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_bootstrap(reference: dict, candidate: dict, *, seed: int = 42, samples: int = 1000) -> dict:
    differences = [int(candidate["by_id"][item_id]["correct"]) - int(row["correct"])
                   for item_id, row in reference["by_id"].items()]
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return {
        "reference": reference["label"],
        "candidate": candidate["label"],
        "delta": statistics.mean(differences),
        "ci95": [percentile(means, 0.025), percentile(means, 0.975)],
        "seed": seed,
        "resamples": samples,
        "scope": "paired evaluation examples only; does not estimate variation across training seeds",
    }


def compare_to_sft(runs: dict[str, dict]) -> dict:
    validate_comparable(runs)
    return {label: paired_bootstrap(runs["sft"], runs[label])
            for label in ("grpo", "ppo", "dpo") if label in runs}


def training_series(run: dict) -> tuple[list[int], list[float]]:
    path = run["directory"] / "train_log.json"
    history = load_json(path)
    fields = ("loss",) if run["label"] in ("sft", "dpo") else ("reward", "rewards/strict_tool_reward/mean")
    for field in fields:
        points = sorted((int(row["step"]), float(row[field])) for row in history if "step" in row and field in row)
        if not points:
            continue
        if not all(math.isfinite(value) for _, value in points):
            raise ValueError(f"训练日志存在非有限指标：{path}")
        return [step for step, _ in points], [value for _, value in points]
    raise ValueError(f"训练日志缺少绘图指标：{path}")


def draw_figures(report: dict, curves: dict, destination: Path) -> list[Path]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    destination.mkdir()
    fig = Figure(figsize=(8.5, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    rows = report["runs"]
    rates = [row["success_rate"] for row in rows]
    bars = ax.bar([LABELS[row["label"]] for row in rows], rates,
                  color=[COLORS[row["label"]] for row in rows], width=0.62)
    for bar, row in zip(bars, rows):
        rate, count = row["success_rate"], row["count"]
        correct = round(rate * count)
        ax.annotate(f"{rate:.1%} ({correct}/{count})", (bar.get_x() + bar.get_width() / 2, rate),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom")
    ax.set(ylim=(0, 1.12), ylabel="Accuracy", title=f"{report['model']} | {report['dataset']}")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#E7EEF0")
    ax.set_axisbelow(True)
    comparison = destination / "model_comparison.png"
    fig.savefig(comparison, dpi=160)
    paths = [comparison]

    if curves:
        fig = Figure(figsize=(4.7 * len(curves), 4), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(ncols=len(curves), squeeze=False)[0]
        for ax, (label, (steps, values)) in zip(axes, curves.items()):
            ax.plot(steps, values, color=COLORS[label], linewidth=1.5, marker="." if len(steps) < 15 else None)
            ax.set_title(f"{label.upper()} training")
            ax.set_xlabel("Rollout" if label == "ppo" else "Optimizer step")
            ax.set_ylabel("Loss" if label in ("sft", "dpo") else "Mean rollout reward")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.grid(axis="y", color="#E7EEF0")
        learning = destination / "learning_curves.png"
        fig.savefig(learning, dpi=160)
        paths.append(learning)
    return paths


def markdown_table(header: str, rows: list[tuple[str, ...]]) -> list[str]:
    def visible_length(cell: str) -> int:
        cell = cell.strip()
        return 0 if re.fullmatch(r"!\[[^]]*\]\([^)]+\)", cell) else len(cell)

    columns = zip(*rows, strict=True) if rows else [()] * len(header.strip("|").split("|"))
    alignment = [":---:" if len({visible_length(cell) for cell in column}) == 1 else ":---"
                 for column in columns]
    return [header, "| " + " | ".join(alignment) + " |",
            *("| " + " | ".join(row) + " |" for row in rows)]


def dataset_readmes(
    dataset_dir: Path, current_runs: dict[str, dict], current_report: dict,
) -> dict[str, str]:
    current = current_report["model"]
    results = []
    comparisons = []
    charts = []
    for model_dir in sorted(dataset_dir.iterdir(), key=lambda path: (path.name.casefold(), path.name)):
        model = model_dir.name
        if not model_dir.is_dir() or not MODEL_NAME.fullmatch(model) or model.endswith("."):
            continue
        runs = current_runs if model == current else load_model_runs(model_dir)
        if not runs:
            continue
        comparison_dir = model_dir / "Comparison"
        images = []
        for path in ("figures/model_comparison.png", "figures/learning_curves.png"):
            exists = path in current_report["figures"] if model == current else (comparison_dir / path).is_file()
            images.append(f"{model}/Comparison/{path}" if exists else None)
        if any(images):
            charts.append((model, images))
        for label, run in runs.items():
            rows = run["by_id"].values()
            correct = sum(row["correct"] for row in rows)
            split = ", ".join(sorted({str(row.get("split") or "—") for row in rows}))
            results.append((model, LABELS[label], split, correct, len(rows)))
        if model == current:
            paired = current_report["paired_bootstrap"]
        else:
            try:
                paired = compare_to_sft(runs)
            except ValueError:
                # 仍在汇总表中列出单次评估，配对比较要求相同评估协议与 SFT 权重。
                continue
        for result in paired.values():
            low, high = result["ci95"]
            comparisons.append((model, result["candidate"].upper(), f"{result['delta'] * 100:+.2f}",
                                f"[{low * 100:+.2f}, {high * 100:+.2f}]"))
    scopes = {(split, count) for _, _, split, _, count in results}
    shared_split, shared_count = next(iter(scopes)) if len(scopes) == 1 else (None, None)
    show_split = shared_split != "val"
    result_rows = []
    for model, method, split, correct, count in results:
        split_cells = (split,) if show_split else ()
        result_rows.append((model, method, *split_cells, f"{correct} / {count}", f"{correct / count:.2%}"))

    readmes = {}
    for filename, labels in README_TEXT.items():
        title = labels["title"].format(dataset=dataset_dir.name)
        lines = [f"# {title}", "", labels["switch"], ""]
        if show_split:
            lines.extend(markdown_table(labels["results_columns"], result_rows))
        else:
            lines.extend([labels["validation_summary"].format(count=shared_count), "",
                          *markdown_table(labels["validation_columns"], result_rows)])
        if comparisons:
            lines.extend([
                "", "<br>", "", f"## {labels['comparison_title']}", "",
                *markdown_table(labels["comparison_columns"], comparisons),
            ])
        if charts:
            chart_rows = []
            for model, images in charts:
                accuracy, training = (
                    f"![{caption.format(model=model)}]({path})" if path else "—"
                    for path, caption in zip(images, labels["captions"], strict=True)
                )
                chart_rows.append((model, accuracy, training))
            lines.extend([
                "", "<br>", "", f"## {labels['charts_title']}", "",
                *markdown_table(labels["charts_columns"], chart_rows),
            ])

        readme = dataset_dir / filename
        suffix = "\n"
        if readme.is_file():
            # 各语言独立保留生成标记后的手写内容及换行。
            _, marker, authored = readme.read_bytes().decode("utf-8").partition(END_GENERATED_RESULTS)
            if marker:
                suffix = authored
        generated = "\n".join(lines).rstrip("\n")
        readmes[filename] = f"{generated}\n\n{END_GENERATED_RESULTS}{suffix}"
    return readmes


def compare_dataset(dataset_dir: Path, model: str) -> dict:
    """校验已完成阶段的评估结果，输出模型对比及数据集汇总。"""
    if not MODEL_NAME.fullmatch(model) or model.endswith("."):
        raise ValueError(f"无效的模型目录名：{model}")
    dataset_dir = Path(dataset_dir)
    model_dir = dataset_dir / model
    runs = load_model_runs(model_dir)
    paired = compare_to_sft(runs)
    curves = {label: training_series(run) for label, run in runs.items() if label != "baseline"}
    reference = runs["baseline"]["metrics"]
    report = {
        "dataset": dataset_dir.name,
        "model": model,
        "data_sha256": reference["data_sha256"],
        "inference_protocol": reference["inference_protocol"],
        "runs": [
            {"label": label, "path": (run["directory"] / "eval").relative_to(dataset_dir).as_posix(),
             **{key: run["metrics"].get(key) for key in METRICS}}
            for label, run in runs.items()
        ],
        "paired_bootstrap": paired,
    }

    output = model_dir / "Comparison"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".comparison-", dir=dataset_dir) as temporary:
        scratch = Path(temporary)
        figures = draw_figures(report, curves, scratch / "figures")
        report["figures"] = [f"figures/{path.name}" for path in figures]
        readmes = dataset_readmes(dataset_dir, runs, report)
        for filename, readme in readmes.items():
            (scratch / filename).write_bytes(readme.encode("utf-8"))
        (output / "figures").mkdir(exist_ok=True)
        for figure in figures:
            figure.replace(output / "figures" / figure.name)
        if not curves:
            (output / "figures" / "learning_curves.png").unlink(missing_ok=True)
        for filename in readmes:
            (scratch / filename).replace(dataset_dir / filename)
    return report
