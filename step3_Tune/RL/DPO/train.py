"""Train sigmoid DPO from SFT, with a fixed, auditable initial reference policy."""
from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import torch
from peft import PeftModel

from step1_LLM.evaluate import adapter_hashes, sha256_file
from step1_LLM.judge_eval import PreferenceEncoder
from step1_LLM.prepare import load_model_and_tokenizer, read_config, seed_everything, write_json
from step2_SFT.train import (load_training_records, measure_update, prepare_output,
                            project_path, snapshot_trainable, state_fingerprint)
from .run.core import DPOSettings, fit_pairs
from .run.pairs import build_pairs
from .run.scoring import PreferencePairScorer, ToolPairScorer


def _save_policy(path, model, tokenizer, config):
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    settings_path = path / "adapter_config.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(base_model_name_or_path=config["model"]["name"], revision=config["model"]["revision"])
    write_json(settings_path, settings)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def train_dpo(config: dict, *, data_path: Path, output: Path, adapter: Path,
              max_steps: int | None = None, limit: int | None = None,
              log_dir: Path | None = None) -> dict:
    """Public API for all three tasks. Human preference uses config['preference_dpo']."""
    config = deepcopy(config)
    preference = config.get("dataset") == "human_preference"
    key = "preference_dpo" if preference else "dpo"
    values = dict(config[key])
    if max_steps is not None:
        values["max_steps"] = max_steps
        values["save_at_steps"] = [step for step in values.get("save_at_steps", []) if step <= max_steps]
    cfg = DPOSettings(**values)
    cfg.validate()
    # Keep the supplied config unchanged for existing experiment identity checks.
    if max_steps is not None:
        config[key] = values
    data_path, output, adapter = map(Path, (data_path, output, adapter))
    logs = Path(log_dir) if log_dir is not None else output
    records = load_training_records(data_path, limit, config)
    source_hashes = adapter_hashes(adapter)
    if "adapter_model.safetensors" not in source_hashes:
        raise FileNotFoundError(f"Missing SFT adapter: {adapter}")
    if source_hashes != config.get("sft_adapter_files_sha256", source_hashes):
        raise ValueError("SFT adapter files differ from the experiment configuration")
    expected_hash = config.get("sft_adapter_sha256")
    if expected_hash is not None and expected_hash != source_hashes["adapter_model.safetensors"]:
        raise ValueError("SFT adapter weight hash differs from the experiment configuration")
    data_hash = sha256_file(data_path)
    if data_hash != config.get("data_sha256", {}).get("train", data_hash):
        raise ValueError("Training data differs from the SFT split recorded in the configuration")
    seed = int(config.get("seed", 42))
    pairs = build_pairs(records, config, seed)
    if not torch.cuda.is_available():
        raise RuntimeError("DPO model training requires CUDA; CPU helper tests do not")
    prepare_output(output)
    logs.mkdir(parents=True, exist_ok=True)
    compact_pairs = [{key: value for key, value in pair.items() if key != "record"} for pair in pairs]
    _write_jsonl(logs / "preference_pairs.jsonl", compact_pairs)
    summary = dict(
        stage="preference_dpo" if preference else "dpo", status="running", config=config,
        actual_dpo=asdict(cfg), data_path=str(data_path), data_sha256=data_hash,
        training_rows=len(records), sft_adapter=str(adapter), sft_adapter_files_sha256=source_hashes,
        output=str(output), pairs_sha256=sha256_file(logs / "preference_pairs.jsonl"),
        pair_protocol=compact_pairs[0]["pair_protocol"],
        loss_protocol="mean sigmoid DPO: -logsigmoid(beta*((logp_chosen-logp_rejected)-(ref_chosen-ref_rejected)))",
        reference_protocol="all scheduled pairs scored by the untouched SFT policy before any update; immutable cached logps",
        probability_protocol=("single A/B/C judge action; conditional label softmax; EOS excluded"
                              if preference else "sum of completion token logps including EOS; prompt/padding masked; no length normalization"),
        action_space=["A", "B", "C"] if preference else None,
        reference_kl="implicit reference-relative DPO objective; no additional sampled KL penalty",
    )
    write_json(output / "run_summary.json", summary)
    try:
        seed_everything(seed)
        model, tokenizer = load_model_and_tokenizer(config, adapter=adapter, trainable=True)
        if not isinstance(model, PeftModel):
            raise TypeError("DPO must continue an existing SFT LoRA adapter")
        model.eval()
        before = snapshot_trainable(model)
        if not before:
            raise ValueError("SFT adapter contains no trainable parameters")
        source_summary = adapter / "run_summary.json"
        expected = None
        if source_summary.is_file():
            source = json.loads(source_summary.read_text(encoding="utf-8"))
            expected = source.get("update", {}).get("after_sha256")
            summary["sft_source"] = dict(path=str(adapter), stage=source.get("stage"),
                                         optimizer_steps=source.get("optimizer_steps"), expected_adapter_sha256=expected)
        summary["initial_state_matches_sft"] = state_fingerprint(before) == expected if expected else None
        if summary["initial_state_matches_sft"] is False:
            raise ValueError("Loaded LoRA parameters do not match the SFT training fingerprint")
        summary["reference_policy_sha256"] = state_fingerprint(before)
        score = (PreferencePairScorer(model, PreferenceEncoder(tokenizer, config)) if preference else
                 ToolPairScorer(model, tokenizer, config, cfg))
        device = next(model.parameters()).device
        history = []
        torch.cuda.reset_peak_memory_stats(device)

        def save_reference(reference, schedule):
            _write_jsonl(logs / "reference_logps.jsonl", reference)
            write_json(logs / "training_schedule.json", {"seed": seed, "pair_ids": [p["id"] for p in pairs],
                                                        "optimizer_batches": schedule})
            summary["reference_logps_sha256"] = sha256_file(logs / "reference_logps.jsonl")
            summary["training_schedule_sha256"] = sha256_file(logs / "training_schedule.json")
            summary["reference_rows"] = len(reference)
            write_json(output / "run_summary.json", summary)

        def log_step(event, optimizer):
            history.append(event)
            write_json(logs / "train_log.json", history)
            if event["step"] == 1 or event["step"] % 16 == 0 or event["step"] == cfg.max_steps:
                print(f"[DPO] {event['step']}/{cfg.max_steps} loss={event['loss']:.6f} "
                      f"margin={event['reward_margin']:.4f}", flush=True)
            if event["step"] in cfg.save_at_steps:
                checkpoint = output / f"checkpoint-{event['step']}"
                _save_policy(checkpoint, model, tokenizer, config)
                torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
                write_json(checkpoint / "run_summary.json", {
                    **summary, "status": "checkpoint", "optimizer_steps": event["step"],
                    "update": measure_update(model, before), "parent_output": str(output),
                })

        result = fit_pairs(model, pairs, score, cfg, seed=seed, on_reference=save_reference, on_step=log_step)
        update = measure_update(model, before)
        if not update["finite_parameters"]:
            raise FloatingPointError("DPO produced non-finite adapter parameters")
        _save_policy(output, model, tokenizer, config)
        summary.update(
            status="completed" if update["nonzero_update"] else "completed_no_update",
            optimizer_steps=result["optimizer_steps"], update=update,
            reference_seconds=result["reference_seconds"], optimization_seconds=result["optimization_seconds"],
            training_seconds=result["reference_seconds"] + result["optimization_seconds"],
            training_pair_exposures=result["training_pair_exposures"],
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
            metrics={"first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
                     "last_reward_margin": history[-1]["reward_margin"]},
        )
        write_json(output / "run_summary.json", summary)
        return summary
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run_summary.json", summary)
        (logs / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run(args: Namespace) -> dict:
    log_dir = getattr(args, "log_dir", None)
    return train_dpo(read_config(project_path(args.config)), data_path=project_path(args.data),
                     output=project_path(args.output), adapter=project_path(args.sft_adapter),
                     max_steps=getattr(args, "max_steps", None), limit=getattr(args, "limit", None),
                     log_dir=project_path(log_dir) if log_dir is not None else None)
