"""读取实验配置并从指定的 SFT 权重启动 GRPO。"""
from __future__ import annotations

from argparse import Namespace

from step1_LLM.prepare import read_config
from step2_SFT.train import project_path

from .run.trainer import train_grpo


def run(args: Namespace) -> dict:
    log_dir = getattr(args, "log_dir", None)
    return train_grpo(
        read_config(project_path(args.config)),
        data_path=project_path(args.data),
        output=project_path(args.output),
        adapter=project_path(args.sft_adapter),
        max_steps=args.max_steps,
        limit=args.limit,
        log_dir=project_path(log_dir) if log_dir is not None else None,
    )
