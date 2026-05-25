"""
Shared training configuration for temporal YOLO scripts.

Provides a TrainingConfig dataclass and helper to add common argparse arguments.
Individual training scripts add model-specific args on top.

Import:
    from saccade.perception.temporal_yolo.train_config import (
        TrainingConfig,
        add_common_training_args,
        build_optimizer_and_scheduler,
        build_v8_detection_loss,
    )
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


@dataclass
class TrainingConfig:
    """Common training hyperparameters shared across temporal YOLO scripts."""

    data_root: str = "datasets/MOT17"
    yolo_weights: str = "models/yolo/yolo26s.pt"
    run_dir: str = "runs/temporal_yolo"
    resume: str = ""
    epochs: int = 30
    batch_size: int = 4
    img_size: int = 640
    clip_len: int = 4
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    lr_gate: float = 1e-3
    num_workers: int = 0
    seqs: str = ""
    save_every: int = 5
    compile: bool = False
    accum_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4
    seed: int = 42

    @property
    def seq_list(self) -> list[str] | None:
        s = self.seqs.strip()
        if not s:
            return None
        return [x.strip() for x in s.split(",")]


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """Register common CLI arguments on an ArgumentParser.

    Callers should create their own ArgumentParser, call this function,
    then add model-specific arguments on top.
    """
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--run-dir", default="runs/temporal_yolo")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-gate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)


def build_optimizer_and_scheduler(
    model: nn.Module,
    args: TrainingConfig | None = None,
    **kwargs: Any,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build AdamW + CosineAnnealingLR for the given model.

    If args is provided, uses its fields (lr, weight_decay, epochs).
    Additional kwargs can override individual settings:
        lr, weight_decay, epochs, eta_min_ratio

    Returns (optimizer, scheduler).
    """
    lr = kwargs.get("lr", args.lr if args else 1e-4)
    weight_decay = kwargs.get("weight_decay", args.weight_decay if args else 1e-4)
    epochs = kwargs.get("epochs", args.epochs if args else 30)
    eta_min_ratio = kwargs.get("eta_min_ratio", 0.01)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * eta_min_ratio,
    )
    return optimizer, scheduler


def build_v8_detection_loss(
    yolo_model: nn.Module,
    cls_weight: float = 0.5,
    box_weight: float = 7.5,
    dfl_weight: float = 1.5,
) -> Any:
    """Build v8DetectionLoss with patched model args.

    The Ultralytics v8DetectionLoss reads model.args at construction time.
    This helper patches model.args into a SimpleNamespace with the given
    loss weights and returns the criterion.
    """
    from ultralytics.utils.loss import v8DetectionLoss

    base_args = dict(yolo_model.args) if isinstance(yolo_model.args, dict) else {}
    base_args["box"] = box_weight
    base_args["cls"] = cls_weight
    base_args["dfl"] = dfl_weight
    yolo_model.args = SimpleNamespace(**base_args)

    return v8DetectionLoss(yolo_model)
