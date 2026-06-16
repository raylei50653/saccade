"""
Shared training utilities for temporal YOLO scripts.

Import:
    from saccade.perception.temporal_yolo.training_utils import (
        save_checkpoint,
        load_checkpoint,
        strip_compiled_keys,
    )
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def strip_compiled_keys(sd: dict[str, Any]) -> dict[str, Any]:
    """Remove _orig_mod. prefix inserted by torch.compile, making
    checkpoints compile-agnostic."""
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch without forcing unsupported deterministic kernels."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def parse_sequence_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_training_sequences(
    data_root: Path,
    requested: str,
    holdout: str,
    *,
    detector: str = "SDP",
) -> tuple[list[str] | None, list[str]]:
    """Resolve an explicit training split and remove held-out sequences."""
    holdout_seqs = parse_sequence_list(holdout)
    if requested:
        train_seqs = parse_sequence_list(requested)
    elif holdout_seqs:
        split_dir = data_root / "train"
        train_seqs = sorted(
            path.name
            for path in split_dir.iterdir()
            if path.is_dir() and path.name.endswith(f"-{detector}")
        )
    else:
        return None, []

    all_requested = sorted(set(train_seqs) | set(holdout_seqs))
    missing = [seq for seq in all_requested if not (data_root / "train" / seq).is_dir()]
    if missing:
        raise ValueError(f"Unknown MOT sequences: {missing}")

    overlap = sorted(set(train_seqs) & set(holdout_seqs))
    train_seqs = [seq for seq in train_seqs if seq not in holdout_seqs]
    if not train_seqs:
        raise ValueError("No training sequences remain after applying --holdout-seqs")
    if overlap:
        print(f"[Split] Removed held-out sequences from training: {overlap}")
    return train_seqs, holdout_seqs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state(
    data_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if data_generator is not None:
        state["dataloader"] = data_generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any],
    data_generator: torch.Generator | None = None,
) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if data_generator is not None and "dataloader" in state:
        data_generator.set_state(state["dataloader"])


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build one LR schedule: linear warmup followed by cosine decay."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be >= 1")
    if not 0 <= warmup_epochs <= total_epochs:
        raise ValueError("warmup_epochs must be between 0 and total_epochs")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    decay_epochs = max(total_epochs - warmup_epochs, 1)

    def lr_lambda(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        progress = min(
            max((epoch_index - warmup_epochs + 1) / decay_epochs, 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    state: dict[str, Any],
    run_dir: Path,
    epoch: int,
    is_best: bool = False,
) -> None:
    """Save state dict as latest.ckpt + epoch_N.ckpt, plus best.ckpt if is_best."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, run_dir / "latest.ckpt")
    torch.save(state, run_dir / f"epoch_{epoch:04d}.ckpt")
    if is_best:
        torch.save(state, run_dir / "best.ckpt")
    tag = " [BEST]" if is_best else ""
    print(f"  Saved epoch_{epoch:04d}.ckpt{tag}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = False,
) -> int:
    """Load checkpoint into model and optionally optimizer.

    Returns the next epoch to start from (resumed epoch + 1).
    Always strips _orig_mod. keys via strip_compiled_keys.
    Optimizer loading is best-effort — failure prints a warning.
    """
    print(f"[Resume] {path}  strict={strict}")
    state = torch.load(path, map_location="cpu", weights_only=False)

    sd = strip_compiled_keys(state.get("model", state.get("student", {})))
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(
            f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)})")

    if optimizer is not None:
        if "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception:
                print("  [Warn] Optimizer state not loaded")
        else:
            print("  [Warn] No optimizer state in checkpoint")

    prev_epoch = state.get("epoch", 0)
    best_loss = state.get("best_loss", float("inf"))
    print(f"  Resumed from epoch {prev_epoch}  best_loss={best_loss:.4f}")
    return prev_epoch + 1
