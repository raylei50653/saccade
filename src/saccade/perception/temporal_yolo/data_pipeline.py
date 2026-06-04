"""
Standard training data pipeline: prepare reusable data before training.

Convention:
  Phase 1 — PREPARE:  preload raw images / precompute frozen encoder outputs
  Phase 2 — TRAIN:    training loop reads from cache, only touch trainable params
  Phase 3 — SAVE:     checkpointing

Provides:
  DataPreloader  — thread-pool JPEG decode → uint8 tensors in RAM
  FeatureCache   — precompute frozen model outputs, persist to disk

Import:
    from saccade.perception.temporal_yolo.data_pipeline import (
        DataPreloader,
        FeatureCache,
    )
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase 1a — Preload raw images to RAM
# ---------------------------------------------------------------------------


class DataPreloader:
    """Decode a list of image paths into uint8 tensors in RAM.

    Uses ThreadPoolExecutor for parallel JPEG decode. Resize is deferred —
    callers can request a transform per-batch to keep RAM usage low.

    Usage:
        preloader = DataPreloader(paths, num_workers=16)
        preloader.load()
        # In training loop:
        img_uint8 = preloader[path]
        img_float = transform_to_gpu(img_uint8, device)
    """

    def __init__(
        self,
        paths: list[Path],
        num_workers: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self._paths = paths
        self._num_workers = num_workers or min(os.cpu_count() or 4, 16)
        self._transform = transform
        self._cache: dict[Path, torch.Tensor] = {}
        self._loaded = False

    def __getitem__(self, path: Path) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError("DataPreloader.load() must be called first")
        return self._cache[path]

    def __len__(self) -> int:
        return len(self._cache)

    def load(self) -> DataPreloader:
        import torchvision.io as tv_io

        print(f"[Prepare] Preloading {len(self._paths)} images to RAM...", flush=True)
        t0 = time.perf_counter()

        def _decode(p: Path) -> tuple[Path, torch.Tensor]:
            img = tv_io.read_image(str(p))
            if self._transform is not None:
                img = self._transform(img)
            return p, img

        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            for path, tensor in pool.map(_decode, self._paths):
                self._cache[path] = tensor

        elapsed = time.perf_counter() - t0
        total_mb = sum(t.numel() * t.element_size() for t in self._cache.values()) / (
            1024 * 1024
        )
        print(f"  {len(self._cache)} images, {total_mb:.1f} MB in {elapsed:.1f}s")
        self._loaded = True
        return self


# ---------------------------------------------------------------------------
# Phase 1b — Precompute frozen encoder outputs
# ---------------------------------------------------------------------------


class FeatureCache:
    """Run a frozen encoder on preloaded images once, cache outputs to disk/RAM.

    The cache file is a .pt dict mapping path → tensor. Subsequent runs
    skip the encoder entirely if the cache file exists.

    Usage:
        cache = FeatureCache("runs/jde/precomputed.pt")
        if not cache.exists():
            pooled = cache.compute(
                preloader, paths, device,
                encoder_fn=lambda imgs, dev: teacher_forward(imgs, dev),
                batch_size=128,
            )
        pooled_batch = cache.get_batch(batch_paths, device)
    """

    def __init__(self, cache_path: str | Path):
        self._path = Path(cache_path)
        self._data: dict[Path, torch.Tensor] = {}

    def exists(self) -> bool:
        return self._path.exists()

    def load(self) -> FeatureCache:
        if self._path.exists():
            print(f"[Prepare] Loading cached features from {self._path}...", flush=True)
            self._data = torch.load(self._path, map_location="cpu", weights_only=False)
            print(f"  {len(self._data)} entries loaded", flush=True)
        return self

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._data, self._path)
        total_mb = sum(t.numel() * t.element_size() for t in self._data.values()) / (
            1024 * 1024
        )
        print(f"  Saved {len(self._data)} entries ({total_mb:.1f} MB) to {self._path}")

    def compute(
        self,
        preloader: DataPreloader,
        paths: list[Path],
        device: torch.device,
        encoder_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
        batch_size: int = 128,
    ) -> dict[Path, torch.Tensor]:
        """Run encoder_fn on all images and store results.

        encoder_fn signature: (imgs_uint8_cpu: (B,C,H,W), device) → (B, dim) on device
        """
        total = len(paths)
        print(f"[Prepare] Precomputing features for {total} images...", flush=True)
        t0 = time.perf_counter()

        for i in range(0, total, batch_size):
            batch_paths = paths[i : i + batch_size]
            imgs_uint8 = torch.stack([preloader[p] for p in batch_paths])

            with torch.no_grad():
                outputs = encoder_fn(imgs_uint8, device)

            for j, p in enumerate(batch_paths):
                self._data[p] = outputs[j].cpu()

            if (i // batch_size) % 20 == 0:
                print(f"  {min(i + batch_size, total)}/{total}", flush=True)

        elapsed = time.perf_counter() - t0
        print(f"  {len(self._data)} features in {elapsed:.1f}s", flush=True)
        self.save()
        return self._data

    def get_batch(
        self,
        batch_paths: list[Path],
        device: torch.device,
    ) -> torch.Tensor:
        """Retrieve precomputed features for a batch of paths."""
        return torch.stack(
            [self._data[p].to(device, non_blocking=True) for p in batch_paths]
        )

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Image transform helpers
# ---------------------------------------------------------------------------


def resize_letterbox(
    img: torch.Tensor,
    img_size: int,
) -> torch.Tensor:
    """Resize a uint8 (C, H, W) tensor to fit img_size×img_size with letterbox.

    Returns a uint8 tensor — caller should convert to float and move to device.
    """
    import torchvision.transforms.functional as TF

    img = TF.resize(img, [img_size, img_size], antialias=True)
    c, h, w = img.shape
    canvas = torch.zeros(c, img_size, img_size, dtype=img.dtype)
    oh = (img_size - h) // 2
    ow = (img_size - w) // 2
    canvas[:, oh : oh + h, ow : ow + w] = img
    return canvas


def resize_stretch_batch_gpu(
    imgs_uint8: torch.Tensor,
    img_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Batch GPU stretch-resize to exactly img_size×img_size (NO letterbox).

    Stretches images to fill the entire canvas — no padding. Use for person
    crops where the entire image is foreground (Market-1501).
    """
    imgs = imgs_uint8.to(device).float().div_(255.0)
    B, C, H, W = imgs.shape
    if H == img_size and W == img_size:
        return imgs
    return F.interpolate(
        imgs, size=(img_size, img_size), mode="bilinear", align_corners=False
    )
