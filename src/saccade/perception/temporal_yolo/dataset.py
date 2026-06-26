"""
MOT17 Temporal Dataset — 供 TemporalYOLOHybrid 訓練使用。

每個 batch 包含連續 T 幀的 clip，讓模型能夠學習 Track Queries 的時序傳遞。

MOT17 資料集目錄結構：
    <data_root>/train/<seq_name>/
        img1/   <- JPEG 影像，名稱為 000001.jpg, ...
        gt/gt.txt <- Ground truth，格式: frame, id, x, y, w, h, conf, cls, vis

gt.txt 格式（每行）：
    frame_id, track_id, x1, y1, w, h, conf, class_id, visibility
    (conf=1 代表有效, class_id=1 代表人)
"""

from __future__ import annotations
import configparser
import csv
from pathlib import Path
from typing import Iterator  # noqa: F401  (used in type hints of _iter_frames in evaluator)
import torch
from torch.utils.data import Dataset, DataLoader


class MOT17TemporalClip(
    Dataset[dict[str, torch.Tensor | list[torch.Tensor] | list[int] | str]]
):
    """
    從 MOT17 訓練集切出長度為 clip_len 的連續幀 clip。

    每個 sample 包含：
      - frames:  (T, 3, H, W) float32
      - gt_boxes:  list[Tensor(N_t, 4)]  每幀的 GT boxes [x1, y1, x2, y2]（絕對像素）
      - gt_ids:    list[Tensor(N_t,)]    每幀的 GT track IDs
      - frame_ids: list[int]             每幀的 frame number

    preload_to_ram=True（預設）：init 時將所有幀以 uint8 載入 RAM（~6 GB），
    __getitem__ 只需做 .float()/255 轉換，消除 JPEG decode 瓶頸。
    搭配 num_workers=0 可避免 CUDA fork 死鎖且 GPU 利用率更高。
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        clip_len: int = 5,
        img_size: int = 640,
        stride: int = 1,
        seqs: list[str] | None = None,
        detector: str | None = "SDP",
        preload_to_ram: bool = True,
        load_images: bool = True,
        use_letterbox: bool = False,
        detail_size: tuple[int, int] | None = None,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.clip_len = clip_len
        self.img_size = img_size
        self.stride = stride
        self.use_letterbox = use_letterbox
        self.detail_size = detail_size
        self.load_images = load_images
        # Strategy B: clip-consistent geometric+photometric augmentation. One
        # random transform is drawn per clip and applied identically to every
        # frame, so temporal continuity (and ID correspondence) is preserved.
        # Default off → bit-exact with the un-augmented loader.
        self.augment = augment
        if not load_images and detail_size is not None:
            raise ValueError("detail_size requires load_images=True")

        split_dir = self.data_root / split
        if seqs is not None:
            self.sequences = seqs
        elif detector is not None:
            self.sequences = sorted(
                [
                    d.name
                    for d in split_dir.iterdir()
                    if d.is_dir() and d.name.endswith(f"-{detector}")
                ]
            )
        else:
            self.sequences = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

        self._clips: list[tuple[str, int]] = []
        self._gt: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self._img_dirs: dict[str, Path] = {}
        self._frame_lists: dict[str, list[Path]] = {}
        self._scale_hw: dict[
            str, tuple[float, ...] | tuple[float, float, int, int]
        ] = {}

        for seq in self.sequences:
            seq_dir = split_dir / seq
            img_dir = seq_dir / "img1"
            gt_file = seq_dir / "gt" / "gt.txt"

            if not img_dir.exists() or not gt_file.exists():
                continue

            frames = sorted(img_dir.glob("*.jpg"))
            if len(frames) < clip_len:
                continue

            self._img_dirs[seq] = img_dir
            self._frame_lists[seq] = frames

            ini = seq_dir / "seqinfo.ini"
            if ini.exists():
                cfg_parser = configparser.ConfigParser()
                cfg_parser.read(ini)
                orig_h = int(cfg_parser["Sequence"]["imHeight"])
                orig_w = int(cfg_parser["Sequence"]["imWidth"])
            else:
                import torchvision.io as tv_io

                _img = tv_io.read_image(str(frames[0]))
                _, orig_h, orig_w = _img.shape

            if self.use_letterbox:
                # Letterbox parameters: keep aspect ratio
                scale = min(img_size / orig_h, img_size / orig_w)
                new_h = int(round(orig_h * scale))
                new_w = int(round(orig_w * scale))
                pad_h = (img_size - new_h) // 2
                pad_w = (img_size - new_w) // 2
                self._scale_hw[seq] = (scale, pad_h, pad_w)
            else:
                # Standard stretch resizing
                self._scale_hw[seq] = (img_size / orig_h, img_size / orig_w)

            gt_per_frame: dict[int, tuple[list[list[float]], list[int]]] = {}
            with gt_file.open() as f:
                for row in csv.reader(f):
                    fid = int(row[0])
                    tid = int(row[1])
                    x, y, w, h = (
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    )
                    conf = int(row[6]) if len(row) > 6 else 1
                    cls = int(row[7]) if len(row) > 7 else 1
                    vis = float(row[8]) if len(row) > 8 else 1.0

                    if conf != 1 or cls != 1 or vis < 0.1:
                        continue

                    if fid not in gt_per_frame:
                        gt_per_frame[fid] = ([], [])
                    gt_per_frame[fid][0].append([x, y, x + w, y + h])
                    gt_per_frame[fid][1].append(tid)

            gt_tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for fid, (boxes, ids) in gt_per_frame.items():
                gt_tensors[fid] = (
                    torch.tensor(boxes, dtype=torch.float32),
                    torch.tensor(ids, dtype=torch.int64),
                )
            self._gt[seq] = gt_tensors

            n_frames = len(frames)
            for start in range(0, n_frames - clip_len + 1, stride):
                self._clips.append((seq, start))

        # Preload all frames as uint8 (3, H, W) — eliminates JPEG decode at getitem time.
        self._img_cache: dict[str, list[torch.Tensor]] | None = None
        if preload_to_ram and load_images:
            self._img_cache = self._preload_images()

    def _preload_images(self) -> dict[str, list[torch.Tensor]]:
        import torchvision.io as tv_io
        import torchvision.transforms.functional as TF
        from concurrent.futures import ThreadPoolExecutor

        all_tasks = []
        for seq, frame_paths in self._frame_lists.items():
            for i, fpath in enumerate(frame_paths):
                all_tasks.append((seq, i, fpath))

        total = len(all_tasks)
        print(
            f"[Dataset] Preloading {total} frames to RAM (uint8 640×640) using multi-threading...",
            flush=True,
        )

        cache: dict[str, list[torch.Tensor | None]] = {
            seq: [None] * len(paths) for seq, paths in self._frame_lists.items()
        }

        def load_and_resize_task(
            task: tuple[str, int, Path],
        ) -> tuple[str, int, torch.Tensor]:
            import torch.nn.functional as F

            seq, idx, fpath = task
            img = tv_io.read_image(str(fpath))  # uint8 (3, H, W)
            if self.use_letterbox:
                scale, pad_h, pad_w = self._scale_hw[seq]  # type: ignore[misc]

                _, orig_h, orig_w = img.shape
                new_h = int(round(orig_h * scale))
                new_w = int(round(orig_w * scale))

                img_resized = TF.resize(img, [new_h, new_w], antialias=True)

                pad_left = pad_w
                pad_right = self.img_size - new_w - pad_left
                pad_top = pad_h
                pad_bottom = self.img_size - new_h - pad_top

                img_padded = F.pad(
                    img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=114
                )
                return seq, idx, img_padded
            else:
                img_resized = TF.resize(
                    img, [self.img_size, self.img_size], antialias=True
                )
                return seq, idx, img_resized

        done = 0
        # Use a reasonable number of workers (e.g., 8 or num_cpus)
        import os

        num_workers = min(os.cpu_count() or 4, 16)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for seq, idx, img in executor.map(load_and_resize_task, all_tasks):
                cache[seq][idx] = img
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{total}", flush=True)

        print(f"  {done}/{total} — done", flush=True)
        return cache  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(
        self, idx: int
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | list[int] | str]:
        seq, start = self._clips[idx]

        frames_list: list[torch.Tensor] = []
        gt_boxes_list: list[torch.Tensor] = []
        gt_ids_list: list[torch.Tensor] = []
        fids: list[int] = []
        detail_frames_list: list[torch.Tensor] = []
        detail_valid_hw_list: list[torch.Tensor] = []

        frame_paths = self._frame_lists[seq]
        if self.use_letterbox:
            scale, pad_h, pad_w = self._scale_hw[seq]  # type: ignore[misc]
        else:
            scale_h, scale_w = self._scale_hw[seq]

        for t in range(self.clip_len):
            frame_id = int(frame_paths[start + t].stem)

            if self.load_images:
                if self._img_cache is not None:
                    img = self._img_cache[seq][start + t]  # uint8 (3, H, W)
                else:
                    if self.use_letterbox:
                        img = _load_and_resize(
                            frame_paths[start + t],
                            self.img_size,
                            True,
                            scale,
                            pad_h,
                            pad_w,
                        )
                    else:
                        img = _load_and_resize(
                            frame_paths[start + t],
                            self.img_size,
                            False,
                            scale_h,
                            scale_w,
                        )
                frames_list.append(img)
            if self.detail_size is not None:
                detail_img, valid_hw = _load_detail_view(
                    frame_paths[start + t], self.detail_size
                )
                detail_frames_list.append(detail_img)
                detail_valid_hw_list.append(valid_hw)

            gt = self._gt[seq].get(
                frame_id, (torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64))
            )
            gt_boxes = gt[0].clone()
            if gt_boxes.numel() > 0:
                if self.use_letterbox:
                    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + pad_w
                    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + pad_h
                else:
                    gt_boxes[:, [0, 2]] *= scale_w
                    gt_boxes[:, [1, 3]] *= scale_h
            gt_boxes_list.append(gt_boxes)
            gt_ids_list.append(gt[1])
            fids.append(frame_id)

        if self.augment and frames_list:
            frames_list, gt_boxes_list, gt_ids_list = _augment_clip(
                frames_list, gt_boxes_list, gt_ids_list, self.img_size
            )

        sample: dict[str, torch.Tensor | list[torch.Tensor] | list[int] | str] = {
            "frames": (
                torch.stack(frames_list)
                if frames_list
                else torch.empty(self.clip_len, 0, 0, 0, dtype=torch.uint8)
            ),
            "gt_boxes": gt_boxes_list,
            "gt_ids": gt_ids_list,
            "frame_ids": fids,
            "seq": seq,
        }
        if detail_frames_list:
            sample["detail_frames"] = torch.stack(detail_frames_list)
            sample["detail_valid_hw"] = torch.stack(detail_valid_hw_list)
        return sample


def _augment_clip(
    frames_list: list[torch.Tensor],
    gt_boxes_list: list[torch.Tensor],
    gt_ids_list: list[torch.Tensor],
    img_size: int,
    *,
    scale_range: tuple[float, float] = (0.8, 1.2),
    translate_frac: float = 0.1,
    hflip_p: float = 0.5,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Apply ONE random transform to every frame of a clip (temporal-consistent).

    Geometric (shared across the clip so GT identities stay aligned): horizontal
    flip + isotropic scale + translation. Photometric: brightness/contrast jitter.
    Boxes are transformed identically; boxes that leave the frame or collapse to
    <1px are dropped together with their IDs. Frames are the square (3, S, S)
    canvas produced by the loader (S = img_size). Strategy B (small/scale robustness).
    """
    import random

    H = W = img_size
    do_flip = random.random() < hflip_p
    s = random.uniform(*scale_range)
    max_t = translate_frac * img_size
    tx = random.uniform(-max_t, max_t)
    ty = random.uniform(-max_t, max_t)
    bright = 1.0 + random.uniform(-brightness, brightness)
    cont = 1.0 + random.uniform(-contrast, contrast)
    cx = cy = (img_size - 1) / 2.0

    # Resampling grid: output(xo,yo) reads input at xi=(xo-cx-tx)/s+cx (inverse of
    # the forward box map below). align_corners=True matches the /(S-1)*2-1 norm.
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    xi = (xs - cx - tx) / s + cx
    yi = (ys - cy - ty) / s + cy
    gx = xi / (W - 1) * 2 - 1
    gy = yi / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    out_frames: list[torch.Tensor] = []
    out_boxes: list[torch.Tensor] = []
    out_ids: list[torch.Tensor] = []
    for img, boxes, ids in zip(frames_list, gt_boxes_list, gt_ids_list):
        f = torch.nn.functional.grid_sample(
            img.float().unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        if do_flip:
            f = torch.flip(f, dims=[-1])
        mean = f.mean(dim=(1, 2), keepdim=True)
        f = ((f - mean) * cont + mean * bright).clamp(0, 255).to(torch.uint8)
        out_frames.append(f)

        if boxes.numel() > 0:
            b = boxes.clone().float()
            # forward box map: out = s*(in - c) + c + t  (inverse of grid above)
            b[:, [0, 2]] = s * (b[:, [0, 2]] - cx) + cx + tx
            b[:, [1, 3]] = s * (b[:, [1, 3]] - cy) + cy + ty
            if do_flip:
                x1 = W - b[:, 2].clone()
                x2 = W - b[:, 0].clone()
                b[:, 0], b[:, 2] = x1, x2
            # No-clip convention: the un-augmented loader (above) and MOT17 GT both
            # keep boxes that run past the frame edge un-clamped, and the detector is
            # trained/deployed on that convention. Decide visibility from the clipped
            # extent (drop a box only when <1px stays on-canvas after scale/translate
            # pushes it off), but keep the UN-clamped coordinates as the regression
            # target — so an edge person is not taught as a truncated-at-edge box.
            # Clamping here was inconsistent with the rest of the pipeline and, with
            # scale<1 + translate pushing objects edge-ward, actively undercut the
            # small/scale-robustness goal of this augmentation.
            vis_w = b[:, [0, 2]].clamp(0, W)
            vis_h = b[:, [1, 3]].clamp(0, H)
            keep = (vis_w[:, 1] - vis_w[:, 0] > 1) & (vis_h[:, 1] - vis_h[:, 0] > 1)
            out_boxes.append(b[keep])
            out_ids.append(ids[keep])
        else:
            out_boxes.append(boxes)
            out_ids.append(ids)
    return out_frames, out_boxes, out_ids


def _load_and_resize(
    path: Path,
    target_size: int,
    use_letterbox: bool = False,
    scale_or_h: float = 1.0,
    pad_h_or_w: float = 1.0,
    pad_w: int = 0,
) -> torch.Tensor:
    """
    讀取 JPEG 並進行縮放至正方形或 Letterbox 縮放（preload_to_ram=False 的 fallback path）。
    """
    import torchvision.io as tv_io
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    img = tv_io.read_image(str(path))
    if use_letterbox:
        scale = scale_or_h
        pad_h = int(pad_h_or_w)
        _, orig_h, orig_w = img.shape
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))

        img_resized = TF.resize(img, [new_h, new_w], antialias=True)

        pad_left = pad_w
        pad_right = target_size - new_w - pad_left
        pad_top = pad_h
        pad_bottom = target_size - new_h - pad_top

        img_padded = F.pad(
            img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=114
        )
        return img_padded
    else:
        img_resized = TF.resize(img, [target_size, target_size], antialias=True)
        return img_resized


def _load_detail_view(
    path: Path, detail_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an aspect-preserving detail image padded at the right and bottom."""
    import torch.nn.functional as F
    import torchvision.io as tv_io
    import torchvision.transforms.functional as TF

    target_h, target_w = detail_size
    img = tv_io.read_image(str(path))
    _, orig_h, orig_w = img.shape
    scale = min(target_h / orig_h, target_w / orig_w)
    valid_h = max(1, min(target_h, int(round(orig_h * scale))))
    valid_w = max(1, min(target_w, int(round(orig_w * scale))))
    resized = TF.resize(img, [valid_h, valid_w], antialias=True)
    padded = F.pad(
        resized,
        (0, target_w - valid_w, 0, target_h - valid_h),
        value=114,
    )
    return padded, torch.tensor([valid_h, valid_w], dtype=torch.int64)


def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    collated = {
        "frames": torch.stack([b["frames"] for b in batch]),  # type: ignore[misc]
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_ids": [b["gt_ids"] for b in batch],
        "frame_ids": [b["frame_ids"] for b in batch],
        "seq": [b["seq"] for b in batch],
    }
    if "detail_frames" in batch[0]:
        collated["detail_frames"] = torch.stack(
            [b["detail_frames"] for b in batch]  # type: ignore[misc]
        )
        collated["detail_valid_hw"] = torch.stack(
            [b["detail_valid_hw"] for b in batch]  # type: ignore[misc]
        )
    return collated


def build_mot17_dataloader(
    data_root: str | Path,
    clip_len: int = 5,
    img_size: int = 640,
    batch_size: int = 8,
    num_workers: int = 0,
    stride: int = 2,
    seqs: list[str] | None = None,
    detector: str = "SDP",
    shuffle: bool = True,
    preload_to_ram: bool = True,
    load_images: bool = True,
    use_letterbox: bool = False,
    detail_size: tuple[int, int] | None = None,
    seed: int | None = None,
    augment: bool = False,
    balance_by_seq: bool = False,
) -> DataLoader[dict[str, object]]:
    dataset = MOT17TemporalClip(
        data_root=data_root,
        split="train",
        clip_len=clip_len,
        img_size=img_size,
        stride=stride,
        seqs=seqs,
        detector=detector,
        preload_to_ram=preload_to_ram,
        load_images=load_images,
        use_letterbox=use_letterbox,
        detail_size=detail_size,
        augment=augment,
    )
    print(f"[Dataset] {len(dataset)} clips from {len(dataset.sequences)} sequences")

    # Strategy B: scene-balanced sampling. With many scenes of wildly different
    # clip counts (PersonPath22 densities 13–96 boxes/frame), uniform shuffling
    # lets a few long/dense sequences dominate a batch. Weight each clip by the
    # inverse of its sequence's clip count so every scene is seen ~equally.
    sampler = None
    if balance_by_seq:
        from collections import Counter

        seq_of = [c[0] for c in dataset._clips]  # (seq, start) per clip
        counts = Counter(seq_of)
        weights = torch.tensor([1.0 / counts[s] for s in seq_of], dtype=torch.double)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True, generator=g
        )
        shuffle = False  # mutually exclusive with a sampler
    # num_workers > 0 with multiprocessing_context='spawn': avoids CUDA fork deadlock
    # (spawn starts fresh processes without inheriting the parent's CUDA context).
    # persistent_workers=True keeps workers alive between epochs to avoid re-init overhead.
    # pin_memory=True is only effective when num_workers > 0; enables DMA H2D transfers.
    use_mp_ctx = "spawn" if num_workers > 0 else None
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=num_workers > 0,
        multiprocessing_context=use_mp_ctx,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
