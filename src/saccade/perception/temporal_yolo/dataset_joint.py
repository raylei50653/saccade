"""
DanceTrack Temporal Dataset — 供 Option D 訓練使用。

DanceTrack 結構：
    <data_root>/train/<seq_name>/
        img1/       <- JPEG 影像
        gt/gt.txt   <- Ground truth
        seqinfo.ini <- 序列信息

gt.txt 格式：
    frame_id, track_id, x1, y1, w, h, conf, class_id, visibility
    (class_id=1 代表人)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from torch.utils.data import ConcatDataset, DataLoader
from .dataset import MOT17TemporalClip, collate_fn


class DanceTrackTemporalClip(MOT17TemporalClip):
    """
    DanceTrack dataset, reusing MOT17TemporalClip with detector=None
    since DanceTrack does not have detector-specific directory suffixes.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        clip_len: int = 5,
        img_size: int = 640,
        stride: int = 1,
        seqs: list[str] | None = None,
        preload_to_ram: bool = True,
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            clip_len=clip_len,
            img_size=img_size,
            stride=stride,
            seqs=seqs,
            detector=None,
            preload_to_ram=preload_to_ram,
        )


def build_joint_dataloader(
    dataset_configs: list[dict[str, Any]],  # list of {name, root, type}
    clip_len: int = 5,
    img_size: int = 640,
    batch_size: int = 8,
    num_workers: int = 0,
    stride: int = 5,
    shuffle: bool = True,
    preload_to_ram: bool = True,
) -> DataLoader[Any]:
    """
    同時加載多個數據集（MOT17, MOT20, DanceTrack）進行聯合訓練。
    """
    datasets = []
    for cfg in dataset_configs:
        dtype = cfg.get("type", "mot")
        if dtype == "mot":
            ds = MOT17TemporalClip(
                data_root=cfg["root"],
                clip_len=clip_len,
                img_size=img_size,
                stride=stride,
                preload_to_ram=preload_to_ram,
            )
        elif dtype == "dancetrack":
            ds = DanceTrackTemporalClip(
                data_root=cfg["root"],
                clip_len=clip_len,
                img_size=img_size,
                stride=stride,
                preload_to_ram=preload_to_ram,
            )
        datasets.append(ds)

    concat_ds: ConcatDataset[Any] = ConcatDataset(datasets)
    print(f"[JointDataset] Total clips: {len(concat_ds)} from {len(datasets)} sources")

    return DataLoader(
        concat_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
