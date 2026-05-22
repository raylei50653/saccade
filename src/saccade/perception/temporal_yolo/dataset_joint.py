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
import configparser
import csv
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import ConcatDataset, DataLoader
from .dataset import MOT17TemporalClip, collate_fn


class DanceTrackTemporalClip(MOT17TemporalClip):
    """
    DanceTrack 數據集。
    繼承自 MOT17TemporalClip，主要差異在於序列發現邏輯（無 detector 後綴）。
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
        # 由於 DanceTrack 沒有 -SDP 這種後綴，我們需要重寫初始化中的序列搜索部分
        self.data_root = Path(data_root)
        self.split = split
        self.clip_len = clip_len
        self.img_size = img_size
        self.stride = stride

        split_dir = self.data_root / split
        if seqs is not None:
            self.sequences = seqs
        else:
            self.sequences = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

        # 其餘邏輯調用父類（解析 gt.txt 的格式是相同的）
        self._clips = []
        self._gt = {}
        self._img_dirs = {}
        self._frame_lists = {}
        self._scale_hw = {}

        # 複用父類的加載邏輯（這部分在 MOT17TemporalClip.__init__ 之後）
        # 但我們不能直接調用 super().__init__ 因為它會過濾 -SDP
        self._init_metadata(split_dir)

        self._img_cache = None
        if preload_to_ram:
            self._img_cache = self._preload_images()

    def _init_metadata(self, split_dir: Path) -> None:
        for seq in self.sequences:
            seq_dir = split_dir / seq
            img_dir = seq_dir / "img1"
            gt_file = seq_dir / "gt" / "gt.txt"

            if not img_dir.exists() or not gt_file.exists():
                continue

            frames = sorted(img_dir.glob("*.jpg"))
            if len(frames) < self.clip_len:
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
            self._scale_hw[seq] = (self.img_size / orig_h, self.img_size / orig_w)

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
                    # DanceTrack gt.txt 格式略有不同，但前 6 位是兼容的
                    if fid not in gt_per_frame:
                        gt_per_frame[fid] = ([], [])
                    gt_per_frame[fid][0].append([x, y, x + w, y + h])
                    gt_per_frame[fid][1].append(tid)

            gt_tensors = {}
            for fid, (boxes, ids) in gt_per_frame.items():
                gt_tensors[fid] = (
                    torch.tensor(boxes, dtype=torch.float32),
                    torch.tensor(ids, dtype=torch.int64),
                )
            self._gt[seq] = gt_tensors

            n_frames = len(frames)
            for start in range(0, n_frames - self.clip_len + 1, self.stride):
                self._clips.append((seq, start))


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
