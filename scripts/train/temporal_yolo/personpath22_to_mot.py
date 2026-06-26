#!/usr/bin/env python3
"""Convert PersonPath22 (gluoncv visible annotations) -> MOTChallenge keyframe format.

Only annotated keyframes are extracted (PersonPath22 train is ~21% densely
annotated, every ~5 frames). Keyframes are renumbered 1..N consecutively so the
output is a dense, frame-adjacent MOT sequence usable by the T3->T1 temporal
curriculum (the keyframe axis becomes the temporal axis; see
docs/modules/detection/research/holdout_generalization_plan.md).

Label rule: keep an entity iff 'person' in its labels. Drops 'reflection' and
'crowd' (no 'person' key); keeps partially/severely-occluded persons (they have
a visible box). Fully-occluded persons have no visible box and are absent from
anno_visible by construction.

Frame alignment: gluoncv blob.frame_idx is 0-based; ffmpeg decodes with
-start_number 0 so output {frame_idx:06d}.jpg matches frame_idx exactly.

Usage:
  personpath22_to_mot.py --videos-zip datasets/PersonPath22/raw/videos.zip \
      --anno-dir datasets/PersonPath22/annotation/anno_visible/anno_visible_2022 \
      --splits datasets/PersonPath22/annotation/splits.json --split train \
      --out-dir datasets/PersonPath22/mot_train [--vids uid_vid_00000.mp4 ...]
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def convert_video(
    vid: str,
    zf: zipfile.ZipFile,
    anno_dir: Path,
    out_dir: Path,
    jpeg_q: int = 2,
    keep_all_frames: bool = False,
) -> dict:
    anno = json.loads((anno_dir / f"{vid}.json").read_text())
    meta = anno["metadata"]
    # 1. filter to person, group boxes by frame
    by_frame: dict[int, list] = {}
    ids: set[int] = set()
    for e in anno["entities"]:
        if "person" not in e.get("labels", {}):
            continue
        f = int(e["blob"]["frame_idx"])
        by_frame.setdefault(f, []).append((int(e["id"]), e["bb"]))
        ids.add(int(e["id"]))
    if not by_frame:
        return {"vid": vid, "skipped": "no person entities"}
    # 2. keyframe rank + id remap (1-based, MOT convention)
    keyframes = sorted(by_frame)
    rank = {f: i + 1 for i, f in enumerate(keyframes)}
    id_map = {pid: i + 1 for i, pid in enumerate(sorted(ids))}

    seq_dir = out_dir / vid
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt").mkdir(exist_ok=True)

    # 3. decode all frames once (start_number 0 => name == frame_idx), pick keyframes
    with tempfile.TemporaryDirectory() as td:
        mp4 = Path(td) / vid
        with zf.open(vid) as src, open(mp4, "wb") as dst:
            shutil.copyfileobj(src, dst)
        frames_dir = Path(td) / "frames"
        frames_dir.mkdir()
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-start_number",
                "0",
                "-qscale:v",
                str(jpeg_q),
                str(frames_dir / "%06d.jpg"),
            ],
            check=True,
        )
        decoded = sorted(int(p.stem) for p in frames_dir.glob("*.jpg"))
        decoded_set = set(decoded)

        # out_index: real frame_idx (0-based) -> 1-based MOT frame number.
        #   keyframe-only (default): renumber keyframes 1..N (keyframe = temporal
        #     axis for the T3→T1 curriculum; sequence is dense/frame-adjacent).
        #   keep_all_frames (eval): every decoded frame kept at its true index
        #     (f+1), so the tracker runs on CONSECUTIVE frames (correct IoU/motion)
        #     while GT exists only at keyframes — score with predictions filtered
        #     to keyframes (see scripts/eval/score_keyframe_filtered.py).
        if keep_all_frames:
            out_index = {f: f + 1 for f in decoded}
            frames_to_copy = decoded
        else:
            out_index = {f: rank[f] for f in keyframes if f in decoded_set}
            frames_to_copy = [f for f in keyframes if f in decoded_set]

        w = h = None
        for f in frames_to_copy:
            src_jpg = frames_dir / f"{f:06d}.jpg"
            shutil.copy(src_jpg, img_dir / f"{out_index[f]:06d}.jpg")
            if w is None:
                from PIL import Image

                with Image.open(src_jpg) as im:
                    w, h = im.size

    # 4. gt.txt — boxes at each keyframe's OUTPUT index (keyframes present in img1)
    kf_extracted = sorted(f for f in keyframes if f in out_index)
    lines = []
    for f in kf_extracted:
        for pid, bb in by_frame[f]:
            x, y, bw, bh = (round(v) for v in bb)
            lines.append(f"{out_index[f]},{id_map[pid]},{x},{y},{bw},{bh},1,1,1")
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")

    # 5. seqinfo.ini — int frameRate (TrackEval casts to int).
    if keep_all_frames:
        seq_len = len(frames_to_copy)
        fps = max(1, round(float(meta["fps"])))  # true frame rate
    else:
        seq_len = len(kf_extracted)
        # effective keyframe rate = fps * extracted/total_frames
        fps = max(
            1,
            round(
                float(meta["fps"]) * seq_len / max(float(meta["number_of_frames"]), 1)
            ),
        )
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={vid}\nimDir=img1\nframeRate={fps}\n"
        f"seqLength={seq_len}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n"
    )
    return {
        "vid": vid,
        "keyframes": len(kf_extracted),
        "seq_len": seq_len,
        "boxes": len(lines),
        "ids": len(id_map),
        "decoded": len(decoded),
        "json_frames": int(meta["number_of_frames"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-zip", required=True)
    ap.add_argument("--anno-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vids", nargs="*", default=None, help="subset (dry-run)")
    ap.add_argument("--jpeg-q", type=int, default=2)
    ap.add_argument(
        "--keep-all-frames",
        action="store_true",
        help="EVAL mode: keep every decoded frame at its true 1-based index (GT "
        "only at keyframes) so the tracker runs on consecutive frames — correct "
        "IoU/motion. Score with predictions filtered to keyframes "
        "(scripts/eval/score_keyframe_filtered.py). Default off = keyframe-only "
        "(renumbered, for the temporal training curriculum).",
    )
    args = ap.parse_args()

    anno_dir = Path(args.anno_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_vids = set(json.loads(Path(args.splits).read_text())[args.split])

    with zipfile.ZipFile(args.videos_zip) as zf:
        in_zip = set(zf.namelist())
        targets = args.vids or sorted(split_vids & in_zip)
        missing = [v for v in targets if v not in in_zip]
        if missing:
            print(
                f"[warn] {len(missing)} target(s) not in zip (external), skipped: {missing[:3]}..."
            )
        targets = [v for v in targets if v in in_zip]
        print(f"[info] converting {len(targets)} videos -> {out_dir}")
        for i, vid in enumerate(targets, 1):
            r = convert_video(
                vid, zf, anno_dir, out_dir, args.jpeg_q, args.keep_all_frames
            )
            print(f"[{i}/{len(targets)}] {r}")


if __name__ == "__main__":
    main()
