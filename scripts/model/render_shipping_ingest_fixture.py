#!/usr/bin/env python3
"""Render the golden fixture for the native ingest (decode + normalize).

Issue #465 Phase B PR-7 (U3b-1; boundary §6). The native ingest
(``shipping/src/ingest_plan.cpp``, ``ingest_host.cpp``, ``ingest_normalize.cu``)
re-implements three oracle facts; this tool runs the oracle's own code for each
and writes what it produced to ``tests/native/fixtures/shipping_ingest.json``:

* **normalize** -- ``torch.div(frame_hwc.permute(2, 0, 1), 255.0,
  out=frame_buffer)`` (``stages._run_detect``) on CUDA, for all 256 input
  values: the float32 bit patterns (CPU torch divides exactly and differs on
  126 of them, so this part needs a GPU);
* **decode** -- ``TorchvisionGpuStreamer`` (``decode_jpeg(device="cuda",
  mode=RGB)``, nvJPEG) over small committed JPEGs in
  ``tests/native/fixtures/shipping_ingest/`` covering 4:2:0 / 4:2:2 / 4:4:4,
  odd sizes, progressive and grayscale bitstreams: the decoded planar RGB
  bytes (also GPU-only);
* **sequence input** -- the streamer's frame listing
  (``sorted(str(p.absolute()) for p in img_dir.glob("*.jpg"))``) and the
  pipeline's ``seqinfo.ini`` reads (``configparser``: ``imWidth``,
  ``imHeight``, ``seqLength``; ``frame_end = min(max_frames or int(1e9),
  seqLength)``) on materialized directory cases. The oracle consumes frame
  ``k`` = ``k``-th listed entry and silently stops early when the listing runs
  out (``truncated``). Cases the
  native reader refuses although the oracle accepts them are marked
  ``"native": "refuse"`` with the reason.

``tests/native/test_shipping_ingest_plan.cpp`` (CPU, CI ``shipping-config-loader``)
replays the normalize table through the host twin and the sequence cases
through ``read_sequence_input``; ``tests/native/test_shipping_ingest_host.cpp``
(GPU) replays the table through the kernel and the JPEGs through the decoder.

Without ``--write-jpegs`` the committed JPEGs are inputs (their sha256 is
recorded and checked); ``--write-jpegs`` regenerates them with Pillow first.
``--check`` compares against the committed fixture; without CUDA it checks the
CPU sections only and says so.

Usage:
  .venv/bin/python scripts/model/render_shipping_ingest_fixture.py [--write-jpegs]
  .venv/bin/python scripts/model/render_shipping_ingest_fixture.py --check
"""

# status: stable

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUTPUT = REPO / "tests" / "native" / "fixtures" / "shipping_ingest.json"
JPEG_DIR = REPO / "tests" / "native" / "fixtures" / "shipping_ingest"
FORMAT = "saccade.shipping_ingest_fixture/v1"
GPU_SECTIONS = ("normalize_f32_hex", "jpegs")

# name -> (width, height, PIL mode, subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0), progressive)
JPEGS: dict[str, tuple[int, int, str, int, bool]] = {
    "baseline_420.jpg": (48, 32, "RGB", 2, False),
    "baseline_422.jpg": (48, 32, "RGB", 1, False),
    "baseline_444.jpg": (48, 32, "RGB", 0, False),
    "odd_420.jpg": (47, 31, "RGB", 2, False),
    "progressive_420.jpg": (48, 32, "RGB", 2, True),
    "gray.jpg": (48, 32, "L", 0, False),
}

SEQINFO = (
    "[Sequence]\nname=SEQ\nimDir=img1\nframeRate=30\nseqLength={n}\n"
    "imWidth=1920\nimHeight=1080\nimExt=.jpg\n"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── JPEG inputs ──────────────────────────────────────────────────────────────


def write_jpegs() -> None:
    from PIL import Image

    rng = np.random.default_rng(465_7)
    JPEG_DIR.mkdir(parents=True, exist_ok=True)
    for name, (w, h, mode, subsampling, progressive) in JPEGS.items():
        y, x = np.mgrid[0:h, 0:w]
        base = np.stack(
            [x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x ^ y) & 255], -1
        )
        noise = rng.integers(-40, 41, size=base.shape)
        rgb = np.clip(base + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, "RGB")
        if mode == "L":
            img = img.convert("L")
        kwargs: dict[str, Any] = {
            "quality": 85,
            "progressive": progressive,
            "optimize": False,
        }
        if mode == "RGB":
            kwargs["subsampling"] = subsampling
        img.save(JPEG_DIR / name, format="JPEG", **kwargs)


# ── oracle runs ──────────────────────────────────────────────────────────────


def oracle_normalize_table() -> list[str]:
    """``_run_detect``'s ingest op on CUDA for every uint8 value."""
    import torch

    values = torch.arange(256, dtype=torch.uint8, device="cuda")
    frame_hwc = values.view(16, 16, 1).expand(16, 16, 3).contiguous()
    frame_buffer = torch.zeros((3, 16, 16), dtype=torch.float32, device="cuda")
    torch.div(frame_hwc.permute(2, 0, 1), 255.0, out=frame_buffer)
    out = frame_buffer[0].flatten().cpu().numpy()
    assert all(
        np.array_equal(frame_buffer[c].flatten().cpu().numpy(), out) for c in range(3)
    )
    return [out[i : i + 1].tobytes().hex() for i in range(256)]


def oracle_decode(path: Path) -> dict[str, Any]:
    """The oracle's streamer over a directory holding only this frame."""
    import torch

    from saccade.perception.eval.streaming import TorchvisionGpuStreamer

    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp)
        (img_dir / "000001.jpg").write_bytes(path.read_bytes())
        streamer = TorchvisionGpuStreamer(img_dir)
        frame_hwc = next(iter(streamer))
        torch.cuda.current_stream().synchronize()
        chw = frame_hwc.permute(2, 0, 1).contiguous().cpu().numpy()
        streamer._stop_worker()
    return {
        "width": int(chw.shape[2]),
        "height": int(chw.shape[1]),
        "rgb_planar_hex": chw.tobytes().hex(),
    }


def _materialize(root: Path, case: dict[str, Any]) -> Path:
    seq = root / case["name"]
    seq.mkdir()
    if case["seqinfo"] is not None:
        (seq / "seqinfo.ini").write_bytes(case["seqinfo"].encode("utf-8"))
    if case["img1"] is not None:
        img = seq / "img1"
        img.mkdir()
        for entry in case["img1"]:
            p = img / entry["name"]
            if entry["kind"] == "file":
                p.write_bytes(b"\xff\xd8\xff")
            elif entry["kind"] == "dir":
                p.mkdir()
            elif entry["kind"] == "symlink":
                os.symlink(entry["target"], p)
            else:
                raise ValueError(entry)
    return seq


def oracle_sequence(seq: Path, max_frames: int) -> dict[str, Any]:
    """pipeline.py's seqinfo reads + the streamer's listing + the frame loop's bound."""
    from saccade.perception.eval.streaming import TorchvisionGpuStreamer

    try:
        config = configparser.ConfigParser()
        config.read(seq / "seqinfo.ini")
        w_orig = config.getint("Sequence", "imWidth")
        h_orig = config.getint("Sequence", "imHeight")
        frame_end = min(max_frames or int(1e9), config.getint("Sequence", "seqLength"))
    except Exception as exc:  # the oracle stops here
        return {"ok": False, "error": type(exc).__name__}
    files = TorchvisionGpuStreamer(seq / "img1").img_files
    listed = [Path(f).name for f in files]
    return {
        "ok": True,
        "im_width": w_orig,
        "im_height": h_orig,
        "seq_length": config.getint("Sequence", "seqLength"),
        "listed": listed,
        "frames": listed[: max(frame_end, 0)],
        # The frame loops stop at the first StopIteration (evaluator.py
        # `_run_frame` returns False, double-buffer `_schedule` returns None):
        # a short listing truncates the sequence instead of failing it.
        "truncated": frame_end > len(listed),
    }


def _files(*names: str) -> list[dict[str, str]]:
    return [{"name": n, "kind": "file"} for n in names]


def sequence_cases() -> list[dict[str, Any]]:
    four = _files("000003.jpg", "000001.jpg", "000004.jpg", "000002.jpg")
    cases: list[dict[str, Any]] = [
        {
            "name": "plain",
            "seqinfo": SEQINFO.format(n=3),
            "img1": four,
            "max_frames": 0,
        },
        {
            "name": "max_frames",
            "seqinfo": SEQINFO.format(n=4),
            "img1": four,
            "max_frames": 2,
        },
        {
            "name": "lexicographic",
            "seqinfo": SEQINFO.format(n=3),
            "img1": _files("10.jpg", "9.jpg", "000002.jpg"),
            "max_frames": 0,
        },
        {
            "name": "glob_edges",
            "seqinfo": SEQINFO.format(n=6),
            "img1": _files(
                "000001.jpg",
                ".hidden.jpg",
                ".jpg",
                "UPPER.JPG",
                "x.jpeg",
                "x.jpg~",
                "x.jpg.bak",
                "b.jpg",
            )
            + [
                {"name": "dir.jpg", "kind": "dir"},
                {"name": "link.jpg", "kind": "symlink", "target": "000001.jpg"},
                {"name": "dangling.jpg", "kind": "symlink", "target": "missing.jpg"},
            ],
            "max_frames": 0,
        },
        {
            "name": "keys_and_comments",
            "seqinfo": "# MOT\r\n[Sequence]\r\n; comment\r\nIMWIDTH : 640\r\nimheight= 480\r\n"
            "  # indented comment\r\nseqlength=2\r\n\r\n[Other]\r\nimWidth=1\r\n",
            "img1": _files("a.jpg", "b.jpg"),
            "max_frames": 0,
        },
        {
            "name": "empty_sequence",
            "seqinfo": SEQINFO.format(n=0),
            "img1": None,
            "max_frames": 0,
        },
        {
            "name": "too_few_files",
            "seqinfo": SEQINFO.format(n=5),
            "img1": four,
            "max_frames": 0,
            "native": "refuse",
            "reason": "a listing shorter than the frames to consume is refused (the oracle truncates the sequence)",
        },
        {
            "name": "no_img1",
            "seqinfo": SEQINFO.format(n=1),
            "img1": None,
            "max_frames": 0,
            "native": "refuse",
            "reason": "a missing img1/ is refused (the oracle truncates the sequence to nothing)",
        },
        {"name": "no_seqinfo", "seqinfo": None, "img1": four, "max_frames": 0},
        {
            "name": "missing_key",
            "seqinfo": "[Sequence]\nimWidth=1920\nseqLength=1\n",
            "img1": four,
            "max_frames": 0,
        },
        {
            "name": "duplicate_key",
            "seqinfo": SEQINFO.format(n=1) + "imWidth=640\n",
            "img1": four,
            "max_frames": 0,
        },
        {
            "name": "inline_comment",
            "seqinfo": "[Sequence]\nimWidth=1920 ; px\nimHeight=1080\nseqLength=1\n",
            "img1": four,
            "max_frames": 0,
        },
        {
            "name": "key_before_section",
            "seqinfo": "imWidth=1920\n[Sequence]\nimHeight=1080\nseqLength=1\n",
            "img1": four,
            "max_frames": 0,
        },
        # The oracle accepts these; the native reader refuses rather than interpret.
        {
            "name": "indented_key",
            "seqinfo": "[Sequence]\n  imWidth=1920\nimHeight=1080\nseqLength=1\n",
            "img1": four,
            "max_frames": 0,
            "native": "refuse",
            "reason": "indented lines (configparser continuation syntax) are refused",
        },
        {
            "name": "underscore_int",
            "seqinfo": "[Sequence]\nimWidth=1_920\nimHeight=1080\nseqLength=1\n",
            "img1": four,
            "max_frames": 0,
            "native": "refuse",
            "reason": "integers other than [+-]digits are refused",
        },
        {
            "name": "default_section",
            "seqinfo": "[DEFAULT]\nimWidth=1920\n[Sequence]\nimHeight=1080\nseqLength=1\n",
            "img1": four,
            "max_frames": 0,
            "native": "refuse",
            "reason": "a DEFAULT section is refused",
        },
        {
            "name": "non_ascii_name",
            "seqinfo": SEQINFO.format(n=1),
            "img1": _files("é.jpg"),
            "max_frames": 0,
            "native": "refuse",
            "reason": "non-ASCII frame names are refused",
        },
        {
            "name": "non_positive_size",
            "seqinfo": "[Sequence]\nimWidth=0\nimHeight=1080\nseqLength=0\n",
            "img1": None,
            "max_frames": 0,
            "native": "refuse",
            "reason": "the oracle's frame buffer cannot have a non-positive size",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for case in cases:
            case.setdefault("native", "same")
            case["oracle"] = oracle_sequence(
                _materialize(root, case), case["max_frames"]
            )
            if case["native"] == "refuse":
                assert case["oracle"]["ok"], (
                    f"{case['name']}: a refuse case must be one the oracle accepts"
                )
            else:
                assert not case["oracle"]["ok"] or not case["oracle"]["truncated"], (
                    f"{case['name']}: a truncated sequence must be a refuse case"
                )
    return cases


# ── fixture ──────────────────────────────────────────────────────────────────


def render(*, gpu: bool) -> dict[str, Any]:
    doc: dict[str, Any] = {"format": FORMAT}
    if gpu:
        doc["normalize_f32_hex"] = oracle_normalize_table()
    jpegs = []
    for name in JPEGS:
        data = (JPEG_DIR / name).read_bytes()
        entry: dict[str, Any] = {"file": name, "sha256": _sha256(data)}
        if gpu:
            entry.update(oracle_decode(JPEG_DIR / name))
        jpegs.append(entry)
    doc["jpegs"] = jpegs
    doc["sequence_cases"] = sequence_cases()
    return doc


def _dump(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=True) + "\n"


def _cpu_view(doc: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k != "normalize_f32_hex"}
    out["jpegs"] = [{"file": j["file"], "sha256": j["sha256"]} for j in doc["jpegs"]]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed fixture is stale"
    )
    parser.add_argument(
        "--write-jpegs", action="store_true", help="regenerate the input JPEGs first"
    )
    args = parser.parse_args(argv)
    import torch

    gpu = torch.cuda.is_available()
    if args.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT.relative_to(REPO)} is missing")
            return 1
        committed = json.loads(OUTPUT.read_text(encoding="utf-8"))
        fresh = render(gpu=gpu)
        ok = (
            _dump(fresh) == OUTPUT.read_text(encoding="utf-8")
            if gpu
            else (_cpu_view(fresh) == _cpu_view(committed))
        )
        scope = (
            "all sections"
            if gpu
            else "CPU sections only (no CUDA: normalize/decode not checked)"
        )
        print(f"{OUTPUT.relative_to(REPO)} is {'fresh' if ok else 'stale'} ({scope})")
        return 0 if ok else 1
    if not gpu:
        print("rendering needs CUDA (normalize table, oracle decode)", file=sys.stderr)
        return 2
    if args.write_jpegs:
        write_jpegs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(_dump(render(gpu=True)), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
