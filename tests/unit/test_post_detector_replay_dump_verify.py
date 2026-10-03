"""``dump_post_detector_replay.py --verify`` on synthetic dumps (#465 PR-6).

The verifier is what makes a dump admissible as the reference for the native
replay (PR-5 stages, PR-6 MOT txt). Byte and sha256 checks alone cannot see a
stream whose metadata was rewritten consistently with a wrong record count, or
streams that drifted out of step; these tests pin that the verifier parses
every record and checks the per-update streams against the non-empty detector
frames, and that it hashes the oracle MOT output it records.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "eval" / "diagnostics" / "dump_post_detector_replay.py"
SEQ = "SYN-01"


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location("dump_post_detector_replay", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(n: int, with_ids: bool = False) -> bytes:
    out = struct.pack(f"<{4 * n}f", *([1.0] * 4 * n)) + struct.pack(
        f"<{n}f", *([0.5] * n)
    )
    if with_ids:
        out += struct.pack(f"<{n}i", *range(1, n + 1))
    return out + struct.pack(f"<{n}i", *([0] * n))


def _streams(frames: list[int], empty: set[int]) -> dict[str, bytes]:
    det = b"".join(
        struct.pack("<3i", f, 0 if f in empty else 2, 0)
        + (b"" if f in empty else _rows(2))
        for f in frames
    )
    upd = [f for f in frames if f not in empty]
    warp = struct.pack("<6f", 1, 0, 0, 0, 1, 0)
    return {
        "detector.bin": det,
        "post_nms.bin": b"".join(struct.pack("<2i", f, 2) + _rows(2) for f in upd),
        "gmc.bin": b"".join(
            struct.pack("<3i", f, i, 1) + warp for i, f in enumerate(upd)
        ),
        "tracker_in.bin": b"".join(
            struct.pack("<3i", f, 2, 1) + warp + _rows(2) for f in upd
        ),
        "tracker_out.bin": b"".join(
            struct.pack("<2i", f, 2) + _rows(2, with_ids=True) for f in upd
        ),
    }


def _entry(data: bytes, records: int) -> dict[str, Any]:
    return {
        "records": records,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write(
    root: Path, streams: dict[str, bytes], records: dict[str, int], empty: list[int]
) -> None:
    tool = _tool()
    seq_dir = root / SEQ
    seq_dir.mkdir(parents=True)
    for name, data in streams.items():
        (seq_dir / name).write_bytes(data)
    meta = {
        "format": tool.FORMAT,
        "sequence": SEQ,
        "width": 8,
        "height": 4,
        "files": {name: _entry(data, records[name]) for name, data in streams.items()},
        "frames_u8": {"records": 0, "bytes": 0, "sha256": "", "stored": False},
        "empty_detector_frames": empty,
    }
    (seq_dir / "meta.json").write_text(json.dumps(meta))
    eval_dir = root / "eval"
    eval_dir.mkdir()
    (eval_dir / f"{SEQ}.txt").write_text("1,1,1.00,1.00,0.00,0.00,0.5000,-1,-1,-1")
    (eval_dir / "_global_id_map.txt").write_text(f"{SEQ}\tlocal_id=1\tglobal_id=1\n")
    manifest = {
        "format": tool.FORMAT,
        "git_head": "0" * 40,
        "git_dirty": False,
        "sequences": {SEQ: meta},
        "mot_reference": tool._mot_reference(eval_dir, [SEQ]),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _good(root: Path) -> None:
    frames, empty = [1, 2, 3, 4], {3}
    streams = _streams(frames, empty)
    records = {n: (4 if n == "detector.bin" else 3) for n in streams}
    _write(root, streams, records, sorted(empty))


def test_consistent_dump_verifies(tmp_path: Path) -> None:
    _good(tmp_path)
    result = _tool().verify(tmp_path, None)
    assert result["integrity_problems"] == []
    assert result["verdict"] == "OK"


def test_wrong_record_count_with_consistent_hashes_fails(tmp_path: Path) -> None:
    frames, empty = [1, 2, 3, 4], {3}
    streams = _streams(frames, empty)
    records = {n: (4 if n == "detector.bin" else 3) for n in streams}
    records["tracker_out.bin"] = 2  # bytes and sha256 still match the file
    _write(tmp_path, streams, records, sorted(empty))
    problems = _tool().verify(tmp_path, None)["integrity_problems"]
    assert problems == [f"{SEQ}/tracker_out.bin: 3 records parsed, meta.json says 2"]


def test_stream_out_of_step_fails(tmp_path: Path) -> None:
    frames, empty = [1, 2, 3, 4], {3}
    streams = _streams(frames, empty)
    # tracker_out lacks frame 4's record; its metadata agrees with the file.
    out = streams["tracker_out.bin"]
    streams["tracker_out.bin"] = out[: len(out) // 3 * 2]
    records = {n: (4 if n == "detector.bin" else 3) for n in streams}
    records["tracker_out.bin"] = 2
    _write(tmp_path, streams, records, sorted(empty))
    problems = _tool().verify(tmp_path, None)["integrity_problems"]
    assert problems == [
        f"{SEQ}/tracker_out.bin: frames differ from the non-empty detector frames"
    ]


def test_truncated_stream_fails(tmp_path: Path) -> None:
    frames, empty = [1, 2], set()
    streams = _streams(frames, empty)
    streams["post_nms.bin"] = streams["post_nms.bin"][:-4]
    records = {n: 2 for n in streams}
    _write(tmp_path, streams, records, [])
    problems = _tool().verify(tmp_path, None)["integrity_problems"]
    assert len(problems) == 1 and "post_nms.bin: truncated record" in problems[0]


def test_tampered_mot_reference_fails(tmp_path: Path) -> None:
    _good(tmp_path)
    (tmp_path / "eval" / f"{SEQ}.txt").write_text(
        "1,1,9.00,1.00,0.00,0.00,0.5000,-1,-1,-1"
    )
    problems = _tool().verify(tmp_path, None)["integrity_problems"]
    assert problems == [f"eval/{SEQ}.txt: content differs from the manifest"]


def test_missing_mot_reference_fails(tmp_path: Path) -> None:
    _good(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["mot_reference"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert _tool().verify(tmp_path, None)["verdict"] == "FAIL"
