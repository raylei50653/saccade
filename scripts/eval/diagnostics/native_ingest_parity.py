#!/usr/bin/env python3
"""Native ingest parity vs the torchvision ingest path (#465 Phase B PR-7).

Issue #465 Phase B PR-7 (U3b-1; docs/reference/native_runtime_shipping_boundary.md
§6: "decoded pixels measured against torchvision on their own; differences are
reported, not folded into later stages"). Developer tooling only
(``developer_build_debug``). Nothing downstream of the ingest runs: no
backbone, head, S2, tracker.

Per sequence it runs, side by side and frame by frame:

* **oracle** -- the eval harness's own ingest objects: ``TorchvisionGpuStreamer``
  (``decode_jpeg(device="cuda", mode=RGB)``, nvJPEG) over ``<seq>/img1``, and
  ``_run_detect``'s ingest op ``torch.div(frame_hwc.permute(2, 0, 1), 255.0,
  out=frame_buffer)`` into a zeroed float32 ``[3, imHeight, imWidth]`` buffer;
  the frame count and geometry from ``seqinfo.ini`` as ``pipeline.py`` reads
  them (``tests/unit/test_native_ingest_oracle_pins.py`` pins these to the
  oracle source);
* **native** -- ``build/shipping/saccade_ingest_probe`` in its own process (no
  Python, no torch), which runs the shipping ingest from the resolved config
  and streams its decoded planar RGB bytes and its frame buffer
  (``saccade.native_ingest_stream/v1``, see the tool's header).

and reports three differences independently, so a decoder difference cannot
hide in (or be blamed on) the normalize:

``decoder``     native uint8 [3, H, W] vs the oracle's decoded frame (CHW);
``normalize``   native frame buffer vs the oracle op applied to the *native*
                decoded bytes (isolates the normalize from the decoder), plus
                ``normalize_table``: the native kernel's output for all 256
                values vs the oracle op on CUDA (contiguous and permuted input);
``ingest``      native frame buffer vs the oracle frame buffer (end to end; a
                consequence of the two above, never used to attribute).

Each is ``EXACT`` (bit-identical on every frame) or ``DIFFERS`` with counts,
maximum absolute difference, a histogram of decoded-byte differences and the
first differing frames. The file listing and geometry must agree before any
frame is compared (a disagreement is an error, not a difference). There is no
tolerance: this tool reports, the owner judges.

Output (the ``--out`` directory is claimed with ``open_run`` before the first
byte): ``report.json`` (``saccade.native_ingest_parity/v1``), ``frames.jsonl``
(per frame: decode path, per-section equality, sha256 of the oracle and native
decoded bytes and frame buffers) and ``probe_<seq>.stderr``. ``--against
<report.json>`` compares the per-frame hashes with an earlier run (run-to-run
identity of both sides). ``--force-decoupled`` runs the probe with every frame
on nvJPEG's decoupled path -- a negative control for the decode-path choice.
Exit 0: every section EXACT (and, with --against, every hash equal); 1: a
difference; 2: error.

Usage (GPU; a formal run holds the gpu0 lease)::

    cmake --build build --target saccade_ingest_probe
    .venv/bin/python tools/resctl.py run gpu0 -- \\
        .venv/bin/python scripts/eval/diagnostics/native_ingest_parity.py \\
        --out results/465_pr7_ingest/<label>/parity [--sequences MOT17-09-SDP] \\
        [--max-frames N] [--force-decoupled] [--against <earlier report.json>]
"""
# status: diagnostic

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import hashlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import IO, Any

project_root = Path(__file__).resolve().parents[3]

SCHEMA = "saccade.native_ingest_parity/v1"
STREAM_FORMAT = "saccade.native_ingest_stream/v1"
RESOLVED_CONFIG = project_root / "configs/shipping/mamba_whole_graph.resolved.json"
PROBE = project_root / "build/shipping/saccade_ingest_probe"
DEFAULT_SEQUENCES = tuple(f"MOT17-{n:02d}-SDP" for n in (2, 4, 5, 9, 10, 11, 13))
ORACLE_INGEST_OP = "torch.div(frame_hwc.permute(2, 0, 1), 255.0, out=frame_buffer)"
FIRST_DIFFS_KEPT = 20


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


# ── the oracle's ingest op and reads (pinned by test_native_ingest_oracle_pins) ──


def oracle_ingest(frame_hwc: Any, frame_buffer: Any) -> None:
    """``stages._run_detect``'s ingest op (non-NV12 branch, no preprocess modes)."""
    import torch

    torch.div(frame_hwc.permute(2, 0, 1), 255.0, out=frame_buffer)


def oracle_sequence_bounds(
    seq_path: Path, max_frames: int | None
) -> tuple[int, int, int]:
    """``pipeline.py``'s seqinfo reads: ``(w_orig, h_orig, frame_end)``."""
    config = configparser.ConfigParser()
    config.read(seq_path / "seqinfo.ini")
    w_orig = config.getint("Sequence", "imWidth")
    h_orig = config.getint("Sequence", "imHeight")
    frame_end = min(max_frames or int(1e9), config.getint("Sequence", "seqLength"))
    return w_orig, h_orig, frame_end


# ── the probe's stream ─────────────────────────────────────────────────────────


def _read_exact(fh: IO[bytes], view: memoryview) -> None:
    got = 0
    while got < len(view):
        n = fh.readinto(view[got:])  # type: ignore[attr-defined]
        if not n:
            raise EOFError(f"probe stream ended after {got} of {len(view)} bytes")
        got += n


def read_record(fh: IO[bytes]) -> dict[str, Any]:
    head = bytearray(8)
    _read_exact(fh, memoryview(head))
    (length,) = struct.unpack("<Q", head)
    body = bytearray(length)
    _read_exact(fh, memoryview(body))
    return json.loads(body.decode("utf-8"))


# ── comparisons (device-agnostic torch ops) ────────────────────────────────────


def compare_decoded(native_u8: Any, oracle_u8: Any) -> dict[str, Any]:
    """Planar uint8 [3, H, W] vs [3, H, W]."""
    import torch

    if torch.equal(native_u8, oracle_u8):
        return {"equal": True}
    d = (native_u8.to(torch.int16) - oracle_u8.to(torch.int16)).abs()
    nz = d[d > 0]
    return {
        "equal": False,
        "differing_values": int(nz.numel()),
        "max_abs": int(d.max()),
        "abs_hist": torch.bincount(nz.flatten().to(torch.int64), minlength=256)
        .cpu()
        .tolist(),
        "differing_per_channel": [int((d[c] > 0).sum()) for c in range(d.shape[0])],
    }


def compare_float_bits(native_f32: Any, ref_f32: Any) -> dict[str, Any]:
    """float32 tensors, bit for bit."""
    import torch

    a = native_f32.view(torch.int32)
    b = ref_f32.view(torch.int32)
    if torch.equal(a, b):
        return {"equal": True}
    ne = a != b
    return {
        "equal": False,
        "differing_values": int(ne.sum()),
        "max_abs": float((native_f32 - ref_f32).abs().max()),
        "max_ulp": int((a.to(torch.int64) - b.to(torch.int64)).abs().max()),
    }


def oracle_normalize_table() -> dict[str, list[str]]:
    """The oracle op on CUDA for 0..255, from a contiguous and a permuted view."""
    import torch

    values = torch.arange(256, dtype=torch.uint8, device="cuda")
    hwc = values.view(16, 16, 1).expand(16, 16, 3).contiguous()
    out = torch.zeros((3, 16, 16), dtype=torch.float32, device="cuda")
    oracle_ingest(hwc, out)
    flat = torch.empty(256, dtype=torch.float32, device="cuda")
    torch.div(values, 255.0, out=flat)
    hexes = lambda t: [t[i : i + 1].numpy().tobytes().hex() for i in range(256)]  # noqa: E731
    return {"permuted": hexes(out[0].flatten().cpu()), "contiguous": hexes(flat.cpu())}


class Section:
    """Accumulates one section's per-frame results for a sequence."""

    def __init__(self) -> None:
        self.frames = 0
        self.equal_frames = 0
        self.differing_values = 0
        self.max_abs: float = 0
        self.max_ulp = 0
        self.abs_hist: list[int] | None = None
        self.per_channel: list[int] | None = None
        self.first_differing: list[dict[str, Any]] = []

    def add(self, frame: int, file: str, r: dict[str, Any]) -> None:
        self.frames += 1
        if r["equal"]:
            self.equal_frames += 1
            return
        self.differing_values += r["differing_values"]
        self.max_abs = max(self.max_abs, r["max_abs"])
        self.max_ulp = max(self.max_ulp, r.get("max_ulp", 0))
        if "abs_hist" in r:
            h = r["abs_hist"]
            self.abs_hist = (
                h
                if self.abs_hist is None
                else [x + y for x, y in zip(self.abs_hist, h)]
            )
            pc = r["differing_per_channel"]
            self.per_channel = (
                pc
                if self.per_channel is None
                else [x + y for x, y in zip(self.per_channel, pc)]
            )
        if len(self.first_differing) < FIRST_DIFFS_KEPT:
            self.first_differing.append(
                {
                    "frame": frame,
                    "file": file,
                    **{k: v for k, v in r.items() if k != "abs_hist"},
                }
            )

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verdict": "EXACT" if self.equal_frames == self.frames else "DIFFERS",
            "frames": self.frames,
            "equal_frames": self.equal_frames,
            "differing_frames": self.frames - self.equal_frames,
        }
        if self.equal_frames != self.frames:
            out.update(
                differing_values=self.differing_values,
                max_abs=self.max_abs,
                first_differing=self.first_differing,
            )
            if self.abs_hist is not None:
                out["abs_hist"] = {str(i): n for i, n in enumerate(self.abs_hist) if n}
                out["differing_per_channel"] = self.per_channel
            else:
                out["max_ulp"] = self.max_ulp
        return out

    @staticmethod
    def merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
        frames = sum(p["frames"] for p in parts)
        equal = sum(p["equal_frames"] for p in parts)
        out: dict[str, Any] = {
            "verdict": "EXACT" if frames == equal else "DIFFERS",
            "frames": frames,
            "equal_frames": equal,
            "differing_frames": frames - equal,
        }
        if frames != equal:
            out["differing_values"] = sum(p.get("differing_values", 0) for p in parts)
            out["max_abs"] = max(p.get("max_abs", 0) for p in parts)
        return out


# ── one sequence ───────────────────────────────────────────────────────────────


def run_sequence(
    args: argparse.Namespace, seq: str, frames_out: IO[str]
) -> dict[str, Any]:
    import numpy as np
    import torch

    from saccade.perception.eval.streaming import TorchvisionGpuStreamer

    seq_path = args.data_root / args.split / seq
    w, h, frame_end = oracle_sequence_bounds(seq_path, args.max_frames)
    streamer = TorchvisionGpuStreamer(seq_path / "img1")
    oracle_files = [Path(f).name for f in streamer.img_files]

    cmd = [str(args.probe), "--config", str(args.config), "--sequence", str(seq_path)]
    if args.max_frames:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.force_decoupled:
        cmd.append("--force-decoupled")
    stderr_path = args.out / f"probe_{seq}.stderr"
    with stderr_path.open("wb") as err:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err, bufsize=1 << 20
        )
        assert proc.stdout is not None
        try:
            pre = read_record(proc.stdout)
            if pre.get("format") != STREAM_FORMAT:
                raise RuntimeError(f"{seq}: probe stream format {pre.get('format')!r}")
            if (pre["im_width"], pre["im_height"]) != (w, h):
                raise RuntimeError(
                    f"{seq}: geometry native {pre['im_width']}x{pre['im_height']} vs oracle {w}x{h}"
                )
            if (
                pre["listed"] != oracle_files
                or pre["frames"] != oracle_files[:frame_end]
            ):
                raise RuntimeError(
                    f"{seq}: native frame listing differs from the oracle's"
                )
            if bool(pre["force_decoupled"]) != bool(args.force_decoupled):
                raise RuntimeError(f"{seq}: probe force_decoupled mismatch")

            n = 3 * h * w
            u8 = np.empty(n, dtype=np.uint8)
            f32 = np.empty(n, dtype=np.float32)
            sections = {k: Section() for k in ("decoder", "normalize", "ingest")}
            paths: dict[str, int] = {}
            oracle_buffer = torch.zeros((3, h, w), dtype=torch.float32, device="cuda")
            ref_buffer = torch.zeros((3, h, w), dtype=torch.float32, device="cuda")
            oracle_iter = iter(streamer)
            for k in range(1, frame_end + 1):
                rec = read_record(proc.stdout)
                if (
                    rec.get("frame") != k
                    or rec["u8_bytes"] != n
                    or rec["f32_bytes"] != 4 * n
                ):
                    raise RuntimeError(f"{seq}: unexpected probe record {rec}")
                _read_exact(proc.stdout, memoryview(u8))
                _read_exact(proc.stdout, memoryview(f32).cast("B"))
                file = rec["file"]
                if file != oracle_files[k - 1]:
                    raise RuntimeError(
                        f"{seq}: frame {k} is {file} natively, {oracle_files[k - 1]} in the oracle"
                    )
                paths[rec["path"]] = paths.get(rec["path"], 0) + 1

                frame_hwc = next(oracle_iter)
                oracle_ingest(frame_hwc, oracle_buffer)
                oracle_u8 = frame_hwc.permute(2, 0, 1).contiguous()
                native_u8 = torch.from_numpy(u8).to("cuda").view(3, h, w)
                native_f32 = torch.from_numpy(f32).to("cuda").view(3, h, w)
                ref_buffer.zero_()
                oracle_ingest(native_u8.permute(1, 2, 0), ref_buffer)

                r_dec = compare_decoded(native_u8, oracle_u8)
                r_norm = compare_float_bits(native_f32, ref_buffer)
                r_ing = compare_float_bits(native_f32, oracle_buffer)
                sections["decoder"].add(k, file, r_dec)
                sections["normalize"].add(k, file, r_norm)
                sections["ingest"].add(k, file, r_ing)
                oracle_u8_host = oracle_u8.cpu().numpy()
                oracle_f32_host = oracle_buffer.cpu().numpy()
                frames_out.write(
                    json.dumps(
                        {
                            "sequence": seq,
                            "frame": k,
                            "file": file,
                            "path": rec["path"],
                            "decoder_equal": r_dec["equal"],
                            "normalize_equal": r_norm["equal"],
                            "ingest_equal": r_ing["equal"],
                            "sha256": {
                                "oracle_u8": hashlib.sha256(
                                    oracle_u8_host.tobytes()
                                ).hexdigest(),
                                "native_u8": hashlib.sha256(memoryview(u8)).hexdigest(),
                                "oracle_f32": hashlib.sha256(
                                    oracle_f32_host.tobytes()
                                ).hexdigest(),
                                "native_f32": hashlib.sha256(
                                    memoryview(f32).cast("B")
                                ).hexdigest(),
                            },
                        }
                    )
                    + "\n"
                )
                if args.progress and k % 100 == 0:
                    print(f"  {seq} {k}/{frame_end}", file=sys.stderr, flush=True)
            end = read_record(proc.stdout)
            if end != {"end": True, "frames": frame_end}:
                raise RuntimeError(f"{seq}: unexpected end record {end}")
        finally:
            streamer._stop_worker()
            proc.stdout.close()
            rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{seq}: probe exited {rc} (see {stderr_path.name})")
    return {
        "im_width": w,
        "im_height": h,
        "frames": frame_end,
        "listed": len(oracle_files),
        "nvjpeg": pre["nvjpeg"],
        "decode_paths": paths,
        "normalize_lut_f32_hex": pre["normalize_lut_f32_hex"],
        **{k: s.summary() for k, s in sections.items()},
    }


# ── provenance of the two decoders ─────────────────────────────────────────────


def _probe_nvjpeg(probe: Path) -> Path | None:
    out = subprocess.run(
        ["ldd", str(probe)], capture_output=True, text=True, check=False
    ).stdout
    for line in out.splitlines():
        if line.strip().startswith("libnvjpeg") and "=>" in line:
            return Path(line.split("=>")[1].split("(")[0].strip())
    return None


def _oracle_nvjpeg() -> Path | None:
    import torchvision

    libs = sorted(
        (Path(torchvision.__file__).parent.parent / "torchvision.libs").glob(
            "libnvjpeg*.so*"
        )
    )
    return libs[0] if len(libs) == 1 else None


def decoder_provenance(probe: Path) -> dict[str, Any]:
    import torch
    import torchvision

    native, oracle = _probe_nvjpeg(probe), _oracle_nvjpeg()
    nat_sha = _sha256_file(native) if native else None
    ora_sha = _sha256_file(oracle) if oracle else None
    probe_ldd = subprocess.run(
        ["ldd", str(probe)], capture_output=True, text=True, check=False
    ).stdout
    return {
        "probe": {
            "path": str(probe),
            "sha256": _sha256_file(probe),
            "nvjpeg": str(native) if native else None,
            "nvjpeg_sha256": nat_sha,
            "links_python_or_torch": any(
                s in probe_ldd for s in ("libpython", "libtorch")
            ),
        },
        "oracle": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "decoder": "saccade.perception.eval.streaming.TorchvisionGpuStreamer",
            "ingest_op": ORACLE_INGEST_OP,
            "nvjpeg": str(oracle) if oracle else None,
            "nvjpeg_sha256": ora_sha,
        },
        "nvjpeg_same_binary": nat_sha is not None and nat_sha == ora_sha,
        "gpu": torch.cuda.get_device_name(0),
    }


# ── run ────────────────────────────────────────────────────────────────────────


def compare_against(frames_path: Path, prior_report: Path) -> dict[str, Any]:
    prior_frames = prior_report.parent / "frames.jsonl"

    def load(p: Path) -> dict[tuple[str, int], dict[str, str]]:
        with p.open() as fh:
            return {
                (r["sequence"], r["frame"]): r["sha256"] for r in map(json.loads, fh)
            }

    now, before = load(frames_path), load(prior_frames)
    keys = sorted(set(now) & set(before))
    differing = {
        side: [list(key) for key in keys if now[key][side] != before[key][side]][
            :FIRST_DIFFS_KEPT
        ]
        for side in ("oracle_u8", "native_u8", "oracle_f32", "native_f32")
    }
    return {
        "report": str(prior_report),
        "frames_compared": len(keys),
        "frames_only_here": len(set(now) - set(before)),
        "frames_only_there": len(set(before) - set(now)),
        "differing_first": differing,
        "verdict": "IDENTICAL"
        if keys and not any(differing.values()) and len(now) == len(before)
        else "DIFFERS",
    }


def run(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("needs CUDA (the oracle decodes on the GPU)")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "host": platform.node(),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "config": {"path": str(args.config), "sha256": _sha256_file(args.config)},
        "force_decoupled": bool(args.force_decoupled),
        "max_frames": args.max_frames,
        **decoder_provenance(args.probe),
    }
    table = oracle_normalize_table()
    seqs: dict[str, Any] = {}
    with (args.out / "frames.jsonl").open("w") as frames_out:
        for seq in args.sequences:
            print(f"{seq} ...", file=sys.stderr, flush=True)
            seqs[seq] = run_sequence(args, seq, frames_out)
    luts = {s["normalize_lut_f32_hex"] == table["permuted"] for s in seqs.values()}
    report["normalize_table"] = {
        "values": 256,
        "oracle_contiguous_equals_permuted": table["contiguous"] == table["permuted"],
        "differing_values": sum(
            a != b
            for a, b in zip(
                next(iter(seqs.values()))["normalize_lut_f32_hex"], table["permuted"]
            )
        ),
        "verdict": "EXACT"
        if luts == {True} and table["contiguous"] == table["permuted"]
        else "DIFFERS",
    }
    for s in seqs.values():
        del s["normalize_lut_f32_hex"]
    report["sequences"] = seqs
    report["totals"] = {
        k: Section.merge([s[k] for s in seqs.values()])
        for k in ("decoder", "normalize", "ingest")
    }
    report["totals"]["decode_paths"] = {
        p: sum(s["decode_paths"].get(p, 0) for s in seqs.values())
        for p in ("hardware_batched", "decoupled")
    }
    report["verdicts"] = {
        "decoder": report["totals"]["decoder"]["verdict"],
        "normalize": report["totals"]["normalize"]["verdict"],
        "normalize_table": report["normalize_table"]["verdict"],
        "ingest": report["totals"]["ingest"]["verdict"],
    }
    ok = set(report["verdicts"].values()) == {"EXACT"}
    if args.against is not None:
        report["against"] = compare_against(args.out / "frames.jsonl", args.against)
        ok = ok and report["against"]["verdict"] == "IDENTICAL"
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verdicts": report["verdicts"],
                "decode_paths": report["totals"]["decode_paths"],
                **({"against": report["against"]["verdict"]} if args.against else {}),
            }
        )
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out", type=Path, required=True, help="run directory (claimed with open_run)"
    )
    ap.add_argument("--probe", type=Path, default=PROBE)
    ap.add_argument("--config", type=Path, default=RESOLVED_CONFIG)
    ap.add_argument("--data-root", type=Path, default=project_root / "datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--sequences",
        type=lambda s: [x for x in s.split(",") if x],
        default=list(DEFAULT_SEQUENCES),
    )
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--force-decoupled",
        action="store_true",
        help="negative control: decoupled path only",
    )
    ap.add_argument(
        "--against",
        type=Path,
        default=None,
        help="earlier report.json to compare hashes with",
    )
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)
    args.out = args.out.resolve()
    args.probe = args.probe.resolve()
    args.config = args.config.resolve()
    if not args.probe.exists():
        ap.error(
            f"{args.probe} not built (cmake --build build --target saccade_ingest_probe)"
        )
    if args.out.exists() and any(args.out.iterdir()):
        ap.error(f"{args.out} is not empty")
    for p in (project_root, project_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from scripts.provenance.run_manifest import open_run

    # ADR 021 AP-2: claim the run directory before the first byte.
    open_run(
        args.out,
        produced_by="diagnostic",
        preset="mamba_whole_graph",
        detector="SDP",
        dataset=f"MOT17 {args.split}",
    )
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 -- reported, exit 2
        print(f"native_ingest_parity: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
