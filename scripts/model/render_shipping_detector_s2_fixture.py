#!/usr/bin/env python3
"""Render the native S2 golden fixture from the oracle's compiled S2 (#465 PR-8).

Developer tooling (``developer_build_debug``). Writes
``tests/native/fixtures/shipping_detector_s2.json`` for
``tests/native/test_shipping_detector_s2.cpp``: per case, a generator spec
(deterministic synthetic head outputs -- splitmix64 integers scaled by a power
of two, so every input is exactly representable and the C++ test rebuilds the
same float32 bits), and the oracle's output for those inputs:
``_whole_graph_fn``'s tail (``native_detector_parity.oracle_s2``: the
torch.compile'd ``_postprocess_mamba_fixed`` + the coordinate scaling) before
and after scaling, as raw float32 bytes (base64). It also records whether the
*eager* S2 gives the same bytes, so the fixture shows it tells the compiled
lowering from eager (§12.1 of docs/reference/native_runtime_resolved_config.md).

The cases target S2's edges: random logits; saturated logits (sigmoid == 1.0
on many classes and anchors: class-argmax and top-k ties); coarse integer
logits (massive ties); very negative logits (subnormal and zero scores,
``div.full``'s scaled branch); dense sweeps of single-class logits over chosen
ranges (every swept anchor reaches the top-k); NaN / +-inf logits and an
infinite box distance; three sequence geometries (exact and inexact
coordinate scales).

Needs CUDA (the oracle's S2 is compiled for it)::

    .venv/bin/python tools/resctl.py run gpu0 -- \\
        .venv/bin/python scripts/model/render_shipping_detector_s2_fixture.py [--check]
"""
# status: active

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[2]
FIXTURE = project_root / "tests/native/fixtures/shipping_detector_s2.json"
HARNESS = project_root / "scripts/eval/diagnostics/native_detector_parity.py"
FORMAT = "saccade.shipping_detector_s2_fixture/v1"
IMG_SIZE = 640
MAX_DET = 300
NUM_CLASSES = 80
CONF_THR = 0.001  # host_params.detector.build.conf_thr (unread by the fixed S2)
STRIDES = (8, 16, 32)
SIDES = tuple(IMG_SIZE // s for s in STRIDES)
UNIFORM_RANDOM_REG = {"mode": "uniform", "lo": 0, "span": 20 << 16, "frac": 16}


def _uniform(lo: float, hi: float, frac: int) -> dict[str, Any]:
    return {
        "mode": "uniform",
        "lo": int(lo * (1 << frac)),
        "span": int((hi - lo) * (1 << frac)),
        "frac": frac,
    }


def _sweep(lo: float, hi: float, frac: int) -> dict[str, Any]:
    lo_i = int(round(lo * (1 << frac)))
    step = int((hi - lo) * (1 << frac)) // MAX_DET
    return {
        "anchor_stride": 28,
        "count": MAX_DET,
        "cls_channel": 0,
        "lo": lo_i,
        "step": step,
        "frac": frac,
    }


NEG_INF_CLS = {"mode": "fill", "value": "-inf"}
CASES: list[dict[str, Any]] = [
    {"name": "random_1080p", "seed": 1, "width": 1920, "height": 1080,
     "cls": _uniform(-14, 4, 16), "reg": UNIFORM_RANDOM_REG},
    {"name": "random_480p", "seed": 2, "width": 640, "height": 480,
     "cls": _uniform(-14, 4, 16), "reg": UNIFORM_RANDOM_REG},
    {"name": "random_inexact_scale", "seed": 3, "width": 1000, "height": 563,
     "cls": _uniform(-14, 4, 16), "reg": UNIFORM_RANDOM_REG},
    {"name": "saturated", "seed": 4, "width": 1920, "height": 1080,
     "cls": _uniform(10, 30, 12), "reg": UNIFORM_RANDOM_REG},
    {"name": "coarse_ties", "seed": 5, "width": 1920, "height": 1080,
     "cls": {"mode": "uniform", "lo": -4, "span": 9, "frac": 0}, "reg": UNIFORM_RANDOM_REG},
    {"name": "deep_negative", "seed": 6, "width": 1920, "height": 1080,
     "cls": _uniform(-104, -80, 16), "reg": UNIFORM_RANDOM_REG},
    {"name": "sweep_wide", "seed": 7, "width": 1920, "height": 1080,
     "cls": NEG_INF_CLS, "reg": UNIFORM_RANDOM_REG, "sweep": _sweep(-20, 20, 16)},
    {"name": "sweep_div_full_scaled", "seed": 8, "width": 1920, "height": 1080,
     "cls": NEG_INF_CLS, "reg": UNIFORM_RANDOM_REG, "sweep": _sweep(-88.8, -86, 16)},
    {"name": "sweep_near_zero", "seed": 9, "width": 1920, "height": 1080,
     "cls": NEG_INF_CLS, "reg": UNIFORM_RANDOM_REG, "sweep": _sweep(-0.01, 0.01, 24)},
    {"name": "sweep_near_saturation", "seed": 10, "width": 640, "height": 480,
     "cls": NEG_INF_CLS, "reg": UNIFORM_RANDOM_REG, "sweep": _sweep(15, 18, 16)},
    {"name": "nonfinite", "seed": 11, "width": 1920, "height": 1080,
     "cls": _uniform(-14, 4, 16), "reg": UNIFORM_RANDOM_REG,
     "specials": [
         {"tensor": "cls", "anchor": 100, "channel": 3, "value": "nan"},
         {"tensor": "cls", "anchor": 200, "channel": 5, "value": "nan"},
         {"tensor": "cls", "anchor": 200, "channel": 7, "value": "nan"},
         {"tensor": "cls", "anchor": 300, "channel": 1, "value": "inf"},
         {"tensor": "cls", "anchor": 400, "channel": 2, "value": "-inf"},
         {"tensor": "cls", "anchor": 6500, "channel": 0, "value": "inf"},
         {"tensor": "reg", "anchor": 6500, "channel": 2, "value": "inf"},
         {"tensor": "cls", "anchor": 8300, "channel": 9, "value": "inf"},
         {"tensor": "cls", "anchor": 8300, "channel": 4, "value": "inf"},
     ]},
]  # fmt: skip


# ── deterministic inputs (mirrored in test_shipping_detector_s2.cpp) ───────────


def splitmix64(x: Any) -> Any:
    import numpy as np

    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _special(name: str) -> float:
    return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}[name]


def make_inputs(case: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """float32 numpy arrays: cls [1, nc, s, s] x3, reg [1, 4, s, s] x3."""
    import numpy as np

    seed = np.uint64(case["seed"])
    tensors = []
    for stream, (kind, channels) in enumerate(
        [("cls", NUM_CLASSES)] * 3 + [("reg", 4)] * 3
    ):
        side = SIDES[stream % 3]
        n = channels * side * side
        spec = case[kind]
        if spec["mode"] == "fill":
            arr = np.full(n, _special(spec["value"]), dtype=np.float32)
        else:
            idx = np.arange(n, dtype=np.uint64)
            u = splitmix64(
                (seed << np.uint64(32)) | (np.uint64(stream) << np.uint64(24)) | idx
            )
            ints = np.int64(spec["lo"]) + (
                (u >> np.uint64(40)) % np.uint64(spec["span"])
            ).astype(np.int64)
            arr = np.ldexp(ints.astype(np.float64), -spec["frac"]).astype(np.float32)
        tensors.append(arr.reshape(1, channels, side, side))
    bases = [0, SIDES[0] ** 2, SIDES[0] ** 2 + SIDES[1] ** 2]

    def locate(anchor: int) -> tuple[int, int, int]:
        level = 0 if anchor < bases[1] else (1 if anchor < bases[2] else 2)
        local = anchor - bases[level]
        return level, local // SIDES[level], local % SIDES[level]

    sw = case.get("sweep")
    if sw is not None:
        for j in range(sw["count"]):
            level, y, x = locate(j * sw["anchor_stride"])
            v = np.ldexp(np.float64(sw["lo"] + j * sw["step"]), -sw["frac"])
            tensors[level][0, sw["cls_channel"], y, x] = np.float32(v)
    for sp in case.get("specials", []):
        level, y, x = locate(sp["anchor"])
        t = tensors[level if sp["tensor"] == "cls" else 3 + level]
        t[0, sp["channel"], y, x] = np.float32(_special(sp["value"]))
    return tensors[:3], tensors[3:]


# ── the oracle ────────────────────────────────────────────────────────────────


def render() -> dict[str, Any]:
    build_path = Path(os.environ.get("SACCADE_BUILD_PATH", project_root / "build"))
    for p in (build_path, project_root / "src", project_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import torch
    import triton

    from saccade.perception.temporal_yolo import mamba_gated_detector as mgd

    spec = importlib.util.spec_from_file_location("native_detector_parity", HARNESS)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    dev = "cuda"
    stride = torch.tensor([8.0, 16.0, 32.0], device=dev)
    shapes = [(1, NUM_CLASSES, s, s) for s in SIDES]
    anchors, anchor_strides = mgd._precompute_anchor_grid(stride, shapes)
    x_idx = torch.tensor([0, 2], device=dev, dtype=torch.long)
    y_idx = torch.tensor([1, 3], device=dev, dtype=torch.long)
    out_cases = []
    with torch.no_grad():
        for case in CASES:
            cls_np, reg_np = make_inputs(case)
            cls = [torch.from_numpy(a).to(dev) for a in cls_np]
            reg = [torch.from_numpy(a).to(dev) for a in reg_np]
            sx = torch.ones(1, device=dev)
            sy = torch.ones(1, device=dev)
            sx.fill_(case["width"] / IMG_SIZE)  # set_whole_graph_img_dims
            sy.fill_(case["height"] / IMG_SIZE)
            common = dict(
                stride=stride, conf_thr=CONF_THR, max_det=MAX_DET, nms_pad=0, anchors=anchors,
                anchor_strides=anchor_strides, small_p3=0.0, sx=sx, sy=sy, x_idx=x_idx, y_idx=y_idx,
            )  # fmt: skip
            raw, scaled = harness.oracle_s2(mgd, cls, reg, **common)
            _, eager = harness.oracle_s2(mgd, cls, reg, **common, compile_=False)
            raw_b = raw[0].cpu().numpy().tobytes()
            scaled_b = scaled[0].cpu().numpy().tobytes()
            eager_b = eager[0].cpu().numpy().tobytes()
            out_cases.append(
                {
                    **case,
                    "raw_f32_b64": base64.b64encode(raw_b).decode(),
                    "scaled_f32_b64": base64.b64encode(scaled_b).decode(),
                    "scaled_sha256": hashlib.sha256(scaled_b).hexdigest(),
                    "eager_scaled_sha256": hashlib.sha256(eager_b).hexdigest(),
                    "eager_equal": eager_b == scaled_b,
                }
            )
    return {
        "format": FORMAT,
        "generator": "scripts/model/render_shipping_detector_s2_fixture.py",
        "oracle": "native_detector_parity.oracle_s2 (torch.compile'd _postprocess_mamba_fixed + coordinate scaling)",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "img_size": IMG_SIZE,
        "max_det": MAX_DET,
        "num_classes": NUM_CLASSES,
        "strides": list(STRIDES),
        "cases": out_cases,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check", action="store_true", help="compare with the committed fixture"
    )
    args = ap.parse_args(argv)
    text = json.dumps(render(), indent=1) + "\n"
    if args.check:
        if FIXTURE.read_text() != text:
            print(
                f"{FIXTURE.relative_to(project_root)} is stale; re-render it",
                file=sys.stderr,
            )
            return 1
        print("fixture is fresh")
        return 0
    FIXTURE.write_text(text)
    print(f"wrote {FIXTURE.relative_to(project_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
