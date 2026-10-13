"""#537 C0: pin what the tracker paths do when the native extension is missing.

Characterization only (C1, B0-F02): these record the behavior at the C0
baseline. They do not choose where a missing capability should be rejected;
that stays open under REQ-535-02-01 / 02-04. Each case runs in a fresh
interpreter with ``saccade_tracking_ext`` blocked, so a locally built
extension does not change the result.
"""

# scope: perception, tracking
# function: diagnostic
# lifecycle: active

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]

KEYS = {"is_cuda", "dets3", "dets0", "binding", "get_tracker"}
_BLOCK = "import sys\nsys.modules['saccade_tracking_ext'] = None\n"

_FACADE = """
import torch
from saccade.perception.tracking.tracker_gpu import GPUByteTracker
t = GPUByteTracker(max_objects=16, embedding_dim=0)
print('is_cuda', t.is_cuda)
b = torch.tensor([[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 90, 90]], dtype=torch.float32)
s = torch.tensor([0.9, 0.8, 0.7])
c = torch.zeros(3, dtype=torch.int32)
for name, n in (('dets3', 3), ('dets0', 0)):
    try:
        print(name, repr(t.update(b[:n], s[:n], c[:n])))
    except Exception as e:
        print(name, type(e).__name__, e)
"""

_DISPATCHER = """
from unittest import mock
import saccade.perception.dispatcher as d
print('binding', d.GPUByteTracker)
disp = d.AsyncDispatcher(detector=mock.MagicMock(), resource_manager=mock.MagicMock())
try:
    disp.get_tracker('cam0')
    print('get_tracker ok')
except Exception as e:
    print('get_tracker', type(e).__name__, e)
"""


def _run(body: str, *, hide_cuda: bool) -> list[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH")) if p
    )
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK + body],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return [line for line in proc.stdout.splitlines() if line.split(" ", 1)[0] in KEYS]


def test_c1_facade_without_extension_or_cuda_raises_before_stub() -> None:
    assert _run(_FACADE, hide_cuda=True) == [
        "is_cuda False",
        "dets3 RuntimeError No CUDA GPUs are available",
        "dets0 RuntimeError No CUDA GPUs are available",
    ]


@pytest.mark.gpu
def test_c1_facade_stub_output_matches_a_legal_empty_frame() -> None:
    assert torch.cuda.is_available()
    # Three detections and zero detections give the same result: a missing
    # capability cannot be told apart from an empty frame (REQ-535-02-01).
    assert _run(_FACADE, hide_cuda=False) == [
        "is_cuda False",
        "dets3 []",
        "dets0 []",
    ]


def test_b0_f02_dispatcher_without_extension_fails_at_first_tracker() -> None:
    assert _run(_DISPATCHER, hide_cuda=True) == [
        "binding typing.Any",
        "get_tracker TypeError Any cannot be instantiated",
    ]
