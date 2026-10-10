"""``saccade_track`` Gate A at the built entrypoint: refusals make no CUDA call (#536).

docs/architecture/ship_export_contracts_536.md CC-536-01-02 (N-T2). The checks
themselves are pinned CUDA-free by ``tests/native/test_shipping_preflight.cpp``
(CI); these pin that the built entrypoints run them after ``<out>`` is taken
and before the first CUDA call, and that a refusal never reaches CUDA.

"No CUDA call" is observed, not inferred. The CUDA driver reads
``CUDA_INJECTION64_PATH`` inside ``cuInit`` and dlopens that path (how
profilers attach); with ``LD_DEBUG=files`` the dynamic loader logs that
dlopen. The path given here does not exist, so the run is otherwise
unchanged. A log without it means the driver was never initialized in that
process. Loading ``libcuda.so.1`` is not the signal: ``libcublasLt``'s
constructor dlopens it before ``main`` in every run.

* positive control: a self-consistent stand-in bundle (a copy of the lineage
  whose three sha256 are those of small stand-in files under a temporary model
  root) passes Gate A -- it checks bytes, not their source -- so the run prints
  ``preflight passed``, builds the runtime (``cuInit`` is logged) and is
  refused by the detector load (Gate B: the stand-in operator library is not
  an ELF file). Needs a CUDA driver on the host;
* negative controls, each one change to that bundle or the real one: a
  replaced head artifact; the stand-in lineage with the committed attestation
  (bound to the frozen lineage's bytes); the real model files without the
  attestation (the operator library built here is not the one the frozen
  lineage names); a sequence without seqinfo.ini; a --report directory that
  does not exist. Each exits 2 with no ``preflight passed``, a ``failed``
  journal whose message starts ``preflight:``, no sequence output and no
  ``cuInit``. They need no GPU.

By default, skips when ``build/shipping/saccade_track`` has not been built.
Set ``SACCADE_SHIPPING_TEST_PREFIX`` to an installed tree to run its actual
``bin/saccade_track`` launcher with its installed config, lineage and
attestation. An explicit prefix must be usable; it never skips or falls back
to a build binary. Installed mode collects only shipping cases. Its positive
control uses the installed model files and invalid JPEG inputs: Gate A and
CUDA initialization pass, then the first sequence fails in the decoder. It
does not copy the operator library outside the path the auditor accepts.
This proves the observer reaches CUDA; successful inference and completion
need separate installed-binary GPU parity evidence.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build" / "shipping"
_PREFIX = os.environ.get("SACCADE_SHIPPING_TEST_PREFIX")
PREFIX = Path(_PREFIX).expanduser().resolve() if _PREFIX else None
INSTALLED = _PREFIX is not None
MODEL_ROOT = PREFIX / "share" / "saccade" if PREFIX else REPO
CONFIG = MODEL_ROOT / "configs" / "shipping" / "mamba_whole_graph.resolved.json"
FROZEN_LINEAGE = (
    MODEL_ROOT
    / "models"
    / "yolo"
    / "mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json"
)
LINEAGE = (
    FROZEN_LINEAGE
    if INSTALLED
    else REPO / "tests" / "native" / "fixtures" / "shipping_head_lineage.json"
)
ATTESTATION = (
    MODEL_ROOT / "configs" / "shipping" / "mamba_head_realization.attestation.json"
)
BINARIES = (
    ["saccade_track"] if INSTALLED else ["saccade_track", "saccade_track_measurement"]
)
JOURNAL = "saccade_track.journal.json"
PROBE = "cuda_init_probe.so"
SEQS = ("SEQ-A", "SEQ-B")
STANDINS = {
    "op_library": b"operator library stand-in",
    "torchscript": b"head artifact stand-in",
    "backbone_engine": b"backbone engine stand-in",
}

pytestmark = pytest.mark.skipif(
    not INSTALLED and not (BUILD / "saccade_track").exists(),
    reason="build/shipping/saccade_track not built",
)


@pytest.fixture(scope="module", autouse=True)
def _require_installed_tree() -> None:
    if INSTALLED:
        assert PREFIX is not None, (
            "SACCADE_SHIPPING_TEST_PREFIX must name an installed tree"
        )
        launcher = PREFIX / "bin" / "saccade_track"
        assert launcher.is_file() and os.access(launcher, os.X_OK), launcher
        assert (PREFIX / "libexec" / "saccade_track").is_file(), PREFIX
        for path in (CONFIG, FROZEN_LINEAGE, ATTESTATION):
            assert path.is_file(), path


def _has_cuda_driver() -> bool:
    return Path("/dev/dxg").exists() or Path("/dev/nvidiactl").exists()


def _write_sequence(seq: Path, seq_length: int = 3) -> None:
    (seq / "img1").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq.name}\nimWidth=8\nimHeight=6\nseqLength={seq_length}\n"
    )
    for k in range(1, seq_length + 1):
        (seq / "img1" / f"{k:06d}.jpg").write_bytes(b"not a jpeg")


def _standin_bundle(tmp_path: Path) -> tuple[Path, Path]:
    """A model root with stand-in files and a lineage bound to them."""
    root = tmp_path / "root"
    lineage = json.loads(LINEAGE.read_text())
    entries = {
        "op_library": lineage["op_library"],
        "torchscript": lineage["torchscript"],
        "backbone_engine": lineage["companions"]["backbone_engine"],
    }
    for key, entry in entries.items():
        f = root / entry["path"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(STANDINS[key])
        entry["sha256"] = hashlib.sha256(STANDINS[key]).hexdigest()
    path = root / "lineage.json"
    path.write_text(json.dumps(lineage, indent=2) + "\n")
    return root, path


def _run(
    tmp_path: Path,
    *,
    binary: str = "saccade_track",
    lineage: Path,
    model_root: Path,
    attestation: Path | None = None,
    report: Path | None = None,
    sequences: list[Path] | None = None,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run the entrypoint; returns its result and whether cuInit ran."""
    exe = PREFIX / "bin" / binary if PREFIX else BUILD / binary
    if not exe.exists():
        if INSTALLED:
            pytest.fail(f"installed entrypoint {exe} is missing")
        pytest.skip(f"{exe.relative_to(REPO)} not built")
    if sequences is None:
        sequences = [tmp_path / "seqs" / s for s in SEQS]
        for s in sequences:
            if not s.exists():
                _write_sequence(s)
    cmd = [str(exe), "--config", str(CONFIG), "--lineage", str(lineage)]
    if attestation is not None:
        cmd += ["--attestation", str(attestation)]
    cmd += ["--model-root", str(model_root), "--out", str(tmp_path / "out")]
    if report is not None:
        cmd += ["--report", str(report)]
    cmd += [str(s) for s in sequences]
    ld = tmp_path / "ld_debug"
    ld.mkdir(exist_ok=True)
    for old in ld.iterdir():
        old.unlink()
    env = dict(os.environ)
    env.update(
        {
            "CUDA_INJECTION64_PATH": str(tmp_path / PROBE),
            "LD_DEBUG": "files",
            "LD_DEBUG_OUTPUT": str(ld / "log"),
        }
    )
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, env=env, check=False
    )
    logs = list(ld.iterdir())
    assert logs, "LD_DEBUG wrote no log: the observer is not working"
    cuda_init = any(PROBE in p.read_text(errors="replace") for p in logs)
    return r, cuda_init


def _run_id(r: subprocess.CompletedProcess[str], name: str) -> str:
    m = re.fullmatch(rf"{name}: run_id ([0-9a-f]{{32}})", r.stderr.splitlines()[0])
    assert m, r.stderr
    return m.group(1)


def _assert_refused_in_gate_a(
    tmp_path: Path,
    r: subprocess.CompletedProcess[str],
    cuda_init: bool,
    needle: str,
    binary: str = "saccade_track",
) -> None:
    assert r.returncode == 2, r.stderr
    assert not cuda_init, "a Gate A refusal initialized CUDA"
    assert f"{binary}: preflight passed" not in r.stderr, r.stderr
    j = json.loads((tmp_path / "out" / JOURNAL).read_text())
    assert j["run_id"] == _run_id(r, binary)
    assert j["state"] == "failed" and j["failure"]["sequence"] is None
    message = j["failure"]["message"]
    assert message.startswith("preflight: ") and needle in message, message
    assert f"{binary}: {message}" in r.stderr.splitlines()
    assert j["identity"] == {"level": None}
    assert all(s["state"] == "pending" for s in j["sequences"])
    assert not list((tmp_path / "out").glob("*.txt"))


@pytest.mark.skipif(not _has_cuda_driver(), reason="no CUDA driver on this host")
def test_positive_control_passes_gate_a_then_initializes_cuda(tmp_path: Path) -> None:
    if INSTALLED:
        r, cuda_init = _run(
            tmp_path,
            lineage=FROZEN_LINEAGE,
            model_root=MODEL_ROOT,
            attestation=ATTESTATION,
        )
    else:
        root, lineage = _standin_bundle(tmp_path)
        r, cuda_init = _run(tmp_path, lineage=lineage, model_root=root)
    assert r.returncode == 2, r.stderr
    assert "saccade_track: preflight passed" in r.stderr, r.stderr
    assert cuda_init, "the observer did not see cuInit after Gate A passed"
    j = json.loads((tmp_path / "out" / JOURNAL).read_text())
    assert j["state"] == "failed"
    message = j["failure"]["message"]
    assert not message.startswith("preflight:"), message
    if INSTALLED:
        assert j["run_id"] == _run_id(r, "saccade_track")
        assert j["identity"] == {"level": None}
        assert j["failure"]["sequence"] == SEQS[0], message
        assert "nvjpeg" in message.lower(), message
        assert all(s["state"] == "pending" for s in j["sequences"])
        assert not list((tmp_path / "out").glob("*.txt"))
    else:
        assert "dlopen" in message, message


@pytest.mark.parametrize("binary", BINARIES)
def test_replaced_head_is_refused_without_cuda(tmp_path: Path, binary: str) -> None:
    root, lineage = _standin_bundle(tmp_path)
    head = root / json.loads(lineage.read_text())["torchscript"]["path"]
    head.write_bytes(b"another head")
    r, cuda_init = _run(tmp_path, binary=binary, lineage=lineage, model_root=root)
    _assert_refused_in_gate_a(tmp_path, r, cuda_init, "head artifact", binary)


def test_self_consistent_lineage_with_the_attestation_is_refused_without_cuda(
    tmp_path: Path,
) -> None:
    root, lineage = _standin_bundle(tmp_path)
    r, cuda_init = _run(
        tmp_path, lineage=lineage, model_root=root, attestation=ATTESTATION
    )
    _assert_refused_in_gate_a(tmp_path, r, cuda_init, "bound to a different lineage")


def test_missing_attestation_is_refused_without_cuda(tmp_path: Path) -> None:
    lineage = (
        json.loads(FROZEN_LINEAGE.read_text()) if FROZEN_LINEAGE.exists() else None
    )
    if lineage is None or not (MODEL_ROOT / lineage["op_library"]["path"]).exists():
        if INSTALLED:
            pytest.fail("the installed frozen lineage or operator library is missing")
        pytest.skip("the frozen lineage or the operator library is not here")
    r, cuda_init = _run(tmp_path, lineage=FROZEN_LINEAGE, model_root=MODEL_ROOT)
    _assert_refused_in_gate_a(tmp_path, r, cuda_init, "operator library")


def test_unreadable_sequence_is_refused_without_cuda(tmp_path: Path) -> None:
    root, lineage = _standin_bundle(tmp_path)
    seqs = [tmp_path / "seqs" / s for s in SEQS]
    for s in seqs:
        _write_sequence(s)
    (seqs[1] / "seqinfo.ini").unlink()
    r, cuda_init = _run(tmp_path, lineage=lineage, model_root=root, sequences=seqs)
    _assert_refused_in_gate_a(tmp_path, r, cuda_init, "seqinfo.ini")


def test_missing_report_directory_is_refused_without_cuda(tmp_path: Path) -> None:
    root, lineage = _standin_bundle(tmp_path)
    report = tmp_path / "no_such_dir" / "track_report.json"
    r, cuda_init = _run(tmp_path, lineage=lineage, model_root=root, report=report)
    _assert_refused_in_gate_a(tmp_path, r, cuda_init, "the directory of --report")
