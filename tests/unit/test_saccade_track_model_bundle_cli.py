"""``saccade_track --model-bundle`` at the built entrypoints (#549 S2-1).

docs/architecture/model_bundle_contract_549.md (sections 2-5); the checks
themselves are pinned CUDA-free by ``tests/native/test_shipping_model_bundle.cpp``
and ``tests/native/test_shipping_preflight_manifest.cpp`` (CI). These pin what
only the built binaries show: the interface, the runtime root the entrypoint
derives from its own location, the allowlist pin built into it, and the
Gate A / Gate B split in a real process.

Each test lays out an install-shaped prefix in ``tmp_path``: the built binary
copied to ``libexec/`` (manifest mode reads ``../share/saccade``) and
``share/saccade/`` holding the operator library and a copy of
``shipping/trusted_model_bundles.json`` (the production allowlist: EMPTY; its
sha256 is the one both binaries are built with). The bundle is a separate
directory. "No CUDA call" is observed as in
``tests/unit/test_saccade_track_preflight_cli.py`` (``CUDA_INJECTION64_PATH``
+ ``LD_DEBUG=files``: ``cuInit`` dlopens the probe path).

* interface: ``--model-bundle`` with any legacy option, or an unknown
  ``--require-identity``, exits 2 before the run id (MB-54) -- no journal, no
  ``<out>``, no CUDA;
* stand-in bundle (small files, a lineage / attestation / manifest bound to
  them): Gate A passes in manifest mode, CUDA initializes, and Gate B refuses
  the stand-in operator (not the pinned N03, before it is opened; #549 S2-2,
  P-N03 rev 3d): ``load_verification`` failed, identity unchanged (MB-58,
  catchable). Needs a CUDA driver;
* refusals without CUDA, each one change: a replaced member, a member of
  another size, a symlinked member, an allowlist other than the pinned one
  (MB-32), no manifest, ``--require-identity expected_source_verified`` with
  the empty production allowlist (MB-31 / MB-33 at the binary), and legacy
  mode under the same policy (MB-57);
* real bundle (N01-N06 copied into a bundle directory, the operator in the
  runtime root; needs the models, MOT17-09 and a GPU): a complete run with
  ``load_verification`` verified / hashed_before_load and the resolved
  bindings across the two roots, whose MOT text is compared with a legacy-mode
  run of the same binary (raw equality; not a qualified parity claim); and a
  SIGKILL sent once the runtime root's operator library is mapped (the load
  has begun) and before any load record, which leaves the journal running with
  ``load_verification`` null (MB-58, uncatchable);
* G-1 heads (#549 MB-41, F-1): the real bundle with its head replaced by a
  head that reads an external sidecar file (a scripted ``__setstate__``,
  ``forward``, ``torch.jit.fork``, or the ``__setstate__`` head saved as
  FlatBuffer), each shown to read the file under plain ``torch.jit.load``.
  The lineage, attestation and manifest are rebound to it, so Gate A matches
  it at ``checksum_matched``. With the sidecar removed, Gate B's head
  admission refuses it: exit 2, ``load_verification`` failed, no sequence
  run. The real-bundle run above is the N02 positive control.

Skips when ``build/shipping/saccade_track`` has not been built; the real-bundle
tests skip without the models, the data or a CUDA driver.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build" / "shipping"
BINARIES = ["saccade_track", "saccade_track_measurement"]
DESIGN = REPO / "docs" / "architecture" / "model_bundle_549"
MANIFEST = DESIGN / "examples" / "headline_s_n01_n06.model_bundle.example.json"
ALLOWLIST = REPO / "shipping" / "trusted_model_bundles.json"
CONFIG = REPO / "configs" / "shipping" / "mamba_whole_graph.resolved.json"
LINEAGE_FIXTURE = REPO / "tests" / "native" / "fixtures" / "shipping_head_lineage.json"
ATTESTATION = REPO / "configs" / "shipping" / "mamba_head_realization.attestation.json"
OP_LIBRARY = REPO / "build" / "libsaccade_scan_torchop.so"
SEQUENCE = REPO / "datasets" / "MOT17" / "train" / "MOT17-09-SDP"
JOURNAL = "saccade_track.journal.json"
PROBE = "cuda_init_probe.so"
SEQS = ("SEQ-A", "SEQ-B")
LOAD_VERIFIED = {"status": "verified", "byte_scope": "hashed_before_load"}
LOAD_FAILED = {"status": "failed", "byte_scope": None}
# Example manifest member index of each slot.
BACKBONE, HEAD, OP, CFG, LINEAGE, ATT = range(6)

pytestmark = pytest.mark.skipif(
    not (BUILD / "saccade_track").exists(),
    reason="build/shipping/saccade_track not built",
)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _has_cuda_driver() -> bool:
    return Path("/dev/dxg").exists() or Path("/dev/nvidiactl").exists()


def _real_models() -> bool:
    manifest = json.loads(MANIFEST.read_text())
    return (
        OP_LIBRARY.is_file()
        and SEQUENCE.is_dir()
        and all(
            (REPO / m["path"]).is_file()
            for m in manifest["members"]
            if m["carried_by"] == "model_bundle"
        )
    )


def _prefix(tmp_path: Path, binary: str, op_bytes: bytes | None) -> Path:
    """An install-shaped prefix: libexec/<binary>, share/saccade/{allowlist, op}."""
    prefix = tmp_path / "prefix"
    exe = BUILD / binary
    if not exe.exists():
        pytest.skip(f"{exe.relative_to(REPO)} not built")
    (prefix / "libexec").mkdir(parents=True)
    shutil.copy2(exe, prefix / "libexec" / binary)
    share = prefix / "share" / "saccade"
    op_rel = json.loads(MANIFEST.read_text())["members"][OP]["path"]
    (share / op_rel).parent.mkdir(parents=True)
    if op_bytes is None:
        shutil.copy2(OP_LIBRARY, share / op_rel)
    else:
        (share / op_rel).write_bytes(op_bytes)
    shutil.copy2(ALLOWLIST, share / "trusted_model_bundles.json")
    return prefix


def _member_path(bundle: Path, prefix: Path, m: dict[str, Any]) -> Path:
    root = bundle if m["carried_by"] == "model_bundle" else prefix / "share" / "saccade"
    return root / m["path"]


def _write_manifest(bundle: Path, prefix: Path, manifest: dict[str, Any]) -> None:
    for m in manifest["members"]:
        data = _member_path(bundle, prefix, m).read_bytes()
        m["bytes"], m["sha256"] = len(data), _sha(data)
    (bundle / "model_bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _standin_bundle(tmp_path: Path, binary: str) -> tuple[Path, Path, dict[str, Any]]:
    """Stand-in head / engine / operator bytes, a lineage, attestation and
    manifest bound to them, and the real config."""
    standins = {
        OP: b"operator library stand-in",
        HEAD: b"head artifact stand-in",
        BACKBONE: b"backbone engine stand-in",
    }
    prefix = _prefix(tmp_path, binary, standins[OP])
    bundle = tmp_path / "bundle"
    manifest = json.loads(MANIFEST.read_text())
    members = manifest["members"]
    for i in (HEAD, BACKBONE):
        p = _member_path(bundle, prefix, members[i])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(standins[i])
    cfg = _member_path(bundle, prefix, members[CFG])
    cfg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, cfg)
    lineage = json.loads(LINEAGE_FIXTURE.read_text())
    lineage["op_library"]["sha256"] = _sha(standins[OP])
    lineage["torchscript"]["sha256"] = _sha(standins[HEAD])
    lineage["companions"]["backbone_engine"]["sha256"] = _sha(standins[BACKBONE])
    lpath = _member_path(bundle, prefix, members[LINEAGE])
    lpath.write_text(json.dumps(lineage, indent=2))
    att = json.loads(ATTESTATION.read_text())
    frozen = att["frozen_lineage"]
    frozen["sha256"] = _sha(lpath.read_bytes())
    frozen["torchscript_sha256"] = lineage["torchscript"]["sha256"]
    frozen["torchscript_content_sha256"] = lineage["torchscript"]["content_sha256"]
    frozen["op_library_sha256"] = lineage["op_library"]["sha256"]
    att["op_library"]["path"] = lineage["op_library"]["path"]
    att["op_library"]["sha256"] = _sha(standins[OP])
    apath = _member_path(bundle, prefix, members[ATT])
    apath.parent.mkdir(parents=True, exist_ok=True)
    apath.write_text(json.dumps(att, indent=2))
    _write_manifest(bundle, prefix, manifest)
    return prefix, bundle, manifest


def _real_bundle(tmp_path: Path, binary: str) -> tuple[Path, Path, dict[str, Any]]:
    """The published example manifest over the real N01-N06 (operator in the
    runtime root): the manifest bytes are the example's."""
    if not _real_models():
        pytest.skip("the models, the operator library or MOT17-09 are not here")
    if not _has_cuda_driver():
        pytest.skip("no CUDA driver on this host")
    prefix = _prefix(tmp_path, binary, None)
    bundle = tmp_path / "bundle"
    manifest = json.loads(MANIFEST.read_text())
    for m in manifest["members"]:
        if m["carried_by"] != "model_bundle":
            continue
        p = _member_path(bundle, prefix, m)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / m["path"], p)
    bundle.mkdir(exist_ok=True)
    shutil.copy2(MANIFEST, bundle / "model_bundle.json")
    return prefix, bundle, manifest


def _sequences(tmp_path: Path) -> list[Path]:
    seqs = []
    for s in SEQS:
        seq = tmp_path / "seqs" / s
        if not seq.exists():
            (seq / "img1").mkdir(parents=True)
            (seq / "seqinfo.ini").write_text(
                f"[Sequence]\nname={s}\nimWidth=8\nimHeight=6\nseqLength=3\n"
            )
            for k in range(1, 4):
                (seq / "img1" / f"{k:06d}.jpg").write_bytes(b"not a jpeg")
        seqs.append(seq)
    return seqs


def _observed_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
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
    return env, ld


def _cuda_initialized(ld: Path) -> bool:
    logs = list(ld.iterdir())
    assert logs, "LD_DEBUG wrote no log: the observer is not working"
    return any(PROBE in p.read_text(errors="replace") for p in logs)


def _run(
    tmp_path: Path, exe: Path, args: list[str], out: str = "out"
) -> tuple[subprocess.CompletedProcess[str], bool]:
    env, ld = _observed_env(tmp_path)
    cmd = [str(exe), *args, "--out", str(tmp_path / out)]
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env, check=False
    )
    return r, _cuda_initialized(ld)


def _run_id(r: subprocess.CompletedProcess[str], binary: str) -> str:
    m = re.fullmatch(rf"{binary}: run_id ([0-9a-f]{{32}})", r.stderr.splitlines()[0])
    assert m, r.stderr
    return m.group(1)


def _journal(tmp_path: Path, out: str = "out") -> dict[str, Any]:
    return json.loads((tmp_path / out / JOURNAL).read_text())


def _assert_refused_in_gate_a(
    tmp_path: Path,
    binary: str,
    r: subprocess.CompletedProcess[str],
    cuda: bool,
    needle: str,
) -> dict[str, Any]:
    assert r.returncode == 2, r.stderr
    assert not cuda, "a Gate A refusal initialized CUDA"
    assert f"{binary}: preflight passed" not in r.stderr, r.stderr
    j = _journal(tmp_path)
    assert j["format"] == "saccade.native_track_journal/v3"
    assert j["run_id"] == _run_id(r, binary)
    assert j["state"] == "failed" and j["failure"]["sequence"] is None
    message = j["failure"]["message"]
    assert message.startswith("preflight: ") and needle in message, message
    assert j["load_verification"] is None
    assert all(s["state"] == "pending" for s in j["sequences"])
    assert not list((tmp_path / "out").glob("*.txt"))
    return j


# ── interface ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("binary", BINARIES)
@pytest.mark.parametrize(
    "option", ["--config", "--lineage", "--attestation", "--model-root"]
)
def test_model_bundle_excludes_legacy_options_before_any_file(
    tmp_path: Path, binary: str, option: str
) -> None:
    """MB-54: refused at argument parsing -- the bundle does not even exist."""
    env, ld = _observed_env(tmp_path)
    out = tmp_path / "out"
    cmd = [
        str(BUILD / binary),
        "--model-bundle",
        str(tmp_path / "no_such_bundle"),
        option,
        str(tmp_path / "no_such_file"),
        "--out",
        str(out),
        str(tmp_path / "seq"),
    ]
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60, env=env, check=False
    )
    assert r.returncode == 2
    assert r.stderr.splitlines() == [
        f"{binary}: --model-bundle cannot be combined with --config, --lineage, "
        "--attestation or --model-root"
    ]
    assert not out.exists()  # no run id, no <out>, no journal
    assert not _cuda_initialized(ld)


@pytest.mark.parametrize("binary", BINARIES)
def test_unknown_identity_policy_is_refused_before_the_run(
    tmp_path: Path, binary: str
) -> None:
    out = tmp_path / "out"
    r = subprocess.run(
        [
            str(BUILD / binary),
            "--model-bundle",
            str(tmp_path),
            "--require-identity",
            "trusted",
            "--out",
            str(out),
            str(tmp_path / "seq"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == 2
    assert "--require-identity trusted is not one of" in r.stderr
    assert "run_id" not in r.stderr and not out.exists()


# ── stand-in bundle ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_cuda_driver(), reason="no CUDA driver on this host")
@pytest.mark.parametrize("binary", BINARIES)
def test_standin_bundle_passes_gate_a_and_gate_b_records_the_failed_load(
    tmp_path: Path, binary: str
) -> None:
    prefix, bundle, manifest = _standin_bundle(tmp_path, binary)
    seqs = _sequences(tmp_path)
    r, cuda = _run(
        tmp_path,
        prefix / "libexec" / binary,
        ["--model-bundle", str(bundle), *map(str, seqs)],
    )
    assert r.returncode == 2, r.stderr
    assert f"{binary}: preflight passed" in r.stderr, r.stderr
    assert cuda, "the observer did not see cuInit after Gate A passed"
    j = _journal(tmp_path)
    assert j["state"] == "failed"
    standin_sha = _sha(b"operator library stand-in")
    assert (
        f"sha256 {standin_sha} is not the admitted N03 operator library's"
        in j["failure"]["message"]
    ), j["failure"]
    assert j["load_verification"] == LOAD_FAILED
    identity = j["identity"]
    assert identity["mode"] == "model_bundle" and identity["required"] == "none"
    assert identity["level"] == "checksum_matched"
    assert identity["expected_source"] is None
    assert identity["allowlist_entry"] == "absent"  # the production allowlist is empty
    assert identity["allowlist_sha256"] == _sha(ALLOWLIST.read_bytes())
    assert identity["bundle_manifest_sha256"] == _sha(
        (bundle / "model_bundle.json").read_bytes()
    )
    bindings = identity["bindings"]
    assert list(bindings) == [
        "bundle_manifest",
        "config",
        "lineage",
        "attestation",
        "op_library",
        "head",
        "engine",
    ]
    share = (prefix / "share" / "saccade").resolve()
    assert bindings["op_library"]["path"] == str(
        share / manifest["members"][OP]["path"]
    )
    assert bindings["head"]["path"] == str(
        bundle.resolve() / manifest["members"][HEAD]["path"]
    )
    for name in ("config", "lineage", "attestation", "op_library", "head", "engine"):
        assert bindings[name]["status"] == "matched", name
        assert bindings[name]["expected_source"]["path"] == str(
            bundle.resolve() / "model_bundle.json"
        )


# ── refusals without CUDA ─────────────────────────────────────────────────────


@pytest.mark.parametrize("binary", BINARIES)
@pytest.mark.parametrize(
    ("field", "unsupported"),
    [("tensorrt", "0.0.0.0"), ("libtorch", "0.0.0+cu130"), ("cuda_runtime", "0.0")],
)
def test_manifest_build_versions_refused_before_cuda(
    tmp_path: Path, binary: str, field: str, unsupported: str
) -> None:
    """MB-43: the actual binary's build tuple must match one full target."""
    prefix, bundle, manifest = _standin_bundle(tmp_path, binary)
    manifest["compatibility"]["targets"][0][field] = unsupported
    _write_manifest(bundle, prefix, manifest)
    r, cuda = _run(
        tmp_path,
        prefix / "libexec" / binary,
        ["--model-bundle", str(bundle), *map(str, _sequences(tmp_path))],
    )
    j = _assert_refused_in_gate_a(tmp_path, binary, r, cuda, "compatibility")
    assert j["identity"]["level"] is None


@pytest.mark.parametrize("binary", BINARIES)
@pytest.mark.skipif(not _has_cuda_driver(), reason="CUDA driver unavailable")
def test_manifest_sm_refused_before_deserialization(
    tmp_path: Path, binary: str
) -> None:
    """MB-44/58: unsupported SM wins over invalid stand-in loader bytes."""
    prefix, bundle, manifest = _standin_bundle(tmp_path, binary)
    manifest["compatibility"]["targets"][0]["gpu_sm"] = ["sm_999"]
    _write_manifest(bundle, prefix, manifest)
    r, cuda = _run(
        tmp_path,
        prefix / "libexec" / binary,
        ["--model-bundle", str(bundle), *map(str, _sequences(tmp_path))],
    )
    assert r.returncode == 2, r.stderr
    assert cuda and f"{binary}: preflight passed" in r.stderr
    j = _journal(tmp_path)
    assert j["state"] == "failed" and j["load_verification"] == LOAD_FAILED
    assert j["identity"]["level"] == "checksum_matched"
    assert "SM" in j["failure"]["message"], j["failure"]
    assert "dlopen" not in j["failure"]["message"]
    assert all(s["state"] == "pending" for s in j["sequences"])


def _replace_head(
    tmp_path: Path, prefix: Path, bundle: Path, m: dict[str, Any]
) -> None:
    p = _member_path(bundle, prefix, m["members"][HEAD])
    data = bytearray(p.read_bytes())
    data[0] ^= 1
    p.write_bytes(bytes(data))


def _grow_engine(tmp_path: Path, prefix: Path, bundle: Path, m: dict[str, Any]) -> None:
    p = _member_path(bundle, prefix, m["members"][BACKBONE])
    p.write_bytes(p.read_bytes() + b"x")


def _symlink_lineage(
    tmp_path: Path, prefix: Path, bundle: Path, m: dict[str, Any]
) -> None:
    p = _member_path(bundle, prefix, m["members"][LINEAGE])
    p.rename(tmp_path / "outside.lineage.json")
    p.symlink_to(tmp_path / "outside.lineage.json")


def _edit_allowlist(
    tmp_path: Path, prefix: Path, bundle: Path, m: dict[str, Any]
) -> None:
    """MB-32: an approval added to the installed allowlist after the build."""
    path = prefix / "share" / "saccade" / "trusted_model_bundles.json"
    a = json.loads(path.read_text())
    a["entries"].append(
        {
            "manifest_sha256": _sha((bundle / "model_bundle.json").read_bytes()),
            "bundle_name": "saccade-headline-s-mamba",
            "bundle_version": "0.1.0",
            "state": "approved",
            "approval": {
                "decision_ref": "https://github.com/raylei50653/saccade/issues/549#issuecomment-1",
                "date": "2026-10-11",
            },
            "revocation": None,
        }
    )
    path.write_text(json.dumps(a, indent=2) + "\n")


def _no_manifest(tmp_path: Path, prefix: Path, bundle: Path, m: dict[str, Any]) -> None:
    (bundle / "model_bundle.json").unlink()


REFUSALS = {
    "member_replaced": (_replace_head, [], "head artifact", ("head", "mismatch")),
    "member_size": (_grow_engine, [], "the manifest says", ("engine", "size_mismatch")),
    "member_symlink": (
        _symlink_lineage,
        [],
        "symbolic link",
        ("lineage", "unsafe_path"),
    ),
    "allowlist_not_pinned": (
        _edit_allowlist,
        [],
        "is not the one this entrypoint was built with",
        None,
    ),
    "manifest_missing": (
        _no_manifest,
        [],
        "does not exist",
        ("bundle_manifest", "missing"),
    ),
    "policy_unlisted": (
        lambda *a: None,
        ["--require-identity", "expected_source_verified"],
        "identity level checksum_matched is below --require-identity expected_source_verified",
        None,
    ),
}


@pytest.mark.parametrize("binary", BINARIES)
@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_manifest_refusals_make_no_cuda_call(
    tmp_path: Path, binary: str, case: str
) -> None:
    edit, extra, needle, status = REFUSALS[case]
    prefix, bundle, manifest = _standin_bundle(tmp_path, binary)
    edit(tmp_path, prefix, bundle, manifest)
    seqs = _sequences(tmp_path)
    r, cuda = _run(
        tmp_path,
        prefix / "libexec" / binary,
        ["--model-bundle", str(bundle), *extra, *map(str, seqs)],
    )
    j = _assert_refused_in_gate_a(tmp_path, binary, r, cuda, needle)
    identity = j["identity"]
    assert identity["mode"] == "model_bundle"
    if case == "policy_unlisted":
        # The policy refusal keeps the proven level and records the request.
        assert identity["level"] == "checksum_matched"
        assert identity["required"] == "expected_source_verified"
        assert identity["allowlist_entry"] == "absent"
    else:
        assert identity["level"] is None
    if status is not None:
        name, want = status
        assert identity["bindings"][name]["status"] == want


@pytest.mark.parametrize("binary", BINARIES)
def test_legacy_mode_cannot_meet_expected_source_verified(
    tmp_path: Path, binary: str
) -> None:
    """MB-57 at the binary: the legacy stand-in passes every Gate A check and
    is refused by the policy, before CUDA."""
    root = tmp_path / "root"
    lineage = json.loads(LINEAGE_FIXTURE.read_text())
    for entry, data in (
        (lineage["op_library"], b"operator library stand-in"),
        (lineage["torchscript"], b"head artifact stand-in"),
        (lineage["companions"]["backbone_engine"], b"backbone engine stand-in"),
    ):
        (root / entry["path"]).parent.mkdir(parents=True, exist_ok=True)
        (root / entry["path"]).write_bytes(data)
        entry["sha256"] = _sha(data)
    lpath = root / "lineage.json"
    lpath.write_text(json.dumps(lineage, indent=2))
    seqs = _sequences(tmp_path)
    r, cuda = _run(
        tmp_path,
        BUILD / binary,
        [
            "--config",
            str(CONFIG),
            "--lineage",
            str(lpath),
            "--model-root",
            str(root),
            "--require-identity",
            "expected_source_verified",
            *map(str, seqs),
        ],
    )
    j = _assert_refused_in_gate_a(
        tmp_path, binary, r, cuda, "(legacy mode cannot exceed checksum_matched)"
    )
    assert j["identity"]["mode"] == "legacy"
    assert j["identity"]["level"] == "checksum_matched"
    assert j["identity"]["required"] == "expected_source_verified"


# ── real bundle (GPU) ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("binary", BINARIES)
@pytest.mark.parametrize("mismatch", ["dtype", "shape", "name"])
def test_real_engine_io_refusal_keeps_gate_a_identity(
    tmp_path: Path, binary: str, mismatch: str
) -> None:
    """MB-45/58: hash-valid artifacts must satisfy the exact manifest I/O."""
    prefix, bundle, manifest = _real_bundle(tmp_path, binary)
    backbone = manifest["io"]["backbone_engine"]["outputs"]
    if mismatch == "dtype":
        manifest["io"]["backbone_engine"]["inputs"][0]["dtype"] = "float16"
    elif mismatch == "shape":
        manifest["io"]["backbone_engine"]["inputs"][0]["shape"][1] = 4
    else:
        backbone[0]["name"], backbone[1]["name"] = (
            backbone[1]["name"],
            backbone[0]["name"],
        )
    _write_manifest(bundle, prefix, manifest)
    r, cuda = _run(
        tmp_path,
        prefix / "libexec" / binary,
        ["--model-bundle", str(bundle), *map(str, _sequences(tmp_path))],
    )
    assert r.returncode == 2, r.stderr
    assert cuda and f"{binary}: preflight passed" in r.stderr
    j = _journal(tmp_path)
    assert j["state"] == "failed" and j["load_verification"] == LOAD_FAILED
    assert j["identity"]["level"] == "checksum_matched"
    bindings = j["identity"]["bindings"]
    assert all(
        bindings[name]["status"] == "matched"
        for name in ("config", "lineage", "attestation", "op_library", "head", "engine")
    )
    assert mismatch in j["failure"]["message"], j["failure"]
    assert all(s["state"] == "pending" for s in j["sequences"])


@pytest.mark.parametrize("binary", BINARIES)
def test_real_bundle_runs_from_two_roots(tmp_path: Path, binary: str) -> None:
    prefix, bundle, manifest = _real_bundle(tmp_path, binary)
    report = tmp_path / "out" / "report.json"
    (tmp_path / "out").mkdir()
    r, cuda = _run(
        tmp_path,
        prefix / "libexec" / binary,
        [
            "--model-bundle",
            str(bundle),
            "--require-identity",
            "checksum_matched",
            "--report",
            str(report),
            str(SEQUENCE),
        ],
    )
    assert r.returncode == 0, r.stderr
    assert cuda
    j = _journal(tmp_path)
    assert j["state"] == "complete"
    assert j["load_verification"] == LOAD_VERIFIED
    identity = j["identity"]
    assert (identity["mode"], identity["level"], identity["required"]) == (
        "model_bundle",
        "checksum_matched",
        "checksum_matched",
    )
    assert identity["allowlist_entry"] == "absent"
    assert identity["bundle_manifest_sha256"] == _sha(MANIFEST.read_bytes())
    rep = json.loads(report.read_text())
    assert rep["format"] == "saccade.native_track_report/v5"
    assert (rep["identity"], rep["load_verification"]) == (identity, LOAD_VERIFIED)
    assert (rep["mode"], rep["model_bundle"], rep["model_root"], rep["config"]) == (
        "model_bundle",
        str(bundle),
        None,
        None,
    )
    resolved = rep["detector"]["plan"]["resolved"]
    share = (prefix / "share" / "saccade").resolve()
    assert resolved["op_library"]["root_kind"] == "runtime_package"
    assert resolved["op_library"]["path"] == str(
        share / manifest["members"][OP]["path"]
    )
    for key, index in (("head_artifact", HEAD), ("backbone_engine", BACKBONE)):
        assert resolved[key]["root_kind"] == "model_bundle"
        assert resolved[key]["path"] == str(
            bundle.resolve() / manifest["members"][index]["path"]
        )
        assert resolved[key]["sha256"] == manifest["members"][index]["sha256"]
    load = rep["detector"]["load"]
    assert load["op_library_sha256"] == manifest["members"][OP]["sha256"]
    assert load["head_artifact_sha256"] == manifest["members"][HEAD]["sha256"]
    assert load["backbone_engine_sha256"] == manifest["members"][BACKBONE]["sha256"]
    # The same binary in legacy mode on the same sequence: raw MOT text equal
    # (a local diagnostic of the two modes, not a qualified parity claim). Its
    # model root is the prefix's share/saccade (the only place its operator
    # library may come from, #549 S2-2 D-N03-5), with the models linked in.
    share_root = prefix / "share" / "saccade"
    (share_root / "models").symlink_to(REPO / "models", target_is_directory=True)
    legacy, _ = _run(
        tmp_path,
        prefix / "libexec" / binary,
        [
            "--config",
            str(CONFIG),
            "--lineage",
            str(REPO / manifest["members"][LINEAGE]["path"]),
            "--attestation",
            str(ATTESTATION),
            "--model-root",
            str(share_root),
            str(SEQUENCE),
        ],
        out="legacy",
    )
    assert legacy.returncode == 0, legacy.stderr
    name = SEQUENCE.name + ".txt"
    assert (tmp_path / "legacy" / name).read_bytes() == (
        tmp_path / "out" / name
    ).read_bytes()
    assert _journal(tmp_path, "legacy")["load_verification"] == LOAD_VERIFIED


# ── G-1 heads (#549 MB-41, F-1) ──────────────────────────────────────────
G1_KINDS = ("setstate", "forward", "fork", "flatbuffer")
G1_REFUSAL = {
    "setstate": "defines __setstate__, which the loader runs",
    "forward": "forward uses aten::from_file, not among the reviewed head operators",
    "fork": "forward uses aten::from_file, aten::wait, prim::fork, not among the reviewed head operators",
    "flatbuffer": "the head is not a ZIP TorchScript archive",
}


def _g1_head(kind: str, sidecar: Path) -> bytes:
    """A TorchScript head that reads ``sidecar`` (outside the archive)."""
    import io

    import torch

    class G1SetState(torch.nn.Module):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path
            self.w = torch.zeros(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.w

        @torch.jit.export
        def __getstate__(self) -> tuple[str, bool]:
            return (self.path, self.training)

        @torch.jit.export
        def __setstate__(self, state: tuple[str, bool]) -> None:
            self.path = state[0]
            self.training = state[1]
            self.w = torch.from_file(self.path, False, 1, dtype=torch.float32)

    class G1Forward(torch.nn.Module):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + torch.from_file(self.path, False, 1, dtype=torch.float32)

    class G1Fork(torch.nn.Module):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            fut = torch.jit.fork(
                torch.from_file, self.path, False, 1, dtype=torch.float32
            )
            return x + torch.jit.wait(fut)

    cls = {"forward": G1Forward, "fork": G1Fork}.get(kind, G1SetState)
    m = torch.jit.script(cls(str(sidecar)))
    buf = io.BytesIO()
    if kind == "flatbuffer":
        torch.jit.save_jit_module_to_flatbuffer(m, buf)
    else:
        torch.jit.save(m, buf)
    return buf.getvalue()


def _g1_value(kind: str, head: bytes) -> float:
    """What the head took from its sidecar under plain torch.jit.load."""
    import io

    import torch

    m = torch.jit.load(io.BytesIO(head), map_location="cpu")
    if kind in ("setstate", "flatbuffer"):
        return float(m.w.item())
    return float(m(torch.zeros(1)).item())


def _rebind_head(
    prefix: Path, bundle: Path, manifest: dict[str, Any], head: bytes
) -> None:
    """The bundle's head replaced by ``head``; lineage, attestation and the
    manifest rebound to it, so Gate A matches every member."""
    members = manifest["members"]
    _member_path(bundle, prefix, members[HEAD]).write_bytes(head)
    lpath = _member_path(bundle, prefix, members[LINEAGE])
    lineage = json.loads(lpath.read_text())
    lineage["torchscript"]["sha256"] = _sha(head)
    lpath.write_text(json.dumps(lineage, indent=2))
    apath = _member_path(bundle, prefix, members[ATT])
    att = json.loads(apath.read_text())
    att["frozen_lineage"]["sha256"] = _sha(lpath.read_bytes())
    att["frozen_lineage"]["torchscript_sha256"] = _sha(head)
    apath.write_text(json.dumps(att, indent=2))
    _write_manifest(bundle, prefix, manifest)


@pytest.mark.parametrize("binary", BINARIES)
@pytest.mark.parametrize("kind", G1_KINDS)
def test_g1_head_is_refused_by_the_head_admission(
    tmp_path: Path, binary: str, kind: str
) -> None:
    prefix, bundle, manifest = _real_bundle(tmp_path, binary)
    sidecar = tmp_path / "sidecar.bin"
    head = _g1_head(kind, sidecar)
    # The control is real: the head's value follows the external file.
    sidecar.write_bytes(struct.pack("<f", 60.0))
    first = _g1_value(kind, head)
    sidecar.write_bytes(struct.pack("<f", 7.0))
    assert (first, _g1_value(kind, head)) == (60.0, 7.0)
    # Without it, any read would fail with LibTorch's own error instead.
    sidecar.unlink()
    _rebind_head(prefix, bundle, manifest, head)
    r, cuda = _run(
        tmp_path,
        prefix / "libexec" / binary,
        ["--model-bundle", str(bundle), *map(str, _sequences(tmp_path))],
    )
    assert r.returncode == 2, r.stderr
    assert cuda and f"{binary}: preflight passed" in r.stderr
    j = _journal(tmp_path)
    assert j["state"] == "failed" and j["load_verification"] == LOAD_FAILED
    assert j["identity"]["level"] == "checksum_matched"
    assert j["identity"]["bindings"]["head"]["status"] == "matched"
    message = j["failure"]["message"]
    assert "shipping detector load: head admission (F-1): " in message, message
    assert G1_REFUSAL[kind] in message, message
    assert all(s["state"] == "pending" for s in j["sequences"])
    assert not list((tmp_path / "out").glob("*.txt"))
    assert not sidecar.exists()


def test_real_bundle_killed_during_the_load_leaves_no_load_claim(
    tmp_path: Path,
) -> None:
    """MB-58, uncatchable: SIGKILL while the runtime is loading the detector.

    "During the load" is observed, not assumed: the kill is sent only once the
    operator library is mapped into the process -- its sealed memfd, made from
    the runtime root's copy (dlopen, Gate B step 2; #549 S2-2) -- while the
    journal still has no load_verification; the head and engine loads that
    follow take seconds. A run that got further (verified) fails this test
    instead of passing it."""
    prefix, bundle, _ = _real_bundle(tmp_path, "saccade_track")
    op = "/memfd:saccade-n03:src-fd="
    journal = tmp_path / "out" / JOURNAL
    p = subprocess.Popen(
        [
            str(prefix / "libexec" / "saccade_track"),
            "--model-bundle",
            str(bundle),
            "--out",
            str(tmp_path / "out"),
            str(SEQUENCE),
        ],
        stderr=subprocess.DEVNULL,
    )
    loading = False
    deadline = time.monotonic() + 120
    try:
        while time.monotonic() < deadline and p.poll() is None:
            try:
                maps = Path(f"/proc/{p.pid}/maps").read_text()
            except OSError:
                maps = ""
            if op in maps:
                loading = json.loads(journal.read_text())["load_verification"] is None
                p.send_signal(signal.SIGKILL)
                break
            time.sleep(0.001)
    finally:
        if p.poll() is None:
            p.kill()
        p.wait(timeout=60)
    assert loading, "the kill did not land after dlopen with no load record"
    assert p.returncode == -signal.SIGKILL
    j = json.loads(journal.read_text())
    assert j["state"] == "running"
    assert j["load_verification"] is None
    assert j["identity"]["level"] == "checksum_matched"
