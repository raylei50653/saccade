"""The headline TorchScript exporter publishes only a checked pair and never replaces the frozen stem by accident.

#536 CC-536-08-02 (docs/architecture/ship_export_contracts_536.md §4), on
``scripts/model/export_headline_mamba_head_torchscript.py``. The GPU steps
(op library, head build, trace, ``torch.jit.save``, structural check) are
replaced by fakes, so these run on a stock runner against a throw-away repo
root; what is tested is the publication logic around them:

* staging: the pair is written and checked in ``<stem>.staging-*``; any gate
  failure (structural check, inventory binding, ``git_dirty``, staged bytes vs
  lineage) or exception before publication leaves the formal paths
  byte-identical and keeps only the lineage plus reasons in ``.rejected-*``;
* commit: ``.pt`` first, lineage last; an abrupt exit between the two renames
  (forked child, ``os._exit`` at the fault seam, no handler runs) leaves a
  half-published pair that the exporter's consumer rule, ``--check`` and
  ``install_model_root.cmake`` all refuse (the cmake cases skip without cmake);
* frozen stem: a stem a committed attestation binds is refused under
  ``--overwrite`` alone, however the path is spelled; the maintenance flag must
  name that same stem, and a complete maintenance publish still breaks the
  attestation binding (the exporter cannot self-accept);
* ``--check`` writes nothing; the lineage keeps the consumer fixture's layout.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "model" / "export_headline_mamba_head_torchscript.py"
INSTALL_SCRIPT = REPO / "shipping" / "cmake" / "install_model_root.cmake"
LINEAGE_FIXTURE = REPO / "tests" / "native" / "fixtures" / "shipping_head_lineage.json"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_headline_mamba_head_torchscript", TOOL
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _tool()

FROZEN = "models/yolo/frozen"
CANDIDATE = "models/yolo/cand"
ATTESTATION = "configs/shipping/head.attestation.json"
OP_SHA = "a" * 64
CONTENT_SHA = "c" * 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lineage_for(pt_rel: str, pt_bytes: bytes) -> dict[str, Any]:
    return {
        "torchscript": {"path": pt_rel, "sha256": _sha(pt_bytes)},
        "companions": {
            "backbone_engine": {
                "path": "models/yolo/backbone.engine",
                "sha256": _sha(b"engine"),
            }
        },
        "op_library": {"path": "build/libop.so", "sha256": "f" * 64},
    }


def _write_pair(repo: Path, stem: str, pt_bytes: bytes) -> None:
    (repo / stem).parent.mkdir(parents=True, exist_ok=True)
    (repo / f"{stem}.pt").write_bytes(pt_bytes)
    lineage = _lineage_for(f"{stem}.pt", pt_bytes)
    (repo / f"{stem}.lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repo root with a frozen pair bound by a committed attestation, an
    installable model root around it, and an older non-frozen candidate pair."""
    root = tmp_path / "repo"
    for rel, data in {
        "models/yolo/backbone.engine": b"engine",
        "build/libop.so": b"op",
        "configs/shipping/config.json": b"{}",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    _write_pair(root, FROZEN, b"frozen head")
    _write_pair(root, CANDIDATE, b"older candidate head")
    att = {
        "frozen_lineage": {
            "path": f"{FROZEN}.lineage.json",
            "sha256": _sha((root / f"{FROZEN}.lineage.json").read_bytes()),
        },
        "op_library": {"path": "build/libop.so", "sha256": _sha(b"op")},
    }
    (root / ATTESTATION).write_text(json.dumps(att))
    monkeypatch.setattr(T, "project_root", root)
    monkeypatch.setattr(T.trt_export, "project_root", root)
    return root


class Pipeline:
    """Fakes for every GPU step of the export and of ``--check``."""

    def __init__(self, mp: pytest.MonkeyPatch, root: Path) -> None:
        self.root = root
        self.payload = b"new head"
        self.structural = True
        self.backbone_match = True
        self.dirty = False
        self.save_fails = False
        self.calls: list[str] = []
        mp.setattr(T, "load_op_library", self.load_op_library)
        mp.setattr(T, "trace_head", self.trace_head)
        mp.setattr(T, "save", self.save)
        mp.setattr(T, "structural_check", self.structural_check)
        mp.setattr(T, "environment", lambda: {"host": "test"})
        mp.setattr(T.trt_export, "resolve_inputs", self.resolve_inputs)
        mp.setattr(T.trt_export, "build_head", self.build_head)
        mp.setattr(T.trt_export, "_git", self.git)

    def load_op_library(self) -> dict[str, Any]:
        self.calls.append("load_op_library")
        return {"path": "build/libop.so", "sha256": OP_SHA, "bytes": 1, "needed": []}

    def resolve_inputs(self, yolo: str, teacher: str) -> dict[str, Any]:
        return {
            "preset": {"path": "configs/presets/p.yaml", "sha256": "d" * 64},
            "ckpt_sha256": "e" * 64,
            "inventory": {
                "ckpt_sha256_match": True,
                "backbone_engine_sha256_match": self.backbone_match,
            },
            "backbone": self.root / "models/yolo/backbone.engine",
            "backbone_sha256": _sha(b"engine"),
        }

    def build_head(self, inputs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return object(), {
            "source": {"mamba_ckpt": {"sha256": "e" * 64}},
            "head_load": {"in_channels": [1, 2, 3]},
        }

    def trace_head(self, head: Any, in_channels: list[int]) -> tuple[Any, Any]:
        return self.payload, {"native_scan_calls": 3, "tracer_warning_sites": []}

    def save(self, traced: bytes, out: Path) -> dict[str, Any]:
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.save_fails:
            out.write_bytes(traced[:3])
            raise OSError(28, "No space left on device")
        out.write_bytes(traced)
        return {**T.trt_export._file_record(out), "content_sha256": CONTENT_SHA}

    def structural_check(self, artifact: Path, head: Any, ch: list[int]) -> Any:
        return {"bitwise_equal_all": self.structural}

    def git(self, *args: str) -> str:
        if args[0] == "rev-parse":
            return "0" * 40
        return " M src/x.py" if self.dirty else ""


@pytest.fixture
def pipe(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    return Pipeline(monkeypatch, repo)


def _args(stem: str, overwrite: bool = False, replace: str | None = None) -> Any:
    return SimpleNamespace(
        stem=stem,
        overwrite=overwrite,
        replace_frozen_stem=replace,
        yolo_weights="y",
        teacher_ckpt="t",
    )


def _tree(root: Path) -> dict[str, tuple[bytes, int] | None]:
    """Every path under ``root``: (bytes, mtime_ns) for files, None for dirs."""
    out: dict[str, tuple[bytes, int] | None] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel] = (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None
    return out


def _formal(root: Path, stem: str) -> dict[str, bytes | None]:
    return {
        sfx: (root / f"{stem}{sfx}").read_bytes()
        if (root / f"{stem}{sfx}").exists()
        else None
        for sfx in (".pt", ".lineage.json")
    }


def _side_dirs(root: Path, kind: str) -> list[Path]:
    return sorted((root / "models/yolo").glob(f"*.{kind}-*"))


def _stem(root: Path, rel: str) -> Path:
    return root / rel


# ── positive control: a complete export ────────────────────────────────────────


def test_export_publishes_a_consistent_pair(repo: Path, pipe: Pipeline) -> None:
    frozen_before = _formal(repo, FROZEN)
    assert T.run_export(_args("models/yolo/fresh")) == 0
    pt = (repo / "models/yolo/fresh.pt").read_bytes()
    lineage = json.loads((repo / "models/yolo/fresh.lineage.json").read_text())
    assert pt == pipe.payload
    assert lineage["torchscript"]["path"] == "models/yolo/fresh.pt"
    assert lineage["torchscript"]["sha256"] == _sha(pt)
    assert T.published_pair_problems(_stem(repo, "models/yolo/fresh")) == []
    assert _side_dirs(repo, "staging") == [] and _side_dirs(repo, "rejected") == []
    assert _formal(repo, FROZEN) == frozen_before
    assert "shipping_accepted" not in json.dumps(lineage)


def test_lineage_keeps_the_consumer_fixture_layout(repo: Path, pipe: Pipeline) -> None:
    """Same top-level and torchscript keys, in order, as the lineage the native
    detector-plan tests consume: staging changes where it is written, not what."""
    assert T.run_export(_args("models/yolo/fresh")) == 0
    got = json.loads((repo / "models/yolo/fresh.lineage.json").read_text())
    want = json.loads(LINEAGE_FIXTURE.read_text())
    assert list(got) == list(want)
    assert list(got["torchscript"]) == list(want["torchscript"])
    assert got["schema"] == want["schema"] == T.SCHEMA


def test_existing_stem_still_needs_overwrite(repo: Path, pipe: Pipeline) -> None:
    before = _tree(repo)
    with pytest.raises(SystemExit, match="exists; pass --overwrite"):
        T.run_export(_args(CANDIDATE))
    assert pipe.calls == [] and _tree(repo) == before
    assert T.run_export(_args(CANDIDATE, overwrite=True)) == 0
    assert (repo / f"{CANDIDATE}.pt").read_bytes() == pipe.payload
    assert T.published_pair_problems(_stem(repo, CANDIDATE)) == []


# ── pre-publication failures: formal paths untouched ───────────────────────────


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("structural", False, "structural check"),
        ("backbone_match", False, "inventory.backbone_engine_sha256_match"),
        ("dirty", True, "tool.git_dirty"),
    ],
)
def test_gate_failure_leaves_formal_paths_untouched(
    repo: Path, pipe: Pipeline, field: str, value: bool, reason: str
) -> None:
    setattr(pipe, field, value)
    before = _tree(repo)
    assert T.run_export(_args(CANDIDATE, overwrite=True)) == 1
    after = _tree(repo)
    rejected = _side_dirs(repo, "rejected")
    assert len(rejected) == 1 and _side_dirs(repo, "staging") == []
    rej = rejected[0].relative_to(repo).as_posix()
    assert {k: v for k, v in after.items() if not k.startswith(rej)} == before
    kept = sorted(p.name for p in rejected[0].iterdir())
    assert kept == ["cand.lineage.json", "rejected.json"]  # unverified .pt deleted
    problems = json.loads((rejected[0] / "rejected.json").read_text())["problems"]
    assert any(reason in p for p in problems)


def test_staged_bytes_that_do_not_match_the_lineage_are_not_published(
    repo: Path, pipe: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def tamper(point: str) -> None:
        if point == "staged":
            (staged,) = (repo / "models/yolo").glob("cand.staging-*/cand.pt")
            staged.write_bytes(b"swapped after hashing")

    monkeypatch.setattr(T, "_fault", tamper)
    before = _formal(repo, CANDIDATE)
    assert T.run_export(_args(CANDIDATE, overwrite=True)) == 1
    assert _formal(repo, CANDIDATE) == before
    (rejected,) = _side_dirs(repo, "rejected")
    problems = json.loads((rejected / "rejected.json").read_text())["problems"]
    assert problems == ["staged .pt sha256 != lineage torchscript.sha256"]


@pytest.mark.parametrize("where", ["save", "after_staging"])
def test_staging_failure_publishes_nothing(
    repo: Path, pipe: Pipeline, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    if where == "save":
        pipe.save_fails = True
    else:

        def boom(point: str) -> None:
            if point == "staged":
                raise OSError(5, "Input/output error")

        monkeypatch.setattr(T, "_fault", boom)
    before = _formal(repo, CANDIDATE)
    with pytest.raises(OSError):
        T.run_export(_args(CANDIDATE, overwrite=True))
    assert _formal(repo, CANDIDATE) == before
    assert _side_dirs(repo, "staging") == []
    (rejected,) = _side_dirs(repo, "rejected")
    assert not list(rejected.glob("*.pt"))
    (reason,) = json.loads((rejected / "rejected.json").read_text())["problems"]
    assert reason.startswith("interrupted before publication, formal paths untouched")


# ── interruption between the renames: half-published, refused ─────────────────


def _exit_between_renames(stem: str, overwrite: bool, replace: str | None) -> None:
    """Run the export in a forked child that dies (``os._exit``, no handler,
    no cleanup) right after ``.pt`` is renamed into place."""

    def die(point: str) -> None:
        if point == "pt_published":
            os._exit(70)

    # The child touches only files through already-imported stdlib code; the
    # alarm turns a fork-time deadlock into a failed assertion, not a hang.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # multi-threaded fork
        pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        try:
            signal.alarm(60)
            setattr(T, "_fault", die)
            T.run_export(_args(stem, overwrite, replace))
        finally:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 70


def test_exit_between_renames_leaves_a_pair_every_consumer_refuses(
    repo: Path, pipe: Pipeline
) -> None:
    pipe.payload = b"exported v1"
    assert T.run_export(_args(CANDIDATE, overwrite=True)) == 0
    assert T.run_check(_args(CANDIDATE)) == 0  # positive control
    old_lineage = (repo / f"{CANDIDATE}.lineage.json").read_bytes()
    pipe.payload = b"exported v2"
    _exit_between_renames(CANDIDATE, overwrite=True, replace=None)
    assert (repo / f"{CANDIDATE}.pt").read_bytes() == pipe.payload
    assert (repo / f"{CANDIDATE}.lineage.json").read_bytes() == old_lineage
    (staging,) = _side_dirs(repo, "staging")  # no handler ran
    assert sorted(p.name for p in staging.iterdir()) == ["cand.lineage.json"]
    (problem,) = T.published_pair_problems(_stem(repo, CANDIDATE))
    assert "half-published" in problem
    assert T.run_check(_args(CANDIDATE)) == 1


def test_exit_between_renames_on_a_new_stem_publishes_nothing(
    repo: Path, pipe: Pipeline
) -> None:
    _exit_between_renames("models/yolo/fresh", overwrite=False, replace=None)
    assert (repo / "models/yolo/fresh.pt").exists()
    (problem,) = T.published_pair_problems(_stem(repo, "models/yolo/fresh"))
    assert "no published pair" in problem


def test_interrupt_between_renames_is_reported_and_kept(
    repo: Path, pipe: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(point: str) -> None:
        if point == "pt_published":
            raise KeyboardInterrupt

    monkeypatch.setattr(T, "_fault", interrupt)
    with pytest.raises(KeyboardInterrupt):
        T.run_export(_args(CANDIDATE, overwrite=True))
    assert T.published_pair_problems(_stem(repo, CANDIDATE)) != []
    (rejected,) = _side_dirs(repo, "rejected")
    (reason,) = json.loads((rejected / "rejected.json").read_text())["problems"]
    assert "HALF-PUBLISHED" in reason


# ── consumer rule ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda r: (r / f"{CANDIDATE}.pt").write_bytes(b"other"), "half-published"),
        (lambda r: (r / f"{CANDIDATE}.pt").unlink(), "missing"),
        (lambda r: (r / f"{CANDIDATE}.lineage.json").unlink(), "no published pair"),
        (lambda r: (r / f"{CANDIDATE}.lineage.json").write_text("{"), "unreadable"),
        (
            lambda r: (r / f"{CANDIDATE}.lineage.json").write_text(
                json.dumps(_lineage_for(f"{FROZEN}.pt", b"frozen head"))
            ),
            "names",
        ),
    ],
)
def test_published_pair_rule_fails_closed(
    repo: Path, damage: Any, message: str
) -> None:
    assert T.published_pair_problems(_stem(repo, CANDIDATE)) == []
    damage(repo)
    (problem,) = T.published_pair_problems(_stem(repo, CANDIDATE))
    assert message in problem


# ── frozen stem ───────────────────────────────────────────────────────────────


def test_the_repository_attestation_binds_the_frozen_stem() -> None:
    frozen = T.frozen_targets()
    for p in T._targets(REPO / T.FROZEN_STEM):
        assert frozen[p.resolve()] == (
            "configs/shipping/mamba_head_realization.attestation.json"
        )
    for p in T._targets(REPO / T.DEFAULT_EXPORT_STEM):
        assert p.resolve() not in frozen


def test_defaults_export_to_a_new_stem_and_check_the_frozen_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}
    monkeypatch.setattr(T, "run_export", lambda a: seen.setdefault("export", a.stem))
    monkeypatch.setattr(T, "run_check", lambda a: seen.setdefault("check", a.stem))
    for argv in ([], ["--check"]):
        monkeypatch.setattr(sys, "argv", [str(TOOL), *argv])
        T.main()
    assert seen == {"export": T.DEFAULT_EXPORT_STEM, "check": T.FROZEN_STEM}
    assert T.DEFAULT_EXPORT_STEM != T.FROZEN_STEM


@pytest.mark.parametrize(
    "spelling", ["plain", "dotdot", "absolute", "symlinked_dir", "suffix_alias"]
)
def test_overwrite_alone_cannot_replace_the_frozen_stem(
    repo: Path, pipe: Pipeline, spelling: str
) -> None:
    if spelling == "symlinked_dir":
        (repo / "alias").symlink_to(repo / "models/yolo")
    stem = {
        "plain": FROZEN,
        "dotdot": "models/yolo/../yolo/frozen",
        "absolute": str(repo / FROZEN),
        "symlinked_dir": "alias/frozen",
        # with_suffix() replaces the last suffix: frozen.x -> frozen.pt
        "suffix_alias": FROZEN + ".x",
    }[spelling]
    before = _tree(repo)
    with pytest.raises(SystemExit, match="frozen stem bound by " + ATTESTATION):
        T.run_export(_args(stem, overwrite=True))
    assert pipe.calls == [] and _tree(repo) == before


@pytest.mark.parametrize(
    ("stem", "overwrite", "replace", "message"),
    [
        (FROZEN, True, CANDIDATE, "frozen stem bound by"),
        (CANDIDATE, True, CANDIDATE, "not bound by any committed attestation"),
        (FROZEN, False, FROZEN, "exists; pass --overwrite"),
    ],
)
def test_the_maintenance_flag_must_name_the_frozen_export_stem(
    repo: Path,
    pipe: Pipeline,
    stem: str,
    overwrite: bool,
    replace: str,
    message: str,
) -> None:
    before = _tree(repo)
    with pytest.raises(SystemExit, match=message):
        T.run_export(_args(stem, overwrite, replace))
    assert pipe.calls == [] and _tree(repo) == before


def test_an_unreadable_attestation_fails_closed(repo: Path, pipe: Pipeline) -> None:
    (repo / "configs/shipping/broken.attestation.json").write_text("{}")
    with pytest.raises(SystemExit, match="cannot read frozen_lineage.path"):
        T.run_export(_args("models/yolo/fresh"))
    assert pipe.calls == []


# ── shipping consumer: install_model_root.cmake ───────────────────────────────


def _install(root: Path) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")
    defs = {
        "SACCADE_REPO_ROOT": str(root),
        "SACCADE_MODEL_ROOT_DEST": str(root.parent / "tree"),
        "SACCADE_SHIPPING_CONFIG": "configs/shipping/config.json",
        "SACCADE_SHIPPING_LINEAGE": f"{FROZEN}.lineage.json",
        "SACCADE_SHIPPING_ATTESTATION": ATTESTATION,
        "SACCADE_ATTESTED_OP_LIBRARY": str(root / "build/libop.so"),
    }
    return subprocess.run(
        [cmake, *(f"-D{k}={v}" for k, v in defs.items()), "-P", str(INSTALL_SCRIPT)],
        capture_output=True,
        text=True,
    )


def test_install_refuses_a_half_published_frozen_stem(
    repo: Path, pipe: Pipeline
) -> None:
    assert _install(repo).returncode == 0  # positive control
    _exit_between_renames(FROZEN, overwrite=True, replace=FROZEN)
    r = _install(repo)
    assert r.returncode != 0
    assert f"{FROZEN}.pt" in r.stderr and "sha256" in r.stderr


def test_a_complete_maintenance_publish_is_not_shipping_accepted(
    repo: Path, pipe: Pipeline, capsys: pytest.CaptureFixture[str]
) -> None:
    assert T.run_export(_args(FROZEN, overwrite=True, replace=FROZEN)) == 0
    assert "no longer binds this lineage" in capsys.readouterr().out
    assert T.published_pair_problems(_stem(repo, FROZEN)) == []
    r = _install(repo)
    assert r.returncode != 0 and f"{FROZEN}.lineage.json" in r.stderr


# ── --check is read-only ──────────────────────────────────────────────────────


def test_check_writes_nothing(repo: Path, pipe: Pipeline) -> None:
    assert T.run_export(_args("models/yolo/fresh")) == 0
    before = _tree(repo)
    assert T.run_check(_args("models/yolo/fresh")) == 0  # positive control
    assert _tree(repo) == before
    (repo / "models/yolo/fresh.pt").write_bytes(b"altered")
    before = _tree(repo)
    assert T.run_check(_args("models/yolo/fresh")) == 1
    assert _tree(repo) == before
