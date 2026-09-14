"""The default dependency set is the tracker core's import closure, nothing more.

``pyproject.toml`` draws the package's dependency boundaries: the default set
is what ``pip install saccade`` gives a third party, and each extra is one
more part of the package with the distributions that part imports. This suite
derives those boundaries from the sources instead of trusting the lists:

  * **the default set is the tracker-core closure** -- walking module-level
    imports from the core names of the public surface reaches exactly the
    default distributions, minus the one named load-time substrate the native
    extension needs (ADR 025);
  * **every module is covered by core or its extra** -- each subpackage is
    owned by one extra, and the module-level closure of that subpackage must
    be satisfied by the default set plus that extra (self-references
    expanded); a module owned by no extra must import only the default set;
  * **nothing is declared twice** -- a distribution belongs to the default set
    or to extras, never both, and ``ultralytics`` in particular is not default
    (ADR 023 follow-up 3);
  * **the dev group is the repository's boundary, not the consumer's** -- it
    unions every extra except ``dali`` so ``uv sync`` still yields the whole
    repository environment;
  * **a build-only extra owns no module** -- ``native-build`` is consumed by
    CMakeLists.txt, not imported, so it has no smoke module and no owned
    subpackage; tests/contract/test_package_native_delivery.py holds its
    contents to the delivery model instead;
  * **a core-only interpreter can use the tracker surface** -- with every
    extras-only distribution made unimportable, ``import saccade`` and the
    core public names still resolve, and the lazily-imported names that need
    an extra say which one;
  * **each extra imports** -- a representative module per extra loads in a
    fresh interpreter of the full development environment.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "saccade"
PYPROJECT = REPO_ROOT / "pyproject.toml"
DISTRIBUTION = "saccade"

# Declared in the default set although nothing under src/saccade imports it on
# the tracker path: saccade_tracking_ext has libnvinfer.so.10 as a direct
# NEEDED and resolves it into this venv's tensorrt_libs, so the tracker
# surface cannot load without the distribution (ADR 025, "runtime loader
# dependency"). The build toolchain used to sit here too; it is the
# `native-build` extra now. Growth here is a review item, not a drive-by.
NATIVE_SUBSTRATE = frozenset({"tensorrt-cu12"})

# Extras nothing imports: their distributions are consumed by the native build
# (CMakeLists.txt), so module ownership and smoke imports do not apply.
BUILD_ONLY_EXTRAS = frozenset({"native-build"})

# Which extra owns which part of the package. Longest prefix wins. A module
# outside every prefix is core and may import only the default set.
MODULE_EXTRA: dict[str, str] = {
    "saccade.api": "serve",
    "saccade.pipeline": "serve",
    "saccade.resource": "serve",
    "saccade.perception.dispatcher": "serve",
    "saccade.perception.drift_handler": "serve",
    "saccade.perception.embedding_dispatcher": "serve",
    "saccade.perception.entropy": "serve",
    "saccade.perception.online_telemetry": "serve",
    "saccade.perception.zero_copy": "serve",
    "saccade.cognition": "cognition",
    "saccade.storage": "storage",
    "saccade.media": "media",
    "saccade.media.dali_pipeline": "dali",
    "saccade.media.rtsp_dali_pipeline": "dali",
    "saccade.perception.temporal_yolo": "yolo",
    "saccade.perception.calibrator": "yolo",
    "saccade.perception.cropper": "yolo",
    "saccade.perception.detector_trt": "yolo",
    "saccade.perception.feature_extractor": "yolo",
    "saccade.perception.feature_bank": "yolo",
    "saccade.perception.multistream_mamba_server": "yolo",
    "saccade.perception.roi_selector": "yolo",
    "saccade.perception.workbench": "yolo",
    "saccade.perception.text_encoder": "train",
    "saccade.perception.eval": "eval",
    "saccade.perception.reid": "eval",
    "saccade.perception.tracking.dynamic_reid": "eval",
    "saccade.perception.tracking.fpn_reid": "eval",
    "saccade.perception.tracking.fpn_reid_cuda": "eval",
}

# Modules whose import name is not a PyPI distribution: the native extensions
# (built into build/, path-loaded) and the vendored TrackEval (sys.path
# insertion from saccade.paths).
PATH_LOADED_IMPORTS = re.compile(r"^(saccade_\w+_ext|trackeval)$")

# Import names whose distribution the installed-metadata lookup cannot name:
# the `nvidia` namespace is shared by every CUDA wheel, and the TensorRT
# bindings report the wheel that carries them rather than the line pinned in
# pyproject.
IMPORT_TO_DISTRIBUTION = {
    "nvidia.dali": "nvidia-dali-cuda120",
    "tensorrt": "tensorrt-cu12",
}

# One import per extra that only that extra (plus core) can satisfy.
EXTRA_SMOKE_MODULES: dict[str, list[str]] = {
    "yolo": [
        "saccade.perception.temporal_yolo.ngla_assigner",
        "saccade.perception.temporal_yolo.yolo_gated_detector",
    ],
    "media": ["saccade.media.ffmpeg_utils"],
    "storage": ["saccade.storage.redis_cache", "saccade.storage.chroma_store"],
    "serve": ["saccade.api.server", "saccade.resource.resource_manager"],
    "cognition": ["saccade.cognition.orchestrator"],
    "eval": [
        "saccade.perception.eval.post_merge",
        "saccade.perception.eval.metrics",
    ],
    "train": ["saccade.perception.text_encoder"],
    "dali": ["saccade.media.dali_pipeline"],
}


# ── pyproject: the declared boundaries ───────────────────────────────────────


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(spec: str) -> str:
    return _normalize(re.split(r"[\s\[<>=!~;@]", spec.strip(), maxsplit=1)[0])


def _self_extras(spec: str) -> list[str] | None:
    m = re.fullmatch(rf"{DISTRIBUTION}\[([^\]]+)\]", spec.strip())
    return [e.strip() for e in m.group(1).split(",")] if m else None


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _default_set() -> frozenset[str]:
    return frozenset(
        _requirement_name(s) for s in _pyproject()["project"]["dependencies"]
    )


def _extras_direct() -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """extra -> (direct distributions, referenced extras), unexpanded."""
    out = {}
    for extra, specs in _pyproject()["project"]["optional-dependencies"].items():
        dists, refs = set(), set()
        for spec in specs:
            selfref = _self_extras(spec)
            if selfref is not None:
                refs.update(selfref)
            else:
                dists.add(_requirement_name(spec))
        out[extra] = (frozenset(dists), frozenset(refs))
    return out


def _extras_expanded() -> dict[str, frozenset[str]]:
    direct = _extras_direct()

    def expand(extra: str, seen: frozenset[str]) -> frozenset[str]:
        assert extra not in seen, f"extra self-reference cycle through {extra!r}"
        dists, refs = direct[extra]
        for ref in refs:
            dists |= expand(ref, seen | {extra})
        return dists

    return {extra: expand(extra, frozenset()) for extra in direct}


# ── sources: the actual boundaries ───────────────────────────────────────────


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_modules() -> dict[str, Path]:
    modules = {
        _module_name(p): p
        for p in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert modules, "no package sources found"
    return modules


class _Imports:
    """Module-level imports of one module, split by when they can fail.

    ``eager`` imports run on ``import module``. ``guarded`` imports also run at
    import time but inside a ``try`` whose handler catches ImportError, so their
    absence is a designed state. ``deferred`` imports sit inside a function or
    method body and run on call.
    """

    def __init__(self) -> None:
        self.eager: set[str] = set()
        self.guarded: set[str] = set()
        self.deferred: set[str] = set()


def _catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:
            return True
        names = (
            [handler.type]
            if not isinstance(handler.type, ast.Tuple)
            else list(handler.type.elts)
        )
        for n in names:
            if isinstance(n, ast.Name) and n.id in {
                "ImportError",
                "ModuleNotFoundError",
                "Exception",
            }:
                return True
    return False


def _resolve_from(module: str, path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    base = module.split(".")
    if path.name != "__init__.py":
        base = base[:-1]
    base = base[: len(base) - (node.level - 1)]
    if node.module:
        base.append(node.module)
    return ".".join(base)


def _scan(module: str, path: Path, modules: dict[str, Path]) -> _Imports:
    found = _Imports()

    def record(node: ast.AST, bucket: set[str]) -> None:
        if isinstance(node, ast.Import):
            bucket.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_from(module, path, node)
            bucket.add(target)
            for alias in node.names:  # `from pkg import submodule`
                if f"{target}.{alias.name}" in modules:
                    bucket.add(f"{target}.{alias.name}")

    def walk(nodes: list[ast.stmt], bucket: set[str]) -> None:
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                record(node, bucket)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.Import, ast.ImportFrom)):
                        record(inner, found.deferred)
            elif isinstance(node, ast.ClassDef):
                walk(node.body, bucket)
            elif isinstance(node, ast.Try):
                walk(
                    node.body,
                    found.guarded if _catches_import_error(node) else bucket,
                )
                for handler in node.handlers:
                    walk(handler.body, bucket)
                walk(node.orelse, bucket)
                walk(node.finalbody, bucket)
            elif isinstance(node, (ast.If, ast.With, ast.For, ast.While)):
                walk(node.body, bucket)
                walk(node.orelse, bucket)

    walk(ast.parse(path.read_text(), str(path)).body, found.eager)
    return found


_STDLIB = frozenset(sys.stdlib_module_names) | {"__future__"}


def _distribution_of(import_name: str) -> str | None:
    """Distribution a third-party import resolves to; None if not third-party."""
    top = import_name.split(".")[0]
    if top in _STDLIB or top == "saccade" or PATH_LOADED_IMPORTS.match(top):
        return None
    for prefix, dist in IMPORT_TO_DISTRIBUTION.items():
        if import_name == prefix or import_name.startswith(prefix + "."):
            return dist
    return top  # resolved against installed metadata by _to_declared


_PACKAGES_TO_DISTS = {
    k: frozenset(_normalize(d) for d in v) for k, v in packages_distributions().items()
}


def _to_declared(import_name: str, declared: frozenset[str]) -> str | None:
    """Pick the declared distribution that provides ``import_name``, if any.

    Returns the raw import name when nothing declared provides it, so the
    caller can report the gap by the name that appears in the source.
    """
    dist = _distribution_of(import_name)
    if dist is None:
        return None
    if dist in declared:
        return dist
    candidates = _PACKAGES_TO_DISTS.get(dist, frozenset())
    hit = candidates & declared
    if hit:
        return sorted(hit)[0]
    return dist


def _closure(entries: list[str], modules: dict[str, Path]) -> tuple[set[str], set[str]]:
    """Modules reached by eager module-level imports, and their third-party
    import names (eager only; guarded and deferred are reported separately)."""
    scans: dict[str, _Imports] = {}
    seen: set[str] = set()
    stack = list(entries)
    third: set[str] = set()
    while stack:
        m = stack.pop()
        if m in seen or m not in modules:
            continue
        seen.add(m)
        parts = m.split(".")
        stack.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        scans.setdefault(m, _scan(m, modules[m], modules))
        for name in scans[m].eager:
            if name.startswith("saccade"):
                stack.append(name)
            elif _distribution_of(name) is not None:
                third.add(name)
    return seen, third


def _owner(module: str) -> str | None:
    best = None
    for prefix, extra in MODULE_EXTRA.items():
        if module == prefix or module.startswith(prefix + "."):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, extra)
    return best[1] if best else None


def _core_entry_modules() -> list[str]:
    import saccade

    return [module for module, _attr, extra in saccade._LAZY.values() if extra is None]


# ── The default set is the tracker-core closure ──────────────────────────────


def test_default_set_is_the_tracker_core_closure() -> None:
    modules = _package_modules()
    default = _default_set()
    reached, third = _closure(_core_entry_modules(), modules)

    needed = {_to_declared(n, default) for n in third}
    missing = {n for n in needed if n not in default}
    assert not missing, (
        f"core surface imports undeclared distributions: {sorted(missing)}"
    )

    surplus = default - needed - NATIVE_SUBSTRATE
    assert not surplus, (
        "default dependencies beyond the tracker-core closure "
        f"(add an extra, or extend NATIVE_SUBSTRATE with a reason): {sorted(surplus)}"
    )
    # The closure is small and named; a change here is a change of surface.
    assert needed == {"numpy", "torch"}, sorted(needed)
    assert "saccade.perception.tracking.tracker_gpu" in reached


def test_native_substrate_is_exactly_what_the_allowlist_says() -> None:
    default = _default_set()
    assert NATIVE_SUBSTRATE <= default, sorted(NATIVE_SUBSTRATE - default)


def test_build_only_extras_own_no_module_and_are_not_default() -> None:
    extras = _extras_expanded()
    default = _default_set()
    assert BUILD_ONLY_EXTRAS <= set(extras), sorted(BUILD_ONLY_EXTRAS - set(extras))
    assert not set(MODULE_EXTRA.values()) & BUILD_ONLY_EXTRAS
    for extra in BUILD_ONLY_EXTRAS:
        assert not extras[extra] & default, sorted(extras[extra] & default)


# ── Every module is covered by core or its extra ─────────────────────────────


def test_every_module_is_covered_by_core_or_its_extra() -> None:
    modules = _package_modules()
    default = _default_set()
    extras = _extras_expanded()
    everything = default | frozenset().union(*extras.values())

    by_owner: dict[str | None, list[str]] = defaultdict(list)
    for m in modules:
        by_owner[_owner(m)].append(m)

    eager_gaps: dict[str, list[str]] = {}
    for owner, owned in sorted(by_owner.items(), key=lambda kv: kv[0] or ""):
        allowed = default | (extras[owner] if owner else frozenset())
        _reached, third = _closure(owned, modules)
        # The closure may cross into modules owned by another extra; the owner
        # then has to carry those distributions too (that is what the
        # self-references are for), so no exemption here.
        gaps = sorted({n for n in third if _to_declared(n, allowed) not in allowed})
        if gaps:
            eager_gaps[owner or "<core>"] = gaps
    assert not eager_gaps, (
        "module-level imports not satisfied by the default set plus the owning "
        f"extra: {json.dumps(eager_gaps, indent=2)}"
    )

    # Deferred and guarded imports only need to be declared somewhere.
    undeclared: dict[str, list[str]] = {}
    for m, path in modules.items():
        scan = _scan(m, path, modules)
        names = {
            n
            for n in scan.deferred | scan.guarded
            if not n.startswith("saccade") and _distribution_of(n) is not None
        }
        gaps = sorted(
            {n for n in names if _to_declared(n, everything) not in everything}
        )
        if gaps:
            undeclared[m] = gaps
    assert not undeclared, (
        "deferred/guarded imports of distributions no extra declares: "
        f"{json.dumps(undeclared, indent=2)}"
    )


def test_module_map_names_real_modules_and_extras() -> None:
    modules = _package_modules()
    extras = _extras_expanded()
    for prefix, extra in MODULE_EXTRA.items():
        assert extra in extras, f"{prefix} -> unknown extra {extra!r}"
        assert any(m == prefix or m.startswith(prefix + ".") for m in modules), (
            f"{prefix} matches no module"
        )


# ── Nothing is declared twice; ultralytics is not default ────────────────────


def test_default_set_and_extras_are_disjoint() -> None:
    default = _default_set()
    for extra, (direct, _refs) in _extras_direct().items():
        both = direct & default
        assert not both, f"{extra!r} redeclares default dependencies: {sorted(both)}"


def test_ultralytics_is_an_extra_not_a_default() -> None:
    assert "ultralytics" not in _default_set()
    assert "ultralytics" in _extras_expanded()["yolo"]


def test_every_extra_is_declared_once_per_distribution_line() -> None:
    # A distribution pinned in two extras must carry the same specifier, or a
    # combined install resolves the intersection and one extra's pin is a lie.
    specs: dict[str, set[str]] = defaultdict(set)
    for _extra, items in _pyproject()["project"]["optional-dependencies"].items():
        for spec in items:
            if _self_extras(spec) is None:
                specs[_requirement_name(spec)].add(spec.strip())
    conflicts = {k: sorted(v) for k, v in specs.items() if len(v) > 1}
    assert not conflicts, conflicts


# ── The dev group is the repository's boundary ───────────────────────────────


def test_dev_group_unions_every_extra_except_dali() -> None:
    dev = _pyproject()["dependency-groups"]["dev"]
    selfrefs = [
        _self_extras(spec)
        for spec in dev
        if isinstance(spec, str) and _self_extras(spec)
    ]
    assert len(selfrefs) == 1, f"expected one saccade[...] entry in dev, got {selfrefs}"
    assert set(selfrefs[0]) == set(_extras_expanded()) - {"dali"}


# ── A core-only interpreter can use the tracker surface ──────────────────────

_CORE_ONLY_PROBE = r"""
import importlib.abc, json, sys

BLOCKED = set(json.loads(sys.argv[1]))

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked for core-only probe: {fullname}")
        return None

for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]
sys.meta_path.insert(0, Block())

import saccade
out = {"version": saccade.__version__, "ok": [], "extra_errors": {}}
for name in sorted(saccade._LAZY):
    _module, _attr, extra = saccade._LAZY[name]
    try:
        getattr(saccade, name)
        out["ok"].append(name)
    except ImportError as exc:
        out["extra_errors"][name] = str(exc)
import saccade.perception.tracking as tracking
out["tracking"] = sorted(tracking.__all__)
out["loaded_blocked"] = sorted(
    {m.split(".")[0] for m in sys.modules} & BLOCKED
)
print(json.dumps(out))
"""


def _blocked_import_names() -> list[str]:
    """Top-level import names of every distribution that lives only in extras."""
    default = _default_set()
    only_extras = frozenset().union(*_extras_expanded().values()) - default
    names = {
        pkg
        for pkg, dists in _PACKAGES_TO_DISTS.items()
        if dists & only_extras and not dists & default
    }
    names -= {"nvidia"}  # namespace shared with the core CUDA wheels
    names.add("ultralytics")
    return sorted(names)


def test_core_only_interpreter_imports_the_tracker_surface() -> None:
    blocked = _blocked_import_names()
    assert {"ultralytics", "cv2", "fastapi", "chromadb", "redis", "llama_index"} <= set(
        blocked
    ), blocked
    proc = subprocess.run(
        [sys.executable, "-c", _CORE_ONLY_PROBE, json.dumps(blocked)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": ""},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    core_names = {n for n, (_m, _a, extra) in _core_lazy_items() if extra is None}
    assert set(out["ok"]) == core_names, out
    for name, (_m, _a, extra) in _core_lazy_items():
        if extra is not None:
            assert name in out["extra_errors"], out
            assert f"saccade[{extra}]" in out["extra_errors"][name], out["extra_errors"]
    assert out["tracking"] == ["GPUByteTracker", "ReorderingBuffer"]
    assert out["loaded_blocked"] == [], out["loaded_blocked"]


def _core_lazy_items() -> list[tuple[str, tuple[str, str, str | None]]]:
    import saccade

    return sorted(saccade._LAZY.items())


# ── Each extra imports ───────────────────────────────────────────────────────


@pytest.mark.parametrize("extra", sorted(EXTRA_SMOKE_MODULES))
def test_extra_smoke_import(extra: str) -> None:
    assert extra in _extras_expanded(), extra
    if extra == "dali":
        pytest.importorskip("nvidia.dali")
    modules = EXTRA_SMOKE_MODULES[extra]
    code = (
        "import importlib, sys\n"
        + "".join(f"importlib.import_module({m!r})\n" for m in modules)
        + "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert proc.returncode == 0, f"{extra}: {modules}\n{proc.stderr[-4000:]}"
    assert proc.stdout.strip().endswith("ok")


def test_every_extra_has_a_smoke_module() -> None:
    assert set(EXTRA_SMOKE_MODULES) == set(_extras_expanded()) - BUILD_ONLY_EXTRAS
    modules = _package_modules()
    for extra, mods in EXTRA_SMOKE_MODULES.items():
        for m in mods:
            assert m in modules, f"{extra}: {m} is not a package module"
            assert _owner(m) == extra, f"{extra}: {m} is owned by {_owner(m)!r}"
