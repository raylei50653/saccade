#!/usr/bin/env python3
"""G2 checks of the installed shipping tree (#465 Phase B PR-12).

Boundary §0 defines G2 (the shipping runtime itself is Python-free) by four
checks, and §6 PR-12 adds ``$ORIGIN`` RUNPATH, an explicit SM list + PTX and a
glibc baseline; the measurement contract is
docs/reference/native_runtime_resolved_config.md §16. Developer tooling only
(``developer_build_debug``): it reads ELF files with ``readelf`` / ``cuobjdump``
and the loader / syscall logs of real runs; it never runs saccade_track itself.

The tree is what ``cmake --install <build> --component shipping`` writes:
``bin/saccade_track`` and the model root ``share/saccade/`` (the attested
operator library sits there at the repository-relative path the frozen lineage
names). Third-party runtime libraries are not in the tree (Phase C decides
whether they ship); a run provides them through ``LD_LIBRARY_PATH``.

Subcommands:

``loaded``   the shared objects a run loaded, from its ``LD_DEBUG=files``
             logs (``LD_DEBUG_OUTPUT=<prefix>``; one file per process):
             requested name, path, sha256.
``deps``     classify a ``loaded`` record into tree / base system / driver /
             third party, hard-link the third-party objects into a directory
             under every name the run asked for (the ``LD_LIBRARY_PATH`` a
             container gets) and write its manifest. With ``--reference`` (the
             same run of the development binary) the third-party sha256 set
             and the operator library must be the reference's.
``static``   G2-1 (NEEDED closure resolved in the tree + the deps directory +
             the base system: complete, no ``libpython*`` /
             ``libtorch_python*``), G2-3 (no ``.py`` / ``.pyc`` /
             ``site-packages`` in the tree), the RUNPATH rule (every ELF the
             build produced: only ``$ORIGIN``-relative RUNPATH, no DT_RPATH;
             the attested operator library is the one enumerated exception,
             identified by its sha256), the SM list + PTX of saccade_track's
             device code, and the glibc baseline of every ELF in the tree and
             the deps directory.
``runtime``  G2-2 and G2-4 from ``strace -ff`` logs of a run: exactly one
             ``execve`` (the entrypoint), no Python library, no Python source,
             no Triton / compiler; the shared objects it opened must be the
             reference run's (by sha256).

Usage::

    check_shipping_tree.py loaded --log-prefix DIR/ld --out loaded.json
    check_shipping_tree.py deps --loaded loaded.json --tree TREE --out DEPS \\
        --manifest deps.json [--reference loaded_dev.json]
    check_shipping_tree.py static --tree TREE --deps-manifest deps.json \\
        --report static.json
    check_shipping_tree.py runtime --strace-prefix DIR/s --tree TREE \\
        --tree-mount /opt/saccade --deps-manifest deps.json \\
        --deps-mount /opt/saccade-deps --reference loaded_dev.json \\
        --report runtime.json

Exit 0: every check passes; 1: a check fails (named in the report); 2: error.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "saccade.shipping_g2/v1"
MODEL_ROOT = "share/saccade"
ENTRYPOINT = "bin/saccade_track"
ATTESTATION = "configs/shipping/mamba_head_realization.attestation.json"

# What the clean image provides (Ubuntu 24.04): glibc, the GCC runtime and zlib
# (zlib1g is Priority: required there; cuDNN's graph library needs it). The
# baseline is the newest symbol version each family may require (owner
# decision, docs §16): glibc 2.39, libstdc++ 6.0.33 (GLIBCXX_3.4.33,
# CXXABI_1.3.15).
BASE_SYSTEM = frozenset(
    {
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libdl.so.2",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "librt.so.1",
        "libstdc++.so.6",
        "libutil.so.1",
        "libz.so.1",
    }
)
VERSION_BASELINE = {"GLIBC": (2, 39), "GLIBCXX": (3, 4, 33), "CXXABI": (1, 3, 15)}
# The display driver's user-mode libraries come from the host (the container
# runtime mounts them); they are not the package's.
_DRIVER = re.compile(
    r"^lib(cuda|cudadebugger|nvidia-[\w.-]+|nvcuvid|dxcore|d3d12(core)?)\.so"
)
_DRIVER_DIRS = ("/usr/lib/wsl/",)
# The shipping SM list (shipping/CMakeLists.txt SACCADE_SHIPPING_TORCH_CUDA_ARCH_LIST).
SHIPPING_SASS = frozenset({"sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"})
SHIPPING_PTX = frozenset({"sm_120"})
_PYTHON_LIB = re.compile(r"^lib(python\d|torch_python)")
_FORBIDDEN_OPEN = re.compile(
    r"(^|/)(lib(python\d|torch_python|triton)[^/]*|[^/]+\.pyc?|site-packages|dist-packages"
    r"|__pycache__|\.triton|torchinductor[^/]*)(/|$)"
)
_SHARED_OBJECT = re.compile(r"\.so(\.[0-9.]+)?$")
_PYTHON_FILE = re.compile(r"\.(py|pyc|pyo|pth)$")
_PYTHON_DIR = frozenset({"__pycache__", "site-packages", "dist-packages"})


class CheckError(RuntimeError):
    pass


Runner = Callable[[list[str]], str]


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise CheckError(f"{' '.join(cmd)} failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_elf(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(4) == b"\x7fELF"


def classify_name(name: str, path: str = "") -> str:
    base = os.path.basename(name)
    if base in BASE_SYSTEM:
        return "base_system"
    if _DRIVER.match(base) or path.startswith(_DRIVER_DIRS):
        return "driver"
    return "third_party"


# ── ELF reading ────────────────────────────────────────────────────────────────


def dynamic_section(path: Path, run: Runner = _run) -> dict[str, Any]:
    out = run(["readelf", "-d", "-W", str(path)])
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)
    soname = re.findall(r"\(SONAME\)\s+Library soname: \[([^\]]+)\]", out)
    runpath = re.findall(r"\(RUNPATH\)\s+Library runpath: \[([^\]]*)\]", out)
    rpath = re.findall(r"\(RPATH\)\s+Library rpath: \[([^\]]*)\]", out)
    return {
        "needed": needed,
        "soname": soname[0] if soname else None,
        "runpath": runpath[0].split(":") if runpath else [],
        "rpath": rpath[0].split(":") if rpath else [],
    }


def _vtuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split("."))


def version_needs(path: Path, run: Runner = _run) -> dict[str, str]:
    """Newest required symbol version per family (GLIBC, GLIBCXX, CXXABI)."""
    out = run(["readelf", "-V", "-W", str(path)])
    section = out.split("Version needs section", 1)
    newest: dict[str, tuple[int, ...]] = {}
    if len(section) == 2:
        for fam, ver in re.findall(
            r"Name: (GLIBC|GLIBCXX|CXXABI)_([0-9.]+)\b", section[1]
        ):
            v = _vtuple(ver)
            if v > newest.get(fam, ()):
                newest[fam] = v
    return {fam: ".".join(map(str, v)) for fam, v in sorted(newest.items())}


def over_baseline(needs: dict[str, str]) -> list[str]:
    return [
        f"{fam}_{ver}"
        for fam, ver in needs.items()
        if _vtuple(ver) > VERSION_BASELINE[fam]
    ]


def cuda_archs(path: Path, run: Runner = _run) -> dict[str, list[str]]:
    """SASS / PTX architectures embedded in an ELF (empty when it has no device
    code: cuobjdump then exits non-zero, which is not an error here)."""

    def listing(flag: str) -> list[str]:
        try:
            text = run(["cuobjdump", flag, str(path)])
        except CheckError:
            return []
        archs = set(re.findall(r"\.(sm_\d+a?)\.", text))
        return sorted(archs, key=lambda a: int(re.sub(r"\D", "", a)))

    return {"sass": listing("--list-elf"), "ptx": listing("--list-ptx")}


# ── loaded ─────────────────────────────────────────────────────────────────────

_LD_MAPPED = re.compile(r"^\s*\d+:\s+file=(\S+) \[\d+\];\s+generating link map")
_LD_INIT = re.compile(r"^\s*\d+:\s+calling init: (\S+)")


def parse_ld_debug(text: str) -> tuple[list[str], list[str]]:
    """What one process mapped, from its ``LD_DEBUG=files`` log: the names the
    loader was asked for ('generating link map'; the path itself for an
    absolute dlopen) and the path of every initialized object ('calling
    init'). glibc prints the path only on the latter."""
    names, paths = [], []
    for line in text.splitlines():
        m = _LD_MAPPED.match(line)
        if m:
            names.append(m.group(1))
            continue
        m = _LD_INIT.match(line)
        if m:
            paths.append(m.group(1))
    return names, paths


def match_loaded(
    names: list[str], paths: list[str], soname: Callable[[str], str | None]
) -> dict[str, list[str]]:
    """path -> the names it was requested under (by path, basename or SONAME).
    Relative paths are as the run gave them (its working directory)."""
    out: dict[str, list[str]] = {p: [] for p in paths}
    for name in names:
        if "/" in name:  # a dlopen by path: logged as given
            hits = [p for p in paths if p == name]
        else:
            hits = [p for p in paths if os.path.basename(p) == name]
            if not hits:
                hits = [p for p in paths if soname(p) == name]
        if len(hits) != 1:
            raise CheckError(f"loader log: {name} matches {len(hits)} loaded objects")
        if name not in out[hits[0]]:
            out[hits[0]].append(name)
    return out


def cmd_loaded(args: argparse.Namespace) -> int:
    logs = sorted(args.log_prefix.parent.glob(args.log_prefix.name + ".*"))
    if not logs:
        raise CheckError(f"no LD_DEBUG logs {args.log_prefix}.*")
    seen: dict[str, dict[str, Any]] = {}
    for log in logs:
        names, paths = parse_ld_debug(log.read_text(errors="replace"))
        matched = match_loaded(
            names, paths, lambda p: dynamic_section(Path(p))["soname"]
        )
        for path, requested in matched.items():
            real = os.path.realpath(path)
            entry = seen.setdefault(
                real, {"path": path, "realpath": real, "requested": [], "sha256": None}
            )
            for n in requested or [os.path.basename(path)]:
                if n not in entry["requested"]:
                    entry["requested"].append(n)
    for entry in seen.values():
        entry["sha256"] = sha256_file(Path(entry["realpath"]))
    record = {
        "schema": SCHEMA,
        "kind": "loaded",
        "logs": [str(p) for p in logs],
        "objects": sorted(seen.values(), key=lambda e: e["realpath"]),
    }
    _write(args.out, record)
    print(f"loaded: {len(seen)} objects from {len(logs)} log(s)")
    return 0


# ── deps ───────────────────────────────────────────────────────────────────────


def classify_loaded(
    loaded: dict[str, Any], tree: Path | None
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "tree": [],
        "base_system": [],
        "driver": [],
        "third_party": [],
    }
    root = str(tree.resolve()) + "/" if tree else None
    for e in loaded["objects"]:
        if root and e["realpath"].startswith(root):
            groups["tree"].append(e)
        else:
            groups[classify_name(e["requested"][0], e["realpath"])].append(e)
    return groups


def _op_library_sha(tree: Path) -> str:
    att = json.loads((tree / MODEL_ROOT / ATTESTATION).read_text())
    return str(att["op_library"]["sha256"])


def cmd_deps(args: argparse.Namespace) -> int:
    loaded = json.loads(args.loaded.read_text())
    groups = classify_loaded(loaded, args.tree)
    problems: list[str] = []
    entry_sha = sha256_file(args.tree / ENTRYPOINT)
    op_sha = _op_library_sha(args.tree)
    tree_shas = {e["sha256"] for e in groups["tree"]}
    if op_sha not in tree_shas:
        problems.append("the run did not load the tree's attested operator library")
    unexpected_tree = tree_shas - {op_sha, entry_sha}
    if unexpected_tree:
        problems.append(f"the run loaded other tree objects: {sorted(unexpected_tree)}")
    for e in groups["third_party"]:
        if any(
            _PYTHON_LIB.match(os.path.basename(n))
            for n in [*e["requested"], e["realpath"]]
        ):
            problems.append(f"the run loaded a Python library: {e['realpath']}")

    args.out.mkdir(parents=True, exist_ok=False)
    entries = []
    for e in groups["third_party"]:
        src = Path(e["realpath"])
        dyn = dynamic_section(src)
        names = sorted(
            {os.path.basename(n) for n in e["requested"]}
            | ({dyn["soname"]} if dyn["soname"] else set())
        )
        for n in names:
            dst = args.out / n
            if dst.exists():
                raise CheckError(f"{n} requested for two different objects")
            try:
                os.link(src, dst)
            except OSError:
                dst.write_bytes(src.read_bytes())
        entries.append(
            {
                "names": names,
                "soname": dyn["soname"],
                "source": e["realpath"],
                "sha256": e["sha256"],
                "bytes": src.stat().st_size,
                "needed": dyn["needed"],
                "version_needs": version_needs(src),
            }
        )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "deps",
        "reading": "Third-party shared objects a run of the shipping tree loaded, provided "
        "outside the tree through LD_LIBRARY_PATH (not bundled; Phase C decides). Base "
        "system = the clean image's glibc / GCC runtime; driver = the host driver the "
        "container runtime mounts.",
        "loaded": str(args.loaded),
        "tree": str(args.tree),
        "entries": sorted(entries, key=lambda x: x["names"][0]),
        "base_system": sorted(e["realpath"] for e in groups["base_system"]),
        "driver": sorted(e["realpath"] for e in groups["driver"]),
        "tree_objects": sorted(e["realpath"] for e in groups["tree"]),
    }
    if args.reference is not None:
        ref = classify_loaded(json.loads(args.reference.read_text()), None)
        ref_third = {e["sha256"] for e in ref["third_party"]}
        ref_op = [e for e in ref["third_party"] if e["sha256"] == op_sha]
        cur_third = {e["sha256"] for e in groups["third_party"]}
        missing = sorted(ref_third - cur_third - {op_sha})
        extra = sorted(cur_third - ref_third)
        manifest["reference"] = {
            "loaded": str(args.reference),
            "missing": missing,
            "extra": extra,
            "operator_library_same": bool(ref_op),
        }
        if missing or extra:
            problems.append(
                f"third-party set differs from the reference: missing {missing}, extra {extra}"
            )
        if not ref_op:
            problems.append("the reference run loaded another operator library")
    manifest["problems"] = problems
    _write(args.manifest, manifest)
    print(
        f"deps: {len(entries)} third-party objects -> {args.out}; problems {problems}"
    )
    return 1 if problems else 0


# ── static ─────────────────────────────────────────────────────────────────────


def python_files(tree: Path) -> list[str]:
    bad = []
    for p in sorted(tree.rglob("*")):
        rel = p.relative_to(tree).as_posix()
        if p.is_dir() and p.name in _PYTHON_DIR:
            bad.append(rel + "/")
        elif p.is_file() and (_PYTHON_FILE.search(p.name) or _PYTHON_LIB.match(p.name)):
            bad.append(rel)
    return bad


def _expand_origin(entry: str, elf_dir: Path) -> Path | None:
    if entry.startswith(("$ORIGIN", "${ORIGIN}")):
        return Path(
            os.path.normpath(
                entry.replace("${ORIGIN}", str(elf_dir)).replace(
                    "$ORIGIN", str(elf_dir)
                )
            )
        )
    return None


def closure(
    roots: list[Path], deps_dir: Path, run: Runner = _run
) -> tuple[list[dict[str, Any]], list[str]]:
    """NEEDED closure as a clean container resolves it: $ORIGIN RUNPATH,
    LD_LIBRARY_PATH (the deps directory), then the base system / driver.
    Absolute RUNPATH entries are ignored (no build or venv path exists there)."""
    resolved: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    queue = [(r, "root") for r in roots]
    visited: set[str] = set()
    while queue:
        elf, why = queue.pop()
        key = str(elf.resolve())
        if key in visited:
            continue
        visited.add(key)
        dyn = dynamic_section(elf, run)
        search = [
            d for d in (_expand_origin(e, elf.parent) for e in dyn["runpath"]) if d
        ]
        search.append(deps_dir)
        for name in dyn["needed"]:
            if _PYTHON_LIB.match(name):
                problems.append(f"{elf.name} needs {name}")
            hit = next((d / name for d in search if (d / name).exists()), None)
            if hit is not None:
                resolved.setdefault(
                    name, {"name": name, "path": str(hit), "class": "package"}
                )
                queue.append((hit, name))
                continue
            cls = classify_name(name)
            if cls == "third_party":
                problems.append(f"{elf.name}: NEEDED {name} is not resolvable")
            resolved.setdefault(name, {"name": name, "path": None, "class": cls})
    return sorted(resolved.values(), key=lambda x: x["name"]), problems


def cmd_static(args: argparse.Namespace) -> int:
    tree: Path = args.tree.resolve()
    deps = json.loads(args.deps_manifest.read_text())
    deps_dir: Path = args.deps_dir.resolve()
    checks: dict[str, Any] = {}
    att = json.loads((tree / MODEL_ROOT / ATTESTATION).read_text())
    op_rel = Path(MODEL_ROOT) / att["op_library"]["path"]
    op_sha = att["op_library"]["sha256"]

    # G2-3
    py = python_files(tree)
    checks["g2_3_no_python_files"] = {"pass": not py, "found": py}

    elves = sorted(p for p in tree.rglob("*") if p.is_file() and is_elf(p))
    produced, exceptions, rp_problems = [], [], []
    per_elf: dict[str, Any] = {}
    for p in elves:
        rel = p.relative_to(tree)
        dyn = dynamic_section(p)
        sha = sha256_file(p)
        info: dict[str, Any] = {
            "sha256": sha,
            "runpath": dyn["runpath"],
            "rpath": dyn["rpath"],
        }
        if rel == op_rel:
            if sha != op_sha:
                rp_problems.append(f"{rel}: sha256 {sha} is not the attested {op_sha}")
            info["exception"] = "attested operator library (frozen bytes, docs §16)"
            exceptions.append(str(rel))
        else:
            produced.append(p)
            if dyn["rpath"]:
                rp_problems.append(f"{rel}: has DT_RPATH {dyn['rpath']}")
            bad = [e for e in dyn["runpath"] if _expand_origin(e, p.parent) is None]
            if bad:
                rp_problems.append(
                    f"{rel}: RUNPATH entries not $ORIGIN-relative: {bad}"
                )
        info["version_needs"] = version_needs(p)
        info["cuda"] = cuda_archs(p)
        per_elf[str(rel)] = info
    if (tree / ENTRYPOINT) not in produced:
        rp_problems.append(f"{ENTRYPOINT} missing")
    if str(op_rel) not in exceptions:
        rp_problems.append(f"{op_rel} (the attested operator library) missing")
    checks["runpath_origin_only"] = {
        "pass": not rp_problems,
        "problems": rp_problems,
        "produced": [str(p.relative_to(tree)) for p in produced],
        "exceptions": exceptions,
    }

    ep = per_elf.get(ENTRYPOINT, {}).get("cuda", {"sass": [], "ptx": []})
    sm_ok = set(ep["sass"]) == SHIPPING_SASS and set(ep["ptx"]) == SHIPPING_PTX
    checks["sm_ptx_entrypoint"] = {
        "pass": sm_ok,
        "expected": {"sass": sorted(SHIPPING_SASS), "ptx": sorted(SHIPPING_PTX)},
        "observed": ep,
        "named_limit": {str(op_rel): per_elf.get(str(op_rel), {}).get("cuda")},
    }

    over: dict[str, list[str]] = {}
    for rel, info in per_elf.items():
        o = over_baseline(info["version_needs"])
        if o:
            over[rel] = o
    for e in deps["entries"]:
        o = over_baseline(e["version_needs"])
        if o:
            over[e["names"][0]] = o
    checks["glibc_baseline"] = {
        "pass": not over,
        "baseline": {k: ".".join(map(str, v)) for k, v in VERSION_BASELINE.items()},
        "over": over,
    }

    clo, clo_problems = closure([tree / ENTRYPOINT, tree / op_rel], deps_dir)
    py_in_closure = [c["name"] for c in clo if _PYTHON_LIB.match(c["name"])]
    checks["g2_1_needed_closure"] = {
        "pass": not clo_problems and not py_in_closure,
        "problems": clo_problems,
        "closure": clo,
    }
    report = {
        "schema": SCHEMA,
        "kind": "static",
        "tree": str(tree),
        "deps_dir": str(deps_dir),
        "elves": per_elf,
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
    }
    _write(args.report, report)
    for name, c in checks.items():
        print(f"{name}: {'PASS' if c['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


# ── runtime ────────────────────────────────────────────────────────────────────

_SYSCALL = re.compile(
    r"^(?:\d+\s+)?(execve|execveat|openat|open)\((.*)\)\s+=\s+(-?\d+)"
)
_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')


def parse_strace(text: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"exec": [], "open": []}
    for line in text.splitlines():
        m = _SYSCALL.match(line.strip())
        if not m:
            continue
        call, argstr, rc = m.group(1), m.group(2), int(m.group(3))
        strings = _STRING.findall(argstr)
        path = strings[0] if strings else ""
        (out["exec"] if call.startswith("exec") else out["open"]).append(
            {"call": call, "path": path, "rc": rc}
        )
    return out


def cmd_runtime(args: argparse.Namespace) -> int:
    logs = sorted(args.strace_prefix.parent.glob(args.strace_prefix.name + ".*"))
    if not logs:
        raise CheckError(f"no strace logs {args.strace_prefix}.*")
    execs: list[dict[str, Any]] = []
    opens: list[dict[str, Any]] = []
    for log in logs:
        parsed = parse_strace(log.read_text(errors="replace"))
        execs += parsed["exec"]
        opens += parsed["open"]
    checks: dict[str, Any] = {}
    entry = f"{args.tree_mount.rstrip('/')}/{ENTRYPOINT}"
    exec_ok = len(execs) == 1 and execs[0]["path"] == entry and execs[0]["rc"] == 0
    checks["g2_2_g2_4_single_exec"] = {
        "pass": exec_ok,
        "expected": entry,
        "execs": execs,
    }
    forbidden = sorted({o["path"] for o in opens if _FORBIDDEN_OPEN.search(o["path"])})
    checks["g2_2_g2_4_no_python_triton_open"] = {
        "pass": not forbidden,
        "attempted": forbidden,
    }

    deps = json.loads(args.deps_manifest.read_text())
    by_name = {n: e["sha256"] for e in deps["entries"] for n in e["names"]}
    tree = args.tree.resolve()
    opened_shas: dict[str, str] = {}
    unmapped: list[str] = []
    for o in opens:
        if o["rc"] < 0 or not _SHARED_OBJECT.search(os.path.basename(o["path"])):
            continue
        p = o["path"]
        if p.startswith(args.deps_mount.rstrip("/") + "/"):
            name = p[len(args.deps_mount.rstrip("/")) + 1 :]
            if name in by_name:
                opened_shas[p] = by_name[name]
            else:
                unmapped.append(p)
        elif p.startswith(args.tree_mount.rstrip("/") + "/"):
            host = tree / p[len(args.tree_mount.rstrip("/")) + 1 :]
            opened_shas[p] = sha256_file(host)
        elif classify_name(os.path.basename(p), p) == "third_party":
            unmapped.append(p)
    ref = classify_loaded(json.loads(args.reference.read_text()), None)
    ref_shas = {e["sha256"] for e in ref["third_party"]}
    cur = set(opened_shas.values())
    checks["loaded_set_equals_reference"] = {
        "pass": cur == ref_shas and not unmapped,
        "missing": sorted(ref_shas - cur),
        "extra": sorted(cur - ref_shas),
        "unmapped_opens": sorted(set(unmapped)),
    }
    nvrtc = sorted(
        {
            o["path"]
            for o in opens
            if o["rc"] >= 0 and "nvrtc" in os.path.basename(o["path"])
        }
    )
    report = {
        "schema": SCHEMA,
        "kind": "runtime",
        "logs": len(logs),
        "checks": checks,
        "observations": {"nvrtc_objects_opened": nvrtc},
        "pass": all(c["pass"] for c in checks.values()),
    }
    _write(args.report, report)
    for name, c in checks.items():
        print(f"{name}: {'PASS' if c['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


# ── main ───────────────────────────────────────────────────────────────────────


def _write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("loaded")
    a.add_argument("--log-prefix", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("deps")
    d.add_argument("--loaded", type=Path, required=True)
    d.add_argument("--tree", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--manifest", type=Path, required=True)
    d.add_argument("--reference", type=Path, default=None)
    s = sub.add_parser("static")
    s.add_argument("--tree", type=Path, required=True)
    s.add_argument("--deps-manifest", type=Path, required=True)
    s.add_argument("--deps-dir", type=Path, required=True)
    s.add_argument("--report", type=Path, required=True)
    r = sub.add_parser("runtime")
    r.add_argument("--strace-prefix", type=Path, required=True)
    r.add_argument("--tree", type=Path, required=True)
    r.add_argument("--tree-mount", required=True)
    r.add_argument("--deps-manifest", type=Path, required=True)
    r.add_argument("--deps-mount", required=True)
    r.add_argument("--reference", type=Path, required=True)
    r.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        return {
            "loaded": cmd_loaded,
            "deps": cmd_deps,
            "static": cmd_static,
            "runtime": cmd_runtime,
        }[args.cmd](args)
    except CheckError as exc:
        print(f"check_shipping_tree: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
