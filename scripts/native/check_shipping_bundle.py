#!/usr/bin/env python3
"""Checks of the bundled shipping tree (#465 Phase C PR-C1).

The tree is what ``cmake --install <build> --component shipping`` writes after
PR-C1 (docs/reference/native_runtime_resolved_config.md §17): the launcher
``bin/saccade_track``, the entrypoint ``libexec/saccade_track`` (the pinned
PR-12 bytes), the loader provenance check ``lib/saccade_loader_audit.so``, the
bundled third-party set ``lib/vendor/`` (shipping/third_party_set.json),
``licenses/`` and the model root ``share/saccade/``. Developer tooling only
(``developer_build_debug``); the ELF and log readers are
``check_shipping_tree.py``'s (PR-12).

Subcommands:

``static``   layout (exactly the expected files), the vendor set's sha256, the
             entrypoint pin, the launcher's bytes, the auditor (libc only, no
             RUNPATH), licenses, G2-3, G2-1 (NEEDED closure resolved the way
             the launcher's loader invocation resolves it: complete inside the
             tree + base system + driver, no Python), search-path containment
             (every relative DT_RPATH / DT_RUNPATH entry of every ELF in the
             tree expands inside the tree; the operator library's absolute
             RUNPATH and libcusparseLt's empty entry are the enumerated
             exceptions), the SM list + PTX of the entrypoint, and the glibc
             baseline of every ELF in the tree.
``sources``  where a host run's loaded objects came from (its
             ``LD_DEBUG=files`` logs, 'calling init' paths): every bundled
             name from the tree's lib/vendor with the pinned bytes, the
             operator library and the auditor from the tree, nothing else.
``runtime``  G2-2 / G2-4 from ``strace -ff`` logs of a run started through
             the launcher: exactly two ``execve`` (the launcher, then the
             system loader with the launcher's arguments), no Python / Triton /
             compiler open, and every shared object it opened from the tree is
             the vendor set + the operator library + the auditor, with no
             bundled name opened from anywhere else.
``rejected`` (PR-C2) a run of the launcher given an option the shipping
             entrypoint does not have (``strace -ff`` logs + its log): exit 2
             with ``saccade_track: unknown argument <option>``, the same exec
             chain, no open (even attempted) of anything under the model root,
             of the operator library or of a GPU device node, and no output.

PR-C2 adds to ``static``: the entrypoint contains none of the byte strings in
shipping/measurement_surface.json (no measurement hook, no developer option).

Usage::

    check_shipping_bundle.py static --tree TREE --report static.json
    check_shipping_bundle.py sources --log-prefix DIR/ld --tree TREE --report sources.json
    check_shipping_bundle.py runtime --strace-prefix DIR/s --tree TREE \\
        --tree-mount /opt/saccade --report runtime.json
    check_shipping_bundle.py rejected --strace-prefix DIR/s --log DIR/saccade_track.log \\
        --out-dir DIR --option=--measurement-mutation --tree-mount /opt/saccade \\
        --report rejected.json

Exit 0: every check passes; 1: a check fails (named in the report); 2: error.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_shipping_tree as g2  # noqa: E402

SCHEMA = "saccade.shipping_bundle/v1"
REPO = Path(__file__).resolve().parents[2]
THIRD_PARTY_SET = REPO / "shipping/third_party_set.json"
ENTRYPOINT_PIN = REPO / "shipping/entrypoint_pin.json"
MEASUREMENT_SURFACE = REPO / "shipping/measurement_surface.json"
LAUNCHER_SOURCE = REPO / "shipping/launcher/saccade_track.sh"
NOTICE_SOURCE = REPO / "shipping/THIRD_PARTY.md"
LAUNCHER = "bin/saccade_track"
ENTRYPOINT = "libexec/saccade_track"
AUDITOR = "lib/saccade_loader_audit.so"
VENDOR = "lib/vendor"
SYSTEM_LOADER = "/lib64/ld-linux-x86-64.so.2"
MODEL_ROOT_FILES = (
    "configs/shipping/mamba_whole_graph.resolved.json",
    "configs/shipping/mamba_head_realization.attestation.json",
    "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json",
    "models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt",
    "models/yolo/yolo26s_backbone_640_best.engine",
    "build/libsaccade_scan_torchop.so",
)
# Search-path entries that do not expand inside the tree, each bound to the
# bytes that carry it.
_CUSPARSELT = "libcusparseLt.so.0"
# GPU device nodes a CUDA context opens (WSL2: /dev/dxg; native: /dev/nvidia*).
_GPU_DEVICE = re.compile(r"^/dev/(dxg$|nvidia)")


def measurement_surface(binary: Path, surface: Path) -> dict[str, Any]:
    """Occurrences in `binary` of each byte string the shipping entrypoint must
    not contain (shipping/measurement_surface.json, PR-C2)."""
    data = binary.read_bytes() if binary.is_file() else b""
    tokens = _load(surface)["forbidden"]
    found = {t: data.count(t.encode()) for t in tokens if t.encode() in data}
    return {
        "pass": bool(data) and not found,
        "found": found,
        "tokens": len(tokens),
        "surface": str(surface),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def expected_files(set_: dict[str, Any]) -> set[str]:
    files = {LAUNCHER, ENTRYPOINT, AUDITOR}
    files |= {f"{g2.MODEL_ROOT}/{f}" for f in MODEL_ROOT_FILES}
    files |= {f"{VENDOR}/{e['soname']}" for e in set_["entries"]}
    files |= {
        "licenses/THIRD_PARTY.md",
        "licenses/saccade/LICENSE",
        "licenses/saccade/NOTICE",
    }
    for e in set_["entries"]:
        for lic in e["license_files"]:
            files.add(f"licenses/{e['wheel']}/{lic['path'].rsplit('/', 1)[-1]}")
    return files


def search_entries(dyn: dict[str, Any]) -> list[tuple[str, str]]:
    return [("RPATH", e) for e in dyn["rpath"]] + [
        ("RUNPATH", e) for e in dyn["runpath"]
    ]


def containment(
    tree: Path,
    elves: dict[str, dict[str, Any]],
    op_rel: str,
    op_sha: str,
    cusparselt_sha: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Every relative search-path entry must expand inside the tree; entries
    that cannot are allowed only on the enumerated objects (by sha256)."""
    problems: list[str] = []
    exceptions: list[dict[str, Any]] = []
    for rel, info in elves.items():
        elf_dir = (tree / rel).parent
        for tag, entry in search_entries(info["dyn"]):
            target = g2._expand_origin(entry, elf_dir)
            if target is not None:
                if target != tree and tree not in target.parents:
                    problems.append(
                        f"{rel}: {tag} {entry} -> {target} is outside the tree"
                    )
                continue
            if rel == op_rel and info["sha256"] == op_sha and tag == "RUNPATH":
                exceptions.append(
                    {
                        "elf": rel,
                        "tag": tag,
                        "entry": entry,
                        "why": "attested operator library (frozen bytes)",
                    }
                )
            elif (
                rel == f"{VENDOR}/{_CUSPARSELT}"
                and info["sha256"] == cusparselt_sha
                and entry == ""
            ):
                exceptions.append(
                    {
                        "elf": rel,
                        "tag": tag,
                        "entry": entry,
                        "why": "empty entry (working directory); the auditor refuses relative candidates",
                    }
                )
            else:
                problems.append(f"{rel}: {tag} entry {entry!r} is not $ORIGIN-relative")
    return problems, exceptions


def bundle_closure(
    tree: Path, roots: list[str], vendor_names: set[str], run: g2.Runner = g2._run
) -> tuple[list[dict[str, Any]], list[str]]:
    """NEEDED closure as the launcher's loader resolves it: a third-party name
    must be in lib/vendor (--library-path; containment guarantees no other
    in-tree candidate exists), base system / driver come from the host."""
    problems: list[str] = []
    resolved: dict[str, dict[str, Any]] = {}
    queue = list(roots)
    seen: set[str] = set()
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for name in g2.dynamic_section(tree / rel, run)["needed"]:
            if g2._PYTHON_LIB.match(name):
                problems.append(f"{rel} needs {name}")
            if name in vendor_names:
                if not (tree / VENDOR / name).is_file():
                    problems.append(f"{rel}: NEEDED {name} is missing from {VENDOR}")
                    resolved.setdefault(
                        name, {"name": name, "path": None, "class": "vendor"}
                    )
                    continue
                resolved.setdefault(
                    name, {"name": name, "path": f"{VENDOR}/{name}", "class": "vendor"}
                )
                queue.append(f"{VENDOR}/{name}")
                continue
            cls = g2.classify_name(name)
            if cls == "third_party":
                problems.append(f"{rel}: NEEDED {name} is not in {VENDOR}")
            resolved.setdefault(name, {"name": name, "path": None, "class": cls})
    return sorted(resolved.values(), key=lambda x: x["name"]), problems


def cmd_static(args: argparse.Namespace) -> int:
    tree: Path = args.tree.resolve()
    set_ = _load(args.third_party_set)
    pin = _load(args.entrypoint_pin)
    att = _load(tree / g2.MODEL_ROOT / g2.ATTESTATION)
    op_rel = f"{g2.MODEL_ROOT}/{att['op_library']['path']}"
    op_sha = att["op_library"]["sha256"]
    by_soname = {e["soname"]: e for e in set_["entries"]}
    checks: dict[str, Any] = {}

    present = {
        p.relative_to(tree).as_posix()
        for p in tree.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    want = expected_files(set_)
    checks["layout_exact"] = {
        "pass": present == want,
        "missing": sorted(want - present),
        "extra": sorted(present - want),
    }

    vendor_bad = []
    for soname, e in by_soname.items():
        p = tree / VENDOR / soname
        if not p.is_file() or p.is_symlink():
            vendor_bad.append(f"{soname}: missing or not a regular file")
        elif g2.sha256_file(p) != e["sha256"]:
            vendor_bad.append(f"{soname}: sha256 is not the pinned {e['sha256']}")
    checks["vendor_set_pinned"] = {
        "pass": not vendor_bad,
        "problems": vendor_bad,
        "objects": len(by_soname),
    }

    ep = tree / ENTRYPOINT
    ep_sha = g2.sha256_file(ep) if ep.is_file() else None
    checks["entrypoint_pinned"] = {
        "pass": ep_sha == pin["sha256"],
        "observed": ep_sha,
        "pinned": pin["sha256"],
    }

    la = tree / LAUNCHER
    la_ok = (
        la.is_file()
        and la.read_bytes() == args.launcher_source.read_bytes()
        and os.access(la, os.X_OK)
    )
    checks["launcher_exact"] = {"pass": la_ok, "source": str(args.launcher_source)}

    lic_bad = []
    for e in set_["entries"]:
        for lic in e["license_files"]:
            p = tree / "licenses" / e["wheel"] / lic["path"].rsplit("/", 1)[-1]
            if not p.is_file() or g2.sha256_file(p) != lic["sha256"]:
                lic_bad.append(str(p.relative_to(tree)))
    notice = tree / "licenses/THIRD_PARTY.md"
    if not notice.is_file() or notice.read_bytes() != NOTICE_SOURCE.read_bytes():
        lic_bad.append("licenses/THIRD_PARTY.md")
    checks["licenses"] = {"pass": not lic_bad, "problems": lic_bad}

    py = g2.python_files(tree)
    checks["g2_3_no_python_files"] = {"pass": not py, "found": py}

    elves: dict[str, dict[str, Any]] = {}
    for rel in sorted(present):
        p = tree / rel
        if p.is_file() and g2.is_elf(p):
            elves[rel] = {
                "sha256": g2.sha256_file(p),
                "dyn": g2.dynamic_section(p),
                "version_needs": g2.version_needs(p),
            }

    rp_bad = []
    ep_dyn = elves.get(ENTRYPOINT, {}).get("dyn")
    if ep_dyn is None or ep_dyn["rpath"] or ep_dyn["runpath"] != ["$ORIGIN/../lib"]:
        rp_bad.append(
            f"{ENTRYPOINT}: search path {ep_dyn and search_entries(ep_dyn)} is not RUNPATH $ORIGIN/../lib only"
        )
    au = elves.get(AUDITOR)
    if (
        au is None
        or au["dyn"]["rpath"]
        or au["dyn"]["runpath"]
        or au["dyn"]["needed"] != ["libc.so.6"]
    ):
        rp_bad.append(
            f"{AUDITOR}: must need libc.so.6 only and carry no search path ({au and au['dyn']})"
        )
    if elves.get(op_rel, {}).get("sha256") != op_sha:
        rp_bad.append(f"{op_rel}: not the attested operator library {op_sha}")
    checks["produced_elves"] = {"pass": not rp_bad, "problems": rp_bad}

    cont_bad, cont_exc = containment(
        tree, elves, op_rel, op_sha, by_soname.get(_CUSPARSELT, {}).get("sha256", "")
    )
    checks["search_path_containment"] = {
        "pass": not cont_bad,
        "problems": cont_bad,
        "exceptions": cont_exc,
    }

    clo, clo_bad = bundle_closure(tree, [ENTRYPOINT, op_rel], set(by_soname))
    checks["g2_1_needed_closure"] = {
        "pass": not clo_bad,
        "problems": clo_bad,
        "closure": clo,
    }

    checks["entrypoint_no_measurement_surface"] = measurement_surface(
        ep, args.measurement_surface
    )

    cuda = g2.cuda_archs(ep) if ep.is_file() else {"sass": [], "ptx": []}
    checks["sm_ptx_entrypoint"] = {
        "pass": set(cuda["sass"]) == g2.SHIPPING_SASS
        and set(cuda["ptx"]) == g2.SHIPPING_PTX,
        "observed": cuda,
        "named_limit": {op_rel: g2.cuda_archs(tree / op_rel)},
    }

    over = {
        rel: o
        for rel, info in elves.items()
        if (o := g2.over_baseline(info["version_needs"]))
    }
    checks["glibc_baseline"] = {
        "pass": not over,
        "baseline": {k: ".".join(map(str, v)) for k, v in g2.VERSION_BASELINE.items()},
        "over": over,
    }

    report = {
        "schema": SCHEMA,
        "kind": "static",
        "tree": str(tree),
        "elves": {rel: {k: v for k, v in info.items()} for rel, info in elves.items()},
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
    }
    g2._write(args.report, report)
    for name, c in checks.items():
        print(f"{name}: {'PASS' if c['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


def _argv(line_strings: list[str]) -> list[str]:
    """execve("path", ["argv0", ...], ...): strace prints the path, then argv."""
    return line_strings[1:]


def strace_records(
    prefix: Path,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    """(log count, execve records, open records) of ``strace -ff -o prefix``."""
    logs = sorted(prefix.parent.glob(prefix.name + ".*"))
    if not logs:
        raise g2.CheckError(f"no strace logs {prefix}.*")
    execs: list[dict[str, Any]] = []
    opens: list[dict[str, Any]] = []
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            m = g2._SYSCALL.match(line.strip())
            if not m:
                continue
            call, argstr, rc = m.group(1), m.group(2), int(m.group(3))
            strings = [
                s.encode().decode("unicode_escape") for s in g2._STRING.findall(argstr)
            ]
            rec = {"call": call, "path": strings[0] if strings else "", "rc": rc}
            if call.startswith("exec"):
                rec["argv"] = _argv(strings)
                execs.append(rec)
            else:
                opens.append(rec)
    return len(logs), execs, opens


def exec_chain(execs: list[dict[str, Any]], mount: str) -> dict[str, Any]:
    """Exactly the launcher, then the system loader with its arguments."""
    want_loader_args = [
        "--library-path",
        f"{mount}/{VENDOR}",
        "--audit",
        f"{mount}/{AUDITOR}",
    ]
    exec_ok = (
        len(execs) == 2
        and all(e["rc"] == 0 for e in execs)
        and execs[0]["path"] == f"{mount}/{LAUNCHER}"
        and execs[1]["path"] == SYSTEM_LOADER
        and execs[1]["argv"][1:5] == want_loader_args
        and f"{mount}/{ENTRYPOINT}" in execs[1]["argv"]
    )
    return {
        "pass": exec_ok,
        "expected": [
            f"{mount}/{LAUNCHER}",
            f"{SYSTEM_LOADER} {' '.join(want_loader_args)} ... {mount}/{ENTRYPOINT}",
        ],
        "execs": execs,
    }


def cmd_runtime(args: argparse.Namespace) -> int:
    n_logs, execs, opens = strace_records(args.strace_prefix)
    mount = args.tree_mount.rstrip("/")
    checks: dict[str, Any] = {}
    checks["g2_2_g2_4_exec_chain"] = exec_chain(execs, mount)
    forbidden = sorted(
        {o["path"] for o in opens if g2._FORBIDDEN_OPEN.search(o["path"])}
    )
    checks["g2_2_g2_4_no_python_triton_open"] = {
        "pass": not forbidden,
        "attempted": forbidden,
    }

    tree = args.tree.resolve()
    set_ = _load(args.third_party_set)
    vendor_shas = {e["sha256"] for e in set_["entries"]}
    bundled = {e["soname"] for e in set_["entries"]}
    att = _load(tree / g2.MODEL_ROOT / g2.ATTESTATION)
    op_sha = att["op_library"]["sha256"]
    auditor_sha = g2.sha256_file(tree / AUDITOR)
    opened: dict[str, str] = {}
    foreign: list[str] = []
    for o in opens:
        p = o["path"]
        if o["rc"] < 0 or not g2._SHARED_OBJECT.search(os.path.basename(p)):
            continue
        norm = os.path.normpath(p)
        if norm.startswith(mount + "/"):
            opened[norm] = g2.sha256_file(tree / norm[len(mount) + 1 :])
            if (
                os.path.basename(norm) in bundled
                and os.path.dirname(norm) != f"{mount}/{VENDOR}"
            ):
                foreign.append(norm)
        elif (
            os.path.basename(norm) in bundled
            or g2.classify_name(os.path.basename(norm), norm) == "third_party"
        ):
            foreign.append(norm)
    cur = set(opened.values())
    want = vendor_shas | {op_sha, auditor_sha}
    checks["opened_set_is_the_bundle"] = {
        "pass": cur == want and not foreign,
        "missing": sorted(want - cur),
        "extra": sorted(cur - want),
        "foreign_opens": sorted(set(foreign)),
    }
    report = {
        "schema": SCHEMA,
        "kind": "runtime",
        "logs": n_logs,
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
    }
    g2._write(args.report, report)
    for name, c in checks.items():
        print(f"{name}: {'PASS' if c['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


def cmd_rejected(args: argparse.Namespace) -> int:
    n_logs, execs, opens = strace_records(args.strace_prefix)
    mount = args.tree_mount.rstrip("/")
    log = args.log.read_text(errors="replace").splitlines()
    want = f"saccade_track: unknown argument {args.option}"
    checks: dict[str, Any] = {}
    checks["exit_2_unknown_argument"] = {
        "pass": bool(log) and log[-1] == "exit=2" and want in log,
        "expected": [want, "exit=2"],
        "log": log[-5:],
    }
    checks["exec_chain"] = exec_chain(execs, mount)
    model_root = f"{mount}/{g2.MODEL_ROOT}/"
    touched = sorted(
        {
            o["path"]
            for o in opens
            if os.path.normpath(o["path"]).startswith(model_root)
            or os.path.basename(o["path"]) == "libsaccade_scan_torchop.so"
        }
    )
    checks["no_model_root_open"] = {"pass": not touched, "attempted": touched}
    devices = sorted({o["path"] for o in opens if _GPU_DEVICE.match(o["path"])})
    checks["no_gpu_device_open"] = {"pass": not devices, "attempted": devices}
    outputs = [
        str(p)
        for p in (args.out_dir / "native", args.out_dir / "track_report.json")
        if p.exists()
    ]
    checks["no_output"] = {"pass": not outputs, "found": outputs}
    report = {
        "schema": SCHEMA,
        "kind": "rejected",
        "option": args.option,
        "logs": n_logs,
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
    }
    g2._write(args.report, report)
    for name, c in checks.items():
        print(f"{name}: {'PASS' if c['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


def init_paths(text: str) -> list[str]:
    """Objects the loader initialized ('calling init', any namespace) once the
    auditor is loaded. The launcher's sh and the loader it execs share a pid,
    so one log file holds both processes; the auditor is the first object the
    loader maps, so what precedes it is the sh's."""
    out: list[str] = []
    started = False
    for line in text.splitlines():
        if not started:
            started = os.path.basename(AUDITOR) in line
            if not started:
                continue
        m = g2._LD_INIT.match(line)
        if m:
            out.append(m.group(1))
    return out


def cmd_sources(args: argparse.Namespace) -> int:
    """Where a run's loaded objects came from (its ``LD_DEBUG=files`` logs):
    every bundled name from the tree's lib/vendor with the pinned bytes, the
    operator library and the auditor from the tree, nothing else from the
    tree, no third-party object from outside it, no Python library."""
    tree = args.tree.resolve()
    set_ = _load(args.third_party_set)
    pinned = {e["soname"]: e["sha256"] for e in set_["entries"]}
    att = _load(tree / g2.MODEL_ROOT / g2.ATTESTATION)
    op_real = str((tree / g2.MODEL_ROOT / att["op_library"]["path"]).resolve())
    auditor = str((tree / AUDITOR).resolve())
    vendor = str((tree / VENDOR).resolve())
    logs = sorted(args.log_prefix.parent.glob(args.log_prefix.name + ".*"))
    if not logs:
        raise g2.CheckError(f"no LD_DEBUG logs {args.log_prefix}.*")
    reals = sorted(
        {
            os.path.realpath(p)
            for log in logs
            for p in init_paths(log.read_text(errors="replace"))
        }
    )
    problems: list[str] = []
    seen: set[str] = set()
    for real in reals:
        base = os.path.basename(real)
        if g2._PYTHON_LIB.match(base):
            problems.append(f"Python library loaded: {real}")
        if base in pinned:
            seen.add(base)
            if os.path.dirname(real) != vendor:
                problems.append(f"{base} loaded from {real}, not {vendor}")
            elif g2.sha256_file(Path(real)) != pinned[base]:
                problems.append(f"{real}: sha256 is not the pinned one")
        elif real.startswith(str(tree) + "/"):
            if real not in (op_real, auditor):
                problems.append(f"unexpected tree object loaded: {real}")
        elif (
            g2.classify_name(g2.dynamic_section(Path(real))["soname"] or base, real)
            == "third_party"
        ):
            problems.append(f"third-party object from outside the tree: {real}")
    for must in (op_real, auditor):
        if must not in reals:
            problems.append(f"not loaded: {must}")
    missing = sorted(set(pinned) - seen)
    if missing:
        problems.append(f"bundled objects not loaded: {missing}")
    report = {
        "schema": SCHEMA,
        "kind": "sources",
        "logs": [str(p) for p in logs],
        "loaded": reals,
        "problems": problems,
        "pass": not problems,
    }
    g2._write(args.report, report)
    print(f"sources: {'PASS' if not problems else 'FAIL'} {problems}")
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--tree", type=Path, required=True)
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--third-party-set", type=Path, default=THIRD_PARTY_SET)
    s.add_argument("--entrypoint-pin", type=Path, default=ENTRYPOINT_PIN)
    s.add_argument("--launcher-source", type=Path, default=LAUNCHER_SOURCE)
    s.add_argument("--measurement-surface", type=Path, default=MEASUREMENT_SURFACE)
    r = sub.add_parser("runtime")
    r.add_argument("--strace-prefix", type=Path, required=True)
    r.add_argument("--tree", type=Path, required=True)
    r.add_argument("--tree-mount", required=True)
    r.add_argument("--report", type=Path, required=True)
    r.add_argument("--third-party-set", type=Path, default=THIRD_PARTY_SET)
    j = sub.add_parser("rejected")
    j.add_argument("--strace-prefix", type=Path, required=True)
    j.add_argument("--log", type=Path, required=True)
    j.add_argument("--out-dir", type=Path, required=True)
    j.add_argument("--option", required=True)
    j.add_argument("--tree-mount", required=True)
    j.add_argument("--report", type=Path, required=True)
    o = sub.add_parser("sources")
    o.add_argument("--log-prefix", type=Path, required=True)
    o.add_argument("--tree", type=Path, required=True)
    o.add_argument("--report", type=Path, required=True)
    o.add_argument("--third-party-set", type=Path, default=THIRD_PARTY_SET)
    args = ap.parse_args(argv)
    try:
        return {
            "static": cmd_static,
            "runtime": cmd_runtime,
            "rejected": cmd_rejected,
            "sources": cmd_sources,
        }[args.cmd](args)
    except g2.CheckError as exc:
        print(f"check_shipping_bundle: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
