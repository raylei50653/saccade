"""The bundled shipping tree's checks, launcher, auditor and install (#465 PR-C1).

docs/reference/native_runtime_resolved_config.md §17. The real checks need an
installed tree, a GPU run and a container; these pin the logic in the ordinary
pytest job:

* ``shipping/third_party_set.json`` and ``shipping/entrypoint_pin.json`` are
  well formed, and ``scripts/native/export_third_party_set.py`` maps objects to
  wheels and refuses other bytes;
* ``scripts/native/check_shipping_bundle.py``: the expected file set, search
  -path containment with its two enumerated exceptions, the NEEDED closure,
  and the runtime verdict (the launcher -> loader exec chain, no Python opens,
  the opened set is the bundle) on synthetic strace logs;
* the launcher passes ``--library-path lib/vendor`` and ``--audit``;
* ``shipping/src/loader_audit.c`` on a real loader: a DT_RPATH candidate is
  searched before ``--library-path`` (so a planted copy is loaded without the
  auditor) and the auditor exits 127 instead (skips without ``cc``);
* ``shipping/cmake/install_third_party.cmake`` copies the pinned objects and
  licenses and refuses one whose sha256 is not pinned (skips without ``cmake``).

PR-C2 (§18): the entrypoint's measurement-surface check (``static``, and the
``check_no_measurement_surface.cmake`` POST_BUILD step) sees one forbidden byte
string, and ``rejected`` (a launcher run given a removed option) fails on a
wrong exit or message, a model-root or operator-library open, a GPU device
open, or any output.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    path = REPO / "scripts" / "native" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bundle = _load("check_shipping_bundle")
export = _load("export_third_party_set")
audit = _load("license_audit")
SET = json.loads((REPO / "shipping" / "third_party_set.json").read_text())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── committed pins ─────────────────────────────────────────────────────────────


def test_third_party_set_is_well_formed() -> None:
    assert SET["schema"] == "saccade.shipping_third_party/v1"
    sonames = [e["soname"] for e in SET["entries"]]
    assert len(sonames) == 27 and len(set(sonames)) == 27
    for e in SET["entries"]:
        assert len(e["sha256"]) == 64 and e["bytes"] > 0
        assert e["source"]["root"] in ("purelib", "nvjpeg_wheel")
        assert e["license_files"], e["soname"]
        assert not bundle.g2._PYTHON_LIB.match(e["soname"])
        assert bundle.g2.classify_name(e["soname"]) == "third_party"


def test_entrypoint_pin_is_well_formed() -> None:
    pin = json.loads((REPO / "shipping" / "entrypoint_pin.json").read_text())
    assert pin["schema"] == "saccade.shipping_entrypoint_pin/v1"
    assert len(pin["sha256"]) == 64 and pin["bytes"] > 0


def test_notice_is_the_audit_rendering() -> None:
    audit_doc = json.loads((REPO / "shipping" / "license_audit.json").read_text())
    assert (REPO / "shipping" / "THIRD_PARTY.md").read_text() == audit.render(
        SET, audit_doc
    )


def test_export_maps_objects_to_wheels_and_refuses_other_bytes(tmp_path: Path) -> None:
    purelib, nvj = tmp_path / "site", tmp_path / "nvj"
    (purelib / "pkg/lib").mkdir(parents=True)
    (purelib / "pkg-1.0.dist-info/licenses").mkdir(parents=True)
    (purelib / "pkg/lib/libx.so.1").write_bytes(b"x")
    (purelib / "pkg-1.0.dist-info/licenses/LICENSE").write_text("terms")
    (purelib / "pkg-1.0.dist-info/RECORD").write_text(
        "pkg/lib/libx.so.1,,\npkg-1.0.dist-info/licenses/LICENSE,,\n"
    )
    nvj.mkdir()
    deps = {
        "loaded": "ld.json",
        "entries": [
            {
                "names": ["libx.so.1"],
                "soname": "libx.so.1",
                "source": str(purelib / "pkg/lib/libx.so.1"),
                "sha256": _sha(b"x"),
                "bytes": 1,
            }
        ],
    }
    out = export.export(deps, {"purelib": purelib, "nvjpeg_wheel": nvj})
    e = out["entries"][0]
    assert e["source"] == {"root": "purelib", "path": "pkg/lib/libx.so.1"}
    assert e["wheel"] == "pkg-1.0"
    assert e["license_files"] == [
        {"path": "pkg-1.0.dist-info/licenses/LICENSE", "sha256": _sha(b"terms")}
    ]
    deps["entries"][0]["sha256"] = _sha(b"y")
    with pytest.raises(export.ExportError, match="not the run's"):
        export.export(deps, {"purelib": purelib, "nvjpeg_wheel": nvj})


# ── static ─────────────────────────────────────────────────────────────────────


def test_expected_files_cover_the_bundle() -> None:
    files = bundle.expected_files(SET)
    assert {
        "bin/saccade_track",
        "libexec/saccade_track",
        "lib/saccade_loader_audit.so",
    } <= files
    assert sum(f.startswith("lib/vendor/") for f in files) == 27
    assert "share/saccade/build/libsaccade_scan_torchop.so" in files
    assert "licenses/THIRD_PARTY.md" in files


def _elf(
    rpath: list[str] | None = None, runpath: list[str] | None = None, sha: str = "s"
) -> dict[str, Any]:
    return {
        "sha256": sha,
        "dyn": {
            "needed": [],
            "soname": None,
            "rpath": rpath or [],
            "runpath": runpath or [],
        },
    }


def test_containment_vendor_layout_keeps_torch_rpath_inside(tmp_path: Path) -> None:
    rp = ["$ORIGIN/../../nvidia/cu13/lib", "$ORIGIN"]
    ok, _ = bundle.containment(
        tmp_path, {"lib/vendor/libtorch_cuda.so": _elf(rpath=rp)}, "op", "o", "c"
    )
    assert ok == []
    flat, _ = bundle.containment(
        tmp_path, {"lib/libtorch_cuda.so": _elf(rpath=rp)}, "op", "o", "c"
    )
    assert len(flat) == 1 and "outside the tree" in flat[0]


def test_containment_enumerated_exceptions_are_bound_to_bytes(tmp_path: Path) -> None:
    op = "share/saccade/build/libsaccade_scan_torchop.so"
    elves = {
        op: _elf(runpath=["/home/x/build"], sha="opsha"),
        "lib/vendor/libcusparseLt.so.0": _elf(runpath=["$ORIGIN", ""], sha="csha"),
    }
    problems, exc = bundle.containment(tmp_path, elves, op, "opsha", "csha")
    assert problems == [] and len(exc) == 2
    problems, _ = bundle.containment(tmp_path, elves, op, "other", "other")
    assert len(problems) == 2
    problems, _ = bundle.containment(
        tmp_path, {"lib/vendor/libz.so": _elf(rpath=["/usr/lib"])}, op, "opsha", "csha"
    )
    assert problems == [
        "lib/vendor/libz.so: RPATH entry '/usr/lib' is not $ORIGIN-relative"
    ]


def test_closure_needs_vendor_and_flags_python(tmp_path: Path) -> None:
    needed = {
        "libexec/saccade_track": ["libtorch.so", "libc.so.6", "libcuda.so.1"],
        "lib/vendor/libtorch.so": ["libmissing.so.1", "libpython3.12.so.1.0"],
    }

    def run(cmd: list[str]) -> str:
        rel = str(Path(cmd[-1]).relative_to(tmp_path))
        return "".join(f" 0x1 (NEEDED) Shared library: [{n}]\n" for n in needed[rel])

    (tmp_path / "lib/vendor").mkdir(parents=True)
    (tmp_path / "lib/vendor/libtorch.so").write_bytes(b"")
    clo, problems = bundle.bundle_closure(
        tmp_path, ["libexec/saccade_track"], {"libtorch.so"}, run
    )
    assert {c["name"]: c["class"] for c in clo}["libtorch.so"] == "vendor"
    assert any("libmissing.so.1 is not in lib/vendor" in p for p in problems)
    assert any("libpython3.12" in p for p in problems)


def test_closure_reports_a_missing_vendor_object(tmp_path: Path) -> None:
    def run(cmd: list[str]) -> str:
        return " 0x1 (NEEDED) Shared library: [libnvrtc.so.13]\n"

    _, problems = bundle.bundle_closure(tmp_path, ["op.so"], {"libnvrtc.so.13"}, run)
    assert problems == ["op.so: NEEDED libnvrtc.so.13 is missing from lib/vendor"]


# ── runtime ────────────────────────────────────────────────────────────────────

# The entrypoint arguments, as the formal runs name them (--model-root given).
_ARGS = '"--config", "/cfg/c.json", "--model-root", "/opt/saccade/share/saccade"'
LOADER = (
    'execve("/lib64/ld-linux-x86-64.so.2", ["/lib64/ld-linux-x86-64.so.2", "--library-path", '
    '"/opt/saccade/lib/vendor", "--audit", "/opt/saccade/lib/saccade_loader_audit.so", "--argv0", '
    f'"/opt/saccade/bin/saccade_track", "/opt/saccade/libexec/saccade_track", {_ARGS}], 0x0 /* 3 vars */) = 0'
)
LAUNCHER = f'execve("/opt/saccade/bin/saccade_track", ["/opt/saccade/bin/saccade_track", {_ARGS}], 0x0 /* 3 vars */) = 0'
# The auditor readiness probe (A2), exec'd by the launcher's command
# substitution: a child process, so strace -ff logs it in a file of its own.
PROBE = (
    'execve("/lib64/ld-linux-x86-64.so.2", ["/lib64/ld-linux-x86-64.so.2", "--library-path", '
    '"/opt/saccade/lib/vendor", "--audit", "/opt/saccade/lib/saccade_loader_audit.so", '
    '"/bin/sh", "-c", ":"], 0x0 /* 4 vars */) = 0'
)


def _yy(line: str) -> str:
    """A plain absolute-path open as strace -yy records it: the cwd on
    AT_FDCWD and, when it succeeded, the opened file on the returned fd."""
    m = re.match(r'^openat\(AT_FDCWD, "([^"]*)"(.*)\) = (\d+)$', line)
    if m:
        return f'openat(AT_FDCWD</>, "{m[1]}"{m[2]}) = {m[3]}<{os.path.normpath(m[1])}>'
    return line.replace("openat(AT_FDCWD, ", "openat(AT_FDCWD</>, ", 1)


def _runtime_tree(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    tree = tmp_path / "tree"
    for d in ("lib/vendor", "share/saccade/configs/shipping", "share/saccade/build"):
        (tree / d).mkdir(parents=True, exist_ok=True)
    (tree / "lib/vendor/libx.so.1").write_bytes(b"x")
    (tree / "lib/saccade_loader_audit.so").write_bytes(b"audit")
    (tree / "share/saccade/build/libop.so").write_bytes(b"op")
    (
        tree / "share/saccade/configs/shipping/mamba_head_realization.attestation.json"
    ).write_text(
        json.dumps({"op_library": {"path": "build/libop.so", "sha256": _sha(b"op")}})
    )
    set_path = tmp_path / "set.json"
    set_path.write_text(
        json.dumps({"entries": [{"soname": "libx.so.1", "sha256": _sha(b"x")}]})
    )
    opens = [
        'openat(AT_FDCWD, "/opt/saccade/lib/saccade_loader_audit.so", O_RDONLY|O_CLOEXEC) = 3',
        'openat(AT_FDCWD, "/opt/saccade/lib/vendor/../../nvidia/cu13/lib/libx.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT',
        'openat(AT_FDCWD, "/opt/saccade/lib/vendor/libx.so.1", O_RDONLY|O_CLOEXEC) = 3',
        'openat(AT_FDCWD, "/opt/saccade/share/saccade/build/libop.so", O_RDONLY|O_CLOEXEC) = 3',
        'openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3',
    ]
    return tree, set_path, [_yy(x) for x in opens]


def _write_strace(st: Path, lines: list[str], probe: list[str] | None) -> None:
    """``lines`` as the launcher process's log, ``probe`` as its child's."""
    st.mkdir(exist_ok=True)
    (st / "s.1").write_text("\n".join(lines) + "\n")
    if probe:
        (st / "s.2").write_text("\n".join(probe) + "\n")


def _runtime(
    tmp_path: Path,
    lines: list[str],
    probe: list[str] | None = None,
    plain: bool = False,
) -> dict[str, Any]:
    tree, set_path, _ = _runtime_tree(tmp_path)
    _write_strace(tmp_path / "st", lines, [PROBE] if probe is None else probe)
    report = tmp_path / "runtime.json"
    bundle.main(
        [
            "runtime",
            "--strace-prefix",
            str(tmp_path / "st" / "s"),
            "--tree",
            str(tree),
            "--tree-mount",
            "/opt/saccade",
            "--report",
            str(report),
            "--third-party-set",
            str(set_path),
            *(["--legacy-plain-trace"] if plain else []),
        ]
    )
    return json.loads(report.read_text())


def test_runtime_pass(tmp_path: Path) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    r = _runtime(tmp_path, [LAUNCHER, LOADER, *opens])
    assert r["pass"], r["checks"]


@pytest.mark.parametrize(
    ("mutate", "check"),
    [
        (
            lambda ls: ['execve("/bin/sh", ["/bin/sh", "-c", "x"], 0x0) = 0', *ls],
            "g2_2_g2_4_exec_chain",
        ),
        (
            lambda ls: [
                ls[0],
                ls[1].replace(
                    '"--audit", "/opt/saccade/lib/saccade_loader_audit.so", ', ""
                ),
                *ls[2:],
            ],
            "g2_2_g2_4_exec_chain",
        ),
        (
            lambda ls: [
                *ls,
                'openat(AT_FDCWD, "/opt/saccade/nvidia/cu13/lib/libx.so.1", O_RDONLY) = 3',
            ],
            "opened_set_is_the_bundle",
        ),
        (
            lambda ls: [
                *ls,
                'openat(AT_FDCWD, "/usr/lib/libpython3.12.so.1.0", O_RDONLY) = -1 ENOENT',
            ],
            "g2_2_g2_4_no_python_triton_open",
        ),
        (lambda ls: [x for x in ls if "libop.so" not in x], "opened_set_is_the_bundle"),
    ],
)
def test_runtime_rejects(tmp_path: Path, mutate: Any, check: str) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    if "nvidia/cu13" in str(
        mutate([LAUNCHER, LOADER])
    ):  # the planted copy must exist to be hashed
        (tmp_path / "tree/nvidia/cu13/lib").mkdir(parents=True)
        (tmp_path / "tree/nvidia/cu13/lib/libx.so.1").write_bytes(b"y")
    r = _runtime(tmp_path, mutate([LAUNCHER, LOADER, *opens]))
    assert not r["checks"][check]["pass"]


_PLAIN_OPENS = [
    'openat(AT_FDCWD, "/opt/saccade/lib/saccade_loader_audit.so", O_RDONLY|O_CLOEXEC) = 3',
    'openat(AT_FDCWD, "/opt/saccade/lib/vendor/libx.so.1", O_RDONLY|O_CLOEXEC) = 3',
    'openat(AT_FDCWD, "/opt/saccade/share/saccade/build/libop.so", O_RDONLY|O_CLOEXEC) = 3',
]


def test_runtime_needs_fd_targets_unless_legacy(tmp_path: Path) -> None:
    _runtime_tree(tmp_path)
    r = _runtime(tmp_path, [LAUNCHER, LOADER, *_PLAIN_OPENS])
    assert not r["pass"] and not r["legacy_plain_trace"]
    for check in ("g2_2_g2_4_no_python_triton_open", "opened_set_is_the_bundle"):
        assert r["checks"][check]["unresolved"] == _PLAIN_OPENS
    legacy = _runtime(tmp_path, [LAUNCHER, LOADER, *_PLAIN_OPENS], plain=True)
    assert legacy["pass"] and legacy["legacy_plain_trace"], legacy["checks"]


@pytest.mark.parametrize(
    "line",
    [
        'openat(7, "payload", O_RDONLY) = 9',
        'openat(AT_FDCWD, "payload", O_RDONLY) = 9',
        'openat(7, "x.so", O_RDONLY <unfinished ...>',
        'openat(AT_FDCWD</>, "/tmp/payload", O_RDONLY) = 9',
    ],
)
def test_runtime_fails_closed_on_unresolved_open(tmp_path: Path, line: str) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    for plain in (False, True):
        r = _runtime(tmp_path, [LAUNCHER, LOADER, *opens, line], plain=plain)
        assert not r["pass"]
        for check in ("g2_2_g2_4_no_python_triton_open", "opened_set_is_the_bundle"):
            assert not r["checks"][check]["pass"]
            assert r["checks"][check]["unresolved"] == [line]


@pytest.mark.parametrize(
    ("line", "ok"),
    [
        # a SONAME symlink to its versioned file: still the base system
        (
            'openat(AT_FDCWD</>, "/usr/lib/x86_64-linux-gnu/libstdc++.so.6", '
            "O_RDONLY|O_CLOEXEC) = 3</usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33>",
            True,
        ),
        # an alias named like the base system, to a third-party object
        (
            'openat(AT_FDCWD</>, "/usr/lib/x86_64-linux-gnu/libstdc++.so.6", '
            "O_RDONLY|O_CLOEXEC) = 3</usr/lib/libnccl.so.2.21.5>",
            False,
        ),
        (
            'openat(AT_FDCWD</>, "/tmp/payload.so", O_RDONLY|O_CLOEXEC) = '
            "3</usr/lib/x86_64-linux-gnu/libz.so.1.3>",
            False,  # the requested name is not the base system's
        ),
    ],
)
def test_runtime_classifies_fd_targets_by_soname_family(
    tmp_path: Path, line: str, ok: bool
) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    r = _runtime(tmp_path, [LAUNCHER, LOADER, *opens, line])
    assert r["checks"]["opened_set_is_the_bundle"]["pass"] is ok, r["checks"]


def test_runtime_accepts_a_non_path_fd_target(tmp_path: Path) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    line = 'openat(AT_FDCWD</>, "/proc/self/fd/0", O_RDONLY) = 9<pipe:[123]>'
    r = _runtime(tmp_path, [LAUNCHER, LOADER, *opens, line])
    assert r["pass"], r["checks"]


_FAILED_PROBE = PROBE.removesuffix(" = 0") + " = -1 ENOENT"


@pytest.mark.parametrize(
    ("lines", "probe"),
    [
        # the PR-C1 form: no readiness probe
        ([LAUNCHER, LOADER], []),
        # the probe in the launcher's own process, not a child
        ([LAUNCHER, PROBE, LOADER], []),
        ([LAUNCHER, LOADER], [PROBE, PROBE]),
        ([LAUNCHER, LOADER], [_FAILED_PROBE]),
        (
            [LAUNCHER, LOADER],
            [
                PROBE.replace(
                    '"--audit", "/opt/saccade/lib/saccade_loader_audit.so", ', ""
                )
            ],
        ),
        ([LAUNCHER, LOADER], [PROBE.replace('"-c", ":"', '"-c", "id"')]),
        ([LOADER, LAUNCHER], [PROBE]),
    ],
)
def test_runtime_exec_chain_needs_the_readiness_probe(
    tmp_path: Path, lines: list[str], probe: list[str]
) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    r = _runtime(tmp_path, [*lines, *opens], probe)
    assert not r["checks"]["g2_2_g2_4_exec_chain"]["pass"]


_LOADER_PREFIX = (
    'execve("/lib64/ld-linux-x86-64.so.2", ["/lib64/ld-linux-x86-64.so.2", "--library-path", '
    '"/opt/saccade/lib/vendor", "--audit", "/opt/saccade/lib/saccade_loader_audit.so", '
)


@pytest.mark.parametrize(
    "loader",
    [
        # the entrypoint named, but /bin/true in the program slot (A4)
        _LOADER_PREFIX
        + '"--argv0", "/opt/saccade/libexec/saccade_track", "/bin/true"], 0x0) = 0',
        # the entrypoint after the program
        _LOADER_PREFIX + '"--argv0", "/opt/saccade/bin/saccade_track", "/bin/true", '
        f'"/opt/saccade/libexec/saccade_track", {_ARGS}], 0x0) = 0',
        # arguments the launcher was not given
        LOADER.replace(_ARGS, f'{_ARGS}, "--trace", "/tmp/t"'),
        # arguments the launcher was given, dropped
        LOADER.replace(_ARGS, '"--config", "/cfg/c.json"'),
        # another argv0
        LOADER.replace('"--argv0", "/opt/saccade/bin/saccade_track"', '"--argv0", "x"'),
        # an extra loader option
        LOADER.replace('"--argv0"', '"--preload", "/tmp/p.so", "--argv0"'),
    ],
)
def test_runtime_exec_chain_checks_the_whole_loader_argv(
    tmp_path: Path, loader: str
) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    r = _runtime(tmp_path, [LAUNCHER, loader, *opens])
    assert not r["pass"] and not r["checks"]["g2_2_g2_4_exec_chain"]["pass"]


@pytest.mark.parametrize(
    "extra",
    [
        'execve("/usr/bin/python3", ["/usr/bin/python3"], 0x0 /* 3 vars */ <unfinished ...>',
        'execve("/usr/bin/python3", ["/usr/bin/python3"], 0x0 /* 3 vars */',
        'execveat(3, "", ["x"], 0x0, AT_EMPTY_PATH <unfinished ...>',
    ],
)
def test_runtime_fails_closed_on_an_incomplete_exec(tmp_path: Path, extra: str) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    r = _runtime(tmp_path, [LAUNCHER, LOADER, *opens, extra])
    c = r["checks"]["g2_2_g2_4_exec_chain"]
    assert not r["pass"] and not c["pass"]
    assert [e["raw"] for e in c["execs"] if e.get("incomplete")] == [extra]


# ── measurement surface (PR-C2) ────────────────────────────────────────────────

SURFACE = REPO / "shipping" / "measurement_surface.json"


@pytest.mark.parametrize(
    ("line", "check", "target"),
    [
        (
            'openat(AT_FDCWD</>, "/tmp/lib-alias", O_RDONLY) = '
            "3</usr/lib/libpython3.12.so.1.0>",
            "g2_2_g2_4_no_python_triton_open",
            "/usr/lib/libpython3.12.so.1.0",
        ),
        (
            'openat(AT_FDCWD</>, "/tmp/lib-alias", O_RDONLY|O_CLOEXEC) = '
            "3</opt/saccade/nvidia/cu13/lib/libx.so.1>",
            "opened_set_is_the_bundle",
            "/opt/saccade/nvidia/cu13/lib/libx.so.1",
        ),
    ],
)
def test_runtime_checks_the_opened_target_of_an_alias(
    tmp_path: Path, line: str, check: str, target: str
) -> None:
    _, _, opens = _runtime_tree(tmp_path)
    r = _runtime(tmp_path, [LAUNCHER, LOADER, *opens, line])
    assert not r["pass"] and not r["checks"][check]["pass"]
    c = r["checks"][check]
    assert target in c.get("attempted", c.get("foreign_opens", []))


def test_measurement_surface_sees_one_token(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.write_bytes(b"\x7fELF\0saccade_track: unknown argument \0double_buffer\0")
    assert bundle.measurement_surface(clean, SURFACE)["pass"]
    assert not bundle.measurement_surface(tmp_path / "missing", SURFACE)["pass"]
    for token in json.loads(SURFACE.read_text())["forbidden"]:
        dirty = tmp_path / "dirty"
        dirty.write_bytes(clean.read_bytes() + token.encode() + b"\0")
        r = bundle.measurement_surface(dirty, SURFACE)
        assert not r["pass"] and token in r["found"]


@pytest.mark.skipif(not shutil.which("cmake"), reason="needs cmake")
def test_post_build_surface_check(tmp_path: Path) -> None:
    def check(data: bytes) -> int:
        b = tmp_path / "bin"
        b.write_bytes(data)
        return subprocess.run(
            [
                "cmake",
                f"-DBINARY={b}",
                f"-DSURFACE={SURFACE}",
                "-P",
                str(REPO / "shipping/cmake/check_no_measurement_surface.cmake"),
            ],
            capture_output=True,
        ).returncode

    assert check(b"\x7fELF\0saccade_track: unknown argument \0") == 0
    assert check(b"\x7fELF\0unknown double-buffer mutation \0") != 0
    assert check(b"\x7fELF\0--max-frames\0") != 0


def _rejected(
    tmp_path: Path,
    lines: list[str],
    log: list[str],
    outputs: bool = False,
    probe: list[str] | None = None,
    plain: bool = False,
) -> dict[str, Any]:
    st = tmp_path / "st"
    _write_strace(st, lines, [PROBE] if probe is None else probe)
    (tmp_path / "saccade_track.log").write_text("\n".join(log) + "\n")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    if outputs:
        (out / "native").mkdir()
    report = tmp_path / "rejected.json"
    bundle.main(
        [
            "rejected",
            "--strace-prefix",
            str(st / "s"),
            "--log",
            str(tmp_path / "saccade_track.log"),
            "--out-dir",
            str(out),
            "--option=--measurement-mutation",
            "--tree-mount",
            "/opt/saccade",
            "--report",
            str(report),
            *(["--legacy-plain-trace"] if plain else []),
        ]
    )
    return json.loads(report.read_text())


_REJECTED_LOG = ["saccade_track: unknown argument --measurement-mutation", "exit=2"]
_REJECTED_PLAIN = [
    'openat(AT_FDCWD, "/opt/saccade/lib/saccade_loader_audit.so", O_RDONLY|O_CLOEXEC) = 3',
    'openat(AT_FDCWD, "/opt/saccade/lib/vendor/libx.so.1", O_RDONLY|O_CLOEXEC) = 3',
    'openat(AT_FDCWD, "/usr/lib/wsl/lib/libcuda.so.1", O_RDONLY|O_CLOEXEC) = 3',
]
_REJECTED_OPENS = [_yy(x) for x in _REJECTED_PLAIN]


def test_rejected_pass(tmp_path: Path) -> None:
    r = _rejected(tmp_path, [LAUNCHER, LOADER, *_REJECTED_OPENS], _REJECTED_LOG)
    assert r["pass"], r["checks"]


@pytest.mark.parametrize(
    ("extra", "log", "outputs", "check"),
    [
        (
            [],
            ["saccade_track: unknown argument --schedule", "exit=2"],
            False,
            "exit_2_unknown_argument",
        ),
        ([], [_REJECTED_LOG[0], "exit=0"], False, "exit_2_unknown_argument"),
        (
            [
                'openat(AT_FDCWD, "/opt/saccade/share/saccade/configs/shipping/x.json", O_RDONLY) = 3'
            ],
            _REJECTED_LOG,
            False,
            "no_model_root_open",
        ),
        (
            [
                'openat(AT_FDCWD, "/opt/saccade/share/saccade/models/yolo/x.engine", O_RDONLY) = -1 ENOENT'
            ],
            _REJECTED_LOG,
            False,
            "no_model_root_open",
        ),
        (
            ['openat(AT_FDCWD, "/elsewhere/libsaccade_scan_torchop.so", O_RDONLY) = 3'],
            _REJECTED_LOG,
            False,
            "no_model_root_open",
        ),
        (
            ['openat(AT_FDCWD, "/dev/dxg", O_RDWR) = 4'],
            _REJECTED_LOG,
            False,
            "no_gpu_device_open",
        ),
        (
            ['openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR) = -1 ENOENT'],
            _REJECTED_LOG,
            False,
            "no_gpu_device_open",
        ),
        ([], _REJECTED_LOG, True, "no_output"),
    ],
)
def test_rejected_rejects(
    tmp_path: Path, extra: list[str], log: list[str], outputs: bool, check: str
) -> None:
    r = _rejected(tmp_path, [LAUNCHER, LOADER, *_REJECTED_OPENS, *extra], log, outputs)
    assert not r["pass"] and not r["checks"][check]["pass"]


@pytest.mark.parametrize(
    ("line", "check"),
    [
        (
            'openat(AT_FDCWD, "/opt/saccade/share/saccade", O_RDONLY|O_DIRECTORY) = 7',
            "no_model_root_open",
        ),
        (
            'openat(7</opt/saccade/share/saccade>, "configs/shipping/x.json", O_RDONLY) = 8',
            "no_model_root_open",
        ),
        (
            'openat(AT_FDCWD</opt/saccade>, "share/saccade/models/x.engine", O_RDONLY) = -1 ENOENT',
            "no_model_root_open",
        ),
        ('openat(7</dev>, "dxg", O_RDWR) = 8', "no_gpu_device_open"),
        (
            'openat(AT_FDCWD, "/dev/../dev/nvidiactl", O_RDWR) = -1 ENOENT',
            "no_gpu_device_open",
        ),
        (
            'openat(AT_FDCWD, "/out/trace/MOT17-05-SDP/detector.bin", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 8',
            "no_output",
        ),
        (
            'openat(7</out/trace>, "detector.bin", O_WRONLY|O_CREAT, 0666) = -1 EACCES',
            "no_output",
        ),
    ],
)
def test_rejected_resolves_paths_and_detects_write_attempts(
    tmp_path: Path, line: str, check: str
) -> None:
    # No output survives on disk: even a failed write must be detected.
    r = _rejected(tmp_path, [LAUNCHER, LOADER, *_REJECTED_OPENS, line], _REJECTED_LOG)
    assert not r["pass"] and not r["checks"][check]["pass"]
    assert r["checks"][check]["attempted"]


@pytest.mark.parametrize(
    "line",
    [
        'openat(7, "configs/shipping/x.json", O_RDONLY) = 8',
        'openat(AT_FDCWD, "share/saccade/configs/shipping/x.json", O_RDONLY) = 8',
        'open("relative.engine", O_RDONLY) = -1 ENOENT',
        'openat(7, "x.engine", O_RDONLY <unfinished ...>',
    ],
)
def test_rejected_fails_closed_on_unresolved_open(tmp_path: Path, line: str) -> None:
    r = _rejected(tmp_path, [LAUNCHER, LOADER, *_REJECTED_OPENS, line], _REJECTED_LOG)
    assert not r["pass"]
    for check in ("no_model_root_open", "no_gpu_device_open", "no_output"):
        assert not r["checks"][check]["pass"]
        assert r["checks"][check]["unresolved"] == [line]


def test_rejected_allows_resolved_unrelated_read(tmp_path: Path) -> None:
    line = 'openat(AT_FDCWD</>, "etc/ld.so.cache", O_RDONLY) = 3</etc/ld.so.cache>'
    r = _rejected(tmp_path, [LAUNCHER, LOADER, *_REJECTED_OPENS, line], _REJECTED_LOG)
    assert r["pass"]


@pytest.mark.parametrize(
    "name", ["trace/MOT17-05-SDP/detector.bin", "track_report.json"]
)
def test_rejected_detects_surviving_output(tmp_path: Path, name: str) -> None:
    output = tmp_path / "out" / name
    output.parent.mkdir(parents=True)
    output.write_bytes(b"output")
    r = _rejected(tmp_path, [LAUNCHER, LOADER, *_REJECTED_OPENS], _REJECTED_LOG)
    assert not r["pass"] and not r["checks"]["no_output"]["pass"]


def test_rejected_checks_output_paths_from_argv(tmp_path: Path) -> None:
    loader = _with_args("--trace", "/out/custom_trace")
    line = (
        'openat(AT_FDCWD, "/out/custom_trace/detector.bin", O_WRONLY|O_CREAT, 0666) = 8'
    )
    r = _rejected(tmp_path, [*loader, *_REJECTED_OPENS, line], _REJECTED_LOG)
    assert not r["pass"] and not r["checks"]["no_output"]["pass"]


def _with_args(*args: str, base: str = _ARGS) -> list[str]:
    """The launcher and loader execs with the entrypoint arguments ``base``
    followed by ``args``."""
    extra = (
        ", ".join([base, *(f'"{a}"' for a in args)])
        if base
        else ", ".join(f'"{a}"' for a in args)
    )
    return [x.replace(_ARGS, extra) for x in (LAUNCHER, LOADER)]


@pytest.mark.parametrize("option", ["--out", "--trace", "--report"])
def test_rejected_resolves_relative_output_args_against_the_cwd(
    tmp_path: Path, option: str
) -> None:
    loader = _with_args(option, "custom_trace")
    line = (
        'openat(AT_FDCWD</out>, "custom_trace/detector.bin", '
        "O_WRONLY|O_CREAT, 0666) = -1 EACCES"
    )
    r = _rejected(tmp_path, [*loader, *_REJECTED_OPENS, line], _REJECTED_LOG)
    assert not r["pass"] and not r["checks"]["no_output"]["pass"]
    assert r["checks"]["no_output"]["attempted"] == ["/out/custom_trace/detector.bin"]


def test_rejected_finds_relative_output_surviving_under_the_cwd(tmp_path: Path) -> None:
    (tmp_path / "out" / "custom_report.json").parent.mkdir(parents=True)
    (tmp_path / "out" / "custom_report.json").write_bytes(b"{}")
    loader = _with_args("--report", "custom_report.json")
    cwd = 'openat(AT_FDCWD</out>, "/etc/ld.so.cache", O_RDONLY) = 3</etc/ld.so.cache>'
    r = _rejected(tmp_path, [*loader, *_REJECTED_OPENS, cwd], _REJECTED_LOG)
    assert not r["pass"] and r["checks"]["no_output"]["found"]


def test_rejected_fails_closed_on_relative_output_without_a_cwd(
    tmp_path: Path,
) -> None:
    # Plain (non -yy) records carry no cwd: the argument's base is unknown.
    loader = _with_args("--trace", "custom_trace")
    r = _rejected(tmp_path, [*loader, *_REJECTED_PLAIN], _REJECTED_LOG, plain=True)
    assert not r["pass"] and not r["checks"]["no_output"]["pass"]
    assert r["checks"]["no_output"]["unresolved_outputs"] == ["--trace custom_trace"]


@pytest.mark.parametrize(
    ("line", "check", "target"),
    [
        (
            'openat(AT_FDCWD</>, "/tmp/config-alias", O_RDONLY) = '
            "3</opt/saccade/share/saccade/configs/shipping/mamba_whole_graph.resolved.json>",
            "no_model_root_open",
            "/opt/saccade/share/saccade/configs/shipping/mamba_whole_graph.resolved.json",
        ),
        (
            'openat(AT_FDCWD</>, "/tmp/op.so", O_RDONLY|O_CLOEXEC) = '
            "3</elsewhere/libsaccade_scan_torchop.so>",
            "no_model_root_open",
            "/elsewhere/libsaccade_scan_torchop.so",
        ),
        (
            'openat(AT_FDCWD</>, "/dev/char/195:0", O_RDWR) = 4</dev/nvidia0<char 195:0>>',
            "no_gpu_device_open",
            "/dev/nvidia0",
        ),
        (
            'openat(AT_FDCWD</>, "/tmp/trace-alias", O_WRONLY|O_CREAT, 0666) = '
            "5</out/trace/detector.bin>",
            "no_output",
            "/out/trace/detector.bin",
        ),
    ],
)
def test_rejected_checks_the_opened_target_of_an_alias(
    tmp_path: Path, line: str, check: str, target: str
) -> None:
    r = _rejected(tmp_path, [LAUNCHER, LOADER, *_REJECTED_OPENS, line], _REJECTED_LOG)
    assert not r["pass"] and not r["checks"][check]["pass"]
    assert target in r["checks"][check]["attempted"]
    assert not r["checks"][check]["unresolved"]


def test_rejected_fails_closed_on_a_yy_open_without_its_target(
    tmp_path: Path,
) -> None:
    line = 'openat(AT_FDCWD</>, "/tmp/config-alias", O_RDONLY) = 3'
    r = _rejected(tmp_path, [LAUNCHER, LOADER, *_REJECTED_OPENS, line], _REJECTED_LOG)
    assert not r["pass"]
    for check in ("no_model_root_open", "no_gpu_device_open", "no_output"):
        assert r["checks"][check]["unresolved"] == [line]


@pytest.mark.parametrize(
    ("args", "line"),
    [
        (
            ("--model-root", "/tmp/models"),
            'openat(AT_FDCWD</>, "/tmp/models/configs/shipping/'
            'mamba_whole_graph.resolved.json", O_RDONLY) = 3</tmp/models/configs/'
            "shipping/mamba_whole_graph.resolved.json>",
        ),
        (
            ("--model-root", "models"),
            'openat(AT_FDCWD</work>, "models/models/yolo/x.engine", O_RDONLY) = -1 ENOENT',
        ),
        (
            (),
            'openat(AT_FDCWD</>, "/cfg/c.json", O_RDONLY) = 3</cfg/c.json>',
        ),
    ],
)
def test_rejected_checks_the_model_inputs_named_by_argv(
    tmp_path: Path, args: tuple[str, ...], line: str
) -> None:
    loader = _with_args(*args) if args else [LAUNCHER, LOADER]
    r = _rejected(tmp_path, [*loader, *_REJECTED_OPENS, line], _REJECTED_LOG)
    assert not r["pass"] and not r["checks"]["no_model_root_open"]["pass"]
    assert r["checks"]["no_model_root_open"]["attempted"]


def test_rejected_fails_closed_on_relative_model_root_without_a_cwd(
    tmp_path: Path,
) -> None:
    loader = _with_args("--model-root", "models")
    r = _rejected(tmp_path, [*loader, *_REJECTED_PLAIN], _REJECTED_LOG, plain=True)
    c = r["checks"]["no_model_root_open"]
    assert not r["pass"] and not c["pass"]
    assert c["unresolved_inputs"] == ["--model-root models"]


_ENGINE = "models/yolo/yolo26s_backbone_640_best.engine"


def test_rejected_checks_the_default_model_root(tmp_path: Path) -> None:
    # no --model-root: the entrypoint's model root is its cwd (A4)
    lines = _with_args(base='"--config", "/cfg/c.json"')
    opens = [x.replace("AT_FDCWD</>", "AT_FDCWD</work>") for x in _REJECTED_OPENS]
    line = f'openat(AT_FDCWD</work>, "/work/{_ENGINE}", O_RDONLY) = 3</work/{_ENGINE}>'
    for d in ("clean", "read"):
        (tmp_path / d).mkdir()
    clean = _rejected(tmp_path / "clean", [*lines, *opens], _REJECTED_LOG)
    assert clean["pass"], clean["checks"]
    assert "/work" in clean["checks"]["no_model_root_open"]["model_inputs"]
    r = _rejected(tmp_path / "read", [*lines, *opens, line], _REJECTED_LOG)
    c = r["checks"]["no_model_root_open"]
    assert not r["pass"] and not c["pass"]
    assert c["attempted"] == [f"/work/{_ENGINE}"]


def test_rejected_default_model_root_at_the_filesystem_root(tmp_path: Path) -> None:
    # cwd "/": every path is under the default model root
    lines = _with_args(base='"--config", "/cfg/c.json"')
    r = _rejected(tmp_path, [*lines, *_REJECTED_OPENS], _REJECTED_LOG)
    c = r["checks"]["no_model_root_open"]
    assert not c["pass"] and "/opt/saccade/lib/vendor/libx.so.1" in c["attempted"]


@pytest.mark.parametrize(
    "args",
    [
        ("--measurement-mutation", "--model-root", "/opt/saccade/share/saccade"),
        ("--model-root",),
    ],
)
def test_rejected_model_root_the_parse_does_not_reach_is_not_set(
    tmp_path: Path, args: tuple[str, ...]
) -> None:
    lines = _with_args(*args, base='"--config", "/cfg/c.json"')
    opens = [x.replace("AT_FDCWD</>", "AT_FDCWD</work>") for x in _REJECTED_OPENS]
    r = _rejected(tmp_path, [*lines, *opens], _REJECTED_LOG)
    assert "/work" in r["checks"]["no_model_root_open"]["model_inputs"]


def test_rejected_fails_closed_on_the_default_model_root_without_a_cwd(
    tmp_path: Path,
) -> None:
    lines = _with_args(base='"--config", "/cfg/c.json"')
    r = _rejected(tmp_path, [*lines, *_REJECTED_PLAIN], _REJECTED_LOG, plain=True)
    c = r["checks"]["no_model_root_open"]
    assert not r["pass"] and not c["pass"]
    assert c["unresolved_inputs"] == ["default --model-root ."]


# ── launcher ───────────────────────────────────────────────────────────────────


def test_launcher_runs_the_entrypoint_through_the_loader() -> None:
    text = (REPO / "shipping" / "launcher" / "saccade_track.sh").read_text()
    assert text.startswith("#!/bin/sh\n")
    assert "exec /lib64/ld-linux-x86-64.so.2" in text
    assert '--library-path "$prefix/lib/vendor"' in text
    assert '--audit "$prefix/lib/saccade_loader_audit.so"' in text
    assert '"$prefix/libexec/saccade_track" "$@"' in text
    assert "unset LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH SACCADE_AUDIT_PROBE" in text
    assert "ready=$(SACCADE_AUDIT_PROBE=1 /lib64/ld-linux-x86-64.so.2" in text
    assert '[ "$ready" != saccade-loader-audit-ready ]' in text
    if shutil.which("sh"):
        subprocess.run(
            ["sh", "-n", str(REPO / "shipping" / "launcher" / "saccade_track.sh")],
            check=True,
        )


# ── auditor on a real loader ───────────────────────────────────────────────────

LDSO = Path("/lib64/ld-linux-x86-64.so.2")


_NEEDS_LOADER = pytest.mark.skipif(
    not shutil.which("cc") or not LDSO.exists(), reason="needs cc and the x86-64 loader"
)


def _cc(*args: str) -> None:
    subprocess.run(["cc", *args], check=True, capture_output=True)


def _audit_prefix(tmp_path: Path) -> Path:
    """<prefix> with lib/vendor/libfoo.so (prints 1), libexec/main (DT_RPATH
    $ORIGIN/../nvidia before --library-path), the real auditor and launcher."""
    p = tmp_path / "prefix"
    for d in ("lib/vendor", "libexec", "nvidia", "gen", "src"):
        (p / d).mkdir(parents=True, exist_ok=True)
    src = p / "src"
    (src / "foo1.c").write_text("int foo(void) { return 1; }\n")
    (src / "foo2.c").write_text("int foo(void) { return 2; }\n")
    (src / "main.c").write_text(
        '#include <stdio.h>\nint foo(void);\nint main(void) { printf("%d\\n", foo()); return 0; }\n'
    )
    (p / "gen" / "loader_audit_names.inc").write_text(
        'static const char *const kBundled[] = {"libfoo.so", NULL};\n'
    )

    _cc(
        "-shared",
        "-fPIC",
        "-Wl,-soname,libfoo.so",
        "-o",
        str(p / "lib/vendor/libfoo.so"),
        str(src / "foo1.c"),
    )
    # DT_RPATH (not RUNPATH): the loader searches it before --library-path
    _cc(
        "-o",
        str(p / "libexec/main"),
        str(src / "main.c"),
        f"-L{p / 'lib/vendor'}",
        "-lfoo",
        "-Wl,--disable-new-dtags,-rpath,$ORIGIN/../nvidia",
    )
    _cc(
        "-shared",
        "-fPIC",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{p / 'gen'}",
        "-o",
        str(p / "lib/saccade_loader_audit.so"),
        str(REPO / "shipping/src/loader_audit.c"),
    )

    (p / "bin").mkdir()
    shutil.copy(REPO / "shipping/launcher/saccade_track.sh", p / "bin/saccade_track")
    (p / "bin/saccade_track").chmod(0o755)
    (p / "libexec/main").rename(p / "libexec/saccade_track")
    return p


def _plant_foreign_copy(p: Path) -> None:
    _cc(
        "-shared",
        "-fPIC",
        "-Wl,-soname,libfoo.so",
        "-o",
        str(p / "nvidia/libfoo.so"),
        str(p / "src/foo2.c"),
    )


@_NEEDS_LOADER
def test_auditor_fails_closed_on_a_planted_rpath_copy(tmp_path: Path) -> None:
    p = _audit_prefix(tmp_path)

    def run(audit: bool) -> subprocess.CompletedProcess[str]:
        cmd = [str(LDSO), "--library-path", str(p / "lib/vendor")]
        if audit:
            cmd += ["--audit", str(p / "lib/saccade_loader_audit.so")]
        return subprocess.run(
            [*cmd, str(p / "libexec/saccade_track")],
            capture_output=True,
            text=True,
            env={},
        )

    clean = run(audit=True)
    assert clean.returncode == 0 and clean.stdout == "1\n", clean.stderr
    _plant_foreign_copy(p)
    silent = run(audit=False)
    assert (
        silent.returncode == 0 and silent.stdout == "2\n"
    )  # the planted copy wins without the auditor
    refused = run(audit=True)
    assert refused.returncode == 127
    assert (
        "foreign copy on the search path" in refused.stderr
        and "nvidia/libfoo.so" in refused.stderr
    )


@_NEEDS_LOADER
def test_auditor_classifies_a_symlink_alias_by_its_real_name(tmp_path: Path) -> None:
    p = _audit_prefix(tmp_path)
    # a foreign libfoo.so (no SONAME, so NEEDED is the link name) and an
    # alias to it under a name that is not bundled
    _cc("-shared", "-fPIC", "-o", str(p / "nvidia/libfoo.so"), str(p / "src/foo2.c"))
    (p / "nvidia/payload").symlink_to("libfoo.so")
    for name, lib in (("direct", "libfoo.so"), ("alias", "payload")):
        _cc(
            "-o",
            str(p / f"libexec/{name}"),
            str(p / "src/main.c"),
            f"-L{p / 'nvidia'}",
            f"-l:{lib}",
            "-Wl,--disable-new-dtags,-rpath,$ORIGIN/../nvidia",
        )

    def run(name: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(LDSO),
                "--library-path",
                str(p / "lib/vendor"),
                "--audit",
                str(p / "lib/saccade_loader_audit.so"),
                str(p / f"libexec/{name}"),
            ],
            capture_output=True,
            text=True,
            env={},
        )

    direct = run("direct")
    assert direct.returncode == 127 and direct.stdout == "", direct.stderr
    alias = run("alias")
    # the foreign foo() must not run through the alias either
    assert alias.returncode == 127 and alias.stdout == "", (alias.stdout, alias.stderr)
    assert "bundled library mapped from outside lib/vendor" in alias.stderr


@_NEEDS_LOADER
@pytest.mark.parametrize(
    ("target", "refused"),
    [
        ("libfoo.so.1.0.0", True),  # the bundled family libfoo.so (A4)
        ("libfoo.so.13", True),
        ("libfoobar.so.1.0", False),  # another family: not the auditor's to refuse
    ],
)
def test_auditor_classifies_a_versioned_alias_by_its_soname_family(
    tmp_path: Path, target: str, refused: bool
) -> None:
    p = _audit_prefix(tmp_path)
    _cc("-shared", "-fPIC", "-o", str(p / "nvidia" / target), str(p / "src/foo2.c"))
    (p / "nvidia/payload").symlink_to(target)
    _cc(
        "-o",
        str(p / "libexec/alias"),
        str(p / "src/main.c"),
        f"-L{p / 'nvidia'}",
        "-l:payload",
        "-Wl,--disable-new-dtags,-rpath,$ORIGIN/../nvidia",
    )
    r = subprocess.run(
        [
            str(LDSO),
            "--library-path",
            str(p / "lib/vendor"),
            "--audit",
            str(p / "lib/saccade_loader_audit.so"),
            str(p / "libexec/alias"),
        ],
        capture_output=True,
        text=True,
        env={},
    )
    if refused:
        assert r.returncode == 127 and r.stdout == "", (r.stdout, r.stderr)
        assert "bundled library mapped from outside lib/vendor" in r.stderr
    else:
        assert r.returncode == 0 and r.stdout == "2\n", r.stderr


def _launch(
    p: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(p / "bin/saccade_track")],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
    )


@_NEEDS_LOADER
def test_launcher_runs_the_entrypoint_when_the_auditor_initializes(
    tmp_path: Path,
) -> None:
    p = _audit_prefix(tmp_path)
    clean = _launch(p)
    assert clean.returncode == 0 and clean.stdout == "1\n", clean.stderr
    # a caller's probe variable does not reach the entrypoint's loader
    probed = _launch(p, {"SACCADE_AUDIT_PROBE": "1"})
    assert probed.returncode == 0 and probed.stdout == "1\n", probed.stderr
    _plant_foreign_copy(p)
    refused = _launch(p)
    assert refused.returncode == 127 and refused.stdout == ""
    assert "foreign copy on the search path" in refused.stderr


@_NEEDS_LOADER
@pytest.mark.parametrize("damage", ["missing", "truncated", "empty", "not_an_auditor"])
def test_launcher_fails_closed_when_the_auditor_does_not_initialize(
    tmp_path: Path, damage: str
) -> None:
    p = _audit_prefix(tmp_path)
    _plant_foreign_copy(p)
    auditor = p / "lib/saccade_loader_audit.so"
    data = auditor.read_bytes()
    auditor.unlink()
    if damage == "truncated":
        auditor.write_bytes(data[: len(data) // 2])
    elif damage == "empty":
        auditor.write_bytes(b"")
    elif damage == "not_an_auditor":  # loads, but has no la_version
        shutil.copy(p / "lib/vendor/libfoo.so", auditor)
    r = _launch(p)
    # the loader alone would ignore the auditor and run the planted copy ("2")
    assert r.returncode == 127 and r.stdout == "", (r.stdout, r.stderr)
    assert "the loader provenance auditor did not initialize" in r.stderr


@_NEEDS_LOADER
@pytest.mark.parametrize("delimiter", [":", ";"])
def test_launcher_refuses_a_library_path_delimiter_in_the_prefix(
    tmp_path: Path, delimiter: str
) -> None:
    # the loader splits --library-path on both, whatever the quoting (A4)
    p = _audit_prefix(tmp_path)
    moved = tmp_path / f"a{delimiter}b" / "prefix"
    moved.parent.mkdir()
    p.rename(moved)
    r = _launch(moved)
    assert r.returncode == 2 and r.stdout == "", (r.stdout, r.stderr)
    assert "must not contain ':' or ';'" in r.stderr


# ── install ────────────────────────────────────────────────────────────────────


def _install(tmp_path: Path, set_: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    set_path = tmp_path / "set.json"
    set_path.write_text(json.dumps(set_))
    return subprocess.run(
        [
            "cmake",
            f"-DSACCADE_REPO_ROOT={REPO}",
            f"-DSACCADE_PREFIX_DEST={tmp_path / 'tree'}",
            f"-DSACCADE_THIRD_PARTY_SET={set_path}",
            f"-DSACCADE_ROOT_purelib={tmp_path / 'site'}",
            f"-DSACCADE_ROOT_nvjpeg_wheel={tmp_path / 'nvj'}",
            "-P",
            str(REPO / "shipping/cmake/install_third_party.cmake"),
        ],
        capture_output=True,
        text=True,
    )


def _fake_set(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "site/pkg/lib").mkdir(parents=True)
    (tmp_path / "site/pkg-1.0.dist-info").mkdir(parents=True)
    (tmp_path / "nvj").mkdir()
    (tmp_path / "site/pkg/lib/libx.so.1").write_bytes(b"x")
    (tmp_path / "site/pkg-1.0.dist-info/LICENSE").write_text("terms")
    return {
        "schema": "saccade.shipping_third_party/v1",
        "entries": [
            {
                "soname": "libx.so.1",
                "sha256": _sha(b"x"),
                "source": {"root": "purelib", "path": "pkg/lib/libx.so.1"},
                "wheel": "pkg-1.0",
                "license_files": [
                    {"path": "pkg-1.0.dist-info/LICENSE", "sha256": _sha(b"terms")}
                ],
            }
        ],
    }


@pytest.mark.skipif(not shutil.which("cmake"), reason="needs cmake")
def test_install_third_party_copies_the_pinned_set(tmp_path: Path) -> None:
    r = _install(tmp_path, _fake_set(tmp_path))
    assert r.returncode == 0, r.stderr
    t = tmp_path / "tree"
    assert (t / "lib/vendor/libx.so.1").read_bytes() == b"x"
    assert (t / "licenses/pkg-1.0/LICENSE").read_text() == "terms"
    assert (t / "licenses/THIRD_PARTY.md").read_bytes() == (
        REPO / "shipping/THIRD_PARTY.md"
    ).read_bytes()
    assert (t / "licenses/saccade/LICENSE").is_file()
    assert (t / "README.txt").read_bytes() == (
        REPO / "shipping/package/README.txt"
    ).read_bytes()


@pytest.mark.skipif(not shutil.which("cmake"), reason="needs cmake")
def test_install_third_party_refuses_other_bytes(tmp_path: Path) -> None:
    set_ = _fake_set(tmp_path)
    (tmp_path / "site/pkg/lib/libx.so.1").write_bytes(b"x+")
    r = _install(tmp_path, set_)
    assert r.returncode != 0 and "third_party_set.json" in r.stderr
    assert not (tmp_path / "tree/lib/vendor/libx.so.1").exists()


# ── sources (host run) ─────────────────────────────────────────────────────────


def test_init_paths_skip_the_launcher_shell() -> None:
    log = (
        "    7:\tcalling init: /usr/lib/libreadline.so.8\n"
        "    7:\tfile=/p/lib/saccade_loader_audit.so [1];  generating link map\n"
        "    7:\tcalling init: /usr/lib/libc.so.6\n"
        "    7:\tcalling init: /p/lib/vendor/libx.so.1\n"
    )
    assert bundle.init_paths(log) == ["/usr/lib/libc.so.6", "/p/lib/vendor/libx.so.1"]
    assert bundle.init_paths("    7:\tcalling init: /usr/lib/libc.so.6\n") == []


def test_sources_pass_and_reject(tmp_path: Path) -> None:
    tree, set_path, _ = _runtime_tree(tmp_path)
    (tmp_path / "ld").mkdir()
    lines = [
        f"    7:\tfile={tree}/lib/saccade_loader_audit.so [1];  generating link map",
        f"    7:\tcalling init: {tree}/lib/saccade_loader_audit.so",
        f"    7:\tcalling init: {tree}/lib/vendor/libx.so.1",
        f"    7:\tcalling init: {tree}/share/saccade/build/libop.so",
    ]

    def sources(extra: list[str]) -> dict[str, Any]:
        (tmp_path / "ld" / "ld.7").write_text("\n".join(lines + extra) + "\n")
        out = tmp_path / "sources.json"
        bundle.main(
            [
                "sources",
                "--log-prefix",
                str(tmp_path / "ld" / "ld"),
                "--tree",
                str(tree),
                "--report",
                str(out),
                "--third-party-set",
                str(set_path),
            ]
        )
        return json.loads(out.read_text())

    assert sources([])["pass"]
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere/libx.so.1").write_bytes(b"x")
    r = sources([f"    7:\tcalling init: {tmp_path}/elsewhere/libx.so.1"])
    assert not r["pass"] and any("not " in p and "vendor" in p for p in r["problems"])
