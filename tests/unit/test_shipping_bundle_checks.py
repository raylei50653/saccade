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
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def test_notice_is_the_exporters_output() -> None:
    assert (REPO / "shipping" / "THIRD_PARTY.md").read_text() == export.notice(SET)


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

    clo, problems = bundle.bundle_closure(
        tmp_path, ["libexec/saccade_track"], {"libtorch.so"}, run
    )
    assert {c["name"]: c["class"] for c in clo}["libtorch.so"] == "vendor"
    assert any("libmissing.so.1 is not in lib/vendor" in p for p in problems)
    assert any("libpython3.12" in p for p in problems)


# ── runtime ────────────────────────────────────────────────────────────────────

LOADER = (
    'execve("/lib64/ld-linux-x86-64.so.2", ["/lib64/ld-linux-x86-64.so.2", "--library-path", '
    '"/opt/saccade/lib/vendor", "--audit", "/opt/saccade/lib/saccade_loader_audit.so", "--argv0", '
    '"/opt/saccade/bin/saccade_track", "/opt/saccade/libexec/saccade_track", "--config", "c"], 0x0 /* 3 vars */) = 0'
)
LAUNCHER = 'execve("/opt/saccade/bin/saccade_track", ["/opt/saccade/bin/saccade_track", "--config", "c"], 0x0 /* 3 vars */) = 0'


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
    return tree, set_path, opens


def _runtime(tmp_path: Path, lines: list[str]) -> dict[str, Any]:
    tree, set_path, _ = _runtime_tree(tmp_path)
    (tmp_path / "st").mkdir(exist_ok=True)
    (tmp_path / "st" / "s.1").write_text("\n".join(lines) + "\n")
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


# ── launcher ───────────────────────────────────────────────────────────────────


def test_launcher_runs_the_entrypoint_through_the_loader() -> None:
    text = (REPO / "shipping" / "launcher" / "saccade_track.sh").read_text()
    assert text.startswith("#!/bin/sh\n")
    assert "exec /lib64/ld-linux-x86-64.so.2" in text
    assert '--library-path "$prefix/lib/vendor"' in text
    assert '--audit "$prefix/lib/saccade_loader_audit.so"' in text
    assert '"$prefix/libexec/saccade_track" "$@"' in text
    assert "unset LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH" in text
    if shutil.which("sh"):
        subprocess.run(
            ["sh", "-n", str(REPO / "shipping" / "launcher" / "saccade_track.sh")],
            check=True,
        )


# ── auditor on a real loader ───────────────────────────────────────────────────

LDSO = Path("/lib64/ld-linux-x86-64.so.2")


@pytest.mark.skipif(
    not shutil.which("cc") or not LDSO.exists(), reason="needs cc and the x86-64 loader"
)
def test_auditor_fails_closed_on_a_planted_rpath_copy(tmp_path: Path) -> None:
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

    def cc(*args: str) -> None:
        subprocess.run(["cc", *args], check=True, capture_output=True)

    cc(
        "-shared",
        "-fPIC",
        "-Wl,-soname,libfoo.so",
        "-o",
        str(p / "lib/vendor/libfoo.so"),
        str(src / "foo1.c"),
    )
    # DT_RPATH (not RUNPATH): the loader searches it before --library-path
    cc(
        "-o",
        str(p / "libexec/main"),
        str(src / "main.c"),
        f"-L{p / 'lib/vendor'}",
        "-lfoo",
        "-Wl,--disable-new-dtags,-rpath,$ORIGIN/../nvidia",
    )
    cc(
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

    def run(audit: bool) -> subprocess.CompletedProcess[str]:
        cmd = [str(LDSO), "--library-path", str(p / "lib/vendor")]
        if audit:
            cmd += ["--audit", str(p / "lib/saccade_loader_audit.so")]
        return subprocess.run(
            [*cmd, str(p / "libexec/main")], capture_output=True, text=True, env={}
        )

    clean = run(audit=True)
    assert clean.returncode == 0 and clean.stdout == "1\n", clean.stderr
    cc(
        "-shared",
        "-fPIC",
        "-Wl,-soname,libfoo.so",
        "-o",
        str(p / "nvidia/libfoo.so"),
        str(src / "foo2.c"),
    )
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
