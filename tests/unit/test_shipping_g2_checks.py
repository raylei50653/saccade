"""The shipping tree's G2 checks and model-root install (#465 PR-12).

Boundary §0 / §6 PR-12, docs/reference/native_runtime_resolved_config.md §16.
The real checks need an installed tree, a GPU run and a container; these pin
the logic in the ordinary pytest job:

* ``scripts/native/check_shipping_tree.py``: the loader-log and strace parsers,
  the glibc baseline comparison, the RUNPATH rule with its one enumerated
  exception (the attested operator library, identified by sha256), the SM +
  PTX check, the Python-file scan, the NEEDED closure, and the runtime verdict
  (one exec, no Python / Triton opens, the reference's loaded set) -- ELF
  readers replaced by fakes;
* the SM list in ``shipping/CMakeLists.txt`` is the one the checker expects;
* ``shipping/cmake/install_model_root.cmake`` copies the bound files and refuses
  one whose sha256 is not the recorded one (skips without ``cmake``).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPO / "shipping" / "cmake" / "install_model_root.cmake"
ATT_REL = "configs/shipping/mamba_head_realization.attestation.json"
OP_REL = "build/libsaccade_scan_torchop.so"


def _load() -> ModuleType:
    path = REPO / "scripts" / "native" / "check_shipping_tree.py"
    spec = importlib.util.spec_from_file_location("check_shipping_tree", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g2 = _load()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── loader log ─────────────────────────────────────────────────────────────────

LD_LOG = """\
    101:	file=libnvinfer.so.10 [0];  needed by /t/bin/saccade_track [0]
    101:	file=libnvinfer.so.10 [0];  generating link map
    101:	  dynamic: 0x1  base: 0x2   size: 0x3
    101:	file=libcuda.so.1 [0];  dynamically loaded by /v/libcudart.so.13 [0]
    101:	file=libcuda.so.1 [0];  generating link map
    101:	file=./build/libsaccade_scan_torchop.so [0];  dynamically loaded by bin [0]
    101:	file=./build/libsaccade_scan_torchop.so [0];  generating link map
    101:	calling init: /usr/lib/ld-linux-x86-64.so.2
    101:	calling init: /v/libnvinfer.so.10
    101:	calling init: /usr/lib/wsl/drivers/x/libcuda.so.1.1
    101:	calling init: ./build/libsaccade_scan_torchop.so
"""


def test_loader_log_pairs_names_with_paths() -> None:
    names, paths = g2.parse_ld_debug(LD_LOG)
    assert names == [
        "libnvinfer.so.10",
        "libcuda.so.1",
        "./build/libsaccade_scan_torchop.so",
    ]
    sonames = {"/usr/lib/wsl/drivers/x/libcuda.so.1.1": "libcuda.so.1"}
    matched = g2.match_loaded(names, paths, lambda p: sonames.get(p))
    assert matched == {
        "/usr/lib/ld-linux-x86-64.so.2": [],
        "/v/libnvinfer.so.10": ["libnvinfer.so.10"],
        "/usr/lib/wsl/drivers/x/libcuda.so.1.1": ["libcuda.so.1"],
        "./build/libsaccade_scan_torchop.so": ["./build/libsaccade_scan_torchop.so"],
    }


def test_loader_log_refuses_an_unmatched_name() -> None:
    with pytest.raises(g2.CheckError, match="matches 0"):
        g2.match_loaded(["libmissing.so.1"], ["/v/libother.so"], lambda p: None)


def test_classification() -> None:
    assert g2.classify_name("libc.so.6") == "base_system"
    assert g2.classify_name("libz.so.1") == "base_system"
    assert g2.classify_name("libcuda.so.1") == "driver"
    assert g2.classify_name("libnvcuvid.so.1") == "driver"
    assert g2.classify_name("libdxcore.so", "/usr/lib/wsl/lib/libdxcore.so") == "driver"
    assert g2.classify_name("libtorch_cuda.so") == "third_party"


# ── ELF readers (fake readelf / cuobjdump) ─────────────────────────────────────

READELF_D = """
Dynamic section at offset 0x1 contains 3 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libtorch.so]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000000e (SONAME)             Library soname: [libx.so.1]
 0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN/../lib:/abs/dir]
"""

READELF_V = """
Version symbols section '.gnu.version' contains 2 entries:
  000:   0 (*local*)       2 (GLIBC_2.40)

Version needs section '.gnu.version_r' contains 2 entries:
  000000: Version: 1  File: libstdc++.so.6  Cnt: 2
  0x0010:   Name: CXXABI_1.3.15  Flags: none  Version: 5
  0x0020:   Name: GLIBCXX_3.4.29  Flags: none  Version: 4
  0x0030: Version: 1  File: libc.so.6  Cnt: 2
  0x0040:   Name: GLIBC_2.38  Flags: none  Version: 3
  0x0050:   Name: GLIBC_2.4  Flags: none  Version: 2
"""


def test_dynamic_section_and_version_needs() -> None:
    dyn = g2.dynamic_section(Path("x"), lambda cmd: READELF_D)
    assert dyn == {
        "needed": ["libtorch.so", "libc.so.6"],
        "soname": "libx.so.1",
        "runpath": ["$ORIGIN/../lib", "/abs/dir"],
        "rpath": [],
    }
    # Only the version-needs section counts (2.40 above is a definition table).
    needs = g2.version_needs(Path("x"), lambda cmd: READELF_V)
    assert needs == {"CXXABI": "1.3.15", "GLIBC": "2.38", "GLIBCXX": "3.4.29"}
    assert g2.over_baseline(needs) == []
    assert g2.over_baseline(
        {"GLIBC": "2.40", "GLIBCXX": "3.4.34", "CXXABI": "1.3.15"}
    ) == [
        "GLIBC_2.40",
        "GLIBCXX_3.4.34",
    ]


def test_cuda_archs() -> None:
    elf = "ELF file    1: x.1.sm_75.cubin\nELF file    2: x.2.sm_120.cubin\nELF file 3: x.3.sm_100.cubin\n"
    ptx = "PTX file    1: x.1.sm_120.ptx\n"

    def run(cmd: list[str]) -> str:
        return elf if cmd[1] == "--list-elf" else ptx

    assert g2.cuda_archs(Path("x"), run) == {
        "sass": ["sm_75", "sm_100", "sm_120"],
        "ptx": ["sm_120"],
    }

    def none(cmd: list[str]) -> str:
        raise g2.CheckError("cuobjdump: no device code")

    assert g2.cuda_archs(Path("x"), none) == {"sass": [], "ptx": []}


def test_shipping_cmake_sm_list_is_the_checkers() -> None:
    text = (REPO / "shipping" / "CMakeLists.txt").read_text()
    m = re.search(r'set\(SACCADE_SHIPPING_TORCH_CUDA_ARCH_LIST "([^"]+)"\)', text)
    assert m
    entries = m.group(1).split(";")
    sass = {"sm_" + e.split("+")[0].replace(".", "") for e in entries}
    ptx = {
        "sm_" + e.split("+")[0].replace(".", "") for e in entries if e.endswith("+PTX")
    }
    assert sass == g2.SHIPPING_SASS
    assert ptx == g2.SHIPPING_PTX


# ── static ─────────────────────────────────────────────────────────────────────


def _tree(tmp_path: Path, op_bytes: bytes = b"\x7fELF-op") -> tuple[Path, Path, Path]:
    tree = tmp_path / "tree"
    (tree / "bin").mkdir(parents=True)
    (tree / "bin" / "saccade_track").write_bytes(b"\x7fELF-track")
    op = tree / g2.MODEL_ROOT / OP_REL
    op.parent.mkdir(parents=True)
    op.write_bytes(op_bytes)
    att = tree / g2.MODEL_ROOT / ATT_REL
    att.parent.mkdir(parents=True)
    att.write_text(
        json.dumps({"op_library": {"path": OP_REL, "sha256": _sha(b"\x7fELF-op")}})
    )
    deps = tmp_path / "deps"
    deps.mkdir()
    manifest = tmp_path / "deps.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {"names": ["libtorch.so"], "version_needs": {"GLIBC": "2.28"}}
                ]
            }
        )
    )
    return tree, deps, manifest


def _static(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    track_runpath: list[str] | None = None,
    track_rpath: list[str] | None = None,
    sass: list[str] | None = None,
    needs: dict[str, str] | None = None,
    op_bytes: bytes = b"\x7fELF-op",
    extra: str | None = None,
) -> dict[str, Any]:
    tree, deps, manifest = _tree(tmp_path, op_bytes)
    if extra:
        (tree / extra).parent.mkdir(parents=True, exist_ok=True)
        (tree / extra).write_text("x")

    def dyn(path: Path, run: Any = None) -> dict[str, Any]:
        if path.name == "saccade_track":
            rp = ["$ORIGIN/../lib"] if track_runpath is None else track_runpath
            return {
                "needed": [],
                "soname": None,
                "runpath": rp,
                "rpath": track_rpath or [],
            }
        return {"needed": [], "soname": None, "runpath": ["/home/x/build"], "rpath": []}

    def cuda(path: Path, run: Any = None) -> dict[str, list[str]]:
        if path.name == "saccade_track":
            return {"sass": sass or sorted(g2.SHIPPING_SASS), "ptx": ["sm_120"]}
        return {"sass": ["sm_120"], "ptx": []}

    monkeypatch.setattr(g2, "dynamic_section", dyn)
    monkeypatch.setattr(g2, "cuda_archs", cuda)
    monkeypatch.setattr(
        g2, "version_needs", lambda p, run=None: needs or {"GLIBC": "2.38"}
    )
    monkeypatch.setattr(g2, "closure", lambda roots, deps_dir, run=None: ([], []))
    report = tmp_path / "static.json"
    rc = g2.main(
        [
            "static",
            "--tree",
            str(tree),
            "--deps-manifest",
            str(manifest),
            "--deps-dir",
            str(deps),
            "--report",
            str(report),
        ]
    )
    out: dict[str, Any] = json.loads(report.read_text())
    assert rc == (0 if out["pass"] else 1)
    return out


def test_static_passes_with_the_op_library_as_the_enumerated_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = _static(monkeypatch, tmp_path)
    assert out["pass"]
    rp = out["checks"]["runpath_origin_only"]
    assert rp["produced"] == ["bin/saccade_track"]
    assert rp["exceptions"] == [f"{g2.MODEL_ROOT}/{OP_REL}"]
    assert out["checks"]["sm_ptx_entrypoint"]["named_limit"] == {
        f"{g2.MODEL_ROOT}/{OP_REL}": {"sass": ["sm_120"], "ptx": []}
    }


@pytest.mark.parametrize(
    ("kwargs", "check"),
    [
        ({"track_runpath": ["/home/x/.venv/lib"]}, "runpath_origin_only"),
        (
            {"track_runpath": ["$ORIGIN/../lib", "/opt/cuda/lib64"]},
            "runpath_origin_only",
        ),
        ({"track_rpath": ["$ORIGIN"]}, "runpath_origin_only"),
        ({"op_bytes": b"\x7fELF-op-rebuilt"}, "runpath_origin_only"),
        ({"sass": ["sm_120"]}, "sm_ptx_entrypoint"),
        ({"needs": {"GLIBC": "2.40"}}, "glibc_baseline"),
        ({"needs": {"CXXABI": "1.3.16"}}, "glibc_baseline"),
        ({"extra": "share/saccade/helper.py"}, "g2_3_no_python_files"),
        ({"extra": "lib/__pycache__/x.cpython-312.pyc"}, "g2_3_no_python_files"),
        ({"extra": "lib/site-packages/README"}, "g2_3_no_python_files"),
    ],
)
def test_static_rejects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kwargs: dict[str, Any], check: str
) -> None:
    out = _static(monkeypatch, tmp_path, **kwargs)
    assert not out["pass"]
    assert [k for k, c in out["checks"].items() if not c["pass"]] == [check]


def test_closure_resolves_origin_and_deps_and_flags_python(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "bin").mkdir(parents=True)
    (tree / "lib").mkdir()
    track = tree / "bin" / "saccade_track"
    track.write_bytes(b"\x7fELF")
    (tree / "lib" / "libown.so").write_bytes(b"\x7fELF")
    deps = tmp_path / "deps"
    deps.mkdir()
    (deps / "libtorch.so").write_bytes(b"\x7fELF")
    needed = {
        "saccade_track": ["libown.so", "libtorch.so", "libc.so.6", "libcuda.so.1"],
        "libown.so": ["libmissing.so.2"],
        "libtorch.so": ["libtorch_python.so"],
    }

    def run(cmd: list[str]) -> str:
        name = Path(cmd[-1]).name
        lines = [
            f" 0x1 (NEEDED)             Shared library: [{n}]"
            for n in needed.get(name, [])
        ]
        if name == "saccade_track":
            lines.append(
                " 0x1d (RUNPATH)            Library runpath: [$ORIGIN/../lib:/nonexistent]"
            )
        return "\n".join(lines)

    clo, problems = g2.closure([track], deps, run)
    by = {c["name"]: c for c in clo}
    assert by["libown.so"]["path"] == str(tree / "lib" / "libown.so")
    assert by["libtorch.so"]["class"] == "package"
    assert by["libc.so.6"]["class"] == "base_system"
    assert by["libcuda.so.1"]["class"] == "driver"
    assert sorted(problems) == [
        "libown.so: NEEDED libmissing.so.2 is not resolvable",
        "libtorch.so needs libtorch_python.so",
        "libtorch.so: NEEDED libtorch_python.so is not resolvable",
    ]


# ── runtime ────────────────────────────────────────────────────────────────────

STRACE_OK = """\
execve("/opt/saccade/bin/saccade_track", ["/opt/saccade/bin/saccade_track", "--config"], 0x7ffd /* 9 vars */) = 0
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, "/opt/saccade-deps/libtorch.so", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, "/opt/saccade-deps/libmissing.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, "/usr/lib/x86_64-linux-gnu/libcuda.so.1", O_RDONLY|O_CLOEXEC) = 3
openat(AT_FDCWD, "/opt/saccade/share/saccade/build/libsaccade_scan_torchop.so", O_RDONLY|O_CLOEXEC) = 4
openat(AT_FDCWD, "/data/MOT17/train/MOT17-02-SDP/img1/000001.jpg", O_RDONLY) = 5
"""


def _runtime(tmp_path: Path, extra: str = "") -> dict[str, Any]:
    tree, _, _ = _tree(tmp_path)
    logs = tmp_path / "strace"
    logs.mkdir()
    (logs / "s.100").write_text(STRACE_OK)
    (logs / "s.101").write_text(extra)
    deps = tmp_path / "deps.json"
    deps.write_text(
        json.dumps({"entries": [{"names": ["libtorch.so"], "sha256": "t" * 64}]})
    )
    ref = tmp_path / "ref.json"
    ref.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "requested": ["libtorch.so"],
                        "realpath": "/v/libtorch.so",
                        "sha256": "t" * 64,
                    },
                    {
                        "requested": ["libc.so.6"],
                        "realpath": "/usr/lib/libc.so.6",
                        "sha256": "c" * 64,
                    },
                    {
                        "requested": [f"./{OP_REL}"],
                        "realpath": "/r/op.so",
                        "sha256": _sha(b"\x7fELF-op"),
                    },
                ]
            }
        )
    )
    report = tmp_path / "runtime.json"
    rc = g2.main(
        [
            "runtime", "--strace-prefix", str(logs / "s"), "--tree", str(tree),
            "--tree-mount", "/opt/saccade", "--deps-manifest", str(deps),
            "--deps-mount", "/opt/saccade-deps", "--reference", str(ref), "--report", str(report),
        ]
    )  # fmt: skip
    out: dict[str, Any] = json.loads(report.read_text())
    assert rc == (0 if out["pass"] else 1)
    return out


def test_runtime_pass(tmp_path: Path) -> None:
    out = _runtime(tmp_path)
    assert out["pass"], out["checks"]


@pytest.mark.parametrize(
    ("extra", "check"),
    [
        (
            'execve("/usr/bin/python3", ["python3"], 0x1 /* 9 vars */) = -1 ENOENT (No such file)\n',
            "g2_2_g2_4_single_exec",
        ),
        (
            'execve("/bin/sh", ["sh", "-c", "cc x.c"], 0x1 /* 9 vars */) = 0\n',
            "g2_2_g2_4_single_exec",
        ),
        (
            'openat(AT_FDCWD, "/opt/saccade-deps/libpython3.12.so.1.0", O_RDONLY) = -1 ENOENT (x)\n',
            "g2_2_g2_4_no_python_triton_open",
        ),
        (
            'openat(AT_FDCWD, "/tmp/torchinductor_x/ab.py", O_RDONLY) = 7\n',
            "g2_2_g2_4_no_python_triton_open",
        ),
        (
            'openat(AT_FDCWD, "/home/u/.triton/cache/x", O_RDONLY) = -1 ENOENT (x)\n',
            "g2_2_g2_4_no_python_triton_open",
        ),
        (
            'openat(AT_FDCWD, "/usr/lib/x86_64-linux-gnu/libfoo.so.3", O_RDONLY) = 7\n',
            "loaded_set_equals_reference",
        ),
    ],
)
def test_runtime_rejects(tmp_path: Path, extra: str, check: str) -> None:
    out = _runtime(tmp_path, extra)
    assert not out["pass"]
    assert [k for k, c in out["checks"].items() if not c["pass"]] == [check]


def test_runtime_rejects_a_missing_reference_object(tmp_path: Path) -> None:
    out = _runtime(tmp_path)
    ref = json.loads((tmp_path / "ref.json").read_text())
    ref["objects"].append(
        {
            "requested": ["libcudnn.so.9"],
            "realpath": "/v/libcudnn.so.9",
            "sha256": "d" * 64,
        }
    )
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    report = tmp_path / "runtime2.json"
    args = [
        "runtime", "--strace-prefix", str(tmp_path / "strace" / "s"), "--tree", str(tmp_path / "tree"),
        "--tree-mount", "/opt/saccade", "--deps-manifest", str(tmp_path / "deps.json"),
        "--deps-mount", "/opt/saccade-deps", "--reference", str(tmp_path / "ref.json"), "--report", str(report),
    ]  # fmt: skip
    assert out["pass"] and g2.main(args) == 1
    assert json.loads(report.read_text())["checks"]["loaded_set_equals_reference"][
        "missing"
    ] == ["d" * 64]


# ── model-root install ─────────────────────────────────────────────────────────


def _fake_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    files = {
        "models/yolo/head.pt": b"head",
        "models/yolo/backbone.engine": b"engine",
        "build/libop.so": b"op",
        "configs/shipping/config.json": b"{}",
    }
    for rel, data in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    lineage = {
        "torchscript": {"path": "models/yolo/head.pt", "sha256": _sha(b"head")},
        "companions": {
            "backbone_engine": {
                "path": "models/yolo/backbone.engine",
                "sha256": _sha(b"engine"),
            }
        },
        "op_library": {"path": "build/libop.so", "sha256": "f" * 64},
    }
    (repo / "models/yolo/lineage.json").write_text(json.dumps(lineage))
    att = {
        "frozen_lineage": {
            "path": "models/yolo/lineage.json",
            "sha256": _sha((repo / "models/yolo/lineage.json").read_bytes()),
        },
        "op_library": {"path": "build/libop.so", "sha256": _sha(b"op")},
    }
    (repo / "configs/shipping/att.json").write_text(json.dumps(att))
    defs = {
        "SACCADE_REPO_ROOT": str(repo),
        "SACCADE_MODEL_ROOT_DEST": str(tmp_path / "tree" / "share" / "saccade"),
        "SACCADE_SHIPPING_CONFIG": "configs/shipping/config.json",
        "SACCADE_SHIPPING_LINEAGE": "models/yolo/lineage.json",
        "SACCADE_SHIPPING_ATTESTATION": "configs/shipping/att.json",
        "SACCADE_ATTESTED_OP_LIBRARY": str(repo / "build/libop.so"),
    }
    return repo, defs


def _install(defs: dict[str, str]) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")
    return subprocess.run(
        [cmake, *(f"-D{k}={v}" for k, v in defs.items()), "-P", str(INSTALL_SCRIPT)],
        capture_output=True,
        text=True,
    )


def test_install_model_root_copies_the_bound_files(tmp_path: Path) -> None:
    _, defs = _fake_repo(tmp_path)
    r = _install(defs)
    assert r.returncode == 0, r.stderr
    dest = Path(defs["SACCADE_MODEL_ROOT_DEST"])
    assert sorted(
        p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()
    ) == [
        "build/libop.so",
        "configs/shipping/att.json",
        "configs/shipping/config.json",
        "models/yolo/backbone.engine",
        "models/yolo/head.pt",
        "models/yolo/lineage.json",
    ]


@pytest.mark.parametrize(
    "victim", ["models/yolo/head.pt", "models/yolo/backbone.engine", "build/libop.so"]
)
def test_install_model_root_refuses_a_changed_file(tmp_path: Path, victim: str) -> None:
    repo, defs = _fake_repo(tmp_path)
    (repo / victim).write_bytes(b"rebuilt")
    r = _install(defs)
    assert r.returncode != 0
    assert "sha256" in r.stderr and victim in r.stderr


def test_install_model_root_refuses_another_lineage(tmp_path: Path) -> None:
    repo, defs = _fake_repo(tmp_path)
    lineage = repo / "models/yolo/lineage.json"
    lineage.write_text(lineage.read_text() + " ")
    r = _install(defs)
    assert r.returncode != 0 and "lineage.json" in r.stderr
