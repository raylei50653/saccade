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
* ``shipping/cmake/install_model_root.cmake`` installs validated text snapshots,
  hashes copied binaries before publishing them, refuses source replacements
  that would install unvalidated bytes (#585), and an attestation whose operator
  library is not the runtime's N03 pin (#549 S2-2, P-N03 rev 3d; skips without
  ``cmake``).
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
        "op_library": {"path": "build/libop.so", "sha256": _sha(b"op"), "bytes": 2},
    }
    (repo / "configs/shipping/att.json").write_text(json.dumps(att))
    (repo / "shipping").mkdir()
    (repo / "shipping/n03_operator_pin.json").write_text(
        json.dumps({"sha256": _sha(b"op"), "bytes": 2})
    )
    defs = {
        "SACCADE_REPO_ROOT": str(repo),
        "SACCADE_MODEL_ROOT_DEST": str(tmp_path / "tree" / "share" / "saccade"),
        "SACCADE_SHIPPING_CONFIG": "configs/shipping/config.json",
        "SACCADE_SHIPPING_LINEAGE": "models/yolo/lineage.json",
        "SACCADE_SHIPPING_ATTESTATION": "configs/shipping/att.json",
        "SACCADE_ATTESTED_OP_LIBRARY": str(repo / "build/libop.so"),
        "SACCADE_N03_OPERATOR_PIN": str(repo / "shipping/n03_operator_pin.json"),
        "SACCADE_N03_BUILT_PIN_SHA256": _sha(b"op"),
        "SACCADE_N03_BUILT_PIN_BYTES": "2",
    }
    return repo, defs


def _install(
    defs: dict[str, str], script: Path = INSTALL_SCRIPT
) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")
    return subprocess.run(
        [cmake, *(f"-D{k}={v}" for k, v in defs.items()), "-P", str(script)],
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
    for installed in dest.rglob("*"):
        if installed.is_file():
            assert (
                installed.read_bytes()
                == (
                    Path(defs["SACCADE_REPO_ROOT"]) / installed.relative_to(dest)
                ).read_bytes()
            )


def test_install_model_root_cleans_snapshots_when_a_destination_directory_is_blocked(
    tmp_path: Path,
) -> None:
    repo, defs = _fake_repo(tmp_path)
    dest = Path(defs["SACCADE_MODEL_ROOT_DEST"])
    config = dest / defs["SACCADE_SHIPPING_CONFIG"]
    config.parent.mkdir(parents=True)
    config.write_bytes(b"existing installed config")
    blocker = dest / "models"
    blocker.write_bytes(b"directory blocker")

    refused = _install(defs)
    assert refused.returncode != 0
    assert config.read_bytes() == b"existing installed config"
    assert blocker.read_bytes() == b"directory blocker"
    assert not list(dest.rglob(".saccade-install-*"))
    assert "cannot create" in refused.stderr
    assert not (dest / defs["SACCADE_SHIPPING_ATTESTATION"]).exists()

    blocker.unlink()
    retried = _install(defs)
    assert retried.returncode == 0, retried.stderr
    assert not list(dest.rglob(".saccade-install-*"))
    for installed in dest.rglob("*"):
        if installed.is_file():
            assert (
                installed.read_bytes()
                == (repo / installed.relative_to(dest)).read_bytes()
            )


def _install_with_source_replacement(
    tmp_path: Path,
    defs: dict[str, str],
    victim: str,
    timing: str,
    read_trigger: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Interpose on the real CMake file command at a deterministic I/O boundary.
    The install script itself has no test hook and all reads/copies/hashes
    still use CMake's built-in command (available as ``_file``)."""
    wrapper = tmp_path / "mutate_install.cmake"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replaced during install\n")
    marker = tmp_path / "mutation-fired"
    wrapper.write_text(
        """
function(_replace_source)
    _file(READ "${MUTATION_REPLACEMENT}" _replacement)
    _file(WRITE "${MUTATE_SOURCE}" "${_replacement}")
    _file(WRITE "${MUTATION_MARKER}" "${MUTATION_TIMING}")
endfunction()

function(file)
    if(ARGV0 STREQUAL "READ" OR ARGV0 STREQUAL "SHA256")
        _file("${ARGV0}" "${ARGV1}" "${ARGV2}")
        set("${ARGV2}" "${${ARGV2}}" PARENT_SCOPE)
        get_property(_read_snapshot GLOBAL PROPERTY MUTATION_SNAPSHOT)
        if(ARGV0 STREQUAL "READ" AND MUTATION_TIMING STREQUAL "after_read"
           AND (ARGV1 STREQUAL MUTATION_READ_TRIGGER OR ARGV1 STREQUAL _read_snapshot))
            _replace_source()
        endif()
    elseif(ARGV0 STREQUAL "COPY_FILE" OR ARGV0 STREQUAL "RENAME")
        if(ARGV0 STREQUAL "COPY_FILE" AND ARGV1 STREQUAL MUTATE_SOURCE
           AND MUTATION_TIMING STREQUAL "before_copy")
            _replace_source()
        endif()
        _file(${ARGV})
        if(ARGV0 STREQUAL "COPY_FILE" AND ARGV1 STREQUAL MUTATION_READ_TRIGGER)
            set_property(GLOBAL PROPERTY MUTATION_SNAPSHOT "${ARGV2}")
        endif()
        if(ARGV3 STREQUAL "RESULT")
            set("${ARGV4}" "${${ARGV4}}" PARENT_SCOPE)
        endif()
        if(ARGV0 STREQUAL "COPY_FILE" AND ARGV1 STREQUAL MUTATE_SOURCE
           AND MUTATION_TIMING STREQUAL "after_copy")
            _replace_source()
        endif()
    elseif(ARGV0 STREQUAL "WRITE")
        _file(WRITE "${ARGV1}" "${ARGV2}")
    else()
        _file(${ARGV})
    endif()
endfunction()

include("${INSTALL_SCRIPT}")
"""
    )
    r = _install(
        {
            **defs,
            "INSTALL_SCRIPT": str(INSTALL_SCRIPT),
            "MUTATE_SOURCE": str(Path(defs["SACCADE_REPO_ROOT"]) / victim),
            "MUTATION_TIMING": timing,
            "MUTATION_READ_TRIGGER": str(
                Path(defs["SACCADE_REPO_ROOT"]) / (read_trigger or victim)
            ),
            "MUTATION_REPLACEMENT": str(replacement),
            "MUTATION_MARKER": str(marker),
        },
        script=wrapper,
    )
    assert marker.exists(), (r.stdout, r.stderr)
    assert marker.read_text() == timing
    return r


@pytest.mark.parametrize(
    "victim",
    [
        "configs/shipping/config.json",
        "configs/shipping/att.json",
        "models/yolo/lineage.json",
    ],
)
def test_install_model_root_installs_the_validated_text_snapshot(
    tmp_path: Path, victim: str
) -> None:
    repo, defs = _fake_repo(tmp_path)
    source = repo / victim
    doc = json.loads(source.read_bytes())
    doc["note"] = "preserve ; ${literal}, 中文 and \\n"
    snapshot = json.dumps(doc, ensure_ascii=False).encode() + b"\r\n"
    source.write_bytes(snapshot)
    if victim == "models/yolo/lineage.json":
        att_path = repo / "configs/shipping/att.json"
        att = json.loads(att_path.read_bytes())
        att["frozen_lineage"]["sha256"] = _sha(snapshot)
        att_path.write_text(json.dumps(att))
    # Config has no content validator; replace it after the attestation read,
    # when the snapshot must already have been captured for this install.
    r = _install_with_source_replacement(
        tmp_path,
        defs,
        victim,
        "after_read",
        read_trigger="configs/shipping/att.json" if "config.json" in victim else victim,
    )
    assert r.returncode == 0, r.stderr
    assert source.read_bytes() != snapshot
    assert (Path(defs["SACCADE_MODEL_ROOT_DEST"]) / victim).read_bytes() == snapshot


@pytest.mark.parametrize(
    "victim", ["models/yolo/head.pt", "models/yolo/backbone.engine", "build/libop.so"]
)
@pytest.mark.parametrize("timing", ["before_copy", "after_copy"])
@pytest.mark.parametrize("existing", [False, True])
def test_install_model_root_checks_the_copied_bytes(
    tmp_path: Path, victim: str, timing: str, existing: bool
) -> None:
    repo, defs = _fake_repo(tmp_path)
    snapshot = (repo / victim).read_bytes()
    if existing:
        r = _install(defs)
        assert r.returncode == 0, r.stderr
    r = _install_with_source_replacement(tmp_path, defs, victim, timing)
    dest = Path(defs["SACCADE_MODEL_ROOT_DEST"])
    installed = dest / victim
    assert (repo / victim).read_bytes() != snapshot
    assert not list(dest.rglob(".saccade-install-*"))
    if timing == "before_copy":
        assert r.returncode != 0, r.stdout
        assert "sha256" in r.stderr and victim in r.stderr
        if existing:
            assert installed.read_bytes() == snapshot
        else:
            assert not installed.exists()
    else:
        assert r.returncode == 0, r.stderr
        assert installed.read_bytes() == snapshot


@pytest.mark.parametrize(
    "victim", ["models/yolo/head.pt", "models/yolo/backbone.engine", "build/libop.so"]
)
def test_install_model_root_refuses_a_changed_file(tmp_path: Path, victim: str) -> None:
    repo, defs = _fake_repo(tmp_path)
    (repo / victim).write_bytes(b"rebuilt")
    r = _install(defs)
    assert r.returncode != 0
    assert "sha256" in r.stderr and victim in r.stderr
    dest = Path(defs["SACCADE_MODEL_ROOT_DEST"])
    assert not (dest / victim).exists()
    assert not list(dest.rglob(".saccade-install-*"))


def test_install_model_root_refuses_another_lineage(tmp_path: Path) -> None:
    repo, defs = _fake_repo(tmp_path)
    lineage = repo / "models/yolo/lineage.json"
    lineage.write_text(lineage.read_text() + " ")
    r = _install(defs)
    assert r.returncode != 0 and "lineage.json" in r.stderr
    dest = Path(defs["SACCADE_MODEL_ROOT_DEST"])
    assert not (dest / "models/yolo/lineage.json").exists()
    assert not list(dest.rglob(".saccade-install-*"))


@pytest.mark.parametrize(
    "victim", ["models/yolo/lineage.json", "configs/shipping/att.json"]
)
@pytest.mark.parametrize("content", [b"{", b"{}"])
def test_install_model_root_removes_snapshots_when_metadata_is_invalid(
    tmp_path: Path, victim: str, content: bytes
) -> None:
    repo, defs = _fake_repo(tmp_path)
    (repo / victim).write_bytes(content)
    r = _install(defs)
    assert r.returncode != 0
    assert "invalid metadata" in r.stderr
    dest = Path(defs["SACCADE_MODEL_ROOT_DEST"])
    assert not list(dest.rglob(".saccade-install-*"))
    assert not any(p.is_file() for p in dest.rglob("*"))


@pytest.mark.parametrize(
    "victim",
    [
        "configs/shipping/config.json",
        "models/yolo/lineage.json",
        "configs/shipping/att.json",
    ],
)
def test_install_model_root_removes_snapshots_when_a_text_source_is_missing(
    tmp_path: Path, victim: str
) -> None:
    repo, defs = _fake_repo(tmp_path)
    (repo / victim).unlink()
    r = _install(defs)
    assert r.returncode != 0
    assert "does not exist" in r.stderr
    dest = Path(defs["SACCADE_MODEL_ROOT_DEST"])
    assert not list(dest.rglob(".saccade-install-*"))
    assert not any(p.is_file() for p in dest.rglob("*"))


@pytest.mark.parametrize(
    "op", [{"sha256": "e" * 64, "bytes": 2}, {"sha256": _sha(b"op"), "bytes": 3}]
)
def test_install_model_root_refuses_an_operator_that_is_not_the_n03_pin(
    tmp_path: Path, op: dict[str, Any]
) -> None:
    """The attestation's op_library (another digest or size) is not the pin
    this build compiled in (and recorded in shipping/n03_operator_pin.json)."""
    repo, defs = _fake_repo(tmp_path)
    att_path = repo / "configs/shipping/att.json"
    att = json.loads(att_path.read_text())
    att["op_library"].update(op)
    att_path.write_text(json.dumps(att))
    r = _install(defs)
    assert r.returncode != 0
    assert "is not the runtime's N03 pin" in r.stderr, r.stderr
    dest = Path(defs["SACCADE_MODEL_ROOT_DEST"])
    assert not (dest / "build/libop.so").exists()
    assert not list(dest.rglob(".saccade-install-*"))


@pytest.mark.parametrize("built", [("e" * 64, "2"), (_sha(b"op"), "3")])
def test_install_model_root_refuses_a_pin_changed_since_the_build(
    tmp_path: Path, built: tuple[str, str]
) -> None:
    """The source pin is no longer the one compiled into the runtime."""
    _, defs = _fake_repo(tmp_path)
    defs["SACCADE_N03_BUILT_PIN_SHA256"], defs["SACCADE_N03_BUILT_PIN_BYTES"] = built
    r = _install(defs)
    assert r.returncode != 0
    assert "rebuild before installing" in r.stderr, r.stderr
    assert not (Path(defs["SACCADE_MODEL_ROOT_DEST"]) / "build/libop.so").exists()


@pytest.mark.parametrize("carries", [True, False])
def test_install_model_root_requires_the_built_entrypoint_to_carry_the_pin(
    tmp_path: Path, carries: bool
) -> None:
    """This build's saccade_track (reconfigured but not rebuilt: another pin)."""
    _, defs = _fake_repo(tmp_path)
    exe = tmp_path / "saccade_track"
    exe.write_bytes(b"\x7fELF..." + (_sha(b"op") if carries else "d" * 64).encode())
    defs["SACCADE_N03_BUILT_ENTRYPOINT"] = str(exe)
    r = _install(defs)
    if carries:
        assert r.returncode == 0, r.stderr
    else:
        assert r.returncode != 0
        assert "does not carry the N03 pin" in r.stderr, r.stderr
        assert not (Path(defs["SACCADE_MODEL_ROOT_DEST"]) / "build/libop.so").exists()
