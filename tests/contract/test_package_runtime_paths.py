"""The package's runtime paths are a contract, not a guess about repository layout.

``saccade.paths`` is the one place where the package turns "where am I" into a
filesystem location. Everything else addresses its own package-owned files
relative to its own ``__file__``, takes external inputs as explicit arguments,
and asks ``saccade.paths`` for repository-provided things (native build
products, the vendored TrackEval, the checkout's sources). This suite pins:

  * **no module reaches above the package** -- ``Path(__file__).parents[4]``
    and its ``.parent.parent.parent`` spelling were how six modules each
    re-derived the repository root; none may come back;
  * **explicit input beats inference** -- ``SACCADE_BUILD_PATH`` and
    ``SACCADE_TRACKEVAL_ROOT`` win over the checkout, and an explicit value
    that points nowhere is surfaced, not papered over;
  * **outside a checkout the answer is None, never a directory four levels
    up** -- the same source loaded from a site-packages-shaped tree yields no
    checkout, no build directory, no TrackEval, and a named error from
    operations that need the repository;
  * **the working directory is not a root** -- resolving from a foreign cwd
    gives the same answers as from the repository root, and relative external
    inputs resolve against the cwd the way a CLI flag does.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import saccade.paths as paths

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "saccade"
RESOLVER = PACKAGE_ROOT / "paths.py"


# ── Static inventory: layout assumptions may live only in the resolver ───────


def _package_sources() -> list[Path]:
    files = sorted(
        p
        for p in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and p != RESOLVER
    )
    assert files, "no package sources found"
    return files


def _strip_comments_and_docstrings(text: str) -> str:
    # Enough for a forbidden-token scan: drop `#` comments and triple-quoted
    # blocks so prose about the old pattern cannot trip the check.
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


# Each pattern is a way of turning a module's location into somewhere outside
# the package, or of treating the process cwd as a repository root.
FORBIDDEN = {
    r"\.parents\[": "walking up from __file__ with parents[N]",
    r"\.parents\s*:": "iterating Path(...).parents to find a repository marker",
    r"\.parent\.parent": "chained .parent hops above the module directory",
    r"Path\.cwd\(\)": "cwd as an implicit root (use saccade.paths.runtime_input)",
    r"os\.getcwd\(\)": "cwd as an implicit root (use saccade.paths.runtime_input)",
    r"SACCADE_BUILD_PATH": "reading the build-path variable outside saccade.paths",
    r"third_party": "hard-coded vendored-tree location outside saccade.paths",
}

# The only permitted shapes for __file__ outside the resolver: the module's own
# bytes, or its own directory (package-owned resources next to it).
ALLOWED_FILE_USE = re.compile(r"Path\(__file__\)\.resolve\(\)(\.parent)?(?![\w.\[])")


def test_layout_assumptions_live_only_in_the_resolver() -> None:
    offences: list[str] = []
    for path in _package_sources():
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for pattern, why in FORBIDDEN.items():
            for match in re.finditer(pattern, code):
                line = code.count("\n", 0, match.start()) + 1
                offences.append(f"{rel}:{line}: {why}")
        for match in re.finditer(r"__file__", code):
            window = code[max(0, match.start() - 5) : match.end() + 40]
            if not ALLOWED_FILE_USE.search(window):
                line = code.count("\n", 0, match.start()) + 1
                offences.append(
                    f"{rel}:{line}: __file__ used other than as Path(__file__)"
                    ".resolve()[.parent]"
                )
    assert not offences, "\n".join(offences)


def test_resolver_is_thin_and_import_light() -> None:
    code = _strip_comments_and_docstrings(RESOLVER.read_text(encoding="utf-8"))
    imports = {
        m.group(1)
        for m in re.finditer(r"^(?:from|import)\s+([\w.]+)", code, re.MULTILINE)
    }
    assert imports <= {"__future__", "os", "pathlib"}, imports


# ── Behaviour inside the checkout ────────────────────────────────────────────


def test_checkout_root_is_this_repository() -> None:
    assert paths.source_checkout_root() == REPO_ROOT
    assert paths.require_source_checkout("test") == REPO_ROOT


def test_build_dir_prefers_the_explicit_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(paths.BUILD_PATH_ENV, raising=False)
    assert paths.build_dir() == REPO_ROOT / "build"

    # Explicit wins even when it does not exist: a wrong pointer must be seen.
    missing = tmp_path / "no-such-build"
    monkeypatch.setenv(paths.BUILD_PATH_ENV, str(missing))
    assert paths.build_dir() == missing.resolve()

    # Blank is "unset", not "the current directory".
    monkeypatch.setenv(paths.BUILD_PATH_ENV, "   ")
    assert paths.build_dir() == REPO_ROOT / "build"


def test_trackeval_root_prefers_the_explicit_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(paths.TRACKEVAL_ROOT_ENV, raising=False)
    vendored = REPO_ROOT / "third_party" / "TrackEval"
    assert paths.trackeval_root() == vendored
    assert (vendored / "trackeval").is_dir()

    explicit = tmp_path / "TrackEval"
    (explicit / "trackeval").mkdir(parents=True)
    monkeypatch.setenv(paths.TRACKEVAL_ROOT_ENV, str(explicit))
    assert paths.trackeval_root() == explicit.resolve()

    monkeypatch.setenv(paths.TRACKEVAL_ROOT_ENV, str(tmp_path / "empty"))
    with pytest.raises(ValueError, match=paths.TRACKEVAL_ROOT_ENV):
        paths.trackeval_root()


def test_runtime_input_is_cwd_relative_like_a_cli_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        paths.runtime_input("models/yolo/x.engine")
        == (tmp_path / "models/yolo/x.engine").resolve()
    )
    absolute = tmp_path / "abs.engine"
    assert paths.runtime_input(str(absolute)) == absolute.resolve()


# ── Behaviour outside a checkout ─────────────────────────────────────────────


def _load_resolver_from(tree: Path) -> ModuleType:
    """Load the resolver's bytes from *tree*/saccade/paths.py as a fresh module."""
    target = tree / "saccade" / "paths.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(RESOLVER, target)
    spec = importlib.util.spec_from_file_location(f"_saccade_paths_{tree.name}", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_copy_has_no_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(paths.BUILD_PATH_ENV, raising=False)
    monkeypatch.delenv(paths.TRACKEVAL_ROOT_ENV, raising=False)
    site = tmp_path / "site-packages"
    module = _load_resolver_from(site)

    assert module.source_checkout_root() is None
    assert module.build_dir() is None
    assert module.trackeval_root() is None
    with pytest.raises(module.SourceCheckoutRequired, match="decimal-hash"):
        module.require_source_checkout("decimal-hash capture")

    # Explicit inputs still work without a checkout.
    build = tmp_path / "build"
    monkeypatch.setenv(paths.BUILD_PATH_ENV, str(build))
    assert module.build_dir() == build.resolve()


def test_src_layout_without_pyproject_is_not_a_checkout(tmp_path: Path) -> None:
    # A bare src/saccade with nothing beside src/ is not a repository either.
    module = _load_resolver_from(tmp_path / "src")
    assert module.source_checkout_root() is None
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert module.source_checkout_root() == tmp_path.resolve()


# ── Independence from the working directory ──────────────────────────────────

_FOREIGN_CWD_PROBE = r"""
import json, os, sys
import saccade.paths as p
from saccade.perception.eval import metrics, consumer_a_bridge_fidelity as cab
from saccade.perception.eval import _decimal_hash_tools as dh
print(json.dumps({
    "cwd": os.getcwd(),
    "checkout": str(p.source_checkout_root()),
    "build": str(p.build_dir()),
    "trackeval": str(p.trackeval_root()),
    "tracker_cu": str(cab._production_tracker_path()),
    "git_commit": dh._git_commit(),
}))
"""


def test_answers_do_not_depend_on_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(paths.BUILD_PATH_ENV, raising=False)
    monkeypatch.delenv(paths.TRACKEVAL_ROOT_ENV, raising=False)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {paths.BUILD_PATH_ENV, paths.TRACKEVAL_ROOT_ENV}
    }
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", _FOREIGN_CWD_PROBE],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    answers = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(answers["cwd"]).resolve() == tmp_path.resolve()
    assert answers["checkout"] == str(REPO_ROOT)
    assert answers["build"] == str(REPO_ROOT / "build")
    assert answers["trackeval"] == str(REPO_ROOT / "third_party" / "TrackEval")
    assert answers["tracker_cu"] == str(REPO_ROOT / "src/tracking/tracker_gpu.cu")
    assert re.fullmatch(r"[0-9a-f]{40}", answers["git_commit"] or "")
