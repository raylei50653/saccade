import importlib.util
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "scripts" / "eval"

COMPAT_WRAPPERS = {
    "analyze_assoc_fn.py": "diagnostics/analyze_assoc_fn.py",
    "analyze_external_fp_rows.py": "appearance/analyze_external_fp_rows.py",
    "analyze_fn.py": "diagnostics/analyze_fn.py",
    "analyze_near_miss_final_output.py": "diagnostics/analyze_near_miss_final_output.py",
    "analyze_near_miss_offsets.py": "diagnostics/analyze_near_miss_offsets.py",
    "analyze_near_miss_stage_attribution.py": (
        "diagnostics/analyze_near_miss_stage_attribution.py"
    ),
    "analyze_score_distribution.py": "detector/analyze_score_distribution.py",
    "export_external_fp_rows.py": "appearance/export_external_fp_rows.py",
    "label_boosted_birth_rows.py": "diagnostics/label_boosted_birth_rows.py",
    "mamba_size_binned_recall.py": "detector/mamba_size_binned_recall.py",
    "mot17_public.py": "baselines/mot17_public.py",
    "oracle_height_birth_ceiling.py": "experiments/oracle_height_birth_ceiling.py",
    "oracle_small_birth_ceiling.py": "experiments/oracle_small_birth_ceiling.py",
    "train_external_fp_classifier.py": "appearance/train_external_fp_classifier.py",
    "ultralytics_official_mot17.py": "baselines/ultralytics_official_mot17.py",
}

SHELL_COMPAT_WRAPPERS = {
    "run_threshold_strategies.sh": "experiments/run_threshold_strategies.sh",
}


def test_nested_eval_scripts_do_not_use_fixed_depth_repo_roots() -> None:
    offenders: list[str] = []
    for script in sorted(EVAL_DIR.glob("*/*.py")):
        text = script.read_text(encoding="utf-8")
        for pattern in ("parent.parent.parent", "parents[2]"):
            if pattern in text:
                offenders.append(f"{script.relative_to(REPO_ROOT)}: {pattern}")

    assert not offenders, (
        "nested scripts/eval entrypoints must find repo root by marker, not by "
        "fixed parent depth:\n" + "\n".join(offenders)
    )


def test_repo_marker_resolution_matches_workspace_root() -> None:
    scripts = sorted(EVAL_DIR.glob("*/*.py"))
    assert scripts

    for script in scripts:
        resolved = next(
            p
            for p in script.resolve().parents
            if (p / "pyproject.toml").exists() and (p / "src" / "saccade").is_dir()
        )
        assert resolved == REPO_ROOT


def test_nested_eval_scripts_with_root_local_imports_add_eval_dir() -> None:
    offenders: list[str] = []
    for script in sorted(EVAL_DIR.glob("*/*.py")):
        text = script.read_text(encoding="utf-8")
        has_root_local_import = "from analyze_score_distribution import" in text
        adds_eval_dir = 'project_root / "scripts" / "eval"' in text
        if has_root_local_import and not adds_eval_dir:
            offenders.append(str(script.relative_to(REPO_ROOT)))

    assert not offenders, (
        "nested scripts/eval entrypoints with unqualified eval-root imports must "
        "add scripts/eval to sys.path:\n" + "\n".join(offenders)
    )


def test_root_duplicate_python_entrypoints_are_compat_wrappers() -> None:
    for wrapper_name, target_name in COMPAT_WRAPPERS.items():
        wrapper = EVAL_DIR / wrapper_name
        target = EVAL_DIR / target_name
        text = wrapper.read_text(encoding="utf-8")

        assert target.exists(), target
        assert "run_eval_script" in text
        assert f'run_eval_script("{target_name}")' in text


def test_root_duplicate_shell_entrypoints_are_compat_wrappers() -> None:
    for wrapper_name, target_name in SHELL_COMPAT_WRAPPERS.items():
        wrapper = EVAL_DIR / wrapper_name
        target = EVAL_DIR / target_name
        text = wrapper.read_text(encoding="utf-8")

        assert target.exists(), target
        assert f'exec "$ROOT/scripts/eval/{target_name}" "$@"' in text


def test_redirect_helper_sets_and_restores_argv0(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "eval_redirect_for_test", EVAL_DIR / "_redirect.py"
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    original_argv0 = sys.argv[0]
    seen: dict[str, str] = {}

    def _fake_run_path(path: str, *, run_name: str) -> None:
        seen["path"] = path
        seen["run_name"] = run_name
        seen["argv0"] = sys.argv[0]

    monkeypatch.setattr(runpy, "run_path", _fake_run_path)

    module.run_eval_script("diagnostics/analyze_fn.py")

    target = EVAL_DIR / "diagnostics" / "analyze_fn.py"
    assert seen == {
        "path": str(target),
        "run_name": "__main__",
        "argv0": str(target),
    }
    assert sys.argv[0] == original_argv0
