"""The H2 firewall must be a partition, and it must fail closed.

Layer-P retryability is only sound if retries cannot reach what the measurement
measures. Two properties carry that:

  * classification is **total and mutually exclusive** — no path lands in two
    classes, and every path lands in exactly one (including `unclassified`);
  * retry admissibility **fails closed** — an unclassified path is rejected, not
    waved through as "probably harmless". Asserting a file harmless without
    classifying it is precisely the move H0's enumerative admission kept
    getting wrong (Amendment 6 Correction 1).

The `decision_relevant` core is also checked against its upstream source rather
than trusted as a transcription: H0's `ADMITTED_RUNTIME_PATHS` is the authority
for the five A6.2 trace paths, and a drift between the two lists would silently
shrink the protected set.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_path_partition as partition  # noqa: E402
import verify_h0_preseal_freeze as h0_freeze  # noqa: E402

_CLASSES = {
    "decision_relevant",
    "identity_semantics",
    "identity_fixture_input",
    "measurement_input",
    "runtime_asset",
    "plumbing_only",
    "non_execution",
    "unclassified",
}


def test_the_five_admitted_runtime_paths_are_all_decision_relevant() -> None:
    """H0's A6.2 set is the authority; H2 may extend it, never shrink it."""
    for path in h0_freeze.ADMITTED_RUNTIME_PATHS:
        assert partition.classify(path) == "decision_relevant", (
            f"{path} is one of H0's admitted runtime paths but H2 classifies it "
            f"{partition.classify(path)!r} — the protected set shrank."
        )


def test_classification_is_total() -> None:
    assert partition.classify("no/such/file/anywhere.bin") == "unclassified"
    for path in ("", ".", "src", "src/"):
        assert partition.classify(path) in _CLASSES


def test_prose_is_non_execution_wherever_it_lives() -> None:
    """Docs never execute, so they never block a retry — but prose inside a
    protected prefix must stay protected rather than become freely editable."""
    assert partition.classify("no/such/place/notes.md") == "non_execution"
    assert partition.classify("scripts/tools/README.md") == "non_execution"
    assert partition.classify("scripts/eval/config/README.md") == "decision_relevant"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/tracking/tracker_gpu.cu", "decision_relevant"),
        ("include/tracking/tracker_gpu.hpp", "decision_relevant"),
        ("src/saccade/perception/eval/pipeline.py", "decision_relevant"),
        ("src/saccade/perception/eval/evaluator.py", "decision_relevant"),
        ("configs/presets/mamba_whole_graph_m.yaml", "decision_relevant"),
        ("scripts/eval/config/geometry.py", "decision_relevant"),
        ("scripts/eval/mot17_args.py", "decision_relevant"),
        ("CMakeLists.txt", "plumbing_only"),
        ("uv.lock", "plumbing_only"),
        ("scripts/tools/h0_runtime_confinement.py", "plumbing_only"),
        ("scripts/tools/run_h2_layer_p.py", "identity_semantics"),
        (".github/workflows/ci.yml", "plumbing_only"),
        (".github/workflows/runtime_identity.yml", "identity_semantics"),
        ("scripts/tools/h2_path_partition.py", "identity_semantics"),
        ("scripts/tools/h2_behavioral_identity.py", "identity_semantics"),
        ("scripts/tools/h2_runtime_inputs.py", "identity_semantics"),
        ("scripts/tools/h2_terminal_partition.py", "identity_semantics"),
        ("scripts/tools/h2_run_spec.py", "identity_semantics"),
        ("scripts/tools/build_runtime_identity.py", "identity_semantics"),
        (
            "docs/research/contracts/h2_phase_a_run_spec_v1.json",
            "identity_semantics",
        ),
        (
            "docs/research/contracts/h2_runtime_binding_v1.json",
            "identity_semantics",
        ),
        (
            "docs/research/contracts/h2_execution_result_v1.json",
            "identity_semantics",
        ),
        (
            "docs/research/contracts/h2_execution_verification_v1.json",
            "identity_semantics",
        ),
        (
            "docs/research/contracts/h2_phase_a_authoring_profile_v1.json",
            "identity_semantics",
        ),
        (
            "docs/research/contracts/h2_phase_a_authoring_profile_v1.schema.json",
            "identity_semantics",
        ),
        (
            "docs/research/contracts/h2_phase_a_run_spec_authoring_decision_v1.json",
            "identity_semantics",
        ),
        ("docs/modules/semantic/TODO.md", "non_execution"),
        ("tests/contract/test_h2_path_partition.py", "non_execution"),
        (
            "datasets/MOT17/train/MOT17-09-SDP/seqinfo.ini",
            "identity_fixture_input",
        ),
        ("datasets/MOT17/train/MOT17-04-SDP/seqinfo.ini", "measurement_input"),
        ("models/yolo/yolo26m.pt", "runtime_asset"),
        ("third_party/plugin.so", "runtime_asset"),
        ("src/saccade/perception/reid/embedder.py", "decision_relevant"),
        ("vendor/unknown.bin", "unclassified"),
    ],
)
def test_representative_classifications(path: str, expected: str) -> None:
    assert partition.classify(path) == expected


def test_classes_are_mutually_exclusive() -> None:
    """No literal path may appear in two class tables."""
    tables = {
        "decision_relevant": set(partition.DECISION_RELEVANT_PATHS),
        "identity_semantics": set(partition.IDENTITY_SEMANTICS_PATHS),
        "plumbing_only": set(partition.PLUMBING_ONLY_PATHS),
        "non_execution": set(partition.NON_EXECUTION_PATHS),
    }
    for left, left_paths in tables.items():
        for right, right_paths in tables.items():
            if left < right:
                assert not left_paths & right_paths, (left, right)

    # A prefix rule must not swallow an exact path of another class.
    for expected, paths in tables.items():
        for path in paths:
            assert partition.classify(path) == expected, path


def test_the_guard_cannot_be_edited_by_the_retry_it_guards() -> None:
    """The digest producer and this classifier live under `scripts/tools/h2_`,
    which is otherwise a plumbing prefix. The exact-path rule must win, or a
    retry could weaken the very check that makes retries sound."""
    for path in partition.IDENTITY_SEMANTICS_PATHS:
        assert partition.classify(path) == "identity_semantics", path

    verdict = partition.check_retry(
        ["CMakeLists.txt", "scripts/tools/h2_behavioral_identity.py"]
    )
    assert not verdict.admissible
    assert verdict.identity_semantics == ("scripts/tools/h2_behavioral_identity.py",)
    assert "cannot edit itself" in verdict.reason()


def test_successor_artifact_authority_cannot_hide_under_broad_prefixes() -> None:
    """Schemas under docs/ and the future h2_* resolver are ruler members."""
    expected = {
        *partition.EXECUTION_ARTIFACT_SCHEMA_PATHS,
        *partition.RUN_SPEC_AUTHORING_PATHS,
        partition.RUN_SPEC_RESOLVER_PATH,
    }
    assert expected <= partition.IDENTITY_SEMANTICS_PATHS
    for path in expected:
        assert partition.classify(path) == "identity_semantics"
        verdict = partition.check_retry([path])
        assert not verdict.admissible
        assert verdict.identity_semantics == (path,)


def test_prefix_rules_do_not_overlap_across_classes() -> None:
    groups = {
        "decision_relevant": partition.DECISION_RELEVANT_PREFIXES,
        "plumbing_only": partition.PLUMBING_ONLY_PREFIXES,
        "non_execution": partition.NON_EXECUTION_PREFIXES,
        "runtime_asset": partition.RUNTIME_ASSET_PREFIXES,
    }
    for left_name, left in groups.items():
        for right_name, right in groups.items():
            if left_name >= right_name:
                continue
            for a in left:
                for b in right:
                    assert not (a.startswith(b) or b.startswith(a)), (
                        f"prefix {a!r} ({left_name}) and {b!r} ({right_name}) overlap: "
                        f"classification would depend on rule order"
                    )


# --------------------------------------------------------------------------- #
# Retry admissibility                                                          #
# --------------------------------------------------------------------------- #
def test_a_plumbing_only_retry_is_admissible() -> None:
    verdict = partition.check_retry(
        ["CMakeLists.txt", "scripts/tools/h0_runtime_confinement.py"]
    )
    assert verdict.admissible
    assert not verdict.decision_relevant and not verdict.unclassified


def test_docs_do_not_block_a_retry() -> None:
    assert partition.check_retry(["docs/modules/semantic/TODO.md"]).admissible


def test_a_decision_relevant_change_blocks_the_retry() -> None:
    verdict = partition.check_retry(["CMakeLists.txt", "src/tracking/tracker_gpu.cu"])
    assert not verdict.admissible
    assert verdict.decision_relevant == ("src/tracking/tracker_gpu.cu",)
    assert "decision-relevant" in verdict.reason()


def test_an_unclassified_change_fails_closed() -> None:
    """The whole point: silence is not permission."""
    verdict = partition.check_retry(["vendor/unknown.bin"])
    assert not verdict.admissible
    assert verdict.unclassified == ("vendor/unknown.bin",)
    assert "fail-closed" in verdict.reason()


def test_a_blocked_path_is_reported_even_when_mixed_with_admissible_ones() -> None:
    verdict = partition.check_retry(
        [
            "uv.lock",
            "docs/TODO.md",
            "src/saccade/perception/eval/stages.py",
            "vendor/unknown.bin",
        ]
    )
    assert not verdict.admissible
    assert verdict.decision_relevant == ("src/saccade/perception/eval/stages.py",)
    assert verdict.unclassified == ("vendor/unknown.bin",)
    assert verdict.plumbing_only == ("uv.lock",)
    assert verdict.non_execution == ("docs/TODO.md",)


def test_absolute_paths_inside_the_repo_normalize() -> None:
    absolute = (_REPO / "src/tracking/tracker_gpu.cu").as_posix()
    assert partition.classify(absolute) == "decision_relevant"


def test_cli_exit_codes_match_the_verdict() -> None:
    assert partition.main(["--check-retry", "CMakeLists.txt"]) == 0
    assert partition.main(["--check-retry", "src/tracking/tracker_gpu.cu"]) == 1


@pytest.mark.parametrize(
    "path,field",
    [
        (
            "datasets/MOT17/train/MOT17-09-SDP/seqinfo.ini",
            "identity_fixture_input",
        ),
        (
            "datasets/MOT17/train/MOT17-04-SDP/seqinfo.ini",
            "measurement_input",
        ),
        ("models/yolo/yolo26m.pt", "runtime_asset"),
    ],
)
def test_consumed_inputs_and_assets_block_retry(path: str, field: str) -> None:
    verdict = partition.check_retry([path])
    assert not verdict.admissible
    assert getattr(verdict, field) == (path,)
