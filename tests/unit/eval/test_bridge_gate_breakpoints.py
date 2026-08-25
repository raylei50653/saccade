"""Unit tests for the bridge-gate breakpoint locator (scripts/eval/diagnostics)."""

# scope: eval
# function: behavior
# lifecycle: active

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = PROJECT_ROOT / "scripts/eval/diagnostics/bridge_gate_breakpoints.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("bridge_gate_breakpoints", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bridge_gate_breakpoints"] = module
    spec.loader.exec_module(module)
    return module


bp = _load_tool()


def _measurement(value: float, per_seq: dict[str, dict[str, int]]):
    return bp.Measurement(value, bp._pooled_idf1(per_seq), per_seq, "fake")


class _StepRunner:
    """Synthetic piecewise-constant metric with jumps at known locations.

    ``inclusive`` models the real gate, whose bounds are closed (the CUDA kernel
    rejects only outside ``[h_lo, h_hi]``): a threshold set exactly to an
    observed ratio already behaves like the values above it.
    """

    def __init__(self, jumps: list[float], inclusive: bool = False) -> None:
        self.jumps = jumps
        self.inclusive = inclusive
        self.runs = 0
        self.seen: dict[float, object] = {}

    def measure(self, value: float):
        self.runs += 1
        if self.inclusive:
            level = sum(1 for j in self.jumps if value >= j)
        else:
            level = sum(1 for j in self.jumps if value > j)
        per_seq = {"S1": {"idtp": 1000 - 10 * level, "idfp": 5, "idfn": 20}}
        m = _measurement(value, per_seq)
        self.seen[value] = m
        return m


def _args(*extra: str, lo: str = "1.2", hi: str = "1.8", work_dir: str = "unused"):
    return bp.parse_args(
        ["--axis", "h_hi", "--lo", lo, "--hi", hi, "--work-dir", work_dir, *extra]
    )


def test_pooled_idf1_pools_counts_instead_of_averaging() -> None:
    """IDF1 is a ratio, so per-sequence values must never be averaged."""
    per_seq = {
        "big": {"idtp": 1000, "idfp": 0, "idfn": 0},
        "tiny": {"idtp": 0, "idfp": 1, "idfn": 1},
    }
    pooled = bp._pooled_idf1(per_seq)
    naive_mean = sum(bp._idf1(c) for c in per_seq.values()) / len(per_seq)

    assert pooled == pytest.approx(100.0 * 2000 / (2000 + 1 + 1))
    assert pooled != pytest.approx(naive_mean)


def test_fingerprint_separates_runs_that_share_a_pooled_idf1() -> None:
    """The equality predicate is the count vector, not the pooled scalar.

    Two different decision outcomes can pool to exactly the same IDF1; using the
    scalar as the bisection predicate would silently merge them into one
    plateau.
    """
    a = {
        "S1": {"idtp": 100, "idfp": 10, "idfn": 10},
        "S2": {"idtp": 200, "idfp": 20, "idfn": 20},
    }
    b = {
        "S1": {"idtp": 200, "idfp": 20, "idfn": 20},
        "S2": {"idtp": 100, "idfp": 10, "idfn": 10},
    }

    left, right = _measurement(1.0, a), _measurement(2.0, b)
    assert left.idf1 == pytest.approx(right.idf1)
    assert left.fingerprint != right.fingerprint


def test_scan_brackets_only_intervals_that_actually_change() -> None:
    runner = _StepRunner([1.35])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)

    assert len(brackets) == 1
    assert brackets[0].low.value < 1.35 <= brackets[0].high.value


def test_bisect_localises_each_jump_to_the_tolerance() -> None:
    jumps = [1.3178, 1.6821]
    runner = _StepRunner(jumps)
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)
    assert len(brackets) == len(jumps)

    for bracket in brackets:
        bp.bisect(runner, bracket, 1e-4)
        contained = [j for j in jumps if bracket.low.value < j <= bracket.high.value]
        assert len(contained) == 1, "a refined bracket must isolate one jump"
        assert bracket.width <= 1e-4


def test_bisection_costs_far_less_than_an_equivalent_grid() -> None:
    """The whole point of bisection on a bit-exact metric: log, not linear."""
    runner = _StepRunner([1.3178, 1.6821])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    for bracket in bp.brackets_from_scan(samples):
        bp.bisect(runner, bracket, 1e-4)

    equivalent_grid = (1.8 - 1.2) / 1e-4
    assert runner.runs < equivalent_grid / 100


def test_report_has_one_more_plateau_than_breakpoints() -> None:
    runner = _StepRunner([1.3178, 1.6821])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)
    for bracket in brackets:
        bp.bisect(runner, bracket, 1e-4)

    args = bp.parse_args(
        ["--axis", "h_hi", "--lo", "1.2", "--hi", "1.8", "--work-dir", "unused"]
    )
    report = bp.build_report(args, list(runner.seen.values()), runner.runs)

    assert len(report["breakpoints"]) == 2
    assert len(report["plateaus"]) == 3
    assert [p["from"] for p in report["plateaus"]] == sorted(
        p["from"] for p in report["plateaus"]
    )


def test_report_states_that_a_plateau_is_not_a_proof() -> None:
    """The tool must never let a 'no jump detected' read as 'no jump exists'."""
    runner = _StepRunner([])
    bp.scan(runner, 1.2, 1.8, 3)
    args = bp.parse_args(
        ["--axis", "h_hi", "--lo", "1.2", "--hi", "1.8", "--work-dir", "unused"]
    )
    report = bp.build_report(args, list(runner.seen.values()), runner.runs)

    assert report["breakpoints"] == []
    assert any("do not prove" in limit for limit in report["limits"])


def test_plateau_never_contradicts_a_measured_point() -> None:
    """Regression: a plateau must not span a value that measured differently.

    The first version derived plateaus from the coarse scan grid only, so the
    bisection's own midpoints could sit inside a reported plateau while holding
    a different value. On real MOT17 data that produced a report claiming
    [1.616, 1.652] was flat at 80.892 while its own measurement at 1.616 said
    80.788 -- the tool asserting more than it had measured.
    """
    runner = _StepRunner([1.62, 1.64, 1.66])
    samples = bp.scan(runner, 1.6, 1.8, 5)
    for bracket in bp.brackets_from_scan(samples):
        bp.bisect(runner, bracket, 1e-3)

    args = bp.parse_args(
        ["--axis", "h_hi", "--lo", "1.6", "--hi", "1.8", "--work-dir", "unused"]
    )
    report = bp.build_report(args, list(runner.seen.values()), runner.runs)

    for plateau in report["plateaus"]:
        inside = [
            m
            for m in runner.seen.values()
            if plateau["from"] <= m.value <= plateau["to"]
        ]
        assert inside, "a plateau must be backed by measurements"
        for m in inside:
            assert m.idf1 == pytest.approx(plateau["idf1"]), (
                f"plateau {plateau['from']}..{plateau['to']} claims "
                f"{plateau['idf1']} but {m.value} measured {m.idf1}"
            )


def test_unrefined_jumps_are_flagged() -> None:
    """A jump wider than --tol may hide siblings and must say so."""
    runner = _StepRunner([1.62, 1.64, 1.66])
    samples = bp.scan(runner, 1.6, 1.8, 5)
    brackets = bp.brackets_from_scan(samples)
    for bracket in brackets:
        bp.bisect(runner, bracket, 1e-3)

    args = _args(lo="1.6", hi="1.8")
    report = bp.build_report(
        args, list(runner.seen.values()), runner.runs, brackets=brackets
    )

    assert any(not b["refined"] for b in report["breakpoints"])
    assert all(b["width"] <= args.tol for b in report["breakpoints"] if b["refined"])


# --------------------------------------------------------------------------
# why a jump is unrefined: reported, never presumed
# --------------------------------------------------------------------------


def test_unrefined_reason_names_multiple_jumps_only_when_that_is_the_cause() -> None:
    """A wide gap after a full bisection is the multiple-jump residual.

    The tool used to print that explanation for *every* unrefined jump, so
    --scan-only, an aborted run and a bracket nobody reached all read as "this
    bracket held several jumps" -- a claim about the data the run never made.
    """
    runner = _StepRunner([1.62, 1.64, 1.66])
    samples = bp.scan(runner, 1.6, 1.8, 5)
    brackets = bp.brackets_from_scan(samples)
    for bracket in brackets:
        bp.bisect(runner, bracket, 1e-3)

    args = _args(lo="1.6", hi="1.8")
    report = bp.build_report(
        args, list(runner.seen.values()), runner.runs, brackets=brackets
    )

    wide = [b for b in report["breakpoints"] if not b["refined"]]
    assert wide, "this fixture must leave a residual gap"
    assert all("more than one jump" in b["reason"] for b in wide)
    for b in report["breakpoints"]:
        if b["refined"]:
            assert "more than one jump" not in b["reason"]
            assert "--tol" in b["reason"]


def test_scan_only_says_bisection_was_never_attempted() -> None:
    runner = _StepRunner([1.35])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)

    args = _args("--scan-only")
    report = bp.build_report(
        args, list(runner.seen.values()), runner.runs, brackets=brackets
    )

    assert len(report["breakpoints"]) == 1
    reason = report["breakpoints"][0]["reason"]
    assert "--scan-only" in reason
    assert "more than one jump" not in reason


def test_aborted_run_says_it_never_reached_the_bracket() -> None:
    runner = _StepRunner([1.35, 1.75])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)
    assert len(brackets) == 2
    bp.bisect(runner, brackets[0], 1e-4)  # the second one never runs

    args = _args()
    report = bp.build_report(
        args,
        list(runner.seen.values()),
        runner.runs,
        brackets=brackets,
        aborted="eval failed (1); see run.log",
    )

    reasons = {b["reason"] for b in report["breakpoints"] if not b["refined"]}
    assert any("aborted" in r for r in reasons), reasons
    assert all("more than one jump" not in r for r in reasons)


def test_bisected_jump_reports_how_it_was_narrowed() -> None:
    runner = _StepRunner([1.35])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)
    bp.bisect(runner, brackets[0], 1e-4)

    args = _args("--tol", "1e-4")
    report = bp.build_report(
        args, list(runner.seen.values()), runner.runs, brackets=brackets
    )

    refined = [b for b in report["breakpoints"] if b["refined"]]
    assert len(refined) == 1
    assert refined[0]["reason"] == "bisected to <= --tol"


# --------------------------------------------------------------------------
# the gate is closed on both bounds
# --------------------------------------------------------------------------


def test_documented_gate_contract_matches_the_cuda_kernel() -> None:
    """The docstring said ``h_lo < ratio < h_hi``; the kernel disagrees.

    tracker_gpu.cu rejects only outside the band, so both bounds are inclusive.
    A tool that documents a strict gate mis-describes which side of a breakpoint
    a threshold sitting exactly on an observed ratio lands on.
    """
    kernel = (PROJECT_ROOT / "src/tracking/tracker_gpu.cu").read_text()
    assert "hr < bridge_h_lo || hr > bridge_h_hi" in kernel, (
        "the gate's comparison moved; re-check the documented contract"
    )

    doc = TOOL_PATH.read_text()
    assert "h_lo <= ratio <= h_hi" in doc
    assert "h_lo < ratio < h_hi" not in doc


def test_closed_gate_boundary_belongs_to_the_upper_plateau() -> None:
    """With a closed gate, the jump happens *at* the observed ratio.

    The reported bracket is half-open on the left, ``(lower, upper]``, so the
    breakpoint value itself must be the upper bound -- not an interior point of
    a plateau claimed to be flat.
    """
    jump = 1.5  # sits exactly on a scan grid point of this range
    runner = _StepRunner([jump], inclusive=True)
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)
    assert len(brackets) == 1
    assert brackets[0].high.value == pytest.approx(jump)

    bp.bisect(runner, brackets[0], 1e-4)
    assert brackets[0].low.value < jump <= brackets[0].high.value

    args = _args("--tol", "1e-4")
    report = bp.build_report(
        args, list(runner.seen.values()), runner.runs, brackets=brackets
    )
    at_boundary = [m for m in runner.seen.values() if m.value == pytest.approx(jump)]
    assert at_boundary
    upper = [p for p in report["plateaus"] if p["from"] <= jump <= p["to"]][0]
    assert upper["from"] == pytest.approx(jump), "the boundary opens the plateau"
    assert report["gate_contract"].startswith("h_lo <= ratio <= h_hi")


# --------------------------------------------------------------------------
# --eval-arg may not override what the tool owns
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        "--relink-bridge-h-hi",
        "--relink-bridge-h-hi=9",
        "--relink-bridge-h-h",  # mot17.py keeps argparse abbreviation on
        "--preset",
        "--output",
        "--data-root",
        "--no-gpu-decode",
        "--reid-mode",
    ],
)
def test_eval_arg_rejects_flags_the_tool_owns(token: str) -> None:
    """A forwarded copy wins in mot17.py, so the report would mislabel the point.

    Reproduced before the fix: --eval-arg=--relink-bridge-h-hi --eval-arg=9 on a
    point measured at 1.2 ran the eval at 9.0 while the report still said 1.2.
    """
    with pytest.raises(SystemExit) as excinfo:
        _args(f"--eval-arg={token}", "--eval-arg=9")
    assert excinfo.value.code == 2


def test_eval_arg_still_forwards_unrelated_flags() -> None:
    args = _args("--eval-arg=--seq-limit", "--eval-arg=2")
    assert args.eval_arg == ["--seq-limit", "2"]


def test_eval_arg_values_are_not_mistaken_for_flags() -> None:
    args = _args("--eval-arg=--sequences", "--eval-arg=MOT17-02-SDP")
    assert args.eval_arg[1] == "MOT17-02-SDP"


# --------------------------------------------------------------------------
# --dry-run must not fabricate measurements
# --------------------------------------------------------------------------


def test_dry_run_writes_nothing_to_the_cache(tmp_path, capsys) -> None:
    """Reproduced before the fix: --dry-run stored IDF1=100 for every point.

    The fake entries were keyed exactly like real ones, so the next real run in
    the same --work-dir served them as cache hits and executed no eval at all.
    """
    work = tmp_path / "bp"
    code = bp.main(
        [
            "--axis",
            "h_hi",
            "--lo",
            "1.2",
            "--hi",
            "1.4",
            "--scan",
            "2",
            "--work-dir",
            str(work),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert not (work / "cache.json").exists()
    # and no report either: every number in one would have been invented
    assert "IDF1" not in out
    assert "plateaus" not in out
    assert out.count("scripts/eval/mot17.py") == 2
    assert "--relink-bridge-h-hi 1.2" in out


# --------------------------------------------------------------------------
# the cache is only valid under the context it was measured in
# --------------------------------------------------------------------------


def _fake_context(digest: str) -> dict:
    return {
        "schema": bp.CACHE_SCHEMA,
        "bound": {"source": digest},
        "component_digests": {"source": digest, "runtime": "same"},
        "digest": digest,
        "informational": {"git_head": "0" * 40},
    }


def _write_cache(work: Path, context: dict, entries: dict) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "cache.json").write_text(
        json.dumps(
            {"schema": bp.CACHE_SCHEMA, "context": context, "entries": entries},
            indent=2,
        ),
        encoding="utf-8",
    )


def test_cache_serves_points_measured_under_the_same_context(
    tmp_path, monkeypatch
) -> None:
    work = tmp_path / "bp"
    monkeypatch.setattr(
        bp, "context_fingerprint", lambda args, env: _fake_context("abc")
    )
    args = _args(work_dir=str(work))
    _write_cache(work, _fake_context("abc"), {})

    runner = bp.Runner(args)
    assert runner.cache == {}


def test_cache_from_a_different_context_is_refused(tmp_path, monkeypatch) -> None:
    """A stale binary or edited source must not be mixed into a fresh report.

    The key held only paths and CLI values, so re-running after a rebuild in the
    same --work-dir silently reused the old numbers.
    """
    work = tmp_path / "bp"
    _write_cache(work, _fake_context("old"), {"{}": {}})
    monkeypatch.setattr(
        bp, "context_fingerprint", lambda args, env: _fake_context("new")
    )

    with pytest.raises(bp.CacheContextMismatch) as excinfo:
        bp.Runner(_args(work_dir=str(work)))
    assert "source" in str(excinfo.value)


def test_cache_without_a_context_fingerprint_is_refused(tmp_path, monkeypatch) -> None:
    """Caches written before this check carry no evidence of what produced them."""
    work = tmp_path / "bp"
    work.mkdir(parents=True)
    (work / "cache.json").write_text(json.dumps({"{}": {"idf1": 80.0}}), "utf-8")
    monkeypatch.setattr(
        bp, "context_fingerprint", lambda args, env: _fake_context("new")
    )

    with pytest.raises(bp.CacheContextMismatch):
        bp.Runner(_args(work_dir=str(work)))


def test_context_mismatch_fails_closed_at_the_command_line(
    tmp_path, monkeypatch, capsys
) -> None:
    work = tmp_path / "bp"
    _write_cache(work, _fake_context("old"), {})
    monkeypatch.setattr(
        bp, "context_fingerprint", lambda args, env: _fake_context("new")
    )

    code = bp.main(
        ["--axis", "h_hi", "--lo", "1.2", "--hi", "1.4", "--work-dir", str(work)]
    )

    assert code == 2
    assert "REFUSING CACHE" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the fingerprint must describe the run that actually happens
# --------------------------------------------------------------------------


def test_environment_is_bound_because_it_changes_the_measurement() -> None:
    """SACCADE_* knobs reach the tracker; the fingerprint missed them entirely.

    Reproduced before the fix: setting SACCADE_SCORE_JITTER (which perturbs
    boxes and scores) left the context digest byte-identical, so the cache
    happily served numbers measured without it.
    """
    base = {"PATH": "/usr/bin", "TERM": "xterm-256color"}
    jittered = dict(base, SACCADE_SCORE_JITTER="7:0.05:1.0")

    assert bp._env_component(base)["digest"] != bp._env_component(jittered)["digest"]
    assert bp._env_component(jittered)["saccade_overrides"] == {
        "SACCADE_SCORE_JITTER": "7:0.05:1.0"
    }


def test_terminal_decoration_does_not_invalidate_the_cache() -> None:
    """Fail-closed is only usable if it closes on things that can matter."""
    a = {"PATH": "/usr/bin", "TERM": "xterm", "SHLVL": "1", "SSH_TTY": "/dev/pts/0"}
    b = {"PATH": "/usr/bin", "TERM": "dumb", "SHLVL": "4"}

    assert bp._env_component(a)["digest"] == bp._env_component(b)["digest"]


def test_env_values_are_not_stored_in_the_clear() -> None:
    """The cache file is a work artefact; the environment carries credentials."""
    component = bp._env_component({"SECRET_TOKEN": "hunter2", "PATH": "/usr/bin"})
    assert "hunter2" not in json.dumps(component)


def test_probe_resolves_extensions_the_way_the_eval_will(tmp_path) -> None:
    """mot17.py honours SACCADE_BUILD_PATH; the probe used to ignore it.

    Reproduced before the fix: with SACCADE_BUILD_PATH pointing elsewhere, the
    eval loaded one saccade_tracking_ext and the fingerprint hashed another.
    """
    fake_build = tmp_path / "build-alt"
    fake_build.mkdir()
    (fake_build / "saccade_tracking_ext.py").write_text("# stand-in\n", "utf-8")

    env = dict(bp.eval_env(), SACCADE_BUILD_PATH=str(fake_build))
    runtime = bp._runtime_component(_args(), env)

    assert runtime["build_path"] == str(fake_build)
    origin = runtime["native_extensions"]["saccade_tracking_ext"]["origin"]
    assert origin == str(fake_build / "saccade_tracking_ext.py")
    assert runtime["native_extensions"]["saccade_tracking_ext"]["sha256"]


def _make_sequence(root: Path, name: str, frame_bytes: bytes) -> None:
    seq = root / name
    (seq / "gt").mkdir(parents=True)
    (seq / "gt" / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n", "utf-8")
    (seq / "seqinfo.ini").write_text("[Sequence]\nseqLength=1\n", "utf-8")
    (seq / "img1").mkdir()
    (seq / "img1" / "000001.jpg").write_bytes(frame_bytes)


def test_frame_contents_are_digested_not_just_their_size(tmp_path) -> None:
    """Same name, same size, different pixels used to pass as the same dataset."""
    root = tmp_path / "DS"
    _make_sequence(root / "train", "SEQ-01-SDP", b"\xff\xd8original")
    args = _args("--data-root", str(root), "--split", "train")
    before = bp._dataset_component(args)

    frame = root / "train" / "SEQ-01-SDP" / "img1" / "000001.jpg"
    frame.write_bytes(b"\xff\xd8repainted"[: len(b"\xff\xd8original")])
    after = bp._dataset_component(args)

    assert before["files"] == after["files"] == 3
    assert frame.stat().st_size == len(b"\xff\xd8original")
    assert before["digest"] != after["digest"]


# --------------------------------------------------------------------------
# bit-exactness is a premise, so it is enforced rather than assumed
# --------------------------------------------------------------------------


def test_eval_command_pins_reid_off() -> None:
    args = _args()
    cmd = bp.eval_command(args, {"px": 0.4, "h_lo": 0.6, "h_hi": 1.7}, Path("out"))
    assert cmd[cmd.index("--reid-mode") + 1] == "off"


@pytest.mark.parametrize(
    "preset", ["fpn_reid_baseline", "mamba_whole_graph_m_extract_ho_live"]
)
def test_preset_that_is_not_reid_off_is_refused(preset: str, capsys) -> None:
    """Bisection on a metric that is not bit-exact measures run-to-run noise."""
    code = bp.main(
        [
            "--axis",
            "h_hi",
            "--lo",
            "1.2",
            "--hi",
            "1.4",
            "--work-dir",
            "unused",
            "--preset",
            preset,
        ]
    )
    err = capsys.readouterr().err

    assert code == 2
    assert "REFUSING TO MEASURE" in err
    assert "reid_mode" in err


def test_shipping_preset_still_satisfies_the_premise() -> None:
    mode, origin = bp.resolve_reid_mode(_args())
    assert mode == "off"
    assert origin.endswith("mamba_whole_graph_m.yaml")
    assert bp.premise_violation(_args()) is None


# --------------------------------------------------------------------------
# an aborted bracket is not a bracket nobody reached
# --------------------------------------------------------------------------


def test_abort_inside_a_bracket_is_not_reported_as_never_reached() -> None:
    """The two states differ: one has measured points inside it, one does not."""
    runner = _StepRunner([1.35, 1.75])
    samples = bp.scan(runner, 1.2, 1.8, 7)
    brackets = bp.brackets_from_scan(samples)
    assert len(brackets) == 2

    # first bracket: bisection starts and dies after one midpoint
    brackets[0].attempted = True
    runner.measure(0.5 * (brackets[0].low.value + brackets[0].high.value))

    args = _args()
    report = bp.build_report(
        args,
        list(runner.seen.values()),
        runner.runs,
        brackets=brackets,
        aborted="eval failed (1); see run.log",
    )

    reasons = {b["reason"] for b in report["breakpoints"] if not b["refined"]}
    started = [r for r in reasons if "stopped part-way" in r]
    never = [r for r in reasons if "before reaching this bracket" in r]
    assert started, reasons
    assert never, reasons
