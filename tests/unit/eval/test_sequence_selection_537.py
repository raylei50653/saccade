"""#537 F06: eval sequences without seqinfo.ini are never skipped silently.

Explicitly named sequences without ``seqinfo.ini`` are an error; discovered
ones are skipped with a logged reason; zero valid sequences is an error. The
Python (``run_eval``) and C++ (``run_eval_cpp``) entries validate before any
config parse, extension import or detector load, and the C++ sequence config
no longer falls back to 1920x1080 when ``seqinfo.ini`` is missing. Contract:
docs/architecture/failure_degradation_537.md §26.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from saccade.perception.eval import cpp_runner, evaluator
from saccade.perception.eval import config as eval_config
from saccade.perception.eval.sequence_selection import (
    SequenceSelectionError,
    filter_discovered,
    requested_sequences,
    select_sequences,
)


def _tree(root: Path, with_ini: list[str], without_ini: list[str]) -> Path:
    for name in with_ini:
        (root / "train" / name / "img1").mkdir(parents=True)
        (root / "train" / name / "seqinfo.ini").write_text(
            "[Sequence]\nimWidth=640\nimHeight=480\nseqLength=1\n"
        )
    for name in without_ini:
        (root / "train" / name / "img1").mkdir(parents=True)
    (root / "train").mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """A-SDP, B-SDP, X-FRCNN have seqinfo.ini; C-SDP does not."""
    return _tree(tmp_path, ["A-SDP", "B-SDP", "X-FRCNN"], ["C-SDP"])


class Log(list[str]):
    def __call__(self, msg: str) -> None:
        self.append(msg)


# ── explicit ────────────────────────────────────────────────────────────────


def test_explicit_all_present(data: Path) -> None:
    log = Log()
    sel = select_sequences(data, "train", "B-SDP,A-SDP", log=log)
    assert sel.explicit is True
    assert sel.valid == ("B-SDP", "A-SDP")  # run order is the request order
    assert sel.skipped == ()
    assert log == []


def test_explicit_partially_missing_is_an_error(data: Path) -> None:
    with pytest.raises(SequenceSelectionError, match="C-SDP") as exc:
        select_sequences(data, "train", "A-SDP,C-SDP,NOPE-SDP")
    assert "NOPE-SDP" in str(exc.value)
    assert "A-SDP" not in str(exc.value)


def test_explicit_all_missing_is_an_error(data: Path) -> None:
    with pytest.raises(SequenceSelectionError, match="without seqinfo.ini"):
        select_sequences(data, "train", "C-SDP,NOPE-SDP")


def test_explicit_under_a_missing_split_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SequenceSelectionError, match="split directory not found"):
        select_sequences(tmp_path, "train", "A-SDP")


def test_empty_names_are_not_sequences(data: Path) -> None:
    sel = select_sequences(data, "train", "A-SDP,,B-SDP,")
    assert sel.valid == ("A-SDP", "B-SDP")


# ── discovered ──────────────────────────────────────────────────────────────


def test_discovered_all_present(tmp_path: Path) -> None:
    root = _tree(tmp_path, ["A-SDP", "B-SDP"], [])
    log = Log()
    sel = select_sequences(root, "train", "", log=log)
    assert sel.explicit is False
    assert sel.valid == ("A-SDP", "B-SDP")
    assert log == []


@pytest.mark.parametrize("sequences", ["", None])
def test_discovered_partially_missing_is_skipped_with_reason(
    data: Path, sequences: str | None
) -> None:
    log = Log()
    sel = select_sequences(data, "train", sequences, log=log)
    assert sel.valid == ("A-SDP", "B-SDP", "X-FRCNN")
    assert sel.skipped == (("C-SDP", "no seqinfo.ini"),)
    assert len(log) == 1 and "C-SDP" in log[0] and "no seqinfo.ini" in log[0]


def test_discovered_all_missing_is_an_error(tmp_path: Path) -> None:
    root = _tree(tmp_path, [], ["C-SDP", "D-SDP"])
    log = Log()
    with pytest.raises(SequenceSelectionError, match="no valid sequence"):
        select_sequences(root, "train", "", log=log)
    assert len(log) == 2  # the reasons are still logged before the error


def test_discovered_empty_split_is_an_error(tmp_path: Path) -> None:
    root = _tree(tmp_path, [], [])
    with pytest.raises(SequenceSelectionError, match="none found"):
        select_sequences(root, "train", "")


def test_suffix_filter_is_discovery(data: Path) -> None:
    log = Log()
    sel = select_sequences(data, "train", "SDP", log=log)
    assert sel.explicit is False
    assert sel.valid == ("A-SDP", "B-SDP")
    assert sel.skipped == (("C-SDP", "no seqinfo.ini"),)
    assert "C-SDP" in log[0]


def test_suffix_filter_all_missing_is_an_error(data: Path) -> None:
    with pytest.raises(SequenceSelectionError, match="none found"):
        select_sequences(data, "train", "DPM")


def test_discovery_under_a_missing_split_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SequenceSelectionError, match="split directory not found"):
        select_sequences(tmp_path, "train", "")


# ── names a caller discovered itself (DanceTrack / SportsMOT style) ─────────


def test_filter_discovered_skips_and_logs(data: Path) -> None:
    log = Log()
    assert filter_discovered(data, "train", ["A-SDP", "C-SDP", "B-SDP"], log=log) == [
        "A-SDP",
        "B-SDP",
    ]
    assert len(log) == 1 and "C-SDP" in log[0]


def test_filter_discovered_zero_valid_is_an_error(data: Path) -> None:
    log = Log()
    with pytest.raises(SequenceSelectionError, match="no valid sequence"):
        filter_discovered(data, "train", ["C-SDP", "NOPE"], log=log)
    assert len(log) == 2


def test_filter_discovered_nothing_found_keeps_the_fallback(data: Path) -> None:
    """Codex F06 r2: DanceTrack scans data_root, so with the usual
    ``--data-root datasets/DanceTrack --split val`` layout it finds nothing
    and used to fall back to run_eval discovery; it still does."""
    log = Log()
    assert filter_discovered(data, "train", [], log=log) == []
    assert filter_discovered(data, "train", ["", ""], log=log) == []
    assert log == []
    # ... and that fallback (empty --sequences) is run_eval's validated discovery.
    assert select_sequences(data, "train", "", log=log).valid == (
        "A-SDP",
        "B-SDP",
        "X-FRCNN",
    )


# ── EvalConfig.seqs keeps the pre-F06 rule ──────────────────────────────────


def _pre_f06_seqs(data_root: str, split: str, sequences: str) -> list[str]:
    """parse_eval_config's sequence block at main@4380c2fa, verbatim."""
    _detector_suffixes = {"SDP", "DPM", "FRCNN"}
    if sequences:
        filters = [s.strip() for s in sequences.split(",")]
        if all(f in _detector_suffixes for f in filters):
            _all = sorted(
                d.name for d in (Path(data_root) / split).iterdir() if d.is_dir()
            )
            seqs = [d for d in _all if any(d.endswith(f"-{f}") for f in filters)]
        else:
            seqs = filters
    else:
        seqs = sorted(d.name for d in (Path(data_root) / split).iterdir() if d.is_dir())
    return seqs


@pytest.mark.parametrize(
    "sequences",
    ["", "SDP", "SDP,FRCNN", "DPM", "A-SDP", "B-SDP, A-SDP", "A-SDP,,", "C-SDP,SDP"],
)
def test_requested_sequences_is_the_pre_f06_rule(data: Path, sequences: str) -> None:
    _, names = requested_sequences(str(data), "train", sequences)
    assert names == _pre_f06_seqs(str(data), "train", sequences)


def test_parse_eval_config_uses_the_rule(data: Path) -> None:
    cfg = eval_config.parse_eval_config(
        output=str(data / "out"),
        data_root=str(data),
        split="train",
        sequences="SDP",
        conf_threshold=0.05,
        reid_mode="off",
        reid_model="siglip2",
        profile_stages=False,
        kwargs={},
    )
    # cfg.seqs is still the requested list; validation is the entries' job.
    assert cfg.seqs == ["A-SDP", "B-SDP", "C-SDP"]


# ── entries validate before running ─────────────────────────────────────────


class Reached(Exception):
    pass


@pytest.fixture
def stop_at_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    def _parse(**_: Any) -> None:
        raise Reached

    monkeypatch.setattr(eval_config, "parse_eval_config", _parse)


def _run_eval(data: Path, sequences: str) -> None:
    evaluator.run_eval(
        engine="unused.engine",
        output=str(data / "out"),
        data_root=str(data),
        split="train",
        sequences=sequences,
        max_frames=1,
        conf_threshold=0.05,
    )


@pytest.mark.usefixtures("stop_at_parse")
@pytest.mark.parametrize("sequences", ["A-SDP,C-SDP", "C-SDP"])
def test_run_eval_refuses_explicit_missing_before_parse(
    data: Path, sequences: str
) -> None:
    with pytest.raises(SequenceSelectionError):
        _run_eval(data, sequences)
    assert not (data / "out").exists()


@pytest.mark.usefixtures("stop_at_parse")
def test_run_eval_refuses_zero_valid_discovered(tmp_path: Path) -> None:
    root = _tree(tmp_path, [], ["C-SDP"])
    with pytest.raises(SequenceSelectionError, match="no valid sequence"):
        _run_eval(root, "")


@pytest.mark.usefixtures("stop_at_parse")
@pytest.mark.parametrize("sequences", ["A-SDP,B-SDP", "", "SDP"])
def test_run_eval_proceeds_for_valid_selections(
    data: Path, sequences: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(Reached):
        _run_eval(data, sequences)
    if sequences != "A-SDP,B-SDP":
        assert "Skipping discovered sequence C-SDP" in capsys.readouterr().out


@pytest.fixture
def no_eval_ext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing saccade_eval_ext fails: run_eval_cpp raises RuntimeError there."""
    monkeypatch.setitem(sys.modules, "saccade_eval_ext", None)


def _run_eval_cpp(data: Path, sequences: str) -> None:
    cpp_runner.run_eval_cpp(
        engine="unused.engine",
        output=str(data / "out"),
        data_root=str(data),
        split="train",
        sequences=sequences,
    )


@pytest.mark.usefixtures("no_eval_ext")
@pytest.mark.parametrize("sequences", ["A-SDP,C-SDP", "C-SDP,NOPE-SDP"])
def test_run_eval_cpp_refuses_explicit_missing_before_the_extension(
    data: Path, sequences: str
) -> None:
    with pytest.raises(SequenceSelectionError):
        _run_eval_cpp(data, sequences)


@pytest.mark.usefixtures("no_eval_ext")
def test_run_eval_cpp_refuses_zero_valid_discovered(tmp_path: Path) -> None:
    root = _tree(tmp_path, [], ["C-SDP"])
    with pytest.raises(SequenceSelectionError, match="no valid sequence"):
        _run_eval_cpp(root, "SDP")


@pytest.mark.usefixtures("no_eval_ext")
@pytest.mark.parametrize("sequences", ["A-SDP", "", "SDP"])
def test_run_eval_cpp_proceeds_for_valid_selections(data: Path, sequences: str) -> None:
    with pytest.raises(RuntimeError, match="saccade_eval_ext not available"):
        _run_eval_cpp(data, sequences)


class _SeqConfig(SimpleNamespace):
    pass


def _cpp_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        core=SimpleNamespace(confirm_streak=2, confirm_score_thresh=0.1),
        kwargs={},
    )


def test_cpp_seq_config_has_no_default_geometry_for_missing_seqinfo(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules, "saccade_eval_ext", SimpleNamespace(CppSequenceConfig=_SeqConfig)
    )
    with pytest.raises(SequenceSelectionError, match="C-SDP"):
        cpp_runner._build_cpp_seq_config(_cpp_cfg(), "C-SDP", str(data), split="train")
