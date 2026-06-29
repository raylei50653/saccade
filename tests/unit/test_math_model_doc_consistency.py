"""Pin the LaTeX math-model document (``docs/latex``) to the code it describes.

``docs/latex/chapters/*.tex`` prints a ``Baseline`` contract for every module
citing concrete default values — ``kalman_r_scale: 2.8``, ``sinkhorn_lambda: 10``,
``relink_bridge_px: 0.25`` and so on. Those numbers are transcribed by hand from
three real sources of truth:

  * the eval preset ``configs/presets/mamba_whole_graph.yaml`` (most knobs);
  * the ``scripts/eval/config/lifecycle.py`` dataclass defaults (knobs the preset
    leaves unset, e.g. the relink anchor mode / rate);
  * the CUDA env-var defaults in ``src/tracking/tracker_gpu.cu`` (the two auction
    bid biases and the DDA cost cap).

When a default is retuned the PDF silently starts documenting a baseline the code
no longer runs. This module triple-locks each cited constant:

    doc text (.tex)  ==  registry below  ==  source of truth (YAML / py / .cu)

``test_doc_baseline_matches_source`` fails if a source-of-truth default drifts
from what the registry (and therefore the doc) claims; ``test_doc_baseline_
printed_in_latex`` fails if someone edits the ``.tex`` number/token without
updating the registry. Adding a new ``Baseline`` constant to the doc means adding
a row here — otherwise it is unguarded, which the module docstring is the only
reminder of.

These are the executable companion to the manual math-vs-code audit of
``main-full.pdf``; the equation-level checks (Kalman Q/R, multiplicative cost,
auction value, bridge ``d_bridge``) live in the kernels themselves and are not
re-derivable here without a GPU, so this guards the part that realistically rots:
the cited constants.
"""

from __future__ import annotations

import re
from collections import namedtuple
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PRESET = _REPO / "configs" / "presets" / "mamba_whole_graph.yaml"
_LIFECYCLE = _REPO / "scripts" / "eval" / "config" / "lifecycle.py"
_CU = _REPO / "src" / "tracking" / "tracker_gpu.cu"
_TEX_DIR = _REPO / "docs" / "latex" / "chapters"


# --------------------------------------------------------------------------- #
# Source-of-truth resolvers (regex-based, so the test adds no import/runtime    #
# dependency on the eval stack or a CUDA build).                                #
# --------------------------------------------------------------------------- #
def _scalar(raw: str) -> object:
    """Parse a YAML/python scalar literal into a Python value."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
        return raw[1:-1]
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _preset_scalar(key: str) -> object:
    text = _PRESET.read_text(encoding="utf-8")
    m = re.search(rf"(?m)^{re.escape(key)}:\s*([^\n]+?)\s*(?:#.*)?$", text)
    assert m, f"key {key!r} not found in {_PRESET}"
    return _scalar(m.group(1))


def _py_default(field: str) -> object:
    """Read a dataclass field default `field: type = <default>` from lifecycle.py."""
    text = _LIFECYCLE.read_text(encoding="utf-8")
    m = re.search(
        rf"(?m)^\s*{re.escape(field)}:\s*[^=]+=\s*([^\n#]+?)\s*(?:#.*)?$", text
    )
    assert m, f"field {field!r} not found in {_LIFECYCLE}"
    return _scalar(m.group(1))


def _cu_env_default(name: str) -> float:
    """Read the fallback default of an env-var knob in tracker_gpu.cu."""
    text = _CU.read_text(encoding="utf-8")
    # Form A:  env_float_value("NAME", 0.12f)
    m = re.search(rf'env_float_value\("{re.escape(name)}",\s*([0-9.]+)f?\)', text)
    if m:
        return float(m.group(1))
    # Form B:  std::getenv("NAME") ... : 0.1f;
    m = re.search(
        rf'std::getenv\("{re.escape(name)}"\).*?:\s*([0-9.]+)f', text, re.DOTALL
    )
    assert m, f"env default for {name!r} not found in {_CU}"
    return float(m.group(1))


_TEX_CACHE: str | None = None


def _tex_blob() -> str:
    """All chapter .tex sources, normalised so `\\texttt{a\\_b: 1}` -> `a_b: 1`."""
    global _TEX_CACHE
    if _TEX_CACHE is None:
        raw = "\n".join(
            p.read_text(encoding="utf-8") for p in sorted(_TEX_DIR.glob("*.tex"))
        )
        raw = raw.replace("\\_", "_").replace("\\texttt{", "")
        raw = raw.replace("{", "").replace("}", "")
        _TEX_CACHE = re.sub(r"\s+", " ", raw)
    return _TEX_CACHE


def _eq(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


# --------------------------------------------------------------------------- #
# Registry: one row per Baseline constant cited in docs/latex/chapters/*.tex.   #
#   key      label / pytest id                                                  #
#   expected the value the doc prints                                           #
#   where    human description of the source of truth (failure message)         #
#   actual   resolver returning the live value from the source of truth         #
#   tex      literal that must appear in the normalised .tex, or None when the  #
#            value is typeset as math (τ_OAO, N_ramp, c_DDA) and only the token  #
#            — not the number — is reliably greppable.                          #
# --------------------------------------------------------------------------- #
Const = namedtuple("Const", "key expected where actual tex")

_CONSTANTS: list[Const] = [
    # ---- GMC (ch.3) ----
    Const(
        "gmc_downscale",
        4,
        "preset:gmc_downscale",
        lambda: _preset_scalar("gmc_downscale"),
        "gmc_downscale: 4",
    ),
    # ---- Kalman (ch.4) ----
    Const(
        "kalman_r_scale",
        2.8,
        "preset:kalman_r_scale",
        lambda: _preset_scalar("kalman_r_scale"),
        "kalman_r_scale: 2.8",
    ),
    # ---- Association cost (ch.5) ----
    Const(
        "fuse_score_weight",
        0.0,
        "preset:fuse_score_weight",
        lambda: _preset_scalar("fuse_score_weight"),
        "fuse_score_weight",
    ),
    Const(
        "multiplicative_cost",
        True,
        "preset:multiplicative_cost",
        lambda: _preset_scalar("multiplicative_cost"),
        "multiplicative_cost: true",
    ),
    Const(
        "stability_cost_w",
        0.20,
        "preset:stability_cost_w",
        lambda: _preset_scalar("stability_cost_w"),
        "stability_cost_w: 0.20",
    ),
    Const("oao_tau", 0.50, "preset:oao_tau", lambda: _preset_scalar("oao_tau"), None),
    Const(
        "oao_ramp_frames",
        25,
        "preset:oao_ramp_frames",
        lambda: _preset_scalar("oao_ramp_frames"),
        None,
    ),
    Const(
        "reid_mode",
        "off",
        "preset:reid_mode",
        lambda: _preset_scalar("reid_mode"),
        "reid_mode: off",
    ),
    # ---- Auction (ch.6) ----
    Const(
        "sinkhorn_lambda",
        10,
        "preset:sinkhorn_lambda",
        lambda: _preset_scalar("sinkhorn_lambda"),
        "sinkhorn_lambda: 10",
    ),
    Const(
        "dda_max_cost",
        0.12,
        "cu-env:SACCADE_DDA_MAX_COST",
        lambda: _cu_env_default("SACCADE_DDA_MAX_COST"),
        None,
    ),
    Const(
        "freshness_w",
        0.0,
        "cu-env:SACCADE_FRESHNESS_W",
        lambda: _cu_env_default("SACCADE_FRESHNESS_W"),
        "SACCADE_FRESHNESS_W",
    ),
    Const(
        "stability_w",
        0.1,
        "cu-env:SACCADE_STABILITY_W",
        lambda: _cu_env_default("SACCADE_STABILITY_W"),
        "SACCADE_STABILITY_W",
    ),
    # ---- Bridge relink (ch.8) ----
    Const(
        "relink_bridge_enabled",
        True,
        "preset:relink_bridge_enabled",
        lambda: _preset_scalar("relink_bridge_enabled"),
        "relink_bridge_enabled: true",
    ),
    Const(
        "relink_bridge_px",
        0.25,
        "preset:relink_bridge_px",
        lambda: _preset_scalar("relink_bridge_px"),
        "relink_bridge_px: 0.25",
    ),
    Const(
        "relink_bridge_margin",
        0.05,
        "preset:relink_bridge_margin",
        lambda: _preset_scalar("relink_bridge_margin"),
        "margin: 0.05",
    ),
    Const(
        "relink_bridge_h_lo",
        0.75,
        "preset:relink_bridge_h_lo",
        lambda: _preset_scalar("relink_bridge_h_lo"),
        "h_lo/h_hi: 0.75/1.33",
    ),
    Const(
        "relink_bridge_h_hi",
        1.33,
        "preset:relink_bridge_h_hi",
        lambda: _preset_scalar("relink_bridge_h_hi"),
        "h_lo/h_hi: 0.75/1.33",
    ),
    Const(
        "relink_bridge_dir_bonus",
        0.8,
        "preset:relink_bridge_dir_bonus",
        lambda: _preset_scalar("relink_bridge_dir_bonus"),
        "dir_bonus: 0.8",
    ),
    Const(
        "relink_bridge_spatial_gate",
        0.0,
        "preset:relink_bridge_spatial_gate",
        lambda: _preset_scalar("relink_bridge_spatial_gate"),
        "spatial_gate",
    ),
    # anchor mode / rate are not set by the preset -> dataclass defaults apply.
    Const(
        "relink_bridge_anchor",
        "adaptive",
        "lifecycle:relink_bridge_anchor",
        lambda: _py_default("relink_bridge_anchor"),
        "adaptive",
    ),
    Const(
        "relink_bridge_anchor_rate",
        0.03,
        "lifecycle:relink_bridge_anchor_rate",
        lambda: _py_default("relink_bridge_anchor_rate"),
        "anchor_rate",
    ),
]


@pytest.mark.parametrize("c", _CONSTANTS, ids=lambda c: c.key)
def test_doc_baseline_matches_source(c: Const) -> None:
    """The value the math-model doc cites must equal the live source of truth."""
    actual = c.actual()
    assert _eq(actual, c.expected), (
        f"docs/latex cites {c.key}={c.expected!r} but {c.where} now holds "
        f"{actual!r} — retune the doc's Baseline contract or revert the default."
    )


@pytest.mark.parametrize("c", [c for c in _CONSTANTS if c.tex], ids=lambda c: c.key)
def test_doc_baseline_printed_in_latex(c: Const) -> None:
    """The literal the registry pins must actually appear in the chapter source."""
    assert c.tex in _tex_blob(), (
        f"expected {c.tex!r} for {c.key} not found in docs/latex/chapters — the "
        f".tex was edited away from the registry (PDF and code would disagree)."
    )
