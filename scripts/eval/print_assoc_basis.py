#!/usr/bin/env python3
"""Print the resolved *association basis* (height / IoU / velocity / cost-weight
primitives + shared physical constants) for a given config / preset.

This makes it possible to discuss a run's *combination* of basic relink/auction
parameters at a glance, instead of cross-referencing 4-5 config dataclasses.

It resolves config exactly like ``scripts/eval/mot17.py`` (argparse defaults <
--config / baseline < --module-* YAMLs < --preset < CLI flags), then groups the
flat config by primitive family (see ``saccade.perception.eval.assoc_basis``).

Examples
--------
    # defaults the daily run would use
    python scripts/eval/print_assoc_basis.py

    # what a preset resolves to
    python scripts/eval/print_assoc_basis.py --preset mamba_whole_graph

    # diff two configs side by side
    python scripts/eval/print_assoc_basis.py --preset speed --diff --preset2 accuracy
"""
# status: stable

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_project_root = Path(__file__).resolve().parent.parent.parent
for _p in (_project_root, _project_root / "src", _project_root / "scripts" / "eval"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mot17_args import build_parser  # noqa: E402
from saccade.perception.eval.assoc_basis import (  # noqa: E402
    ASSOC_BASIS,
    format_basis,
    format_signals,
    resolve,
)

_MODULE_FLAGS = (
    "module_detection",
    "module_geometry",
    "module_motion",
    "module_reid",
    "module_semantic",
    "module_trigger",
    "module_lifecycle",
)


def _load_config_defaults(argv: list[str]) -> dict:
    """Replicate mot17.py's file-based default merge for the given argv.

    Kept in lockstep with ``scripts/eval/mot17.py::_load_config_defaults`` —
    same precedence so the printed basis matches a real run.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--preset", default=None)
    for flag in _MODULE_FLAGS:
        pre.add_argument(f"--{flag.replace('_', '-')}", default=None, dest=flag)
    pre_args, _ = pre.parse_known_args(argv)

    defaults: dict = {}
    if pre_args.config:
        config_path = Path(pre_args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        with config_path.open() as f:
            defaults.update(yaml.safe_load(f) or {})
    elif not pre_args.preset:
        fallback = _project_root / "configs" / "mot17_baseline.yaml"
        if fallback.exists():
            with fallback.open() as f:
                defaults.update(yaml.safe_load(f) or {})

    for flag in _MODULE_FLAGS:
        path_str = getattr(pre_args, flag, None)
        if path_str:
            module_path = Path(path_str)
            if not module_path.is_absolute():
                module_path = Path.cwd() / module_path
            with module_path.open() as f:
                defaults.update(yaml.safe_load(f) or {})

    if pre_args.preset:
        preset_path = _project_root / "configs" / "presets" / f"{pre_args.preset}.yaml"
        with preset_path.open() as f:
            defaults.update(yaml.safe_load(f) or {})
    return defaults


def _resolve_flat(argv: list[str]) -> dict:
    parser = build_parser()
    config_defaults = _load_config_defaults(argv)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args, _ = parser.parse_known_args(argv)
    return vars(args)


def _print_diff(flat_a: dict, flat_b: dict, label_a: str, label_b: str) -> None:
    ga, gb = resolve(flat_a), resolve(flat_b)
    print(f"assoc-basis diff: {label_a}  ->  {label_b}")
    print("=" * 72)
    any_diff = False
    fam_desc = {f.name: f.description for f in ASSOC_BASIS}
    for fam in ASSOC_BASIS:
        rows = [
            (k, ga[fam.name][k], gb[fam.name][k])
            for k in fam.keys
            if ga[fam.name][k] != gb[fam.name][k]
        ]
        if not rows:
            continue
        any_diff = True
        print(f"\n[{fam.name}] {fam_desc[fam.name]}")
        width = max(len(k) for k, _, _ in rows)
        for k, va, vb in rows:
            print(f"  {k:<{width}}  {va}  ->  {vb}")
    if not any_diff:
        print("\n(no differences in any association-basis family)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False
    )
    ap.add_argument(
        "--signals",
        action="store_true",
        help="Print the signal x consumer matrix + unmanaged constants "
        "(config-independent) and exit.",
    )
    ap.add_argument("--diff", action="store_true", help="Diff two configs.")
    ap.add_argument("--preset2", default=None, help="Second preset for --diff (RHS).")
    ap.add_argument(
        "--config2", default=None, help="Second config file for --diff (RHS)."
    )
    known, passthrough = ap.parse_known_args()

    if known.signals:
        print(format_signals())
        return

    if known.diff:
        rhs_argv = []
        if known.preset2:
            rhs_argv += ["--preset", known.preset2]
        if known.config2:
            rhs_argv += ["--config", known.config2]
        if not rhs_argv:
            ap.error("--diff requires --preset2 or --config2")
        flat_a = _resolve_flat(passthrough)
        flat_b = _resolve_flat(rhs_argv)
        label_a = " ".join(passthrough) or "<defaults>"
        label_b = " ".join(rhs_argv)
        _print_diff(flat_a, flat_b, label_a, label_b)
        return

    flat = _resolve_flat(passthrough)
    title = "association basis: " + (" ".join(passthrough) or "<defaults>")
    print(format_basis(flat, title=title))


if __name__ == "__main__":
    main()
