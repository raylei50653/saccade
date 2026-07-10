"""Frozen portable OR-tail policy loader + evaluator (M-B1 Stage 1).

Research-only. Fail-closed on schema / identity mismatch.
Does not search, refit, or repair thresholds.

Contract:
  docs/modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md
  docs/modules/semantic/research/m_b1_to_m_b1_5_two_stage_plan_20260710.md
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Locked freeze identity for the Stage 1 portable policy.
EXPECTED_CANDIDATE_ID = "m_b1_repaired_eps0_loo_pass_20260709"

# Canonical ordered singleton atoms (OR of 5 tails). Order is ABI for native thr[].
ORDERED_ATOM_IDS: tuple[str, ...] = (
    "score_m_bridge:tail_q85",
    "abs_log_h:tail_q85",
    "dist_h:tail_q85",
    "abs_ratio_m1:tail_q85",
    "resid_mean:tail_q85",
)

ORDERED_SIGNALS: tuple[str, ...] = (
    "score_m_bridge",
    "abs_log_h",
    "dist_h",
    "abs_ratio_m1",
    "resid_mean",
)

ALLOWED_OPS = frozenset({">", ">=", "<", "in_range"})
BANNED_KINDS = frozenset({"zone_q", "gap", "gap_bin", "hard_zone"})


class PortablePolicyError(ValueError):
    """Fail-closed portable policy validation error."""


@dataclass(frozen=True)
class PortableAtom:
    atom_id: str
    signal: str
    thr: float
    op: str
    kind: str
    thr_hi: float | None = None
    quantile: float | None = None
    description: str = ""
    role: str = ""


@dataclass(frozen=True)
class PortablePolicy:
    """Frozen hard-OR of singleton tail atoms."""

    path: Path
    file_hash: str
    candidate_id: str
    clauses: tuple[tuple[str, ...], ...]
    atoms: tuple[PortableAtom, ...]
    atom_by_id: Mapping[str, PortableAtom]
    eps: float
    thr_vector: tuple[float, ...]  # length 5, ORDERED_ATOM_IDS order
    schema_version: str = "portable_or_tail_v1"
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PortablePolicyError(msg)


def load_portable_policy(
    path: str | Path,
    *,
    expected_candidate_id: str | None = EXPECTED_CANDIDATE_ID,
    allow_zone_gap: bool = False,
) -> PortablePolicy:
    """Load and validate frozen portable_policy.json (fail-closed)."""
    p = Path(path).expanduser().resolve()
    _require(p.is_file(), f"portable policy file missing: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PortablePolicyError(f"portable policy JSON invalid: {p}: {e}") from e

    _require(isinstance(raw, dict), "portable policy root must be object")
    clauses_raw = raw.get("clauses")
    atom_specs = raw.get("atom_specs")
    _require(isinstance(clauses_raw, list) and clauses_raw, "missing clauses")
    _require(isinstance(atom_specs, dict) and atom_specs, "missing atom_specs")

    candidate_id = str(
        raw.get("candidate_id")
        or raw.get("policy_candidate_id")
        or expected_candidate_id
        or ""
    )
    if expected_candidate_id is not None:
        # Allow missing candidate_id in freeze file when parent dir/name matches.
        if "candidate_id" in raw:
            _require(
                str(raw["candidate_id"]) == expected_candidate_id,
                f"candidate_id mismatch: got {raw['candidate_id']!r}, "
                f"expected {expected_candidate_id!r}",
            )
        else:
            # Infer from parent directory name (freeze layout).
            if (
                p.parent.name != expected_candidate_id
                and candidate_id != expected_candidate_id
            ):
                # Soft identity: accept freeze file when parent is freeze study id.
                if p.parent.name.startswith("m_b1_repaired_eps0_loo_pass"):
                    candidate_id = expected_candidate_id
                else:
                    raise PortablePolicyError(
                        f"cannot verify candidate_id for {p}; "
                        f"parent={p.parent.name!r}, expected={expected_candidate_id!r}"
                    )
            else:
                candidate_id = expected_candidate_id

    clauses: list[tuple[str, ...]] = []
    for i, cl in enumerate(clauses_raw):
        _require(isinstance(cl, list) and cl, f"clause[{i}] must be non-empty list")
        aids = tuple(str(a) for a in cl)
        clauses.append(aids)

    # Stage 1: only singleton-OR of ordered tail atoms.
    _require(
        all(len(c) == 1 for c in clauses),
        "Stage 1 portable policy must be OR of singleton clauses "
        f"(got AND sizes {[len(c) for c in clauses]})",
    )
    clause_atom_ids = [c[0] for c in clauses]
    _require(
        set(clause_atom_ids) == set(ORDERED_ATOM_IDS),
        f"clause atom set mismatch: got {sorted(clause_atom_ids)}, "
        f"expected {list(ORDERED_ATOM_IDS)}",
    )
    # Preserve freeze clause order for audit; thr_vector uses ORDERED_ATOM_IDS.

    atoms: list[PortableAtom] = []
    atom_by_id: dict[str, PortableAtom] = {}
    for aid in ORDERED_ATOM_IDS:
        _require(aid in atom_specs, f"missing atom_specs[{aid!r}]")
        spec = atom_specs[aid]
        _require(isinstance(spec, dict), f"atom_specs[{aid!r}] must be object")
        signal = str(spec.get("signal", ""))
        kind = str(spec.get("kind", ""))
        op = str(spec.get("op", ">"))
        thr = spec.get("thr")
        _require(signal in ORDERED_SIGNALS, f"{aid}: unexpected signal {signal!r}")
        _require(aid.startswith(signal + ":"), f"{aid}: atom_id/signal mismatch")
        _require(op in ALLOWED_OPS, f"{aid}: unsupported op {op!r}")
        _require(thr is not None and np.isfinite(float(thr)), f"{aid}: invalid thr")
        if not allow_zone_gap:
            _require(
                "zone" not in kind
                and "gap" not in kind
                and "zone" not in aid
                and "gap" not in aid,
                f"{aid}: zone/gap atoms banned in Stage 1 (kind={kind!r})",
            )
            _require(
                kind in {"tail_q", "tail"}, f"{aid}: expected tail kind, got {kind!r}"
            )
        thr_hi = spec.get("thr_hi")
        atom = PortableAtom(
            atom_id=aid,
            signal=signal,
            thr=float(thr),
            op=op,
            kind=kind,
            thr_hi=float(thr_hi) if thr_hi is not None else None,
            quantile=float(spec["quantile"])
            if spec.get("quantile") is not None
            else None,
            description=str(spec.get("description", "")),
            role=str(spec.get("role", "")),
        )
        atoms.append(atom)
        atom_by_id[aid] = atom

    thr_vector = tuple(atom_by_id[aid].thr for aid in ORDERED_ATOM_IDS)
    eps = float(raw.get("eps", 0.0))
    file_hash = _sha256_file(p)

    return PortablePolicy(
        path=p,
        file_hash=file_hash,
        candidate_id=candidate_id,
        clauses=tuple(clauses),
        atoms=tuple(atoms),
        atom_by_id=atom_by_id,
        eps=eps,
        thr_vector=thr_vector,
        raw=raw,
    )


def apply_atom(atom: PortableAtom, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if atom.op == "in_range":
        assert atom.thr_hi is not None
        return (x >= atom.thr) & (x <= float(atom.thr_hi))
    thr = atom.thr
    if atom.op == ">":
        return x > thr
    if atom.op == ">=":
        return x >= thr
    if atom.op == "<":
        return x < thr
    raise PortablePolicyError(f"unknown op {atom.op}")


def signals_from_arrays(
    *,
    score_m_bridge: np.ndarray,
    abs_log_h: np.ndarray,
    dist_h: np.ndarray,
    abs_ratio_m1: np.ndarray,
    resid_mean: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "score_m_bridge": np.asarray(score_m_bridge, dtype=float),
        "abs_log_h": np.asarray(abs_log_h, dtype=float),
        "dist_h": np.asarray(dist_h, dtype=float),
        "abs_ratio_m1": np.asarray(abs_ratio_m1, dtype=float),
        "resid_mean": np.asarray(resid_mean, dtype=float),
    }


def evaluate_policy(
    policy: PortablePolicy,
    signals: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate frozen OR policy on equal-length signal arrays.

    Returns reject mask, per-atom masks, fire bitmask, and fire class labels.
    """
    n = next(iter(signals.values())).shape[0]
    for s in ORDERED_SIGNALS:
        _require(s in signals, f"missing signal {s!r}")
        _require(signals[s].shape[0] == n, f"signal {s!r} length mismatch")

    atom_masks: dict[str, np.ndarray] = {}
    bitmask = np.zeros(n, dtype=np.int32)
    for i, aid in enumerate(ORDERED_ATOM_IDS):
        atom = policy.atom_by_id[aid]
        m = apply_atom(atom, signals[atom.signal])
        atom_masks[aid] = m
        bitmask = bitmask | (m.astype(np.int32) << i)

    reject = np.zeros(n, dtype=bool)
    for m in atom_masks.values():
        reject |= m

    n_fired = np.zeros(n, dtype=np.int32)
    for m in atom_masks.values():
        n_fired += m.astype(np.int32)

    fire_class = np.full(n, "zero", dtype=object)
    fire_class[n_fired == 1] = "singleton"
    fire_class[n_fired >= 2] = "cofire"

    fired_atom_ids = []
    for j in range(n):
        ids = [ORDERED_ATOM_IDS[i] for i in range(5) if (int(bitmask[j]) >> i) & 1]
        fired_atom_ids.append(ids)

    return {
        "reject": reject,
        "atom_masks": atom_masks,
        "atom_bitmask": bitmask,
        "n_atoms_fired": n_fired,
        "fire_class": fire_class,
        "fired_atom_ids": fired_atom_ids,
    }


def evaluate_policy_row(
    policy: PortablePolicy,
    row_signals: Mapping[str, float],
) -> dict[str, Any]:
    """Scalar convenience wrapper for a single candidate."""
    sig = {k: np.asarray([float(row_signals[k])], dtype=float) for k in ORDERED_SIGNALS}
    out = evaluate_policy(policy, sig)
    return {
        "reject": bool(out["reject"][0]),
        "atom_bitmask": int(out["atom_bitmask"][0]),
        "n_atoms_fired": int(out["n_atoms_fired"][0]),
        "fire_class": str(out["fire_class"][0]),
        "fired_atom_ids": list(out["fired_atom_ids"][0]),
        "atom_fires": {
            aid: bool(out["atom_masks"][aid][0]) for aid in ORDERED_ATOM_IDS
        },
    }


def snapshot_policy(policy: PortablePolicy) -> dict[str, Any]:
    """Serializable snapshot for study artifacts (not a second evidence home)."""
    return {
        "schema_version": policy.schema_version,
        "candidate_id": policy.candidate_id,
        "policy_path": str(policy.path),
        "policy_file_hash": policy.file_hash,
        "eps": policy.eps,
        "ordered_atom_ids": list(ORDERED_ATOM_IDS),
        "thr_vector": list(policy.thr_vector),
        "clauses": [list(c) for c in policy.clauses],
        "atoms": [
            {
                "atom_id": a.atom_id,
                "signal": a.signal,
                "op": a.op,
                "thr": a.thr,
                "kind": a.kind,
                "quantile": a.quantile,
                "role": a.role,
                "description": a.description,
            }
            for a in policy.atoms
        ],
    }


def resolve_policy_path_from_env(
    cli_path: str | None = None,
    *,
    env_var: str = "SACCADE_RESEARCH_PORTABLE_OR_TAIL_POLICY",
) -> Path | None:
    """Resolve default-off policy path: CLI wins, then env. None => hook off."""
    if cli_path is not None and str(cli_path).strip():
        return Path(str(cli_path).strip()).expanduser()
    env = os.environ.get(env_var, "").strip()
    if env:
        return Path(env).expanduser()
    return None


def fire_class_counts(fire_class: Sequence[str] | np.ndarray) -> dict[str, int]:
    arr = np.asarray(fire_class, dtype=object)
    return {
        "n_zero_fire": int((arr == "zero").sum()),
        "n_singleton": int((arr == "singleton").sum()),
        "n_cofire": int((arr == "cofire").sum()),
    }


def derive_atom_summary(
    policy: PortablePolicy,
    atom_masks: Mapping[str, np.ndarray],
    reject: np.ndarray,
    sequences: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Per-atom summary rows (derived from full event masks)."""
    rows = []
    reject = np.asarray(reject, dtype=bool)
    for aid in ORDERED_ATOM_IDS:
        m = np.asarray(atom_masks[aid], dtype=bool)
        others = np.zeros_like(m)
        for oid, om in atom_masks.items():
            if oid != aid:
                others |= np.asarray(om, dtype=bool)
        singleton = m & ~others
        cofired = m & others
        row: dict[str, Any] = {
            "atom_id": aid,
            "n_fired": int(m.sum()),
            "n_singleton": int(singleton.sum()),
            "n_cofired": int(cofired.sum()),
            "n_rejected": int((m & reject).sum()),
            "n_decision_changed": -1,  # filled by online audit when available
        }
        if sequences is not None:
            seq = np.asarray(sequences)
            row["n_sequences_fired"] = int(len({str(s) for s in seq[m]}))
            row["n_sequences_singleton"] = int(len({str(s) for s in seq[singleton]}))
        else:
            row["n_sequences_fired"] = -1
            row["n_sequences_singleton"] = -1
        rows.append(row)
    return rows


def reconcile_fire_classes(
    n_hook_eligible: int,
    n_zero: int,
    n_singleton: int,
    n_cofire: int,
    n_rejected: int,
) -> list[str]:
    """Programmatic reconciliation assertions from plan §10. Returns errors."""
    errors: list[str] = []
    if n_hook_eligible != n_zero + n_singleton + n_cofire:
        errors.append(
            f"eligible != zero+singleton+cofire: "
            f"{n_hook_eligible} != {n_zero}+{n_singleton}+{n_cofire}"
        )
    if n_rejected > n_hook_eligible:
        errors.append(f"n_rejected {n_rejected} > eligible {n_hook_eligible}")
    if n_rejected > n_singleton + n_cofire:
        errors.append(
            f"n_rejected {n_rejected} > singleton+cofire {n_singleton + n_cofire}"
        )
    return errors


def classify_e2e_status(
    *,
    hook_off_identity_ok: bool,
    n_rejected: int,
    metrics_delta: Mapping[str, float] | None = None,
    per_seq_regression: bool = False,
    determinism_ok: bool = True,
    runtime_ok: bool = True,
) -> str:
    """Suggested Stage 1 classification (plan §11). Not a production claim."""
    if not hook_off_identity_ok:
        return "online_inconclusive"
    if not determinism_ok:
        return "online_inconclusive"
    if per_seq_regression:
        return "online_unsafe"
    md = metrics_delta or {}
    # Hard safety heuristics (research defaults; headline remains e2e_safe yes/no).
    if md.get("GT_hurt_proxy", 0.0) > 0:
        return "online_unsafe"
    if md.get("IDF1", 0.0) < -0.5 or md.get("AssA", 0.0) < -0.5:
        return "online_unsafe"
    if n_rejected == 0 and abs(md.get("IDF1", 0.0)) < 1e-9:
        return "online_effect_neutral_but_safe"
    if not runtime_ok:
        return "online_inconclusive"
    if md.get("IDF1", 0.0) >= -0.2 and not per_seq_regression:
        return "e2e_safe_for_default_off"
    return "online_inconclusive"
