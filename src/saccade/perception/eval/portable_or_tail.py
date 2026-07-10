"""Frozen portable OR-tail policy loader + evaluator (M-B1 Stage 1).

Research-only. Fail-closed on schema / identity mismatch.
Does not search, refit, or repair thresholds.

Online Stage 1 path (``enforce_freeze_lock=True``, default):
  locks candidate identity, thr vector, file hash, and op='>'
  to match the CUDA hard-OR ABI.

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

# Locked freeze identity for the Stage 1 portable policy (online hook ABI).
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

# CUDA propose-kernel only implements strict greater-than (x > thr).
STAGE1_OP = ">"
ALLOWED_OPS_STAGE1 = frozenset({STAGE1_OP})

# Freeze thr vector (7-seq fit) — must match CUDA thr[] order exactly.
FROZEN_THR_VECTOR: tuple[float, ...] = (
    11.908911563050141,  # score_m_bridge:tail_q85
    1.3485465824400666,  # abs_log_h:tail_q85
    6.732025413759512,  # dist_h:tail_q85
    2.085930035366586,  # abs_ratio_m1:tail_q85
    14.043463872732945,  # resid_mean:tail_q85
)
FROZEN_EPS = 0.0
# SHA-256 of freeze portable_policy.json bytes (content lock).
FROZEN_POLICY_SHA256 = (
    "3638c2ef48e84d5f7cd3c3ef8ad1fca8414588005a2c33f0d4f6a9490595818b"
)
_THR_ATOL = 1e-12
_THR_RTOL = 0.0

# Online B-audit (full candidate-event export) is NOT implemented.
ONLINE_BAUDIT_IMPLEMENTED = False

# Stage 1b plumbing controls (not production / not freeze candidates).
# Explicit control_arm in JSON skips freeze thr/hash lock so we can prove
# signal → atom → reject → decision change without Stage 2 thr search.
ALLOWED_CONTROL_ARMS: frozenset[str] = frozenset({"activation", "force_reject"})
# atom0 midpoint of production bridge_px=0.4 — pre-specified, not metric-picked.
ACTIVATION_CONTROL_ATOM0_THR = 0.2
# Unreachable thr for non-tested atoms on control arms.
CONTROL_DISABLED_THR = 1.0e9
# force_reject: every finite score_m_bridge / bdist > -1 fires atom0.
FORCE_REJECT_ATOM0_THR = -1.0


class PortablePolicyError(ValueError):
    """Fail-closed portable policy validation error."""


class PortableAuditNotImplementedError(RuntimeError):
    """Raised when --research-portable-or-tail-audit is requested but unimplemented."""


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
    """Hard-OR of singleton tail atoms (freeze or Stage 1b control)."""

    path: Path
    file_hash: str
    candidate_id: str
    clauses: tuple[tuple[str, ...], ...]
    atoms: tuple[PortableAtom, ...]
    atom_by_id: Mapping[str, PortableAtom]
    eps: float
    thr_vector: tuple[float, ...]  # length 5, ORDERED_ATOM_IDS order
    schema_version: str = "portable_or_tail_v1"
    freeze_locked: bool = False
    control_arm: str | None = None  # None | activation | force_reject
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def is_control_arm(self) -> bool:
        return self.control_arm is not None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PortablePolicyError(msg)


def _thr_close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= _THR_ATOL + _THR_RTOL * abs(float(b))


def resolve_candidate_id(
    raw: Mapping[str, Any],
    path: Path,
    *,
    expected_candidate_id: str | None,
) -> str:
    """Resolve candidate identity fail-closed.

    Accepted identity sources (in order):
      1. explicit ``candidate_id`` (or ``policy_candidate_id``) field in JSON
      2. parent directory name **exactly** equal to expected_candidate_id

    Soft prefix / inferred fallbacks are rejected.
    """
    explicit = raw.get("candidate_id")
    if explicit is None:
        explicit = raw.get("policy_candidate_id")

    if explicit is not None:
        cid = str(explicit)
        if expected_candidate_id is not None:
            _require(
                cid == expected_candidate_id,
                f"candidate_id mismatch: got {cid!r}, expected {expected_candidate_id!r}",
            )
        return cid

    # No field: only exact parent-dir identity (no soft prefix).
    if expected_candidate_id is not None and path.parent.name == expected_candidate_id:
        return expected_candidate_id

    raise PortablePolicyError(
        f"portable policy missing candidate_id and parent dir "
        f"{path.parent.name!r} is not exactly {expected_candidate_id!r}; "
        f"refuse soft fallback"
    )


def require_online_audit_available(*, audit_enabled: bool) -> None:
    """Fail-closed: online B-audit flag must not claim unimplemented export."""
    if not audit_enabled:
        return
    if ONLINE_BAUDIT_IMPLEMENTED:
        return
    raise PortableAuditNotImplementedError(
        "--research-portable-or-tail-audit / research_portable_or_tail_audit "
        "requests online full candidate-event export, which is NOT implemented "
        "(Stage 1 online B-audit still PENDING). "
        "For offline pairs-replay tables use: "
        "scripts/tools/run_m_b1_hook_ab.py --offline-events-only. "
        "For A1/B e2e use hook without --research-portable-or-tail-audit "
        "(native counters via get_relink_debug only)."
    )


def load_portable_policy(
    path: str | Path,
    *,
    expected_candidate_id: str | None = EXPECTED_CANDIDATE_ID,
    enforce_freeze_lock: bool = True,
    allow_zone_gap: bool = False,
) -> PortablePolicy:
    """Load and validate portable_policy.json (fail-closed).

    Parameters
    ----------
    enforce_freeze_lock:
        When True (default, online Stage 1 freeze path): lock thr vector, file
        hash, eps, and op='>' to the freeze constants; CUDA ABI alignment
        required. **Exception:** JSON with explicit ``control_arm`` in
        ``ALLOWED_CONTROL_ARMS`` skips freeze thr/hash lock (Stage 1b plumbing
        only; never a production candidate).
        Set False only for offline research tools / unit tests with synthetic thr.
    """
    p = Path(path).expanduser().resolve()
    _require(p.is_file(), f"portable policy file missing: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PortablePolicyError(f"portable policy JSON invalid: {p}: {e}") from e

    _require(isinstance(raw, dict), "portable policy root must be object")
    clauses_raw = raw.get("clauses")
    atom_specs = raw.get("atom_specs")
    _require(
        isinstance(clauses_raw, list) and bool(clauses_raw),
        "missing clauses",
    )
    _require(
        isinstance(atom_specs, dict) and bool(atom_specs),
        "missing atom_specs",
    )

    control_raw = raw.get("control_arm")
    control_arm: str | None = None
    if control_raw is not None and str(control_raw).strip() != "":
        control_arm = str(control_raw).strip()
        _require(
            control_arm in ALLOWED_CONTROL_ARMS,
            f"unknown control_arm {control_arm!r}; "
            f"allowed={sorted(ALLOWED_CONTROL_ARMS)}",
        )

    # Control arms carry their own candidate_id; do not force freeze id.
    cid_expected = None if control_arm is not None else expected_candidate_id
    candidate_id = resolve_candidate_id(raw, p, expected_candidate_id=cid_expected)

    clauses: list[tuple[str, ...]] = []
    for i, cl in enumerate(clauses_raw):
        _require(
            isinstance(cl, list) and bool(cl),
            f"clause[{i}] must be non-empty list",
        )
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

    atoms: list[PortableAtom] = []
    atom_by_id: dict[str, PortableAtom] = {}
    for aid in ORDERED_ATOM_IDS:
        _require(aid in atom_specs, f"missing atom_specs[{aid!r}]")
        spec = atom_specs[aid]
        _require(isinstance(spec, dict), f"atom_specs[{aid!r}] must be object")
        signal = str(spec.get("signal", ""))
        kind = str(spec.get("kind", ""))
        op = str(spec.get("op", STAGE1_OP))
        thr = spec.get("thr")
        _require(signal in ORDERED_SIGNALS, f"{aid}: unexpected signal {signal!r}")
        _require(aid.startswith(signal + ":"), f"{aid}: atom_id/signal mismatch")
        # CUDA only implements '>' — reject other ops even offline-schema path.
        _require(
            op in ALLOWED_OPS_STAGE1,
            f"{aid}: op {op!r} unsupported; Stage 1 / CUDA ABI requires op={STAGE1_OP!r}",
        )
        _require(thr is not None and np.isfinite(float(thr)), f"{aid}: invalid thr")
        if not allow_zone_gap:
            _require(
                "zone" not in kind
                and "gap" not in kind
                and "zone" not in aid
                and "gap" not in aid,
                f"{aid}: zone/gap atoms banned in Stage 1 (kind={kind!r})",
            )
            if control_arm is None:
                _require(
                    kind in {"tail_q", "tail"},
                    f"{aid}: expected tail kind, got {kind!r}",
                )
            else:
                # Control arms may use kind=control / thr_fixed plumbing labels.
                _require(
                    kind in {"tail_q", "tail", "control", "thr_fixed"},
                    f"{aid}: control arm kind {kind!r} not allowed",
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
    freeze_locked = False

    if control_arm is not None:
        # Stage 1b plumbing: validate intentional thr shapes; never freeze-lock.
        freeze_locked = False
        _require(
            abs(eps - FROZEN_EPS) <= _THR_ATOL,
            f"control_arm eps must be 0, got {eps}",
        )
        if control_arm == "activation":
            _require(
                _thr_close(thr_vector[0], ACTIVATION_CONTROL_ATOM0_THR),
                f"activation control atom0 thr must be "
                f"{ACTIVATION_CONTROL_ATOM0_THR}, got {thr_vector[0]}",
            )
            for i in range(1, 5):
                _require(
                    thr_vector[i] >= CONTROL_DISABLED_THR * 0.5,
                    f"activation control atom{i} must be disabled "
                    f"(thr>={CONTROL_DISABLED_THR * 0.5}), got {thr_vector[i]}",
                )
        elif control_arm == "force_reject":
            _require(
                thr_vector[0] <= FORCE_REJECT_ATOM0_THR + 1e-9,
                f"force_reject atom0 thr must be <= {FORCE_REJECT_ATOM0_THR}, "
                f"got {thr_vector[0]}",
            )
    elif enforce_freeze_lock:
        freeze_locked = True
        _require(
            candidate_id == EXPECTED_CANDIDATE_ID,
            f"enforce_freeze_lock requires candidate_id={EXPECTED_CANDIDATE_ID!r}, "
            f"got {candidate_id!r}",
        )
        _require(
            abs(eps - FROZEN_EPS) <= _THR_ATOL,
            f"freeze eps mismatch: got {eps}, expected {FROZEN_EPS}",
        )
        mismatches = [
            f"{ORDERED_ATOM_IDS[i]}: got {thr_vector[i]}, expected {FROZEN_THR_VECTOR[i]}"
            for i in range(5)
            if not _thr_close(thr_vector[i], FROZEN_THR_VECTOR[i])
        ]
        _require(
            not mismatches,
            "freeze thr_vector mismatch (CUDA ABI lock):\n  " + "\n  ".join(mismatches),
        )
        _require(
            file_hash == FROZEN_POLICY_SHA256,
            f"freeze policy file hash mismatch: got {file_hash}, "
            f"expected {FROZEN_POLICY_SHA256} "
            f"(refuse silent thr/schema drift)",
        )

    return PortablePolicy(
        path=p,
        file_hash=file_hash,
        candidate_id=candidate_id,
        clauses=tuple(clauses),
        atoms=tuple(atoms),
        atom_by_id=atom_by_id,
        eps=eps,
        thr_vector=thr_vector,
        freeze_locked=freeze_locked,
        control_arm=control_arm,
        raw=raw,
    )


def apply_atom(atom: PortableAtom, values: np.ndarray) -> np.ndarray:
    """Apply a single atom. Stage 1 only supports op '>' (CUDA ABI)."""
    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if atom.op != STAGE1_OP:
        raise PortablePolicyError(
            f"apply_atom: op {atom.op!r} unsupported; Stage 1 requires {STAGE1_OP!r}"
        )
    return x > float(atom.thr)


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
        "freeze_locked": policy.freeze_locked,
        "control_arm": policy.control_arm,
        "frozen_policy_sha256": FROZEN_POLICY_SHA256,
        "eps": policy.eps,
        "ordered_atom_ids": list(ORDERED_ATOM_IDS),
        "thr_vector": list(policy.thr_vector),
        "stage1_op": STAGE1_OP,
        "online_baudit_implemented": ONLINE_BAUDIT_IMPLEMENTED,
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


def classify_stage1_milestones(
    *,
    evaluation_entry_ok: bool,
    frozen_policy_null_effect: bool,
    activation_action_path_ok: bool | None,
    force_reject_path_ok: bool | None,
    online_baudit_ok: bool = False,
    strict_a0_identity_ok: bool | None = None,
    soft_a0_identity_ok: bool | None = None,
    determinism_repeated_ok: bool = False,
    runtime_overhead_ok: bool = False,
) -> dict[str, Any]:
    """Split Stage 1a evaluation-entry vs Stage 1b action-path claims.

    Does **not** promote soft A0 identity or vacuous freeze-B into full Stage 1
    CLOSED. Full Stage 1 remains OPEN until action-path (+ preferred B-audit /
    strict A0 / determinism) land.
    """
    stage1a = "PASSED" if evaluation_entry_ok else "FAILED"
    frozen_relevance = (
        "NULL_support_mismatch"
        if frozen_policy_null_effect
        else ("UNKNOWN" if evaluation_entry_ok else "NOT_EVALUATED")
    )
    if activation_action_path_ok is None and force_reject_path_ok is None:
        action = "NOT_ACTIVATED"
    elif activation_action_path_ok is False or force_reject_path_ok is False:
        action = "FAILED"
    elif activation_action_path_ok is True and force_reject_path_ok is True:
        action = "PASSED"
    elif activation_action_path_ok is True or force_reject_path_ok is True:
        action = "PARTIAL"
    else:
        action = "NOT_ACTIVATED"

    baudit = "PASSED" if online_baudit_ok else "PENDING"
    if strict_a0_identity_ok is True:
        a0 = "strict_pass"
    elif soft_a0_identity_ok is True:
        a0 = "soft_pass_strict_unresolved"
    elif strict_a0_identity_ok is False:
        a0 = "strict_fail"
    else:
        a0 = "not_compared"

    # Full Stage 1 closed only when evaluation-entry + both action controls pass.
    # B-audit / strict A0 / determinism remain separate contract rows.
    stage1_overall = (
        "CLOSED"
        if (
            stage1a == "PASSED"
            and action == "PASSED"
            and online_baudit_ok
            and strict_a0_identity_ok is True
            and determinism_repeated_ok
            and runtime_overhead_ok
        )
        else "OPEN"
    )
    # Engineering milestone for "action-path proven" without full audit freeze.
    stage1b_eng = (
        "PASSED"
        if stage1a == "PASSED" and action == "PASSED"
        else ("PARTIAL" if action == "PARTIAL" else "OPEN")
    )
    return {
        "stage1a_evaluation_entry": stage1a,
        "frozen_policy_online_relevance": frozen_relevance,
        "stage1b_action_path": action,
        "stage1b_eng_milestone": stage1b_eng,
        "online_baudit": baudit,
        "a0_identity": a0,
        "determinism_repeated_run": "PASSED" if determinism_repeated_ok else "PENDING",
        "runtime_overhead": "PASSED" if runtime_overhead_ok else "PENDING",
        "stage1_overall": stage1_overall,
        "headline_claim_allowed": (
            "policy loading and evaluation-entry wiring are valid; "
            "online rejection/action chain "
            + (
                "proven under control arms"
                if action == "PASSED"
                else "remains unactivated or incomplete e2e"
            )
        ),
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


# Host get_relink_debug() layout (cursor + d_relink_dbg_[0..11]).
# Use these indices when reading counters from Python.
RELINK_DEBUG_HOST_INDEX: dict[str, int] = {
    "archived_cursor": 0,
    "births": 1,
    "revived": 2,
    "bridge_attempts": 3,
    "bridge_accepts": 4,
    "hook_eligible": 5,
    "hook_rejected": 6,
    "atom0_score_m_bridge": 7,
    "atom1_abs_log_h": 8,
    "atom2_dist_h": 9,
    "atom3_abs_ratio_m1": 10,
    "app_veto": 11,
    "atom4_resid_mean": 12,
}
