"""The behavior axis must digest the declared members, and only those.

Three properties carry the axis, and each has a way of being quietly wrong:

  * **member set** — the digest covers exactly the four § 4.0 members. Digesting
    the whole inventory dict would fold in mode, fixture, and any future key, so
    an identity-mode digest and a production digest of the same behavior would
    differ and the axis would stop being about behavior.
  * **sensitivity** — every member must actually move the digest. A member that
    is collected but not digested is worse than absent: it looks like coverage.
  * **fail-closed intake** — an axis may not be built from a run whose repeats
    disagreed. A digest that averages over a non-reproducible run is a fiction.

No GPU here: these exercise the pure digest/intake logic. The runs themselves are
gates G1/G2, recorded in the declaration § 5.1.2–5.1.3.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import h2_behavioral_identity as behavior  # noqa: E402


def _inventory() -> dict:
    return {
        "active_tid_slot_pairs": [{"frame": 1, "pairs": [[7, 0], [3, 1]]}],
        "final_track_rows": [
            {
                "binary32_bits": [1065353216, 0, 1073741824, 0, 1056964608],
                "class": 0,
                "frame": 1,
                "row_index": 0,
                "track_id": 7,
            }
        ],
        "mot_output": {"length": 42, "sha256": "ab" * 32},
        "relink_debug_raw": list(range(13)),
        "schema": behavior.BEHAVIOR_SCHEMA,
    }


def test_the_declared_member_set_is_exactly_four() -> None:
    assert behavior.BEHAVIOR_MEMBERS == (
        "active_tid_slot_pairs",
        "final_track_rows",
        "mot_output",
        "relink_debug_raw",
    )


def test_trace_only_members_are_not_in_the_axis() -> None:
    """§ 4.0: they cannot exist capture-off, so including them would make the
    axis uncomputable in identity mode."""
    for excluded in (
        "proposal_projection",
        "winner_commit_projection",
        "overflow_vector",
    ):
        assert excluded not in behavior.BEHAVIOR_MEMBERS


def test_the_digest_ignores_provenance_keys() -> None:
    """Mode, fixture and future keys must not enter the digest."""
    base = _inventory()
    decorated = {**base, "mode": "identity", "sequence": "MOT17-09-SDP", "future": 1}
    assert behavior.behavior_digest(decorated) == behavior.behavior_digest(base)


@pytest.mark.parametrize("member", behavior.BEHAVIOR_MEMBERS)
def test_every_member_moves_the_digest(member: str) -> None:
    base = _inventory()
    before = behavior.behavior_digest(base)
    mutated = copy.deepcopy(base)
    if member == "mot_output":
        mutated[member] = {"length": 43, "sha256": "cd" * 32}
    elif member == "relink_debug_raw":
        mutated[member] = [99] + list(range(1, 13))
    elif member == "active_tid_slot_pairs":
        mutated[member] = [{"frame": 1, "pairs": [[7, 0], [4, 1]]}]
    else:
        mutated[member][0]["track_id"] = 8
    assert behavior.behavior_digest(mutated) != before, (
        f"{member} is collected but does not affect the digest — coverage in name only"
    )


def test_slot_order_is_normalized_not_asserted() -> None:
    """The recorder owes A7.6's canonical order; the native call does not provide it.

    `get_active_tid_slot_pairs()` iterates `std::unordered_map<int,int>` in
    track-id bucket order (tracker_gpu.cu:5084). Two runs that observe the same
    active set in different bucket orders must produce the same digest.
    """
    ordered = _inventory()
    ordered["active_tid_slot_pairs"] = [{"frame": 1, "pairs": [[7, 0], [3, 1]]}]
    same_set_other_order = _inventory()
    same_set_other_order["active_tid_slot_pairs"] = [
        {"frame": 1, "pairs": sorted([[3, 1], [7, 0]], key=lambda p: p[1])}
    ]
    assert behavior.behavior_digest(ordered) == behavior.behavior_digest(
        same_set_other_order
    )


def test_an_incomplete_inventory_fails_closed() -> None:
    for member in behavior.BEHAVIOR_MEMBERS:
        broken = _inventory()
        del broken[member]
        with pytest.raises(behavior.BehavioralIdentityError):
            behavior.behavior_digest(broken)


def test_an_empty_inventory_is_not_an_identity() -> None:
    empty = _inventory()
    empty["final_track_rows"] = []
    with pytest.raises(behavior.BehavioralIdentityError):
        behavior.behavior_digest(empty)


def test_a_wrong_schema_fails_closed() -> None:
    wrong = _inventory()
    wrong["schema"] = "something_else_v1"
    with pytest.raises(behavior.BehavioralIdentityError):
        behavior.behavior_digest(wrong)


def test_canonical_json_is_h0s_convention() -> None:
    raw = behavior.canonical_json_bytes({"b": 1, "a": [1, 2]})
    assert raw == b'{"a":[1,2],"b":1}'
    with pytest.raises(ValueError):
        behavior.canonical_json_bytes({"nan": float("nan")})


# --------------------------------------------------------------------------- #
# Identity intake                                                              #
# --------------------------------------------------------------------------- #
def _behavior_payload(**overrides) -> dict:
    payload = {
        "digest": "ab" * 32,
        "digests": ["ab" * 32],
        "identical": True,
        "mode": "identity",
        "preset": behavior.POLICY_PRESET_REL,
        "repeats": 1,
        "resolved_fingerprint": "cd" * 32,
        "schema": "h2_behavior_result_v1",
        "sequence": behavior.IDENTITY_SEQUENCE,
    }
    payload.update(overrides)
    return payload


def test_a_non_reproducible_run_cannot_define_an_axis(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(_behavior_payload(identical=False, repeats=3, digest=None)),
        encoding="utf-8",
    )
    with pytest.raises(identity.IdentityError, match="repeats disagreed"):
        identity.load_behavior(path)


def test_a_foreign_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_behavior_payload(schema="other_v1")), encoding="utf-8")
    with pytest.raises(identity.IdentityError):
        identity.load_behavior(path)


def test_a_valid_payload_loads(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_behavior_payload()), encoding="utf-8")
    loaded = identity.load_behavior(path)
    assert loaded["digest"] == "ab" * 32
    assert loaded["state"] == "computed"


# --------------------------------------------------------------------------- #
# Axis assembly                                                                #
# --------------------------------------------------------------------------- #
def test_an_identity_without_a_behavior_axis_is_incomplete() -> None:
    built = identity.build_identity(behavior=None, build_dir=None)
    assert built["complete"] is False
    assert built["identity"]["behavior"] is None
    assert built["axes"]["behavior"]["state"] == "not_computed"


def test_the_three_static_axes_are_reproducible() -> None:
    first = identity.build_identity(behavior=None, build_dir=None)
    second = identity.build_identity(behavior=None, build_dir=None)
    for axis in ("decision_surface", "environment", "implementation"):
        assert first["identity"][axis] == second["identity"][axis]
        assert first["identity"][axis] is not None


def test_the_implementation_axis_covers_the_kernel_and_the_preset() -> None:
    files = set(identity.decision_relevant_files())
    assert "src/tracking/tracker_gpu.cu" in files
    assert "include/tracking/tracker_gpu.hpp" in files
    assert "configs/presets/mamba_whole_graph_m.yaml" in files
    # Prose is classified decision-relevant but kept out of the axis, so a README
    # edit cannot bump `implementation`.
    assert not any(path.endswith(".md") for path in files)


def test_witness_fields_are_marked_as_carrying_no_authority() -> None:
    built = identity.build_identity(behavior=None, build_dir=None)
    note = built["witness"]["note"]
    assert "no decision authority" in note
    assert "not a function of source" in note
