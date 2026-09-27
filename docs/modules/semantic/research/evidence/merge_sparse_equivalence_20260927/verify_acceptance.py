"""Verify supplemental F2 evidence; run from any working directory.

Use --local-mot to additionally read the original local MOT/substrate files.
The default also requires current sources to match the captured qualification.
Use --archive-only to audit historical evidence without qualifying current code.
"""

import argparse
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
REPO = next(p for p in PACKET.parents if (p / "pyproject.toml").exists())
ORIGINAL = (
    REPO / "docs/modules/semantic/research/evidence/merge_only_cross_dataset_20260924"
)
# Trust anchor is the independently reviewed PR head, not a packet-controlled ref.
SEALED_REF = "31c36d78a5f8197300127052b54a6a6e62b8a1cb"
PACKET_REL = "docs/modules/semantic/research/evidence/merge_sparse_equivalence_20260927"
ORIGINAL_REL = (
    "docs/modules/semantic/research/evidence/merge_only_cross_dataset_20260924"
)
MAINTAINED_FILES = {"verify_acceptance.py", "commands.md"}


@lru_cache(maxsize=None)
def sealed_bytes(relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{SEALED_REF}:{relative}"], cwd=REPO
    )


def sealed_json(relative: str) -> Any:
    return json.loads(sealed_bytes(relative))


def verify_inventory(root: Path, relative: str, maintained: set[str]) -> None:
    expected = sealed_json(f"{relative}/SHA256SUMS.json")
    inventory = load(root / "SHA256SUMS.json")
    expected_rows = {row["file"]: row for row in expected["files"]}
    rows = inventory["files"]
    physical = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p != root / "SHA256SUMS.json"
    }
    require(physical == set(expected_rows), f"{relative}: unsealed/missing files")
    require(
        len(rows) == len(expected_rows)
        and {row["file"] for row in rows} == set(expected_rows),
        f"{relative}: inventory membership drift",
    )
    if not maintained:
        require(inventory == expected, f"{relative}: sealed inventory drift")
    for row in rows:
        name = row["file"]
        if name not in maintained:
            require(row == expected_rows[name], f"{name}: sealed checksum drift")
        path = root / name
        require(
            path.is_file() and not path.is_symlink(), f"{name}: missing sealed file"
        )
        require(
            path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
            f"{name}: sealed bytes drift",
        )


def verify_identity(qualified: dict[str, Any], dataset: str) -> None:
    expected = sealed_json(f"{PACKET_REL}/qualification/{dataset}.json")
    # Compare every captured identity field, including the complete numeric
    # contract, source inventory/hashes, GPU, software, merge and block policy.
    require(
        {k: v for k, v in qualified.items() if k != "sequences"}
        == {k: v for k, v in expected.items() if k != "sequences"},
        f"{dataset}: qualification runtime/source identity drift; requalification required",
    )
    require(
        set(qualified["sequences"]) == set(expected["sequences"]),
        f"{dataset}: qualification sequence coverage drift",
    )
    for name, digest in qualified["source_sha256"].items():
        captured = subprocess.check_output(
            ["git", "show", f"{qualified['commit']}:{name}"], cwd=REPO
        )
        require(
            hashlib.sha256(captured).hexdigest() == digest,
            f"{name}: captured source hash mismatch",
        )
    for seq, row in qualified["sequences"].items():
        for key in (
            "embedding_sha256",
            "substrate_sha256",
            "samples",
            "graph_nodes",
            "row_chunk",
            "sparse_blocked_regime",
            "sparse_blocked_forced_regime",
        ):
            require(row[key] == expected["sequences"][seq][key], f"{seq}: {key} drift")


def current_source_drift() -> list[str]:
    expected = sealed_json(f"{PACKET_REL}/qualification/mot17.json")
    return [
        name
        for name, digest in expected["source_sha256"].items()
        if not (REPO / name).is_file() or sha(REPO / name) != digest
    ]


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm(payload: dict[str, Any], name: str = "merge_only") -> Any:
    return next(row for row in payload["primary"] if row["arm"] == name)


def partition_digest(ids: set[int], edges: list[tuple[int, int]]) -> str:
    parent = {i: i for i in ids}

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in edges:
        parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for i in ids:
        groups.setdefault(root(i), []).append(i)
    partition = sorted(sorted(group) for group in groups.values())
    return hashlib.sha256(json.dumps(partition).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-mot", action="store_true")
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Verify the sealed capture only; does not qualify current sources/runtime",
    )
    args = parser.parse_args()
    verify_inventory(PACKET, PACKET_REL, MAINTAINED_FILES)
    verify_inventory(ORIGINAL, ORIGINAL_REL, set())
    rows = {}
    bindings = load(PACKET / "substrate_bindings.json")
    prior_acceptance = load(PACKET / "prior_acceptance.json")
    for ds, count, replay in [
        ("mot17", 7, "mot17"),
        ("mot20", 2, "mot20_2"),
        ("dancetrack", 40, "dancetrack"),
    ]:
        original = load(ORIGINAL / ds / "results.json")
        old = arm(original)
        measured = arm(load(PACKET / "replay" / replay / "results.json"))
        prior = load(PACKET / "prior_equivalence" / f"{ds}.json")
        qualified = load(PACKET / "qualification" / f"{ds}.json")
        verify_identity(qualified, ds)
        require(qualified["numeric_contract"]["matmul_precision"] == "highest", ds)
        require(not qualified["numeric_contract"]["allow_tf32"], ds)
        require(not qualified["numeric_contract"]["autocast"], ds)
        require(not qualified["dirty"], ds)
        require(old["metrics"] == measured["metrics"], f"{ds}: full metrics mismatch")
        require(len(old["mot_sha256"]) == count, f"{ds}: sequence coverage")
        for seq, final_sha in old["mot_sha256"].items():
            q = qualified["sequences"][seq]
            require(
                q["sparse_repeat_costs_exact"] and q["sparse_repeat_output_exact"],
                f"{seq}: repeat",
            )
            for mode in ["sparse", "sparse_blocked"]:
                require(
                    q[mode]["out_sha256"] == q["dense"]["out_sha256"],
                    f"{seq}: output hash mismatch {mode}",
                )
                require(
                    q[mode]["stats"] == q["dense"]["stats"]
                    and q[mode]["accepted"] == q["dense"]["accepted"],
                    f"{seq}: stats/accepted count mismatch {mode}",
                )
                v = q[mode]["vs_dense"]
                require(
                    all(
                        v[k]
                        for k in [
                            "out_identical",
                            "stats_identical",
                            "accepted_identical_in_order",
                        ]
                    ),
                    f"{seq}: decisions {mode}",
                )
                require(v["eligible_verdict_mismatches"] == 0, f"{seq}: verdicts")
            require(
                q["dense"]["out_sha256"]
                == prior["sequences"][seq]["dense"]["out_sha256"],
                f"{seq}: historical dense output",
            )
            require(
                q["sparse"]["out_sha256"]
                == prior["sequences"][seq]["sparse"]["out_sha256"],
                f"{seq}: pre-interpolation output",
            )
            require(
                q["final_mot_sha256"] == final_sha == measured["mot_sha256"][seq],
                f"{seq}: final output",
            )
            require(
                q["substrate_sha256"] == bindings[seq]["normalized_lines_sha256"]
                and bindings[seq]["raw_file_sha256"]
                == original["identity"]["substrate_sha256"][seq],
                f"{seq}: substrate",
            )
            old_edges = [
                (p["a_id"], p["b_id"])
                for p in old["seq_stage_records"][seq][0]["accepted_pairs"]
            ]
            measured_edges = [
                (p["a_id"], p["b_id"])
                for p in measured["seq_stage_records"][seq][0]["accepted_pairs"]
            ]
            require(old_edges == measured_edges, f"{seq}: historical accepted pairs")
            require(q["sparse"]["accepted"] == len(old_edges), f"{seq}: accepted count")
            if args.local_mot:
                for row in [old, measured]:
                    require(
                        sha(Path(row["output_dir"]) / f"{seq}.txt") == final_sha,
                        f"{seq}: local bytes",
                    )
                require(
                    sha(Path(original["identity"]["substrate"]) / f"{seq}.txt")
                    == bindings[seq]["raw_file_sha256"],
                    f"{seq}: local substrate",
                )
                substrate = Path(original["identity"]["substrate"]) / f"{seq}.txt"
                ids = {
                    int(line.split(",")[1])
                    for line in substrate.read_text().splitlines()
                    if line.strip()
                }
                require(
                    partition_digest(ids, old_edges)
                    == partition_digest(ids, measured_edges)
                    == prior_acceptance["sequences"][seq]["partition_sha256"],
                    f"{seq}: reconstructed partition",
                )
            rows[seq] = dict(
                final_mot_sha256=final_sha,
                pre_interpolation_sha256=q["sparse"]["out_sha256"],
                accepted_merges=len(old_edges),
            )
    require(len(rows) == 49, "49 reference sequences required")
    qualified = load(PACKET / "qualification/mot20.json")
    full = load(PACKET / "replay/mot20_full/results.json")
    noint = load(PACKET / "replay/mot20_full_noint/results.json")
    frozen_mot20 = load(ORIGINAL / "mot20/substrate_sha256.json")
    for seq in ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]:
        q = qualified["sequences"][seq]
        require(
            bindings[seq]["raw_file_sha256"] == frozen_mot20[f"{seq}.txt"]["sha256"],
            f"{seq}: frozen MOT20 substrate",
        )
        require(
            q["sparse_repeat_costs_exact"] and q["sparse_repeat_output_exact"],
            f"{seq}: repeat",
        )
        require(
            q["final_mot_sha256"] == arm(full)["mot_sha256"][seq],
            f"{seq}: full MOT20 final bytes",
        )
        require(
            q["sparse"]["out_sha256"] == arm(noint)["mot_sha256"][seq],
            f"{seq}: full MOT20 no-interpolation bytes",
        )
        for key in ["peak_allocated_bytes", "peak_reserved_bytes", "wall_s"]:
            require(q["sparse"][key] > 0, f"{seq}: measurement {key}")
    for name in ["base", "merge_only"]:
        v = full["variation"][name]
        require(
            v["n"] == 3 and v["mot_files_identical"],
            f"{name}: full MOT20 repeat outputs",
        )
        require(
            all(v[k]["range"] == 0 for k in ["IDF1", "HOTA", "AssA", "MOTA"]),
            f"{name}: full metrics repeats",
        )
    drift = current_source_drift()
    if not args.archive_only:
        require(
            not drift,
            f"Current source drift requires full reference requalification: {drift}",
        )
    print(
        json.dumps(
            dict(
                all_passed=True,
                verification_scope="sealed_capture"
                if args.archive_only
                else "sealed_capture_and_source_identity",
                current_source_drift=drift,
                current_runtime_qualified=False,
                production_eligible=False,
                reference_sequences=49,
                mot20_sequences=4,
                local_mot_checked=args.local_mot,
                metric_reuse="Exact final MOT byte identity binds the qualified implementation to the scored full-precision rows; no approximate metric tolerance.",
                sequences=rows,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
