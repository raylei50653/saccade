"""Verify supplemental F2 evidence; run from any working directory.

Use --local-mot to additionally read the original local MOT/substrate files.
The default verification uses the sealed packets and exact recorded hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
REPO = next(p for p in PACKET.parents if (p / "pyproject.toml").exists())
ORIGINAL = (
    REPO / "docs/modules/semantic/research/evidence/merge_only_cross_dataset_20260924"
)


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
    args = parser.parse_args()
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
    print(
        json.dumps(
            dict(
                all_passed=True,
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
