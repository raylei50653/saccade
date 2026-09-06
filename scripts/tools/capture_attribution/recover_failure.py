"""Extract surviving primary tool evidence for #340; never rerun the workload."""

# status: diagnostic

import argparse
import hashlib
import json
from pathlib import Path


TOOL_IDS = {
    "toolu_01N8b4K2DHkNQTH9eZ5osR6G": "harness_creation",
    "toolu_01BJExtLyMEY8vQ8UpiTnemz": "host",
    "toolu_01RETdTPD64ru7h9iSdFSvFU": "source_amend",
    "toolu_01HYW4W44Lb5eYr8iSBNWq2K": "launch",
    "toolu_01BebPzotydyAfmiYF8pAcjV": "run_table",
    "toolu_01PM2BKuiHAJrCqRM1yi3QiN": "failure_excerpt",
    "toolu_01RvN6GSgA7iQyvuWD4qy2MH": "head_observation",
    "toolu_01KzYmHcpjgNy33nMybRREez": "traceback",
    "toolu_0194m4enxjhaYYQcBXGqbjrt": "precapture_added",
    "toolu_01TdJ4fx4HE4JyFeWjDwkBFU": "precapture_amend",
    "toolu_01U6r5AC5S2eymbsUJFXMgu6": "original_logs_removed",
}


def recover(transcript: Path, output: Path) -> None:
    raw = transcript.read_bytes()
    evidence = []
    for number, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        content = row.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            key = block.get("tool_use_id", block.get("id"))
            if key in TOOL_IDS:
                evidence.append(
                    {
                        "source_line": number,
                        "timestamp": row.get("timestamp"),
                        "evidence_role": TOOL_IDS[key],
                        "block": block,
                    }
                )
    seen = {r["evidence_role"] for r in evidence}
    if seen != set(TOOL_IDS.values()):
        raise ValueError(f"Missing evidence roles: {set(TOOL_IDS.values()) - seen}")
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "capture_failure_recovery_v1",
        "status": "partial_recovery_not_exact_runtime_attestation",
        "transcript": str(transcript.resolve()),
        "transcript_sha256_at_recovery": hashlib.sha256(raw).hexdigest(),
        "original_run": "b2_1_B before 2026-09-06T02:36:42Z reset",
        "candidate_source": "806c52cf8ced0836c80606559f7c38a5fcc546a3",
        "missing": [
            "original full stdout/stderr and per-run JSONL (deleted and names reused)",
            "per-run source file hashes and dirty-worktree snapshot",
            "per-run environment, loaded library versions/hashes and asset hashes",
            "failing CUDA API return trace, stream handle/flags and capture timeline",
        ],
        "evidence": evidence,
    }
    (output / "recovered.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(output / "recovered.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("output", type=Path, help="new directory; never overwrites")
    args = parser.parse_args()
    recover(args.transcript, args.output)
