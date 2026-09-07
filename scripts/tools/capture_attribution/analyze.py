"""Validate trace structure and attribute observed errors without exclusion claims."""

# status: diagnostic

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
from typing import Callable


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _maps_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _base_api_name(value: str) -> str:
    previous = None
    while value != previous:
        previous = value
        value = re.sub(r"_(?:v\d+|ptsz|ptds)$", "", value)
    return value


def _stream_create(api: str) -> bool:
    base = _base_api_name(api)
    return base.startswith(("cudaStreamCreate", "cuStreamCreate"))


def _stream_destroy(api: str) -> bool:
    base = _base_api_name(api)
    return base.startswith(("cudaStreamDestroy", "cuStreamDestroy"))


class NativeStackResolver:
    """Resolve captured process addresses only through attested run binaries."""

    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.expected = manifest.get("mapped_file_sha256_after", {})
        self.maps = None
        self.binary_cache = {}
        self.frame_cache = {}

    def _load_maps(self):
        if self.maps is not None:
            return self.maps
        path = self.root / "maps.txt"
        if not path.is_file():
            self.maps = []
            return self.maps
        entries = []
        for line in path.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 5:
                continue
            try:
                start, end = (int(value, 16) for value in fields[0].split("-"))
                offset = int(fields[2], 16)
            except ValueError:
                continue
            entries.append(
                {
                    "start": start,
                    "end": end,
                    "permissions": fields[1],
                    "offset": offset,
                    "path": _maps_path(fields[5]) if len(fields) == 6 else "",
                }
            )
        self.maps = entries
        return entries

    @staticmethod
    def _elf_metadata(path: Path):
        with path.open("rb") as source:
            ident = source.read(16)
            if len(ident) != 16 or ident[:4] != b"\x7fELF":
                raise ValueError("not_elf")
            bits, byte_order = ident[4], ident[5]
            endian = "<" if byte_order == 1 else ">" if byte_order == 2 else None
            if endian is None:
                raise ValueError("unknown_elf_endianness")
            if bits == 2:
                values = struct.unpack(endian + "HHIQQQIHHHHHH", source.read(48))
                elf_type, phoff, phentsize, phnum = (
                    values[0],
                    values[4],
                    values[8],
                    values[9],
                )
                ph_format = endian + "IIQQQQQQ"
            elif bits == 1:
                values = struct.unpack(endian + "HHIIIIIHHHHHH", source.read(36))
                elf_type, phoff, phentsize, phnum = (
                    values[0],
                    values[4],
                    values[8],
                    values[9],
                )
                ph_format = endian + "IIIIIIII"
            else:
                raise ValueError("unknown_elf_class")
            segments = []
            for index in range(phnum):
                source.seek(phoff + index * phentsize)
                raw = source.read(struct.calcsize(ph_format))
                if len(raw) != struct.calcsize(ph_format):
                    raise ValueError("short_program_header")
                fields = struct.unpack(ph_format, raw)
                if bits == 2:
                    kind, flags, offset, vaddr = fields[:4]
                else:
                    kind, offset, vaddr, _, _, _, flags, _ = fields
                if kind == 1:  # PT_LOAD
                    segments.append({"flags": flags, "offset": offset, "vaddr": vaddr})
        return elf_type, segments

    def _binary(self, value: str):
        if value in self.binary_cache:
            return self.binary_cache[value]
        result = {"status": "unresolved", "reason": "binary_unavailable"}
        path = Path(value)
        recorded = self.expected.get(value)
        if recorded is None:
            # Compatibility with an escaped path emitted by an older harness.
            recorded = self.expected.get(
                value.replace(" ", "\\040").replace("\t", "\\011")
            )
        try:
            if not path.is_file():
                raise OSError("not a regular file")
            current = _sha256(path)
            if recorded is None:
                result = {"status": "unresolved", "reason": "binary_unattested"}
            elif current != recorded:
                result = {
                    "status": "unresolved",
                    "reason": "binary_hash_mismatch",
                    "expected_sha256": recorded,
                    "observed_sha256": current,
                }
            else:
                elf_type, segments = self._elf_metadata(path)
                result = {
                    "status": "verified",
                    "sha256": current,
                    "elf_type": elf_type,
                    "segments": segments,
                }
        except (OSError, ValueError, struct.error) as exc:
            result = {
                "status": "unresolved",
                "reason": f"binary_unreadable:{type(exc).__name__}",
            }
        self.binary_cache[value] = result
        return result

    def _locate(self, address: int):
        matches = [
            entry
            for entry in self._load_maps()
            if entry["start"] <= address < entry["end"]
        ]
        if len(matches) != 1:
            return {
                "address": address,
                "status": "unresolved",
                "reason": "address_unmapped" if not matches else "mapping_ambiguous",
            }
        mapping = matches[0]
        path = mapping["path"]
        if not path.startswith("/") or path.endswith(" (deleted)"):
            return {
                "address": address,
                "mapping": mapping,
                "status": "unresolved",
                "reason": "mapping_has_no_live_binary",
            }
        binary = self._binary(path)
        if binary["status"] != "verified":
            return {
                "address": address,
                "mapping": mapping,
                "module": {"path": path, **binary},
                "status": "unresolved",
                "reason": binary["reason"],
            }
        if binary["elf_type"] == 2:  # ET_EXEC
            vma = address
        elif binary["elf_type"] == 3:  # ET_DYN
            page = os.sysconf("SC_PAGE_SIZE")
            candidates = {
                segment["vaddr"]
                - (segment["vaddr"] % page)
                + address
                - mapping["start"]
                for segment in binary["segments"]
                if segment["offset"] - (segment["offset"] % page) == mapping["offset"]
            }
            if len(candidates) != 1:
                return {
                    "address": address,
                    "mapping": mapping,
                    "module": {"path": path, **binary},
                    "status": "unresolved",
                    "reason": "elf_load_segment_ambiguous",
                }
            vma = candidates.pop()
        else:
            return {
                "address": address,
                "mapping": mapping,
                "module": {"path": path, **binary},
                "status": "unresolved",
                "reason": f"unsupported_elf_type:{binary['elf_type']}",
            }
        return {
            "address": address,
            "mapping": mapping,
            "module": {"path": path, "sha256": binary["sha256"]},
            "status": "located",
            "vma": vma,
        }

    def resolve_addresses(self, addresses: list[int]) -> list[dict]:
        frames = []
        pending = defaultdict(list)
        for address in addresses:
            located = self._locate(address)
            if located["status"] != "located":
                frames.append(located)
                continue
            key = (located["module"]["path"], located["vma"])
            cached = self.frame_cache.get(key)
            if cached is None:
                pending[located["module"]["path"]].append(located["vma"])
            frames.append(located)
        for path, values in pending.items():
            unique = list(dict.fromkeys(values))
            try:
                child = subprocess.run(
                    ["addr2line", "-f", "-C", "-e", path, *map(hex, unique)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as exc:
                child = None
                failure = f"symbolizer_unavailable:{type(exc).__name__}"
            if child is None or child.returncode != 0:
                reason = (
                    failure if child is None else f"symbolizer_exit:{child.returncode}"
                )
                for vma in unique:
                    self.frame_cache[(path, vma)] = {
                        "status": "module_only",
                        "symbol": None,
                        "location": None,
                        "symbol_gap": reason,
                    }
                continue
            lines = child.stdout.splitlines()
            if len(lines) != 2 * len(unique):
                for vma in unique:
                    self.frame_cache[(path, vma)] = {
                        "status": "module_only",
                        "symbol": None,
                        "location": None,
                        "symbol_gap": "symbolizer_output_cardinality",
                    }
                continue
            for index, vma in enumerate(unique):
                symbol, location = lines[2 * index : 2 * index + 2]
                known = symbol not in {"", "??"}
                self.frame_cache[(path, vma)] = {
                    "status": "resolved" if known else "module_only",
                    "symbol": symbol if known else None,
                    "location": location
                    if location not in {"", "??:0", "??:?"}
                    else None,
                    **({} if known else {"symbol_gap": "symbol_unknown"}),
                }
        for frame in frames:
            if frame["status"] == "located":
                frame.update(self.frame_cache[(frame["module"]["path"], frame["vma"])])
        return frames

    def __call__(self, stack: list[int], api: str, truncated) -> dict:
        if not stack:
            return {"status": "gap", "gap": "missing_native_stack", "frames": []}
        frames = self.resolve_addresses(stack)
        if truncated is None:
            return {
                "status": "gap",
                "gap": "stack_truncation_unknown",
                "frames": frames,
            }
        if truncated:
            return {"status": "gap", "gap": "native_stack_truncated", "frames": frames}
        expected = _base_api_name(api)
        boundaries = [
            index
            for index, frame in enumerate(frames)
            if frame.get("symbol")
            and _base_api_name(frame["symbol"].split("(", 1)[0]) == expected
        ]
        if len(boundaries) != 1:
            return {
                "status": "gap",
                "gap": "creation_api_frame_unresolved"
                if not boundaries
                else "creation_api_frame_ambiguous",
                "frames": frames,
            }
        caller_index = boundaries[0] + 1
        if caller_index >= len(frames):
            return {
                "status": "gap",
                "gap": "creation_caller_missing",
                "api_frame_index": boundaries[0],
                "frames": frames,
            }
        caller = frames[caller_index]
        if caller.get("status") not in {"resolved", "module_only"}:
            return {
                "status": "gap",
                "gap": "creation_caller_module_unresolved",
                "api_frame_index": boundaries[0],
                "caller_frame_index": caller_index,
                "frames": frames,
            }
        return {
            "status": "resolved",
            "api_frame_index": boundaries[0],
            "caller_frame_index": caller_index,
            "caller": caller,
            "frames": frames,
        }


def _nested(left: dict, right: dict) -> bool:
    return (
        left["enter"]["ns"] <= right["enter"]["ns"]
        and right["exit"]["ns"] <= left["exit"]["ns"]
    ) or (
        right["enter"]["ns"] <= left["enter"]["ns"]
        and left["exit"]["ns"] <= right["exit"]["ns"]
    )


def _group_operations(calls: list[dict]) -> list[list[dict]]:
    groups = []
    for call in sorted(calls, key=lambda item: item["enter"]["seq"]):
        matching = [
            group
            for group in groups
            if group[0]["exit"]["context"] == call["exit"]["context"]
            and group[0]["exit"]["stream"] == call["exit"]["stream"]
            and group[0]["exit"]["tid"] == call["exit"]["tid"]
            and any(_nested(member, call) for member in group)
        ]
        if len(matching) == 1:
            matching[0].append(call)
        else:
            groups.append([call])
    return groups


def _primary(group: list[dict]):
    candidates = [
        call
        for call in group
        if all(
            call["enter"]["ns"] <= other["enter"]["ns"]
            and other["exit"]["ns"] <= call["exit"]["ns"]
            for other in group
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def _build_lifetimes(
    completed_calls: list[dict], resolve_stack: Callable, problems: list, gaps: list
):
    def evidence_gap(value):
        gaps.append(value)
        problems.append(value)

    creates = [
        call
        for call in completed_calls
        if _stream_create(call["exit"]["api"]) and call["exit"]["rc"] == 0
    ]
    destroys = [
        call
        for call in completed_calls
        if _stream_destroy(call["exit"]["api"]) and call["exit"]["rc"] == 0
    ]
    create_groups = _group_operations(creates)
    destroy_groups = _group_operations(destroys)
    operations = [
        (min(call["enter"]["seq"] for call in group), "create", group)
        for group in create_groups
    ] + [
        (min(call["enter"]["seq"] for call in group), "destroy", group)
        for group in destroy_groups
    ]
    active = {}
    generations = Counter()
    lifetimes = []
    for _, kind, group in sorted(operations):
        exit_row = group[0]["exit"]
        key = (exit_row["context"], exit_row["stream"])
        domains = [call["exit"]["domain"] for call in group]
        primary = _primary(group)
        operation_id = f"0x{key[0]:x}:0x{key[1]:x}:seq{group[0]['enter']['seq']}"
        if len(domains) != len(set(domains)) or primary is None:
            evidence_gap(f"lifetime_operation_ambiguous:{operation_id}")
        if kind == "destroy":
            lifetime = active.pop(key, None)
            if lifetime is None:
                evidence_gap(f"destroy_without_observed_lifetime:{operation_id}")
                continue
            lifetime["destroy"] = {
                "enter_ns": min(call["enter"]["ns"] for call in group),
                "exit_ns": max(call["exit"]["ns"] for call in group),
                "observed_apis": [
                    {
                        "api": call["exit"]["api"],
                        "domain": call["exit"]["domain"],
                        "correlation": call["exit"]["correlation"],
                    }
                    for call in group
                ],
            }
            continue
        if not exit_row.get("has_stream") or not exit_row["stream"]:
            evidence_gap(f"successful_create_missing_handle:{operation_id}")
            continue
        if key in active:
            evidence_gap(f"create_while_lifetime_active:{operation_id}")
        generations[key] += 1
        generation = generations[key]
        lifetime_id = f"0x{key[0]:x}:0x{key[1]:x}:g{generation}"
        observed = []
        for call in sorted(group, key=lambda item: item["enter"]["seq"]):
            entered, exited = call["enter"], call["exit"]
            resolution = resolve_stack(
                entered.get("native_stack", []),
                exited["api"],
                entered.get("native_stack_truncated"),
            )
            if resolution["status"] != "resolved":
                evidence_gap(
                    f"creation_stack_gap:{lifetime_id}:{exited['api']}:{resolution['gap']}"
                )
            observed.append(
                {
                    "api": exited["api"],
                    "domain": exited["domain"],
                    "correlation": exited["correlation"],
                    "enter_ns": entered["ns"],
                    "exit_ns": exited["ns"],
                    "flags": exited.get("flags", -1),
                    "has_priority": exited.get("has_priority", False),
                    "priority": exited.get("priority")
                    if exited.get("has_priority", False)
                    else None,
                    "native_stack": entered.get("native_stack", []),
                    "native_stack_truncated": entered.get("native_stack_truncated"),
                    "stack_resolution": resolution,
                }
            )
        flag_values = {item["flags"] for item in observed}
        priority_values = {
            item["priority"] for item in observed if item["has_priority"]
        }
        if len(flag_values) != 1 or next(iter(flag_values), -1) < 0:
            evidence_gap(f"creation_flags_ambiguous:{lifetime_id}")
        if len(priority_values) > 1:
            evidence_gap(f"creation_priority_ambiguous:{lifetime_id}")
        primary_index = group.index(primary) if primary is not None else None
        primary_resolution = (
            observed[primary_index]["stack_resolution"]
            if primary_index is not None
            else {"status": "gap", "gap": "logical_creator_ambiguous"}
        )
        owner = {
            "status": primary_resolution["status"],
            "evidence_rule": "direct caller frame above the outermost observed create API",
            **(
                {"frame": primary_resolution["caller"]}
                if primary_resolution["status"] == "resolved"
                else {"gap": primary_resolution["gap"]}
            ),
        }
        lifetime = {
            "lifetime_id": lifetime_id,
            "context": key[0],
            "stream": key[1],
            "generation": generation,
            "created_ns": max(call["exit"]["ns"] for call in group),
            "destroy": None,
            "flags": next(iter(flag_values)) if len(flag_values) == 1 else None,
            "priority": next(iter(priority_values))
            if len(priority_values) == 1
            else None,
            "logical_creation_api": primary["exit"]["api"] if primary else None,
            "observed_creation_apis": observed,
            "owner": owner,
        }
        lifetimes.append(lifetime)
        active[key] = lifetime
    return lifetimes


def _lifetime_at(lifetimes: list[dict], row: dict):
    stream = row.get("stream", 0)
    if stream in (0, 1, 2):
        return {"status": "special_stream", "stream": stream}
    matches = [
        lifetime
        for lifetime in lifetimes
        if lifetime["context"] == row["context"]
        and lifetime["stream"] == stream
        and lifetime["created_ns"] <= row["ns"]
        and (
            lifetime["destroy"] is None or row["ns"] <= lifetime["destroy"]["enter_ns"]
        )
    ]
    if len(matches) != 1:
        return {
            "status": "gap",
            "gap": "no_observed_lifetime" if not matches else "lifetime_ambiguous",
            "stream": stream,
        }
    lifetime = matches[0]
    return {
        "status": "resolved",
        "lifetime_id": lifetime["lifetime_id"],
        "flags": lifetime["flags"],
        "owner": lifetime["owner"],
    }


def analyze(root: Path, *, stack_resolver: Callable | None = None) -> dict:
    rows = read_rows(root / "cuda.jsonl")
    python = read_rows(root / "python.jsonl")
    manifest = json.loads((root / "manifest.json").read_text())
    problems = []
    evidence_gaps = []

    def evidence_gap(value):
        evidence_gaps.append(value)
        problems.append(value)

    build_record_path = root / "observer_build.json"
    if not build_record_path.is_file():
        evidence_gap("missing_observer_build_record")
    else:
        build_record = json.loads(build_record_path.read_text())
        if build_record.get("schema") != "capture_attribution_observer_build_v2":
            evidence_gap("observer_build_schema_gap")
        required_creation = {
            "RUNTIME": {
                "cudaStreamCreate_v3020",
                "cudaStreamCreateWithFlags_v5000",
                "cudaStreamCreateWithPriority_v5050",
            },
            "DRIVER": {"cuStreamCreate", "cuStreamCreateWithPriority"},
        }
        creation_callbacks = build_record.get("stream_creation_callbacks", {})
        for domain, required in required_creation.items():
            missing = required - set(creation_callbacks.get(domain, []))
            if missing:
                evidence_gap(
                    f"stream_creation_callback_coverage_gap:{domain}:{','.join(sorted(missing))}"
                )
        unparsed_creation = [
            name
            for name in build_record.get("unparsed_callbacks", [])
            if name.startswith(("cudaStreamCreate", "cuStreamCreate"))
        ]
        if unparsed_creation:
            evidence_gap(
                "unparsed_stream_creation_callbacks:" + ",".join(unparsed_creation)
            )

    for name, expected in manifest.get("artifacts_sha256", {}).items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            problems.append(f"artifact_hash_mismatch:{name}")
    if not manifest.get("artifacts_sha256"):
        problems.append("missing_final_manifest")
    if manifest.get("source_drift"):
        problems.append("source_drift")
    if manifest.get("asset_drift") or manifest.get("asset_inventory_added"):
        problems.append("asset_drift")
    if [r["seq"] for r in rows] != list(range(1, len(rows) + 1)):
        problems.append("noncontiguous_event_sequence")
    stopped = [r for r in python if r["event"] == "harness_stopped"]
    if len(stopped) != 1 or stopped[0]["cupti_rc"] != 0 or stopped[0]["live_threads"]:
        problems.append("observer_shutdown_incomplete_or_workers_alive")
    sites = {r["site_id"]: r for r in python if r["event"] == "python_begin_enter"}
    pending = {}
    completed_calls = []
    intervals = []
    open_captures = {}
    errors = []
    event_records = {}
    event_edges = []
    status_observations = []
    for row in rows:
        api = row["api"]
        capture = "StreamBeginCapture" in api or "StreamBeginRecapture" in api
        end = "StreamEndCapture" in api
        call = (row["tid"], row["domain"], row["cbid"], row["correlation"])
        if row["phase"] == "enter":
            if call in pending:
                problems.append(f"duplicate_api_enter:{call}")
            pending[call] = row
            if end:
                key = (row["domain"], row["context"], row["stream"])
                if key in open_captures:
                    open_captures[key]["end_enter_ns"] = row["ns"]
            continue
        entered = pending.pop(call, None)
        if entered is None and row.get("selected", False):
            problems.append(f"selected_api_exit_without_enter:{row['seq']}")
        if entered is not None:
            completed_calls.append({"enter": entered, "exit": row})
        if row["rc"] != 0:
            errors.append(row)
        if row["rc"] == 0 and row.get("status", -1) >= 0:
            status_observations.append(row)
        event_key = (row["domain"], row["context"], row.get("event"))
        if row["rc"] == 0 and row.get("event"):
            if api.startswith(("cudaEventDestroy", "cuEventDestroy")):
                event_records.pop(event_key, None)
            elif api.startswith(("cudaEventRecord", "cuEventRecord")):
                event_records[event_key] = row
            elif "StreamWaitEvent" in api:
                event_edges.append(
                    {
                        "wait": row,
                        "last_observed_record": event_records.get(event_key),
                        "interpretation": "observed event edge; check external flags and capture status",
                    }
                )
        if capture:
            if not entered:
                problems.append(f"capture_without_enter:{row['seq']}")
            if not row["has_stream"] or row["flags"] < 0 or row["mode"] < 0:
                problems.append(f"unknown_capture_metadata:{row['seq']}")
            if row["rc"] != 0:
                continue
            key = (row["domain"], row["context"], row["stream"])
            if key in open_captures:
                problems.append(f"overlapping_capture_same_stream:{row['seq']}")
            record = {
                "domain": row["domain"],
                "context": row["context"],
                "stream": row["stream"],
                "flags": row["flags"],
                "flags_source": row["flags_source"],
                "mode": row["mode"],
                "tid": row["tid"],
                "site_id": row["site_id"],
                "label": sites.get(row["site_id"], {}).get(
                    "label", "unclassified.native"
                ),
                "begin_enter_ns": entered["ns"] if entered else None,
                "begin_exit_ns": row["ns"],
                "end_enter_ns": None,
                "end_exit_ns": None,
                "end_rc": None,
                "native_stack": entered["native_stack"] if entered else [],
            }
            intervals.append(record)
            open_captures[key] = record
        if end:
            key = (row["domain"], row["context"], row["stream"])
            record = open_captures.pop(key, None)
            if record is None:
                problems.append(f"unmatched_capture_end:{row['seq']}")
            else:
                record["end_exit_ns"], record["end_rc"] = row["ns"], row["rc"]
    if pending:
        problems.append(f"unpaired_api_calls:{len(pending)}")
    if open_captures:
        problems.append(f"unclosed_captures:{len(open_captures)}")
    if not intervals:
        problems.append("no_successful_capture_observed")

    resolver = stack_resolver or NativeStackResolver(root, manifest)
    lifetimes = _build_lifetimes(completed_calls, resolver, problems, evidence_gaps)

    def attach_lifetime(row, kind):
        reference = _lifetime_at(lifetimes, row)
        if reference["status"] == "gap":
            evidence_gap(f"{kind}_lifetime_gap:{row['seq']}:{reference['gap']}")
        return reference

    for capture in intervals:
        capture["stream_lifetime"] = attach_lifetime(
            {
                "context": capture["context"],
                "stream": capture["stream"],
                "ns": capture["begin_exit_ns"],
                "seq": capture["site_id"],
            },
            "capture",
        )
    for edge in event_edges:
        edge["wait_stream_lifetime"] = attach_lifetime(edge["wait"], "event_wait")
        if edge["last_observed_record"] is not None:
            edge["record_stream_lifetime"] = attach_lifetime(
                edge["last_observed_record"], "event_record"
            )
        else:
            edge["record_stream_lifetime"] = {
                "status": "gap",
                "gap": "no_observed_event_record",
            }
    for observation in status_observations:
        observation["stream_lifetime"] = attach_lifetime(
            observation, "status_observation"
        )

    correlations = []
    for error in errors:
        if error["rc"] not in (900, 901, 906):
            continue
        matching = [
            i
            for i in intervals
            if i["context"] == error["context"]
            and i["begin_exit_ns"] <= error["ns"]
            and (i["end_enter_ns"] is None or error["ns"] <= i["end_enter_ns"])
        ]
        correlations.append(
            {
                "error": error,
                "observed_open_captures": matching,
                "interpretation": "temporal overlap, not proof of causation",
            }
        )
    return {
        "trace_structure_ok": not problems,
        "ownership_evidence_ok": not evidence_gaps,
        "problems": problems,
        "evidence_gaps": evidence_gaps,
        "scope": "observed callbacks only; nested runtime/driver rows are retained in one logical stream lifetime",
        "api_counts": dict(Counter(r["api"] for r in rows if r["phase"] == "enter")),
        "stream_lifetimes": lifetimes,
        "captures": intervals,
        "event_edges": event_edges,
        "stream_status_observations": status_observations,
        "capture_errors": correlations,
        "unclassified_captures": sum(
            i["label"].startswith("unclassified") for i in intervals
        ),
        "native_stack_note": "creation callers are resolved from this run's maps.txt only after the mapped binary hash matches manifest.json",
        "root_cause_closed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    result = analyze(args.trace)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["trace_structure_ok"] else 1)
