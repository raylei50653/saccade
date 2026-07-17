#!/usr/bin/env python3
"""Linux fail-closed runtime confinement and file-input attestation.

The Phase-A parent forks the final child itself so that Landlock is installed
before ``execve(2)`` starts the ELF loader.  A small seccomp filter reports only
file-consuming syscalls to the ptrace supervisor; all other syscalls run at
normal speed.  Landlock is the enforcing boundary and the supervisor is the
independent recorder/fail-closed classifier.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import signal
import stat
import struct
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "h0_runtime_confinement_plan_v1"
ATTESTATION_SCHEMA = "h0_runtime_inputs_v1"
BACKEND = "landlock_seccomp_ptrace_v1"
INGRESS_POLICY = "deny_external_bytes_v1"
TRACE_SCOPE = ("execve", "execveat", "mmap", "open", "openat", "openat2")

_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_SYS_SECCOMP = 317

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
_LANDLOCK_ACCESS_FS_REFER = 1 << 13
_LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14

_PR_SET_NO_NEW_PRIVS = 38
_SECCOMP_SET_MODE_FILTER = 1
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_TRACE = 0x7FF00000
_SECCOMP_RET_ALLOW = 0x7FFF0000
_AUDIT_ARCH_X86_64 = 0xC000003E

_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_RET_K = 0x06

_PTRACE_TRACEME = 0
_PTRACE_CONT = 7
_PTRACE_PEEKDATA = 2
_PTRACE_GETREGS = 12
_PTRACE_SETOPTIONS = 0x4200
_PTRACE_GETEVENTMSG = 0x4201
_PTRACE_O_TRACEFORK = 1 << 1
_PTRACE_O_TRACEVFORK = 1 << 2
_PTRACE_O_TRACECLONE = 1 << 3
_PTRACE_O_TRACEEXEC = 1 << 4
_PTRACE_O_TRACESECCOMP = 1 << 7
_PTRACE_O_EXITKILL = 1 << 20
_PTRACE_EVENT_FORK = 1
_PTRACE_EVENT_VFORK = 2
_PTRACE_EVENT_CLONE = 3
_PTRACE_EVENT_EXEC = 4
_PTRACE_EVENT_SECCOMP = 7
_PTRACE_OPTIONS = (
    _PTRACE_O_TRACEFORK
    | _PTRACE_O_TRACEVFORK
    | _PTRACE_O_TRACECLONE
    | _PTRACE_O_TRACEEXEC
    | _PTRACE_O_TRACESECCOMP
    | _PTRACE_O_EXITKILL
)
_WAIT_WALL = 0x40000000

_NR_OPEN = 2
_NR_MMAP = 9
_NR_EXECVE = 59
_NR_CREAT = 85
_NR_OPENAT = 257
_NR_EXECVEAT = 322
_NR_OPENAT2 = 437
_BLOCKED_INGRESS_BY_NR = {
    22: "pipe",
    29: "shmget",
    30: "shmat",
    31: "shmctl",
    41: "socket",
    42: "connect",
    43: "accept",
    44: "sendto",
    45: "recvfrom",
    46: "sendmsg",
    47: "recvmsg",
    48: "shutdown",
    49: "bind",
    50: "listen",
    51: "getsockname",
    52: "getpeername",
    53: "socketpair",
    54: "setsockopt",
    55: "getsockopt",
    64: "semget",
    65: "semop",
    66: "semctl",
    67: "shmdt",
    68: "msgget",
    69: "msgsnd",
    70: "msgrcv",
    71: "msgctl",
    101: "ptrace",
    103: "syslog",
    240: "mq_open",
    241: "mq_unlink",
    242: "mq_timedsend",
    243: "mq_timedreceive",
    244: "mq_notify",
    245: "mq_getsetattr",
    248: "add_key",
    249: "request_key",
    250: "keyctl",
    253: "inotify_init",
    254: "inotify_add_watch",
    255: "inotify_rm_watch",
    275: "splice",
    276: "tee",
    278: "vmsplice",
    288: "accept4",
    293: "pipe2",
    294: "inotify_init1",
    298: "perf_event_open",
    299: "recvmmsg",
    300: "fanotify_init",
    301: "fanotify_mark",
    303: "name_to_handle_at",
    304: "open_by_handle_at",
    307: "sendmmsg",
    310: "process_vm_readv",
    311: "process_vm_writev",
    312: "kcmp",
    319: "memfd_create",
    321: "bpf",
    323: "userfaultfd",
    425: "io_uring_setup",
    426: "io_uring_enter",
    427: "io_uring_register",
    434: "pidfd_open",
    438: "pidfd_getfd",
    440: "process_madvise",
}
_EXPLICIT_KERNEL_RESOURCE_BY_NR = {318: "getrandom"}
BLOCKED_INGRESS_SYSCALLS = tuple(_BLOCKED_INGRESS_BY_NR.values())
KERNEL_RESOURCES = ("exec_auxv", "getrandom")
_TRACED_SYSCALLS = (
    (
        _NR_OPEN,
        _NR_MMAP,
        _NR_EXECVE,
        _NR_CREAT,
        _NR_OPENAT,
        _NR_EXECVEAT,
        _NR_OPENAT2,
    )
    + tuple(_BLOCKED_INGRESS_BY_NR)
    + tuple(_EXPLICIT_KERNEL_RESOURCE_BY_NR)
)
_SYSCALL_NAMES = {
    _NR_OPEN: "open",
    _NR_MMAP: "mmap",
    _NR_EXECVE: "execve",
    _NR_CREAT: "open",
    _NR_OPENAT: "openat",
    _NR_EXECVEAT: "execveat",
    _NR_OPENAT2: "openat2",
    **_BLOCKED_INGRESS_BY_NR,
    **_EXPLICIT_KERNEL_RESOURCE_BY_NR,
}
_AT_FDCWD = -100
_O_ACCMODE = 3
_O_WRONLY = 1
_O_RDWR = 2
_PROT_READ = 1
_PROT_WRITE = 2
_PROT_EXEC = 4


class ConfinementError(RuntimeError):
    """The kernel boundary could not be installed or classified exactly."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


class _UserRegsStruct(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "r15",
            "r14",
            "r13",
            "r12",
            "rbp",
            "rbx",
            "r11",
            "r10",
            "r9",
            "r8",
            "rax",
            "rcx",
            "rdx",
            "rsi",
            "rdi",
            "orig_rax",
            "rip",
            "cs",
            "eflags",
            "rsp",
            "ss",
            "fs_base",
            "gs_base",
            "ds",
            "es",
            "fs",
            "gs",
        )
    ]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ConfinementError(f"runtime input is not a physical regular file: {path}")
    data = path.read_bytes()
    if len(data) != info.st_size:
        raise ConfinementError(f"runtime input changed while hashing: {path}")
    return len(data), hashlib.sha256(data).hexdigest()


def _absolute(path: Path) -> str:
    value = path.absolute().as_posix()
    if not path.is_absolute() or path.as_posix() != value:
        raise ConfinementError(f"runtime path is not canonical absolute: {path}")
    return value


def build_plan(
    *,
    root: Path,
    incomplete: Path,
    inventory: Mapping[str, Any],
    build_identity: Mapping[str, Any],
    denial_probe: Path,
    run_ids: Sequence[str],
) -> dict[str, Any]:
    """Freeze every file inode admitted to the child before it is forked."""
    files: dict[str, dict[str, Any]] = {}

    def admit(
        path: Path,
        *,
        binding: str,
        logical_paths: Sequence[Path],
        expected_length: int | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        real = path.resolve(strict=True)
        length, sha256 = _sha256_file(real)
        if expected_length is not None and (length, sha256) != (
            expected_length,
            expected_sha256,
        ):
            raise ConfinementError(f"runtime binding identity mismatch: {path}")
        aliases = {_absolute(real)}
        for logical in logical_paths:
            aliases.add(_absolute(logical))
        key = real.as_posix()
        existing = files.get(key)
        if existing is None:
            info = real.stat(follow_symlinks=False)
            files[key] = {
                "bindings": {binding},
                "executable": bool(info.st_mode & 0o111),
                "length": length,
                "logical_paths": aliases,
                "realpath": key,
                "sha256": sha256,
            }
        else:
            if (existing["length"], existing["sha256"]) != (length, sha256):
                raise ConfinementError(f"conflicting runtime identity: {real}")
            existing["bindings"].add(binding)
            existing["logical_paths"].update(aliases)

    for record in inventory["repository"]:
        if record["kind"] != "regular":
            continue
        path = root / record["path"]
        admit(
            path,
            binding="repository",
            logical_paths=(path,),
            expected_length=record["length"],
            expected_sha256=record["sha256"],
        )
    sequence_root = root / inventory["sequence"]["root"]
    for record in inventory["sequence"]["files"]:
        path = sequence_root / record["path"]
        admit(
            path,
            binding="sequence",
            logical_paths=(path,),
            expected_length=record["length"],
            expected_sha256=record["sha256"],
        )
    for binding in ("models_engines", "tool_runtime"):
        for record in inventory[binding]:
            logical = Path(record["logical_path"])
            logical = logical if logical.is_absolute() else root / logical
            admit(
                Path(record["realpath"]),
                binding=binding,
                logical_paths=(logical, Path(record["realpath"])),
                expected_length=record["length"],
                expected_sha256=record["sha256"],
            )
    for record in build_identity["artifacts"]:
        path = root / record["path"]
        admit(
            path,
            binding="build_artifact",
            logical_paths=(path,),
            expected_length=record["length"],
            expected_sha256=record["sha256"],
        )
    python_identity = build_identity["python"]
    admit(
        Path(python_identity["path"]),
        binding="tool_runtime",
        logical_paths=(root / ".venv/bin/python", Path(python_identity["path"])),
        expected_length=python_identity["length"],
        expected_sha256=python_identity["sha256"],
    )
    lookup_directories: set[str] = set()
    for record in files.values():
        for alias in record["logical_paths"]:
            lookup_directories.add(Path(alias).parent.as_posix())
            if "tool_runtime" in record["bindings"]:
                lookup_directories.update(
                    parent.as_posix()
                    for parent in Path(alias).parents
                    if parent.as_posix() != "/"
                )
    python_library_root = Path(python_identity["path"]).parent.parent / "lib"
    if python_library_root.is_dir():
        lookup_directories.add(python_library_root.resolve(strict=True).as_posix())
    output_directories = [
        (incomplete / "runs" / run_id).resolve(strict=True).as_posix()
        for run_id in run_ids
    ]
    resource_rules: list[dict[str, str]] = []
    for path, kind in ((Path("/proc"), "procfs"), (Path("/sys"), "sysfs")):
        if path.is_dir() and path.resolve(strict=True) == path:
            resource_rules.append({"kind": kind, "path": path.as_posix()})
    device_candidates = [Path("/dev/null"), Path("/dev/zero"), Path("/dev/urandom")]
    device_candidates.extend(sorted(Path("/dev").glob("nvidia*")))
    device_candidates.extend(sorted(Path("/dev/nvidia-caps").glob("nvidia-cap*")))
    device_candidates.extend(sorted(Path("/dev/dri").glob("renderD*")))
    for path in device_candidates:
        try:
            info = path.stat(follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISCHR(info.st_mode) and not path.is_symlink():
            resource_rules.append({"kind": "device", "path": path.as_posix()})
    resource_rules.sort(key=lambda item: item["path"].encode("utf-8"))

    public_files = []
    for record in files.values():
        public_files.append(
            {
                "bindings": sorted(record["bindings"]),
                "executable": record["executable"],
                "length": record["length"],
                "logical_paths": sorted(
                    record["logical_paths"], key=lambda value: value.encode("utf-8")
                ),
                "realpath": record["realpath"],
                "sha256": record["sha256"],
            }
        )
    public_files.sort(key=lambda item: item["realpath"].encode("utf-8"))
    public = {
        "backend": BACKEND,
        "blocked_ingress_syscalls": list(BLOCKED_INGRESS_SYSCALLS),
        "denial_probe": denial_probe.absolute().as_posix(),
        "files": public_files,
        "ingress_policy": INGRESS_POLICY,
        "kernel_resources": list(KERNEL_RESOURCES),
        "lookup_directories": sorted(
            lookup_directories, key=lambda value: value.encode("utf-8")
        ),
        "output_directories": output_directories,
        "resource_rules": resource_rules,
        "schema": PLAN_SCHEMA,
        "trace_scope": list(TRACE_SCOPE),
    }
    return {**public, "digest": _digest(public)}


def _libc() -> ctypes.CDLL:
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    library.ptrace.restype = ctypes.c_long
    return library


def _syscall(library: ctypes.CDLL, number: int, *args: object) -> int:
    ctypes.set_errno(0)
    value = int(library.syscall(number, *args))
    if value < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return value


def _add_landlock_rule(
    library: ctypes.CDLL, ruleset_fd: int, path: str, access: int
) -> None:
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = _PathBeneathAttr(access, fd, 0)
        _syscall(
            library,
            _SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )
    finally:
        os.close(fd)


def _install_landlock(plan: Mapping[str, Any]) -> int:
    library = _libc()
    abi = _syscall(
        library,
        _SYS_LANDLOCK_CREATE_RULESET,
        0,
        0,
        _LANDLOCK_CREATE_RULESET_VERSION,
    )
    if abi < 3:
        raise ConfinementError(f"Landlock ABI {abi} lacks required truncate handling")
    handled = (
        _LANDLOCK_ACCESS_FS_EXECUTE
        | _LANDLOCK_ACCESS_FS_WRITE_FILE
        | _LANDLOCK_ACCESS_FS_READ_FILE
        | _LANDLOCK_ACCESS_FS_READ_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_FILE
        | _LANDLOCK_ACCESS_FS_MAKE_CHAR
        | _LANDLOCK_ACCESS_FS_MAKE_DIR
        | _LANDLOCK_ACCESS_FS_MAKE_REG
        | _LANDLOCK_ACCESS_FS_MAKE_SOCK
        | _LANDLOCK_ACCESS_FS_MAKE_FIFO
        | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | _LANDLOCK_ACCESS_FS_MAKE_SYM
        | _LANDLOCK_ACCESS_FS_REFER
        | _LANDLOCK_ACCESS_FS_TRUNCATE
    )
    attr = _RulesetAttr(handled)
    ruleset_fd = _syscall(
        library,
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        0,
    )
    try:
        for record in plan["files"]:
            access = _LANDLOCK_ACCESS_FS_READ_FILE
            if record["executable"]:
                access |= _LANDLOCK_ACCESS_FS_EXECUTE
            _add_landlock_rule(library, ruleset_fd, record["realpath"], access)
        for path in plan["lookup_directories"]:
            _add_landlock_rule(library, ruleset_fd, path, _LANDLOCK_ACCESS_FS_READ_DIR)
        output_access = handled & ~(
            _LANDLOCK_ACCESS_FS_EXECUTE
            | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
            | _LANDLOCK_ACCESS_FS_MAKE_CHAR
        )
        for path in plan["output_directories"]:
            _add_landlock_rule(library, ruleset_fd, path, output_access)
        for resource in plan["resource_rules"]:
            access = _LANDLOCK_ACCESS_FS_READ_FILE
            if resource["kind"] in {"procfs", "sysfs"}:
                access |= _LANDLOCK_ACCESS_FS_READ_DIR
            else:
                access |= _LANDLOCK_ACCESS_FS_WRITE_FILE
            _add_landlock_rule(library, ruleset_fd, resource["path"], access)
        if int(library.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        _syscall(library, _SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)
    return abi


def _install_seccomp_trace() -> None:
    instructions = [
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 4),
        _SockFilter(_BPF_JMP_JEQ_K, 1, 0, _AUDIT_ARCH_X86_64),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_KILL_PROCESS),
        _SockFilter(_BPF_LD_W_ABS, 0, 0, 0),
    ]
    for number in _TRACED_SYSCALLS:
        instructions.extend(
            (
                _SockFilter(_BPF_JMP_JEQ_K, 0, 1, number),
                _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_TRACE),
            )
        )
    instructions.append(_SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW))
    filters = (_SockFilter * len(instructions))(*instructions)
    program = _SockFprog(len(instructions), filters)
    _syscall(
        _libc(),
        _SYS_SECCOMP,
        _SECCOMP_SET_MODE_FILTER,
        0,
        ctypes.byref(program),
    )


def _ptrace(request: int, pid: int, address: object = 0, data: object = 0) -> int:
    library = _libc()
    ctypes.set_errno(0)
    result = int(library.ptrace(request, pid, address, data))
    error = ctypes.get_errno()
    if result == -1 and error:
        raise OSError(error, os.strerror(error))
    return result


def _read_memory_word(pid: int, address: int) -> bytes:
    value = _ptrace(_PTRACE_PEEKDATA, pid, ctypes.c_void_p(address), 0)
    return struct.pack("l", value)


def _read_memory(pid: int, address: int, length: int) -> bytes:
    result = bytearray()
    while len(result) < length:
        result.extend(_read_memory_word(pid, address + len(result)))
    return bytes(result[:length])


def _read_c_string(pid: int, address: int, limit: int = 65536) -> str:
    if not address:
        raise ConfinementError("null path pointer")
    result = bytearray()
    word_size = struct.calcsize("l")
    while len(result) < limit:
        word = _read_memory_word(pid, address + len(result))
        end = word.find(b"\0")
        result.extend(word if end < 0 else word[:end])
        if end >= 0:
            try:
                return os.fsdecode(bytes(result))
            except UnicodeError as exc:
                raise ConfinementError("non-decodable syscall path") from exc
        if len(word) != word_size:
            break
    raise ConfinementError("unterminated or oversized syscall path")


def _signed(value: int, bits: int = 64) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


class _Recorder:
    def __init__(self, plan: Mapping[str, Any]) -> None:
        self.plan = plan
        self.files = {record["realpath"]: record for record in plan["files"]}
        self.aliases = {
            alias: record
            for record in plan["files"]
            for alias in record["logical_paths"]
        }
        self.lookup = frozenset(plan["lookup_directories"])
        self.tool_lookup = frozenset(
            parent
            for record in plan["files"]
            if "tool_runtime" in record["bindings"]
            for alias in record["logical_paths"]
            for parent in Path(alias).parents
        )
        self.outputs = tuple(Path(path) for path in plan["output_directories"])
        self.resources = tuple(
            (Path(item["path"]), item["kind"]) for item in plan["resource_rules"]
        )
        self.observed_files: dict[str, set[str]] = {}
        self.observed_resources: dict[tuple[str, str], set[str]] = {}
        self.violations: list[dict[str, str]] = []
        self.denial_probe_count = 0

    def _violate(self, operation: str, path: str, reason: str) -> None:
        self.violations.append({"operation": operation, "path": path, "reason": reason})

    def _candidate(self, pid: int, raw: str, dirfd: int) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        base_link = (
            Path(f"/proc/{pid}/cwd")
            if dirfd == _AT_FDCWD
            else Path(f"/proc/{pid}/fd/{dirfd}")
        )
        try:
            base = Path(os.readlink(base_link))
        except OSError as exc:
            raise ConfinementError(f"unresolvable syscall dirfd {dirfd}") from exc
        return base / path

    def observe_path(
        self, pid: int, raw: str, dirfd: int, operation: str, *, write: bool = False
    ) -> bool:
        try:
            candidate = self._candidate(pid, raw, dirfd)
        except ConfinementError as exc:
            self._violate(operation, raw, str(exc))
            return False
        lexical = candidate.as_posix()
        parts = PurePosixPath(lexical).parts
        if not candidate.is_absolute():
            self._violate(operation, lexical, "non_canonical_path")
            return False
        non_canonical = any(part in {".", ".."} for part in parts)
        if lexical == self.plan["denial_probe"]:
            self.denial_probe_count += 1
            return True
        for output in self.outputs:
            if candidate == output or output in candidate.parents:
                try:
                    resolved_output = candidate.resolve(strict=False)
                except OSError:
                    self._violate(operation, lexical, "unclassifiable_output_path")
                    return False
                if non_canonical:
                    self._violate(operation, lexical, "non_canonical_path")
                    return False
                if resolved_output != candidate:
                    self._violate(operation, lexical, "unbound_output_alias")
                    return False
                self.observed_resources.setdefault(("run_output", lexical), set()).add(
                    operation
                )
                return True
        for resource_root, kind in self.resources:
            beneath = candidate == resource_root or (
                kind in {"procfs", "sysfs"} and resource_root in candidate.parents
            )
            if beneath:
                try:
                    resolved_resource = candidate.resolve(strict=False)
                except OSError:
                    self._violate(operation, lexical, "unclassifiable_resource_path")
                    return False
                if non_canonical:
                    self._violate(operation, lexical, "non_canonical_path")
                    return False
                if not (
                    resolved_resource == resource_root
                    or resource_root in resolved_resource.parents
                ):
                    self._violate(operation, lexical, "resource_path_escape")
                    return False
                self.observed_resources.setdefault((kind, lexical), set()).add(
                    operation
                )
                return True
        try:
            real = candidate.resolve(strict=True)
            info = real.stat(follow_symlinks=False)
        except FileNotFoundError:
            normalized_parent = candidate.resolve(strict=False).parent.as_posix()
            normalized = candidate.resolve(strict=False)
            if (
                candidate.parent.as_posix() in self.lookup
                or any(
                    root == normalized or root in normalized.parents
                    for root in self.tool_lookup
                )
                or (
                    non_canonical
                    and (
                        normalized_parent in self.lookup
                        or any(
                            root == candidate.resolve(strict=False)
                            or root in candidate.resolve(strict=False).parents
                            for root in self.tool_lookup
                        )
                        or any(
                            Path(root) == candidate.resolve(strict=False)
                            or Path(root) in candidate.resolve(strict=False).parents
                            for root in self.lookup
                        )
                    )
                )
            ):
                return True
            self._violate(operation, lexical, "unbound_missing_lookup")
            return False
        except OSError:
            self._violate(operation, lexical, "unclassifiable_path")
            return False
        if stat.S_ISDIR(info.st_mode):
            if real.as_posix() in self.lookup or any(
                Path(root) in real.parents for root in self.lookup
            ):
                self.observed_resources.setdefault(
                    ("bound_directory", real.as_posix()), set()
                ).add(operation)
                return True
            self._violate(operation, lexical, "unbound_directory")
            return False
        if not stat.S_ISREG(info.st_mode):
            self._violate(operation, lexical, "unbound_non_file_resource")
            return False
        record = self.files.get(real.as_posix())
        alias_record = self.aliases.get(lexical)
        loader_traversal = (
            non_canonical
            and record is not None
            and "tool_runtime" in record["bindings"]
        )
        if non_canonical and not loader_traversal:
            self._violate(operation, lexical, "non_canonical_path")
            return False
        if record is None or (alias_record is not record and not loader_traversal):
            self._violate(operation, lexical, "unbound_regular_file")
            return False
        if write:
            self._violate(operation, lexical, "write_to_bound_input")
            return False
        if (info.st_size, info.st_mode & 0o111) != (
            record["length"],
            (0o111 if record["executable"] else 0),
        ):
            # Compare executable as a boolean; normalize the mode expression.
            if info.st_size != record["length"] or bool(info.st_mode & 0o111) != bool(
                record["executable"]
            ):
                self._violate(operation, lexical, "runtime_identity_drift")
                return False
        self.observed_files.setdefault(real.as_posix(), set()).add(operation)
        return True

    def observe_fd(self, pid: int, fd: int, operation: str, *, write: bool) -> bool:
        if fd < 0:
            return True
        try:
            raw = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            self._violate(operation, f"fd:{fd}", "unclassifiable_file_descriptor")
            return False
        if raw.startswith(("anon_inode:", "socket:", "pipe:", "/memfd:")):
            self._violate(operation, raw, "unplanned_kernel_object")
            return False
        return self.observe_path(pid, raw, _AT_FDCWD, operation, write=write)

    def observe_syscall(self, pid: int, registers: _UserRegsStruct) -> bool:
        number = int(registers.orig_rax)
        operation = _SYSCALL_NAMES.get(number, f"syscall_{number}")
        try:
            if number in _EXPLICIT_KERNEL_RESOURCE_BY_NR:
                self.observed_resources.setdefault(
                    ("kernel_random", f"syscall:{operation}"), set()
                ).add(operation)
                return True
            if number in _BLOCKED_INGRESS_BY_NR:
                self._violate(
                    operation,
                    f"syscall:{operation}",
                    "forbidden_runtime_ingress",
                )
                return False
            if number == _NR_OPEN:
                raw = _read_c_string(pid, int(registers.rdi))
                flags = int(registers.rsi)
                return self.observe_path(
                    pid,
                    raw,
                    _AT_FDCWD,
                    operation,
                    write=(flags & _O_ACCMODE) in {_O_WRONLY, _O_RDWR},
                )
            if number == _NR_CREAT:
                raw = _read_c_string(pid, int(registers.rdi))
                return self.observe_path(pid, raw, _AT_FDCWD, operation, write=True)
            if number in {_NR_OPENAT, _NR_OPENAT2}:
                raw = _read_c_string(pid, int(registers.rsi))
                dirfd = _signed(int(registers.rdi), 32)
                if number == _NR_OPENAT:
                    flags = int(registers.rdx)
                else:
                    flags = struct.unpack(
                        "Q", _read_memory(pid, int(registers.rdx), 8)
                    )[0]
                return self.observe_path(
                    pid,
                    raw,
                    dirfd,
                    operation,
                    write=(flags & _O_ACCMODE) in {_O_WRONLY, _O_RDWR},
                )
            if number == _NR_EXECVE:
                raw = _read_c_string(pid, int(registers.rdi))
                return self.observe_path(pid, raw, _AT_FDCWD, operation)
            if number == _NR_EXECVEAT:
                raw = _read_c_string(pid, int(registers.rsi))
                return self.observe_path(
                    pid, raw, _signed(int(registers.rdi), 32), operation
                )
            if number == _NR_MMAP:
                prot = int(registers.rdx)
                flags = int(registers.r10)
                fd = _signed(int(registers.r8), 32)
                mmap_operation = (
                    "mmap_exec"
                    if prot & _PROT_EXEC
                    else "mmap_read"
                    if prot & _PROT_READ
                    else "mmap"
                )
                return self.observe_fd(
                    pid,
                    fd,
                    mmap_operation,
                    write=bool(prot & _PROT_WRITE) and (flags & 3) in {1, 3},
                )
        except (ConfinementError, OSError, struct.error) as exc:
            self._violate(operation, "", f"unclassifiable_syscall:{exc}")
            return False
        self._violate(operation, "", "unknown_traced_syscall")
        return False

    def observe_startup_mappings(self, pid: int) -> bool:
        """Record ELF mappings created by the kernel itself during execve."""
        self.observed_resources.setdefault(
            ("kernel_auxv", "kernel:exec_auxv"), set()
        ).add("execve")
        try:
            lines = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            self._violate(
                "startup_mapping", "", f"unclassifiable_startup_mappings:{exc}"
            )
            return False
        accepted = True
        for line in lines:
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            if not self.observe_path(pid, fields[5], _AT_FDCWD, "startup_mapping"):
                accepted = False
        return accepted

    def attestation(
        self,
        *,
        landlock_abi: int,
        boundary_installed: bool,
        process_tree_terminal: bool,
    ) -> dict[str, Any]:
        regular_files: list[dict[str, Any]] = []
        for realpath, operations in self.observed_files.items():
            source = self.files[realpath]
            try:
                length, sha256 = _sha256_file(Path(realpath))
            except (ConfinementError, OSError) as exc:
                self._violate(
                    "final_identity", realpath, f"runtime_identity_drift:{exc}"
                )
                length, sha256 = source["length"], source["sha256"]
            if (length, sha256) != (source["length"], source["sha256"]):
                self._violate("final_identity", realpath, "runtime_identity_drift")
            roles = set(source["bindings"])
            suffixes = Path(realpath).suffixes
            if any(suffix in {".py", ".pyc"} for suffix in suffixes):
                roles.add("python_module")
            if ".so" in suffixes or ".so." in Path(realpath).name:
                roles.add("shared_library")
            if "execve" in operations or "execveat" in operations:
                roles.add("interpreter_or_executable")
            regular_files.append(
                {
                    "bindings": sorted(source["bindings"]),
                    "length": source["length"],
                    "logical_paths": list(source["logical_paths"]),
                    "operations": sorted(operations),
                    "realpath": realpath,
                    "roles": sorted(roles),
                    "sha256": source["sha256"],
                }
            )
        regular_files.sort(key=lambda item: item["realpath"].encode("utf-8"))
        resources = [
            {"kind": kind, "operations": sorted(operations), "path": path}
            for (kind, path), operations in self.observed_resources.items()
        ]
        resources.sort(key=lambda item: (item["path"].encode("utf-8"), item["kind"]))
        if self.denial_probe_count != 1:
            self._violate(
                "denial_probe",
                self.plan["denial_probe"],
                f"expected_once_observed_{self.denial_probe_count}",
            )
        state = (
            "complete"
            if boundary_installed and process_tree_terminal and not self.violations
            else "rejected"
        )
        return {
            "backend": BACKEND,
            "confinement_plan": {
                key: value for key, value in self.plan.items() if key != "digest"
            },
            "confinement_plan_digest": self.plan["digest"],
            "denial_probe_observed": self.denial_probe_count == 1,
            "ingress_policy": INGRESS_POLICY,
            "installed_before_exec": boundary_installed,
            "landlock_abi": landlock_abi,
            "process_tree_terminal": process_tree_terminal,
            "regular_files": regular_files,
            "resources": resources,
            "schema": ATTESTATION_SCHEMA,
            "state": state,
            "trace_scope": list(TRACE_SCOPE),
            "violations": list(self.violations),
        }


class ConfinedProcess:
    """Small ``Popen``-compatible ptrace supervisor used by the controller."""

    def __init__(self, pid: int, plan: Mapping[str, Any], landlock_abi: int) -> None:
        self.pid = pid
        self._tracees = {pid}
        self._main_returncode: int | None = None
        self._recorder = _Recorder(plan)
        self._landlock_abi = landlock_abi
        self._boundary_installed = True

    def _resume(self, pid: int, delivered_signal: int = 0) -> None:
        try:
            _ptrace(_PTRACE_CONT, pid, 0, delivered_signal)
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                raise

    def _kill_tree(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for tracee in tuple(self._tracees):
            try:
                os.kill(tracee, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _event(self, pid: int, status: int) -> None:
        if os.WIFEXITED(status):
            self._tracees.discard(pid)
            if pid == self.pid:
                self._main_returncode = os.WEXITSTATUS(status)
            return
        if os.WIFSIGNALED(status):
            self._tracees.discard(pid)
            if pid == self.pid:
                self._main_returncode = -os.WTERMSIG(status)
            return
        if not os.WIFSTOPPED(status):
            return
        event = status >> 16
        stop_signal = os.WSTOPSIG(status)
        if event == _PTRACE_EVENT_SECCOMP:
            registers = _UserRegsStruct()
            _ptrace(_PTRACE_GETREGS, pid, 0, ctypes.byref(registers))
            if not self._recorder.observe_syscall(pid, registers):
                self._kill_tree()
            self._resume(pid)
            return
        if event in {_PTRACE_EVENT_FORK, _PTRACE_EVENT_VFORK, _PTRACE_EVENT_CLONE}:
            new_pid = ctypes.c_ulonglong()
            _ptrace(_PTRACE_GETEVENTMSG, pid, 0, ctypes.byref(new_pid))
            self._tracees.add(int(new_pid.value))
            self._resume(pid)
            return
        if event == _PTRACE_EVENT_EXEC:
            if not self._recorder.observe_startup_mappings(pid):
                self._kill_tree()
            self._resume(pid)
            return
        self._resume(
            pid, 0 if stop_signal in {signal.SIGSTOP, signal.SIGTRAP} else stop_signal
        )

    def _pump(self, deadline: float | None) -> None:
        first_pass = True
        while self._tracees:
            if not first_pass and deadline is not None and time.monotonic() >= deadline:
                return
            first_pass = False
            progressed = False
            for tracee in tuple(sorted(self._tracees)):
                try:
                    pid, status = os.waitpid(tracee, os.WNOHANG | _WAIT_WALL)
                except ChildProcessError as exc:
                    self._recorder._violate(
                        "process_tree",
                        f"pid:{tracee}",
                        "unclassifiable_tracee_terminal",
                    )
                    try:
                        os.kill(tracee, 0)
                    except ProcessLookupError:
                        self._tracees.discard(tracee)
                        if tracee == self.pid and self._main_returncode is None:
                            self._main_returncode = -signal.SIGKILL
                        progressed = True
                        continue
                    self._kill_tree()
                    raise ConfinementError(
                        f"lost supervision of live tracee {tracee}"
                    ) from exc
                if pid == 0:
                    continue
                progressed = True
                self._event(pid, status)
            if not progressed:
                time.sleep(0.001)
        if self._main_returncode is None:
            self._recorder._violate(
                "process_tree", f"pid:{self.pid}", "missing_main_terminal_status"
            )
            self._main_returncode = -signal.SIGKILL

    def poll(self) -> int | None:
        self._pump(time.monotonic())
        return self._main_returncode if not self._tracees else None

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        self._pump(deadline)
        if self._tracees or self._main_returncode is None:
            import subprocess

            raise subprocess.TimeoutExpired(self.pid, timeout)
        return self._main_returncode

    def terminate_tree(self) -> None:
        self._kill_tree()
        self._pump(None)

    @property
    def live_tracee_count(self) -> int:
        return len(self._tracees)

    def runtime_attestation(self) -> dict[str, Any]:
        if self._tracees or self._main_returncode is None:
            raise ConfinementError(
                "runtime attestation requested before process-tree terminal"
            )
        return self._recorder.attestation(
            landlock_abi=self._landlock_abi,
            boundary_installed=self._boundary_installed,
            process_tree_terminal=True,
        )


def _validate_standard_streams(
    stdin: Any, stdout: Any, stderr: Any, plan: Mapping[str, Any]
) -> None:
    """Admit no inherited byte channel beyond /dev/null and evidence logs."""

    def target(stream: Any, name: str) -> tuple[Path, os.stat_result]:
        try:
            fd = int(stream.fileno())
            info = os.fstat(fd)
            raw = os.readlink(f"/proc/self/fd/{fd}")
        except (OSError, TypeError, ValueError) as exc:
            raise ConfinementError(f"unclassifiable {name} stream: {exc}") from exc
        path = Path(raw)
        if not path.is_absolute() or raw.endswith(" (deleted)"):
            raise ConfinementError(f"non-physical {name} stream: {raw}")
        return path, info

    stdin_path, stdin_info = target(stdin, "stdin")
    if stdin_path != Path("/dev/null") or not stat.S_ISCHR(stdin_info.st_mode):
        raise ConfinementError("stdin is not the admitted /dev/null device")
    output_roots = tuple(Path(path) for path in plan["output_directories"])
    for stream, name in ((stdout, "stdout"), (stderr, "stderr")):
        path, info = target(stream, name)
        if (
            not stat.S_ISREG(info.st_mode)
            or not any(path == root or root in path.parents for root in output_roots)
            or path.resolve(strict=True) != path
        ):
            raise ConfinementError(f"{name} is not a physical regular evidence output")


def spawn_confined(
    vector: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdin: Any,
    stdout: Any,
    stderr: Any,
    plan: Mapping[str, Any],
) -> ConfinedProcess:
    """Fork, install the boundary, and exec the literal RC1.1 vector."""
    if os.uname().machine != "x86_64":
        raise ConfinementError("runtime syscall attestation requires x86_64")
    _validate_standard_streams(stdin, stdout, stderr, plan)
    ready_read, ready_write = os.pipe2(os.O_CLOEXEC)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - asserted through the supervising parent
        try:
            os.close(ready_read)
            os.setsid()
            os.chdir(cwd)
            for source, target in (
                (stdin.fileno(), 0),
                (stdout.fileno(), 1),
                (stderr.fileno(), 2),
            ):
                if source != target:
                    os.dup2(source, target, inheritable=True)
            if ready_write != 3:
                os.dup2(ready_write, 3, inheritable=True)
                os.close(ready_write)
                ready_write = 3
            maximum = min(int(os.sysconf("SC_OPEN_MAX")), 1_048_576)
            os.closerange(4, maximum)
            _ptrace(_PTRACE_TRACEME, 0, 0, 0)
            os.kill(os.getpid(), signal.SIGSTOP)
            abi = _install_landlock(plan)
            _install_seccomp_trace()
            os.write(ready_write, f"READY {abi}\n".encode("ascii"))
            os.close(ready_write)
            os.execve(vector[0], list(vector), dict(env))
        except BaseException as exc:
            try:
                os.write(ready_write, f"ERROR {type(exc).__name__}: {exc}\n".encode())
            except OSError:
                pass
            os._exit(127)
    os.close(ready_write)
    try:
        waited, status = os.waitpid(pid, _WAIT_WALL)
        if waited != pid or not os.WIFSTOPPED(status):
            raise ConfinementError("child did not stop before confinement setup")
        _ptrace(_PTRACE_SETOPTIONS, pid, 0, _PTRACE_OPTIONS)
        _ptrace(_PTRACE_CONT, pid, 0, 0)
        ready = bytearray()
        while True:
            chunk = os.read(ready_read, 4096)
            if not chunk:
                break
            ready.extend(chunk)
            if len(ready) > 65536:
                raise ConfinementError("oversized confinement setup response")
        message = ready.decode("utf-8", errors="replace").strip()
        if not message.startswith("READY "):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise ConfinementError(f"runtime confinement setup failed: {message}")
        abi = int(message.split(" ", 1)[1])
        return ConfinedProcess(pid, plan, abi)
    finally:
        os.close(ready_read)
