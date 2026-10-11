"""N03 sealed-memfd load and the loader auditor on a real loader (#549 S2-2).

docs/reference/model_bundle_s2_2_n03_549.md (P-N03 rev 3c, D-N03-2..5). An
install-shaped prefix is built in ``tmp_path`` with the real
``shipping/src/loader_audit.c`` and launcher, a bundled ``lib/vendor/libfoo.so``
and a stand-in operator library at ``share/saccade/build/
libsaccade_scan_torchop.so`` (``op()`` returns 1; a byte-different copy
returns 2). ``libexec/saccade_track`` is a host driver compiled from the
production ``shipping/src/operator_loader.cpp``: it names its runtime
package's operator from its own mapping (D-N03-5), then prepares (read once,
hash, sealed labeled memfd, readback) and loads it (dlopen of the memfd,
mapping post-check). A forger (C) maps objects the ways the auditor must
refuse.

* positives: the driver through the launcher (explicit loader, --audit) and
  by direct exec with LD_AUDIT both derive <prefix>/share/saccade/build/...
  and run the operator from its memfd; the vDSO and the vendor library pass;
* D-N03-5: a plan naming another file is refused by the host (exit 2);
* races (production host, real auditor): rename-replace or in-place write of
  the source after the hash is refused by the auditor (127); an in-place
  write after the load does not change what runs;
* auditor refusals (127): an unlabeled, unsealed, forged-content or
  forged-source memfd; a source descriptor at a foreign path or a foreign
  hardlink; a deleted or regular file via /proc/self/fd; a relative alias to
  a memfd; an alias whose realpath is a planted "... (deleted)" file; an
  ordinary file whose name ends in " (deleted)" (compatibility restriction);
  the operator by path from a foreign directory;
* the operator by path at the fixed location still passes (path mode).

Skips without cc, g++ or the x86-64 loader.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LDSO = Path("/lib64/ld-linux-x86-64.so.2")
OP_REL = "share/saccade/build/libsaccade_scan_torchop.so"

pytestmark = pytest.mark.skipif(
    not shutil.which("cc") or not shutil.which("g++") or not LDSO.exists(),
    reason="needs cc, g++ and the x86-64 loader",
)

_OP_SRC = "int op(void) { return VALUE; }\n"

_DRIVER_SRC = r"""
// The production N03 host sequence (operator_loader.hpp) as an entrypoint.
//   load <sha>             prepare + load the runtime package's operator,
//                          print "value=<op()> mapped=<kernel path>"
//   foreign <path> <sha>   DetectorHost's D-N03-5 check for a binding <path>
// Env N03_HOOK runs (sh -c) at N03_HOOK_STAGE after_prepare | after_load.
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include "saccade_shipping/detector_plan.hpp"
#include "saccade_shipping/operator_loader.hpp"
namespace sh = saccade::shipping;
extern char** environ;
static void hook(const char* stage) {
    const char* want = std::getenv("N03_HOOK_STAGE");
    const char* cmd = std::getenv("N03_HOOK");
    if (!want || !cmd || std::strcmp(want, stage) != 0) return;
    pid_t pid;
    char* argv[] = {(char*)"/bin/sh", (char*)"-c", (char*)cmd, nullptr};
    char* envp[] = {(char*)"PATH=/usr/bin:/bin", nullptr};
    int st = 0;
    if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, argv, envp) != 0 || waitpid(pid, &st, 0) < 0 || st != 0) {
        std::fprintf(stderr, "hook failed\n");
        std::_Exit(65);
    }
}
static int call(void* h) {
    auto fn = reinterpret_cast<int (*)()>(dlsym(h, "op"));
    return fn ? fn() : -1;
}
int main(int argc, char** argv) {
    try {
        const std::string legal = sh::entrypoint_runtime_operator_path();
        if (argc == 3 && std::string(argv[1]) == "load") {
            sh::VerifiedOperator op = sh::VerifiedOperator::prepare(legal, argv[2]);
            hook("after_prepare");
            void* h = op.load();
            std::printf("value=%d mapped=%s\n", call(h), op.mapped().path.c_str());
            std::fflush(stdout);
            hook("after_load");
            std::printf("value_after=%d\n", call(h));
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "foreign") {
            const std::string bound = std::filesystem::path(argv[2]).lexically_normal().string();
            if (bound != legal) {
                std::printf("refused: %s is not the runtime package's %s\n", bound.c_str(), legal.c_str());
                return 2;
            }
            (void)sh::VerifiedOperator::prepare(legal, argv[3]).load();
            return 0;
        }
    } catch (const sh::DetectorLoadError& e) {
        std::printf("refused: %s\n", e.what());
        return 2;
    }
    return 64;
}
"""

_FORGER_SRC = r"""
/* Maps the stand-in operator the ways the auditor must refuse (or, for
 * path-pinned, accept). argv: <mode> <fixed op> <other file> <work dir> */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
static char buf[1 << 20];
static size_t slurp(const char *p) {
    int fd = open(p, O_RDONLY);
    ssize_t n = fd < 0 ? -1 : read(fd, buf, sizeof buf);
    if (n <= 0) { perror(p); exit(64); }
    close(fd);
    return (size_t)n;
}
static int memfd(const char *label, const char *src, int seals) {
    size_t n = slurp(src);
    int m = memfd_create(label, MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (m < 0 || write(m, buf, n) != (ssize_t)n) exit(65);
    if (seals && fcntl(m, F_ADD_SEALS, seals) != 0) exit(66);
    return m;
}
int main(int argc, char **argv) {
    if (argc != 5) return 64;
    const char *mode = argv[1], *fixed = argv[2], *other = argv[3], *work = argv[4];
    int all = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    char label[64], load[4096], alias[4096];
    int s = -1, m = -1;
    if (!strcmp(mode, "unlabeled")) {
        m = memfd("saccade_operator_candidate", fixed, all);
    } else if (!strcmp(mode, "unsealed")) {
        s = open(fixed, O_RDONLY);
        snprintf(label, sizeof label, "saccade-n03:src-fd=%d", s);
        m = memfd(label, fixed, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL);
    } else if (!strcmp(mode, "forged-content")) {  /* label names the fixed source, other bytes */
        s = open(fixed, O_RDONLY);
        snprintf(label, sizeof label, "saccade-n03:src-fd=%d", s);
        m = memfd(label, other, all);
    } else if (!strcmp(mode, "closed-source")) {
        m = memfd("saccade-n03:src-fd=987", fixed, all);
    } else if (!strcmp(mode, "foreign-source")) {  /* same bytes, source opened elsewhere */
        s = open(other, O_RDONLY);
        snprintf(label, sizeof label, "saccade-n03:src-fd=%d", s);
        m = memfd(label, other, all);
    } else if (!strcmp(mode, "fd-regular")) {
        s = open(fixed, O_RDONLY);
    } else if (!strcmp(mode, "fd-deleted")) {
        s = open(other, O_RDONLY);
        unlink(other);
    } else if (!strcmp(mode, "alias-relative")) {
        m = memfd("saccade_operator_candidate", fixed, all);
        snprintf(load, sizeof load, "/proc/self/fd/%d", m);
        snprintf(alias, sizeof alias, "%s/alias.so", work);
        if (symlink(load, alias) != 0 || chdir(work) != 0) return 67;
        m = -1;
        strcpy(load, "./alias.so");
    } else if (!strcmp(mode, "alias-spoof")) {
        s = open(other, O_RDONLY);
        char planted[4096];
        snprintf(planted, sizeof planted, "%s (deleted)", other);
        unlink(other);
        int pf = open(planted, O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (pf < 0 || write(pf, "x", 1) != 1) return 68;
        close(pf);
        snprintf(load, sizeof load, "/proc/self/fd/%d", s);
        snprintf(alias, sizeof alias, "%s/spoof.so", work);
        if (symlink(load, alias) != 0) return 69;
        s = -1;
        strcpy(load, alias);
    } else if (!strcmp(mode, "path")) {
        strcpy(load, other);
    } else {
        return 64;
    }
    if (m >= 0) snprintf(load, sizeof load, "/proc/self/fd/%d", m);
    else if (s >= 0) snprintf(load, sizeof load, "/proc/self/fd/%d", s);
    void *h = dlopen(load, RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 2; }
    int (*op)(void) = (int (*)(void))dlsym(h, "op");
    printf("loaded value=%d\n", op ? op() : -1);
    return 0;
}
"""


def _cc(*args: str) -> None:
    subprocess.run(["cc", *args], check=True, capture_output=True)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def prefix(tmp_path: Path) -> Path:
    p = tmp_path / "prefix"
    for d in (
        "lib/vendor",
        "libexec",
        "gen",
        "src",
        "bin",
        "share/saccade/build",
        "other",
    ):
        (p / d).mkdir(parents=True, exist_ok=True)
    src = p / "src"
    (src / "foo.c").write_text("int foo(void) { return 1; }\n")
    (src / "op.c").write_text(_OP_SRC)
    (src / "driver.cpp").write_text(_DRIVER_SRC)
    (src / "forger.c").write_text(_FORGER_SRC)
    (p / "gen" / "loader_audit_names.inc").write_text(
        'static const char *const kBundled[] = {"libfoo.so", NULL};\n'
    )
    _cc(
        "-shared",
        "-fPIC",
        "-Wl,-soname,libfoo.so",
        "-o",
        str(p / "lib/vendor/libfoo.so"),
        str(src / "foo.c"),
    )
    _cc(
        "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror", f"-I{p / 'gen'}",
        "-o", str(p / "lib/saccade_loader_audit.so"), str(REPO / "shipping/src/loader_audit.c"),
    )  # fmt: skip
    # the stand-in operator (1) at the fixed path, a byte-different copy (2)
    for value, out in ((1, p / OP_REL), (2, p / "other" / "op2.so")):
        _cc("-shared", "-fPIC", f"-DVALUE={value}", "-o", str(out), str(src / "op.c"))
    subprocess.run(
        [
            "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{REPO / 'shipping/include'}",
            "-o", str(p / "libexec/saccade_track"), str(src / "driver.cpp"),
            str(REPO / "shipping/src/operator_loader.cpp"), str(REPO / "shipping/src/sha256.cpp"),
            f"-L{p / 'lib/vendor'}", "-Wl,--no-as-needed", "-lfoo", "-ldl",
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    _cc(
        "-std=gnu11",
        "-Wall",
        "-o",
        str(p / "libexec/forger"),
        str(src / "forger.c"),
        "-ldl",
    )
    shutil.copy(REPO / "shipping/launcher/saccade_track.sh", p / "bin/saccade_track")
    (p / "bin/saccade_track").chmod(0o755)
    return p


def _audited(
    p: Path, exe: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LDSO), "--library-path", str(p / "lib/vendor"), "--audit", str(p / "lib/saccade_loader_audit.so"),
         str(p / "libexec" / exe), *args],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
    )  # fmt: skip


def _forge(
    p: Path, mode: str, other: Path, work: Path
) -> subprocess.CompletedProcess[str]:
    work.mkdir(exist_ok=True)
    return _audited(p, "forger", mode, str(p / OP_REL), str(other), str(work))


def test_launcher_runs_the_runtime_operator_from_its_sealed_memfd(prefix: Path) -> None:
    sha = _sha(prefix / OP_REL)
    r = subprocess.run(
        [str(prefix / "bin/saccade_track"), "load", sha],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "value=1 mapped=/memfd:saccade-n03:src-fd=" in r.stdout
    assert r.stdout.rstrip().endswith("value_after=1")


def test_direct_exec_with_ld_audit_derives_the_same_operator(prefix: Path) -> None:
    sha = _sha(prefix / OP_REL)
    r = subprocess.run(
        [str(prefix / "libexec/saccade_track"), "load", sha],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LD_AUDIT": str(prefix / "lib/saccade_loader_audit.so"),
            "LD_LIBRARY_PATH": str(prefix / "lib/vendor"),
        },
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "value=1 mapped=/memfd:saccade-n03:src-fd=" in r.stdout


def test_a_binding_to_another_file_is_refused_by_the_host(prefix: Path) -> None:
    shutil.copy(prefix / OP_REL, prefix / "other" / "same_bytes.so")
    r = _audited(
        prefix,
        "saccade_track",
        "foreign",
        str(prefix / "other" / "same_bytes.so"),
        _sha(prefix / OP_REL),
    )
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert f"is not the runtime package's {(prefix / OP_REL)}" in r.stdout


def test_a_wrong_sha256_is_refused_before_anything_is_mapped(prefix: Path) -> None:
    r = _audited(prefix, "saccade_track", "load", "0" * 64)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "sha256" in r.stdout and "memfd" not in r.stdout


@pytest.mark.parametrize(
    ("hook", "message"),
    [
        ("mv -f '{other}' '{op}'", "operator library mapped from another path"),
        ("cat '{other}' > '{op}'", "operator memfd differs from the operator library"),
    ],
    ids=["rename_after_hash", "in_place_after_hash"],
)
def test_source_changed_after_the_hash_is_refused_by_the_auditor(
    prefix: Path, hook: str, message: str
) -> None:
    op = prefix / OP_REL
    sha = _sha(op)
    other = prefix / "other" / "op2.so"
    r = _audited(
        prefix, "saccade_track", "load", sha,
        env={"N03_HOOK_STAGE": "after_prepare", "N03_HOOK": hook.format(other=other, op=op)},
    )  # fmt: skip
    assert r.returncode == 127, (r.stdout, r.stderr)
    assert message in r.stderr
    assert "value=" not in r.stdout  # the operator's code never ran


def test_source_rewritten_after_the_load_does_not_change_what_runs(
    prefix: Path,
) -> None:
    op = prefix / OP_REL
    sha = _sha(op)
    r = _audited(
        prefix, "saccade_track", "load", sha,
        env={"N03_HOOK_STAGE": "after_load", "N03_HOOK": f"cat '{prefix / 'other' / 'op2.so'}' > '{op}'"},
    )  # fmt: skip
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "value=1" in r.stdout and "value_after=1" in r.stdout


@pytest.mark.parametrize(
    ("mode", "other", "message"),
    [
        ("unlabeled", "op2", "descriptor-backed object is not the operator memfd"),
        ("unsealed", "op2", "operator memfd is not sealed"),
        ("forged-content", "op2", "operator memfd differs from the operator library"),
        ("closed-source", "op2", "operator source descriptor is not open"),
        ("foreign-source", "copy", "operator library mapped from another path"),
        ("foreign-source", "hardlink", "operator library mapped from another path"),
        ("fd-regular", "op2", "descriptor-backed object is not the operator memfd"),
        ("fd-deleted", "copy", "descriptor-backed object is not the operator memfd"),
        (
            "alias-relative",
            "op2",
            "mapped object is not the object its name resolves to",
        ),
        ("alias-spoof", "copy", "mapped object is not the object its name resolves to"),
        (
            "path",
            "deleted_name",
            "mapped object is not the object its name resolves to",
        ),
        ("path", "copy", "operator library mapped from another path"),
    ],
)
def test_auditor_refuses_objects_that_are_not_the_verified_operator(
    prefix: Path, tmp_path: Path, mode: str, other: str, message: str
) -> None:
    op = prefix / OP_REL
    if other == "op2":
        target = prefix / "other" / "op2.so"
    elif other == "copy":
        target = prefix / "other" / "libsaccade_scan_torchop.so"
        shutil.copy(op, target)
    elif other == "hardlink":
        target = prefix / "other" / "libsaccade_scan_torchop.so"
        os.link(op, target)
    else:  # an ordinary file named "... (deleted)"
        target = prefix / "other" / "libplugin.so (deleted)"
        shutil.copy(op, target)
    r = _forge(prefix, mode, target, tmp_path / "work")
    assert r.returncode == 127, (mode, r.stdout, r.stderr)
    assert message in r.stderr, r.stderr
    assert "loaded value" not in r.stdout


def test_the_operator_by_path_at_the_fixed_location_still_passes(
    prefix: Path, tmp_path: Path
) -> None:
    r = _forge(prefix, "path", prefix / OP_REL, tmp_path / "work")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "loaded value=1\n"
