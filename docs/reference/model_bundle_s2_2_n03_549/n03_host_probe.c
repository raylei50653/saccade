/* #549 S2-2 N03 research probe (evidence only, not production).
 *
 * usage: n03_host_probe <mode> <op_path> <expected_sha256> [<aux_path>]
 *
 *   current          sha256(op_path) by path, hook(after_hash), dlopen(op_path)
 *                    -- the main@74f2f5cb DetectorHost sequence.
 *   proposed         S = open(op_path); fstat regular; read all from S into a
 *                    buffer; sha256 == expected; M = memfd_create(
 *                    "saccade-n03:src-fd=<S>"); write; seal (WRITE|GROW|SHRINK|
 *                    SEAL); fstat size; pread M back, sha256 == expected;
 *                    hook(after_hash); dlopen("/proc/self/fd/<M>"); close S, M.
 *   proposed-exec    proposed with MFD_EXEC in place of MFD_NOEXEC_SEAL (r2
 *                    showed both load; MFD_EXEC fails where vm.memfd_noexec=2).
 *   alias-relative   unlabeled sealed memfd M; <aux_dir>/alias.so -> /proc/self/fd/M;
 *                    chdir(aux_dir); dlopen("./alias.so")  (review r1 finding 1).
 *   alias-spoof      S = open(aux) then unlink(aux); plant a file at "<aux> (deleted)";
 *                    <dir>/alias.so -> /proc/self/fd/S; dlopen(<dir>/alias.so).
 *   twice            proposed load of op_path, then a second proposed load in the
 *                    same process of aux with env N03_SECOND_SHA (review finding 3).
 *                    env N03_TWICE: legacy (r1 host: close M, no checks), postcheck
 *                    (M kept open, mapping identity check, no registry), closecheck
 *                    (post-check, then M closed so its number is reused), registry (default: one
 *                    verified operator per process, keyed by SHA-256 and source path,
 *                    plus the postcheck).
 *   forge-unlabeled  #582 candidate: sealed memfd without label from op_path.
 *   forge-unsealed   labeled memfd (S = op_path) without F_SEAL_WRITE.
 *   forge-content    label S = op_path, bytes from aux_path (expected = aux sha).
 *   forge-badfd      label names a descriptor that is not open.
 *   fd-regular       dlopen("/proc/self/fd/<S>") of S = open(op_path).
 *   fd-deleted       S = open(aux_path) then unlink(aux_path); dlopen /proc/self/fd/<S>.
 *
 * Hook: env N03_HOOK (a /bin/sh command) runs at stage env N03_HOOK_STAGE
 * (after_open | after_hash | after_load); posix_spawn, empty environment.
 * After a successful dlopen the probe prints every mapping of the loaded
 * operator (dev:inode and path, /proc/self/maps) and compares, in memory, each
 * non-writable PT_LOAD segment outside PT_GNU_RELRO with the verified buffer at
 * the same file offsets (located by the verified ELF headers and l_addr). Exit codes: 0 loaded, 2 refused by
 * the host (catchable), 3 loaded bytes differ from the verified buffer,
 * 64+ usage/setup. The auditor's refusal is its own _exit(127).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <openssl/evp.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <unistd.h>

static unsigned char *g_buf;  // verified bytes
static size_t g_len;

static void sha_hex(const unsigned char *b, size_t n, char out[65]) {
    unsigned char d[32];
    unsigned int dl = 0;
    EVP_Digest(b, n, d, &dl, EVP_sha256(), NULL);
    for (int i = 0; i < 32; ++i) sprintf(out + 2 * i, "%02x", d[i]);
}

static int read_fd_all(int fd, unsigned char **out, size_t *len) {
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) return -1;
    unsigned char *b = malloc(st.st_size ? (size_t)st.st_size : 1);
    size_t got = 0;
    while (got < (size_t)st.st_size) {
        ssize_t k = pread(fd, b + got, (size_t)st.st_size - got, (off_t)got);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        got += (size_t)k;
    }
    char extra;
    if (pread(fd, &extra, 1, (off_t)got) != 0) return -1;  // grew while read
    *out = b;
    *len = got;
    return 0;
}

static void hook(const char *stage) {
    const char *want = getenv("N03_HOOK_STAGE"), *cmd = getenv("N03_HOOK");
    if (!want || !cmd || strcmp(want, stage) != 0) return;
    printf("hook %s: %s\n", stage, cmd);
    fflush(stdout);
    pid_t pid;
    char *argv[] = {"/bin/sh", "-c", (char *)cmd, NULL};
    char *envp[] = {"PATH=/usr/bin:/bin", NULL};
    int st = 0;
    if (posix_spawn(&pid, "/bin/sh", NULL, NULL, argv, envp) != 0 || waitpid(pid, &st, 0) < 0 || st != 0) {
        fprintf(stderr, "hook failed\n");
        _exit(65);
    }
}

static void refuse(const char *what) {
    printf("host refused: %s\n", what);
    fflush(stdout);
    _exit(2);
}

/* r2 finding: dlopen maps a MFD_NOEXEC_SEAL memfd (that flag stops execve, not
 * mmap PROT_EXEC), while MFD_EXEC is refused where vm.memfd_noexec=2. The
 * proposal therefore asks for MFD_NOEXEC_SEAL and retries without it only on
 * EINVAL (a kernel before 6.3, which has no such flag). */
static unsigned g_mfd_exec = MFD_NOEXEC_SEAL;

static int make_memfd(const char *label, const unsigned char *b, size_t n, int seals) {
    int m = memfd_create(label, MFD_CLOEXEC | MFD_ALLOW_SEALING | g_mfd_exec);
    if (m < 0 && errno == EINVAL && g_mfd_exec == MFD_NOEXEC_SEAL) m = memfd_create(label, MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (m < 0) refuse("memfd_create");
    size_t sent = 0;
    while (sent < n) {
        ssize_t k = write(m, b + sent, n - sent);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) refuse("memfd write");
        sent += (size_t)k;
    }
    if (seals && fcntl(m, F_ADD_SEALS, seals) != 0) refuse("F_ADD_SEALS");
    return m;
}

/* Verify the sealed memfd: seals, size and SHA-256 of what it now holds. */
static void verify_memfd(int m, const char *expected) {
    int need = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    if ((fcntl(m, F_GET_SEALS) & need) != need) refuse("memfd not sealed");
    unsigned char *b;
    size_t n;
    char hex[65];
    if (read_fd_all(m, &b, &n) != 0 || n != g_len) refuse("memfd readback");
    sha_hex(b, n, hex);
    if (strcmp(hex, expected) != 0) refuse("memfd readback sha256 mismatch");
    printf("memfd readback sha256=%s seals=0x%x\n", hex, fcntl(m, F_GET_SEALS));
    free(b);
}

/* Host post-check (review finding 3): the mapping holding the returned
 * object's dynamic section is the memfd just verified (glibc returns an
 * already-loaded object by name, without mapping or auditing anything). */
static int mapped_is(void *h, int m) {
    struct link_map *lm = NULL;
    struct stat ms;
    if (dlinfo(h, RTLD_DI_LINKMAP, &lm) != 0 || lm == NULL || fstat(m, &ms) != 0) return 0;
    FILE *f = fopen("/proc/self/maps", "r");
    char line[8192];
    int ok = 0, continuation = 0;
    while (f && fgets(line, sizeof line, f)) {
        /* review r3: parse complete records only; a chunk without '\n' is part of
         * an overlong record, discarded through its newline. */
        int complete = strchr(line, '\n') != NULL || feof(f);
        int skip = continuation || !complete;
        continuation = !complete;
        if (skip) continue;
        unsigned long lo, hi, off, ino;
        unsigned maj, mnr;
        char perm[8];
        if (sscanf(line, "%lx-%lx %7s %lx %x:%x %lu", &lo, &hi, perm, &off, &maj, &mnr, &ino) == 7 &&
            (uintptr_t)lm->l_ld >= lo && (uintptr_t)lm->l_ld < hi) {
            ok = major(ms.st_dev) == maj && minor(ms.st_dev) == mnr && ms.st_ino == ino;
            break;
        }
    }
    if (f) fclose(f);
    printf("postcheck mapped_is_verified_memfd=%d\n", ok);
    return ok;
}

/* Process-wide: one verified operator (SHA-256, source path, handle; memfd kept open). */
static char g_loaded_sha[65];
static void *g_loaded_handle;
static char g_loaded_path[4096];

static int load_verified(const char *path, const char *expected, int *src_out) {
    int s = open(path, O_RDONLY | O_CLOEXEC);
    if (s < 0) refuse("open");
    hook("after_open");
    char hex[65];
    if (read_fd_all(s, &g_buf, &g_len) != 0) refuse("not a regular file or short read");
    sha_hex(g_buf, g_len, hex);
    if (strcmp(hex, expected) != 0) refuse("sha256 mismatch");
    printf("source sha256=%s bytes=%zu src_fd=%d\n", hex, g_len, s);
    *src_out = s;
    return s;
}

/* Print the operator's mappings (dev:inode, path) from /proc/self/maps. */
static void print_maps(const char *needle) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[8192];
    while (f && fgets(line, sizeof line, f))
        if (strstr(line, needle)) printf("map %s", line);
    if (f) fclose(f);
}

/* Compare, in memory, every non-writable PT_LOAD segment of the loaded
 * operator (outside PT_GNU_RELRO, which relocation writes) with the verified
 * buffer at its file offsets. Returns 1 when every compared byte is equal. */
static int compare_loaded(void *h) {
    struct link_map *lm = NULL;
    if (dlinfo(h, RTLD_DI_LINKMAP, &lm) != 0 || lm == NULL) return 0;
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)g_buf;
    const Elf64_Phdr *ph = (const Elf64_Phdr *)(g_buf + eh->e_phoff);
    uint64_t rlo = 0, rhi = 0;
    for (int i = 0; i < eh->e_phnum; ++i)
        if (ph[i].p_type == PT_GNU_RELRO) rlo = ph[i].p_vaddr, rhi = ph[i].p_vaddr + ph[i].p_memsz;
    size_t compared = 0;
    int equal = 1, segments = 0;
    for (int i = 0; i < eh->e_phnum; ++i) {
        if (ph[i].p_type != PT_LOAD || (ph[i].p_flags & PF_W)) continue;
        for (uint64_t v = ph[i].p_vaddr; v < ph[i].p_vaddr + ph[i].p_filesz; ++v) {
            if (v >= rlo && v < rhi) continue;
            uint64_t off = ph[i].p_offset + (v - ph[i].p_vaddr);
            if (off >= g_len) break;
            compared++;
            if (*(const unsigned char *)(lm->l_addr + v) != g_buf[off]) equal = 0;
        }
        segments++;
        printf("segment flags=%c%c%c vaddr=0x%" PRIx64 " filesz=%" PRIu64 "\n", ph[i].p_flags & PF_R ? 'r' : '-',
               ph[i].p_flags & PF_W ? 'w' : '-', ph[i].p_flags & PF_X ? 'x' : '-', (uint64_t)ph[i].p_vaddr,
               (uint64_t)ph[i].p_filesz);
    }
    printf("l_name=%s compared_segments=%d compared_bytes=%zu relro=[0x%" PRIx64 ",0x%" PRIx64 ") all_equal=%d\n",
           lm->l_name, segments, compared, rlo, rhi, equal);
    return segments > 0 && compared > 0 && equal;
}

/* Mapped vendor/system objects (dependency evidence). */
static void print_objects(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    char line[8192];
    while (f && fgets(line, sizeof line, f)) {
        char *p = strchr(line, '/');
        if (p && strstr(line, " r--p 00000000 ")) printf("object %s", p);
    }
    if (f) fclose(f);
}

/* The proposal's host sequence. legacy=1 reproduces the r1 host (closes M,
 * no post-check); registry=1 refuses a second, different operator. */
static void *proposed_load(const char *op, const char *expected, int legacy, int registry) {
    if (registry && g_loaded_handle) {
        if (strcmp(g_loaded_sha, expected) != 0) refuse("operator already loaded with different bytes");
        if (strcmp(g_loaded_path, op) != 0) refuse("operator already loaded from another source path");
        printf("registry: reuse verified operator %s\n", g_loaded_sha);
        return g_loaded_handle;
    }
    int s = -1;
    char label[64], load[64];
    free(g_buf);
    load_verified(op, expected, &s);
    snprintf(label, sizeof label, "saccade-n03:src-fd=%d", s);
    int m = make_memfd(label, g_buf, g_len, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL);
    verify_memfd(m, expected);
    hook("after_hash");
    snprintf(load, sizeof load, "/proc/self/fd/%d", m);
    printf("dlopen %s\n", load);
    fflush(stdout);
    void *h = dlopen(load, RTLD_NOW | RTLD_LOCAL);
    if (!h) refuse(dlerror());
    close(s);
    if (legacy) {
        close(m);
        return h;
    }
    if (!mapped_is(h, m)) refuse("loaded object is not the verified memfd");
    if (getenv("N03_TWICE") && strcmp(getenv("N03_TWICE"), "closecheck") == 0) {
        close(m); /* isolate the post-check: let the next memfd reuse this number */
        return h;
    }
    /* M stays open: its number is never reused for another operator. */
    snprintf(g_loaded_sha, sizeof g_loaded_sha, "%s", expected);
    snprintf(g_loaded_path, sizeof g_loaded_path, "%s", op);
    g_loaded_handle = h;
    return h;
}

int main(int argc, char **argv) {
    if (argc < 4) return 64;
    const char *mode = argv[1], *op = argv[2], *expected = argv[3], *aux = argc > 4 ? argv[4] : NULL;
    char load[64];
    const char *load_path = op;
    const char *needle = "libsaccade_scan_torchop";
    int s = -1, m = -1;
    int all = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    char label[64];

    if (strcmp(mode, "twice") == 0) {
        const char *how = getenv("N03_TWICE") ? getenv("N03_TWICE") : "registry";
        const char *second = getenv("N03_SECOND_SHA");
        if (!aux || !second) return 64;
        int legacy = strcmp(how, "legacy") == 0, registry = strcmp(how, "registry") == 0;
        void *h1 = proposed_load(op, expected, legacy, registry);
        printf("first load equals verified=%d\n", compare_loaded(h1));
        void *h2 = proposed_load(aux, second, legacy, registry);
        printf("second handle %s first\n", h2 == h1 ? "==" : "!=");
        int eq = compare_loaded(h2);
        printf("second load equals verified=%d\n", eq);
        fflush(stdout);
        _exit(eq ? 0 : 3);
    }
    if (strcmp(mode, "alias-relative") == 0 || strcmp(mode, "alias-spoof") == 0) {
        if (!aux) return 64;
        char target[64], alias[4096];
        if (strcmp(mode, "alias-relative") == 0) {
            load_verified(op, expected, &s);
            m = make_memfd("saccade_operator_candidate", g_buf, g_len, all);
            snprintf(target, sizeof target, "/proc/self/fd/%d", m);
            snprintf(alias, sizeof alias, "%s/alias.so", aux);
            if (symlink(target, alias) != 0 || chdir(aux) != 0) refuse("alias setup");
            load_path = "./alias.so";
        } else {
            load_verified(aux, expected, &s);
            char planted[4096];
            snprintf(planted, sizeof planted, "%s (deleted)", aux);
            if (unlink(aux) != 0) refuse("unlink");
            int pf = open(planted, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
            if (pf < 0 || write(pf, "planted\n", 8) != 8) refuse("plant");
            close(pf);
            snprintf(target, sizeof target, "/proc/self/fd/%d", s);
            snprintf(alias, sizeof alias, "%s.alias.so", aux);
            if (symlink(target, alias) != 0) refuse("alias setup");
            char real[4096];
            printf("planted %s; realpath(alias)=%s\n", planted, realpath(alias, real) ? real : "UNRESOLVED");
            snprintf(load, sizeof load, "%s", "");
            load_path = alias;
        }
        printf("dlopen %s\n", load_path);
        fflush(stdout);
        void *h = dlopen(load_path, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            printf("dlopen failed: %s\n", dlerror());
            fflush(stdout);
            _exit(2);
        }
        printf("dlopen succeeded\n");
        fflush(stdout);
        _exit(0);
    }
    if (strcmp(mode, "current") == 0) {
        int fd = open(op, O_RDONLY | O_CLOEXEC);
        char hex[65];
        if (fd < 0 || read_fd_all(fd, &g_buf, &g_len) != 0) refuse("open/read");
        close(fd);
        sha_hex(g_buf, g_len, hex);
        if (strcmp(hex, expected) != 0) refuse("sha256 mismatch");
        printf("source sha256=%s bytes=%zu (path, fd closed)\n", hex, g_len);
    } else if (strcmp(mode, "proposed") == 0 || strcmp(mode, "proposed-exec") == 0) {
        if (strcmp(mode, "proposed-exec") == 0) g_mfd_exec = MFD_EXEC;
        void *h = proposed_load(op, expected, 0, 1);
        printf("dlopen succeeded\n");
        print_maps("memfd:saccade-n03");
        int eq = compare_loaded(h);
        printf("loaded_equals_verified=%d\n", eq);
        hook("after_load");
        if (getenv("N03_HOOK_STAGE") && strcmp(getenv("N03_HOOK_STAGE"), "after_load") == 0) {
            int after = compare_loaded(h);
            printf("after_load_mutation loaded_equals_verified=%d\n", after);
            eq = eq && after;
        }
        print_objects();
        fflush(stdout);
        _exit(eq ? 0 : 3);
    } else if (strcmp(mode, "forge-unlabeled") == 0) {
        load_verified(op, expected, &s);
        m = make_memfd("saccade_operator_candidate", g_buf, g_len, all);
        needle = "memfd:saccade_operator_candidate";
    } else if (strcmp(mode, "forge-unsealed") == 0) {
        load_verified(op, expected, &s);
        snprintf(label, sizeof label, "saccade-n03:src-fd=%d", s);
        m = make_memfd(label, g_buf, g_len, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL);
        needle = "memfd:saccade-n03";
    } else if (strcmp(mode, "forge-content") == 0) {
        if (!aux) return 64;
        int a = -1;
        load_verified(aux, expected, &a);  // bytes and their sha from aux
        s = open(op, O_RDONLY | O_CLOEXEC);  // label names the fixed-path source
        if (s < 0) refuse("open");
        snprintf(label, sizeof label, "saccade-n03:src-fd=%d", s);
        m = make_memfd(label, g_buf, g_len, all);
        verify_memfd(m, expected);
        needle = "memfd:saccade-n03";
    } else if (strcmp(mode, "forge-badfd") == 0) {
        load_verified(op, expected, &s);
        m = make_memfd("saccade-n03:src-fd=987", g_buf, g_len, all);
        needle = "memfd:saccade-n03";
    } else if (strcmp(mode, "fd-regular") == 0) {
        load_verified(op, expected, &s);
        snprintf(load, sizeof load, "/proc/self/fd/%d", s);
        load_path = load;
    } else if (strcmp(mode, "fd-deleted") == 0) {
        if (!aux) return 64;
        load_verified(aux, expected, &s);
        if (unlink(aux) != 0) refuse("unlink");
        snprintf(load, sizeof load, "/proc/self/fd/%d", s);
        load_path = load;
    } else {
        return 64;
    }
    if (m >= 0) {
        snprintf(load, sizeof load, "/proc/self/fd/%d", m);
        load_path = load;
    }
    hook("after_hash");
    printf("dlopen %s\n", load_path);
    fflush(stdout);
    void *h = dlopen(load_path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        fflush(stdout);
        _exit(2);
    }
    printf("dlopen succeeded\n");
    if (s >= 0) close(s);
    if (m >= 0) close(m);
    print_maps(needle);
    int before = compare_loaded(h);
    printf("loaded_equals_verified=%d\n", before);
    hook("after_load");
    if (getenv("N03_HOOK_STAGE") && strcmp(getenv("N03_HOOK_STAGE"), "after_load") == 0) {
        int after = compare_loaded(h);
        printf("after_load_mutation loaded_equals_verified=%d\n", after);
        before = before && after;
    }
    print_objects();
    fflush(stdout);
    _exit(before ? 0 : 3);  // no destructors: a mutated text page may be mapped
}
