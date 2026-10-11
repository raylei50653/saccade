/* Loader provenance check of the shipping package (#465 Phase C PR-C1).
 *
 * An rtld-audit library (rtld-audit(7)) that the launcher bin/saccade_track
 * hands to the dynamic loader with --audit. It is the secondary check: the
 * primary mechanism is --library-path <prefix>/lib/vendor. Torch, TensorRT and
 * nvshmem carry DT_RPATH entries (e.g. $ORIGIN/../../nvidia/cu13/lib) that the
 * loader searches *before* --library-path, so a file planted at one of those
 * places would be loaded silently (docs §17). This library fails closed
 * instead:
 *
 *   la_objsearch  for a bundled name (shipping/third_party_set.json), every
 *                 search candidate outside <prefix>/lib/vendor/ is skipped;
 *                 if such a candidate exists on disk, the process exits 127.
 *                 Relative candidates (an empty RUNPATH entry is the working
 *                 directory) are treated the same for every name.
 *   la_objopen    every mapped object is resolved (realpath) before it is
 *                 classified, by its mapped name and by its real name, so a
 *                 symlink alias (payload -> libfoo.so) is classified as
 *                 libfoo.so (A3). Names are compared by SONAME family
 *                 (trailing numeric version components dropped), so an alias
 *                 to libfoo.so.13.1.0 is in the family of a bundled
 *                 libfoo.so.13 (A4). A bundled family must come from
 *                 <prefix>/lib/vendor/, the operator library from its model
 *                 root path; nothing named libpython* / libtorch_python*. An
 *                 object that cannot be resolved (the vDSO) passes only when
 *                 neither check applies to its mapped name.
 *
 *   la_version    with SACCADE_AUDIT_PROBE=1, once <prefix> is derived,
 *                 writes kReady to stdout and exits 0 before the program
 *                 runs. The loader ignores an audit library it cannot load
 *                 (missing, truncated, no la_version), so the launcher first
 *                 runs this probe and refuses to start the entrypoint unless
 *                 it reads kReady (docs §17.8, A2).
 *
 * <prefix> is derived from this library's own path (<prefix>/lib/; the
 * launcher passes it as a physical path). It asks
 * for no symbol-binding events (la_objopen returns 0), so binding is the
 * loader's default. C, libc only: the auditor runs in its own link-map
 * namespace and must not bring a C++ runtime there.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "loader_audit_names.inc" /* static const char *const kBundled[] = {..., NULL}; */

static const char kOpLibraryRel[] = "/share/saccade/build/libsaccade_scan_torchop.so";
static const char kReady[] = "saccade-loader-audit-ready\n";
/* PROTOTYPE (#549 S2-2 N03 proposal): the label the host gives the sealed
 * memfd it loads N03 from; readlink(/proc/self/fd/M) = "/memfd:<label><S> (deleted)",
 * S = the host's still-open source descriptor of the fixed-path N03. */
static const char kOpMemfdLabel[] = "/memfd:saccade-n03:src-fd=";
static const char kProcFd[] = "/proc/self/fd/";
static unsigned char g_buf_a[1 << 16], g_buf_b[1 << 16];
static char g_vendor[PATH_MAX];
static char g_op_library[PATH_MAX];

static void fail(const char *what, const char *path) {
    dprintf(STDERR_FILENO, "saccade_track: loader provenance check failed: %s: %s\n", what, path);
    _exit(127);
}

static const char *base_name(const char *p) {
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

static int is_bundled(const char *name) {
    for (int i = 0; kBundled[i] != NULL; ++i)
        if (strcmp(kBundled[i], name) == 0) return 1;
    return 0;
}

/* Length of `name` without its trailing numeric version components
 * (libfoo.so.13.1.0 -> libfoo.so); 0 when what remains does not end in ".so". */
static size_t family_len(const char *name) {
    size_t n = strlen(name);
    for (;;) {
        size_t k = n;
        while (k > 0 && name[k - 1] >= '0' && name[k - 1] <= '9') --k;
        if (k == n || k == 0 || name[k - 1] != '.') break;
        n = k - 1;
    }
    return n >= 3 && memcmp(name + n - 3, ".so", 3) == 0 ? n : 0;
}

static int same_family(const char *a, const char *b) {
    size_t n = family_len(a);
    return n != 0 && n == family_len(b) && memcmp(a, b, n) == 0;
}

/* In the SONAME family of a bundled name (exact names first). */
static int is_bundled_family(const char *name) {
    if (is_bundled(name)) return 1;
    for (int i = 0; kBundled[i] != NULL; ++i)
        if (same_family(kBundled[i], name)) return 1;
    return 0;
}

static int is_python(const char *name) {
    return strncmp(name, "libpython", 9) == 0 || strncmp(name, "libtorch_python", 15) == 0;
}

/* Lexical normalization of an absolute path: no symlink resolution, the file
 * need not exist (search candidates usually do not). */
static int normalize(const char *in, char *out) {
    size_t n = 0;
    if (in[0] != '/') return 0;
    out[0] = '\0';
    while (*in) {
        while (*in == '/') ++in;
        const char *end = strchr(in, '/');
        size_t len = end ? (size_t)(end - in) : strlen(in);
        if (len == 0) break;
        if (len == 1 && in[0] == '.') {
        } else if (len == 2 && in[0] == '.' && in[1] == '.') {
            while (n > 0 && out[--n] != '/') {
            }
            out[n] = '\0';
        } else {
            if (n + 1 + len >= PATH_MAX) return 0;
            out[n++] = '/';
            memcpy(out + n, in, len);
            n += len;
            out[n] = '\0';
        }
        in += len;
    }
    if (n == 0) strcpy(out, "/");
    return 1;
}

/* A file directly in <prefix>/lib/vendor/. */
static int in_vendor(const char *path) {
    char p[PATH_MAX];
    size_t v = strlen(g_vendor);
    if (!normalize(path, p)) return 0;
    return strncmp(p, g_vendor, v) == 0 && p[v] == '/' && strchr(p + v + 1, '/') == NULL;
}

unsigned int la_version(unsigned int version) {
    (void)version;
    Dl_info info;
    char self[PATH_MAX];
    if (!dladdr((void *)&la_version, &info) || info.dli_fname == NULL || !normalize(info.dli_fname, self))
        fail("cannot locate the auditor", info.dli_fname ? info.dli_fname : "?");
    char *slash = strrchr(self, '/'); /* <prefix>/lib/saccade_loader_audit.so */
    *slash = '\0';
    if (snprintf(g_vendor, sizeof g_vendor, "%s/vendor", self) >= (int)sizeof g_vendor)
        fail("prefix too long", self);
    slash = strrchr(self, '/'); /* <prefix>/lib */
    if (slash == NULL) fail("auditor is not under <prefix>/lib", self);
    *slash = '\0';
    if (snprintf(g_op_library, sizeof g_op_library, "%s%s", self, kOpLibraryRel) >= (int)sizeof g_op_library)
        fail("prefix too long", self);
    const char *probe = getenv("SACCADE_AUDIT_PROBE");
    if (probe != NULL && strcmp(probe, "1") == 0) {
        if (write(STDOUT_FILENO, kReady, sizeof kReady - 1) != (ssize_t)(sizeof kReady - 1)) _exit(127);
        _exit(0);
    }
    return LAV_CURRENT;
}

char *la_objsearch(const char *name, uintptr_t *cookie, unsigned int flag) {
    (void)cookie;
    const char *base = base_name(name);
    if (is_python(base)) fail("Python library requested", name);
    if (flag == LA_SER_ORIG) return (char *)name;
    int foreign = name[0] != '/' || (is_bundled(base) && !in_vendor(name));
    if (!foreign) return (char *)name;
    if (access(name, F_OK) == 0) fail("foreign copy on the search path", name);
    return NULL; /* skip a candidate that does not exist */
}

/* Strict non-negative decimal int, whole string. */
static int parse_fd(const char *s, const char *end) {
    long v = 0;
    if (s == end || (end - s > 1 && s[0] == '0')) return -1;
    for (const char *c = s; c < end; ++c) {
        if (*c < '0' || *c > '9') return -1;
        v = v * 10 + (*c - '0');
        if (v > 1000000) return -1;
    }
    return (int)v;
}

static int read_full(int fd, unsigned char *b, size_t n, off_t off) {
    size_t got = 0;
    while (got < n) {
        ssize_t k = pread(fd, b + got, n - got, off + (off_t)got);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return 0;
        got += (size_t)k;
    }
    return 1;
}

/* PROTOTYPE: an object mapped from /proc/self/fd/M passes only as N03 loaded
 * from the host's sealed memfd, and only if (1) M is a memfd carrying the
 * N03 label and all four seals, (2) the label's source descriptor S was opened
 * as <prefix>/share/saccade/build/libsaccade_scan_torchop.so (kernel d_path,
 * the realpath analogue of the path rule) and is the regular file now at that
 * path (dev/ino, final component not a symlink), and (3) M's bytes equal the
 * bytes of that file, read now. Anything else is refused. */
static void check_op_memfd(const char *path) {
    size_t pl = sizeof kProcFd - 1;
    if (strncmp(path, kProcFd, pl) != 0) fail("unresolvable mapped object", path);
    int m = parse_fd(path + pl, path + strlen(path));
    if (m < 0) fail("unresolvable mapped object", path);
    char link[PATH_MAX], want[PATH_MAX];
    ssize_t n = readlink(path, link, sizeof link - 1);
    if (n <= 0) fail("unresolvable mapped object", path);
    link[n] = '\0';
    size_t ll = sizeof kOpMemfdLabel - 1;
    static const char kDeleted[] = " (deleted)";
    size_t dl = sizeof kDeleted - 1;
    if ((size_t)n <= ll + dl || strncmp(link, kOpMemfdLabel, ll) != 0 || strcmp(link + n - dl, kDeleted) != 0)
        fail("descriptor-backed object is not the operator memfd", link);
    int s = parse_fd(link + ll, link + n - dl);
    if (s < 0 || s == m) fail("operator memfd names no source descriptor", link);
    int seals = fcntl(m, F_GET_SEALS);
    int need = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
    if (seals < 0 || (seals & need) != need) fail("operator memfd is not sealed", link);
    /* (2) source descriptor provenance */
    if (snprintf(want, sizeof want, "%s%d", kProcFd, s) >= (int)sizeof want) fail("operator source", link);
    n = readlink(want, link, sizeof link - 1);
    if (n <= 0) fail("operator source descriptor is not open", want);
    link[n] = '\0';
    if (strcmp(link, g_op_library) != 0) fail("operator library mapped from another path", link);
    struct stat ss, fs, ms;
    int f = open(g_op_library, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (f < 0) fail("cannot open the operator library", g_op_library);
    if (fstat(s, &ss) != 0 || fstat(f, &fs) != 0 || fstat(m, &ms) != 0 || !S_ISREG(ss.st_mode) ||
        !S_ISREG(fs.st_mode) || ss.st_dev != fs.st_dev || ss.st_ino != fs.st_ino)
        fail("operator source descriptor is not the operator library", g_op_library);
    /* (3) byte equality: sealed memfd == operator library */
    if (ms.st_size != fs.st_size) fail("operator memfd differs from the operator library", g_op_library);
    for (off_t off = 0; off < fs.st_size;) {
        size_t k = (size_t)(fs.st_size - off) < sizeof g_buf_a ? (size_t)(fs.st_size - off) : sizeof g_buf_a;
        if (!read_full(m, g_buf_a, k, off) || !read_full(f, g_buf_b, k, off) || memcmp(g_buf_a, g_buf_b, k) != 0)
            fail("operator memfd differs from the operator library", g_op_library);
        off += (off_t)k;
    }
    close(f);
}

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
    (void)lmid;
    (void)cookie;
    const char *path = map->l_name;
    if (path == NULL || path[0] == '\0') return 0;
    const char *base = base_name(path);
    char real[PATH_MAX];
    int resolved = realpath(path, real) != NULL;
    /* PROTOTYPE: a descriptor name (/proc/..., /dev/fd/...) or an absolute name
     * that does not resolve (a memfd, a deleted or O_TMPFILE file) has no path
     * to classify; realpath of a descriptor name is not trusted (a file planted
     * at its readlink text would resolve). Such an object is refused unless
     * it is the verified operator memfd. */
    if (strncmp(path, "/proc/", 6) == 0 || strncmp(path, "/dev/fd/", 8) == 0 || (!resolved && path[0] == '/')) {
        check_op_memfd(path);
        return 0;
    }
    const char *real_base = resolved ? base_name(real) : base;
    if (is_python(base) || is_python(real_base)) fail("Python library mapped", path);
    const char *op_base = base_name(g_op_library);
    int bundled = is_bundled_family(base) || is_bundled_family(real_base);
    int op = strcmp(base, op_base) == 0 || strcmp(real_base, op_base) == 0 || same_family(base, op_base) ||
             same_family(real_base, op_base);
    if (!bundled && !op) return 0;
    if (!resolved) fail("cannot resolve a mapped object", path);
    if (bundled && !in_vendor(real)) fail("bundled library mapped from outside lib/vendor", path);
    if (op && strcmp(real, g_op_library) != 0) fail("operator library mapped from another path", path);
    return 0;
}
