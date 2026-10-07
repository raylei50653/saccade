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
 *                 libfoo.so (A3). A bundled name must come from
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
#include <limits.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "loader_audit_names.inc" /* static const char *const kBundled[] = {..., NULL}; */

static const char kOpLibraryRel[] = "/share/saccade/build/libsaccade_scan_torchop.so";
static const char kReady[] = "saccade-loader-audit-ready\n";
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

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
    (void)lmid;
    (void)cookie;
    const char *path = map->l_name;
    if (path == NULL || path[0] == '\0') return 0;
    const char *base = base_name(path);
    char real[PATH_MAX];
    int resolved = realpath(path, real) != NULL;
    const char *real_base = resolved ? base_name(real) : base;
    if (is_python(base) || is_python(real_base)) fail("Python library mapped", path);
    const char *op_base = base_name(g_op_library);
    int bundled = is_bundled(base) || is_bundled(real_base);
    int op = strcmp(base, op_base) == 0 || strcmp(real_base, op_base) == 0;
    if (!bundled && !op) return 0;
    if (!resolved) fail("cannot resolve a mapped object", path);
    if (bundled && !in_vendor(real)) fail("bundled library mapped from outside lib/vendor", path);
    if (op && strcmp(real, g_op_library) != 0) fail("operator library mapped from another path", path);
    return 0;
}
