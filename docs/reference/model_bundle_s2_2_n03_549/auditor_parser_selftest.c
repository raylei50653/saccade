/* Auditor mapping_from_fd on the synthetic texts of parser_cases.h (research). */
#define _GNU_SOURCE
#include <sys/mman.h>
#include "loader_audit_proto.c" /* tracked as loader_audit_prototype.c */
#include "parser_cases.h"

int main(void) {
    int fails = 0;
    for (int t = 0; t < (int)(sizeof kParserCases / sizeof kParserCases[0]); ++t) {
        char *text = build_case(t);
        int fd = memfd_create("maps-case", 0);
        size_t n = strlen(text);
        if (fd < 0 || write(fd, text, n) != (ssize_t)n || lseek(fd, 0, SEEK_SET) != 0) return 2;
        struct mapped_id id;
        memset(&id, 0, sizeof id);
        int found = mapping_from_fd(fd, 0x5000, &id);
        close(fd);
        /* expected found==2: the auditor (64 KiB buffer) discards an oversized
         * target record, so it must refuse (found=0) rather than mis-parse. */
        int want = kParserCases[t].found == 2 ? 0 : kParserCases[t].found;
        /* the auditor refuses (does not identify) a mapped path that does not
         * fit PATH_MAX - 1: a fail-closed compatibility restriction */
        size_t plen = kParserCases[t].kind <= 1 ? TARGET_PATH_LEN(kParserCases[t].len) : 0;
        if (plen >= PATH_MAX) want = 0;
        int ok = found == want && (!found || id.ino == kParserCases[t].ino);
        printf("auditor-parser case %-38s path_len=%zu want=%d found=%d ino=%lu %s\n", kParserCases[t].name, plen, want, found, found ? id.ino : 0UL, ok ? "PASS" : "FAIL");
        fails += !ok;
        free(text);
    }
    return fails ? 1 : 0;
}
