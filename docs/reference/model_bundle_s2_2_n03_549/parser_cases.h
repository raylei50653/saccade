/* Synthetic /proc/self/maps texts for the N03 research parser self-tests
 * (review r3/r4/r5). Long *target* cases also exceed PATH_MAX, so the
 * auditor's refusal of them is its PATH_MAX rule, not its buffer limit; the
 * long *earlier* cases exercise the buffer limits. The target address is 0x5000; a matching record is
 * "1000-9000 r--p 00000000 08:30 777 /path...". The auditor's buffer is
 * 64 KiB, the host's former buffer 8 KiB: lengths straddle both. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TARGET_HEAD "1000-9000 r--p 00000000 08:30 777 /"
/* Length of the target's path ("/ppp..."), given the whole record length. */
#define TARGET_PATH_LEN(len) ((len) - (sizeof TARGET_HEAD - 1) + 1)

struct parser_case {
    const char *name;
    int kind;      /* see build_case */
    size_t len;    /* length of the long record, excluding '\n' */
    int found;     /* expected */
    unsigned long ino;
};

static const struct parser_case kParserCases[] = {
    {"short_target_newline", 0, 60, 1, 777},
    {"short_target_last_no_newline", 1, 60, 1, 777},
    {"target_path_4095_newline", 0, sizeof TARGET_HEAD - 2 + 4095, 1, 777},
    {"target_path_4096_newline", 0, sizeof TARGET_HEAD - 2 + 4096, 1, 777},
    {"target_8191_newline", 0, 8191, 1, 777},
    {"target_8191_last_no_newline", 1, 8191, 1, 777},
    {"target_8192_newline", 0, 8192, 1, 777},
    {"target_8192_last_no_newline", 1, 8192, 1, 777},
    {"target_65535_newline", 0, 65535, 1, 777},
    {"target_65535_last_no_newline", 1, 65535, 1, 777},
    {"target_70000_newline_oversized_auditor", 0, 70000, 2, 777},
    {"earlier_oversized_with_header_text", 2, 70000, 1, 777},
    {"earlier_8191_with_header_text", 2, 8191, 1, 777},
    {"no_target", 3, 60, 0, 0},
};

/* The record covering 0x5000, padded to `len` with a path. */
static void put_target(char *out, size_t len, int newline) {
    const char *head = TARGET_HEAD;
    size_t h = strlen(head);
    memcpy(out, head, h);
    for (size_t i = h; i < len; ++i) out[i] = 'p';
    out[len] = newline ? '\n' : '\0';
    out[len + 1] = '\0';
}

/* An earlier record whose path, at every 4096-byte chunk boundary, at 8191
 * (the former host buffer) and at 65535 (the auditor buffer), carries
 * header-shaped text claiming 0x5000 with inode 666. */
static void put_decoy(char *out, size_t len) {
    const char *head = "a000-b000 r--p 00000000 08:30 1 /";
    const char *fake = "1000-9000 r--p 00000000 08:30 666 /fake";
    size_t h = strlen(head), fl = strlen(fake);
    memcpy(out, head, h);
    for (size_t i = h; i < len; ++i) out[i] = 'q';
    for (size_t b = 4096; b + fl < len; b += 4096) memcpy(out + b, fake, fl);
    if (8191 + fl < len) memcpy(out + 8191, fake, fl);
    /* review r5: the auditor's 64 KiB buffer fills at 65535 bytes; its
     * continuation then starts with a header-shaped decoy */
    if (65535 + fl < len) memcpy(out + 65535, fake, fl);
    if (len > fl + h) memcpy(out + len - fl, fake, fl);
    out[len] = '\n';
}

static char *build_case(int t) {
    const struct parser_case *c = &kParserCases[t];
    size_t cap = 2 * c->len + 256;
    char *text = (char *)calloc(1, cap);
    const char *pre = "0400-0800 r--p 00000000 08:30 5 /pre\n";
    strcpy(text, pre);
    char *p = text + strlen(pre);
    switch (c->kind) {
        case 0: put_target(p, c->len, 1); break;
        case 1: put_target(p, c->len, 0); break;
        case 2:
            put_decoy(p, c->len);
            put_target(p + c->len + 1, 60, 1);
            break;
        case 3: strcpy(p, "c000-d000 r--p 00000000 08:30 9 /other\n"); break;
    }
    return text;
}
