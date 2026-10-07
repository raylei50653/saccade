#!/bin/sh
# Saccade shipping package installer (#465 Phase C PR-C3); released next to the
# package as <name>.install.sh (docs/reference/native_runtime_resolved_config.md
# §19).
#
#   sh <name>.install.sh <dir>/<name>.tar.gz TARGET
#       Checks the tarball against <dir>/<name>.sha256, extracts it into a
#       private staging directory next to TARGET (same file system), checks
#       every file against the package's MANIFEST.json (exactly the listed
#       files, each a regular file with the listed size, sha256 and mode), and
#       only then moves the tree to TARGET with one rename that refuses to
#       replace anything (renameat2 RENAME_NOREPLACE through mv -n -T). On any
#       failure or signal the staging directory is removed and TARGET is never
#       created. TARGET must not exist.
#   sh <name>.install.sh --verify PREFIX
#       Checks an installed tree against its own MANIFEST.json; changes nothing.
#
# The package digest is not a signature: whoever can replace the tarball can
# replace <name>.sha256 too (signing is PR-C4). Base system tools only (dash,
# coreutils, tar, gzip, sed, grep, findutils; Ubuntu 24.04). No option or
# environment variable changes what is checked.
#
# Exit 0: installed (or verified); 1: a check failed; 2: usage, or TARGET
# exists / cannot be used; 128+N: interrupted by signal N.
set -eu
umask 022
LC_ALL=C
export LC_ALL

SCHEMA=saccade.shipping_manifest/v1
staging=

say() { echo "saccade-install: $*" >&2; }
fail() { say "$*"; exit 1; }
usage() {
    say "usage: sh install.sh <dir>/<name>.tar.gz TARGET | sh install.sh --verify PREFIX"
    exit 2
}
cleanup() {
    if [ -n "$staging" ]; then
        rm -rf -- "$staging"
        staging=
    fi
}
trap cleanup EXIT
trap 'cleanup; exit 129' HUP
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# manifest_list ROOT NAME OUT: parse ROOT/MANIFEST.json into OUT as
# "sha256 mode bytes path" lines (sorted by path). The builder writes one file
# object per line in a fixed form; any line that does not have exactly that
# form, or a count that disagrees with "file_count", fails.
manifest_list() {
    m=$1/MANIFEST.json
    [ -f "$m" ] && [ ! -L "$m" ] || fail "no MANIFEST.json in the package"
    grep -qx '  "schema": "'"$SCHEMA"'",' "$m" || fail "MANIFEST.json: schema is not $SCHEMA"
    grep -qx '  "package": "'"$2"'",' "$m" || fail "MANIFEST.json: package is not $2"
    count=$(sed -n 's/^  "file_count": \([1-9][0-9]*\),$/\1/p' "$m")
    case $count in
        '' | *[!0-9]*) fail "MANIFEST.json: no file_count" ;;
    esac
    seg='[A-Za-z0-9_+-][A-Za-z0-9._+-]*'
    sed -n 's#^    {"path": "\('"$seg"'\(/'"$seg"'\)*\)", "sha256": "\([0-9a-f]\{64\}\)", "bytes": \(0\|[1-9][0-9]*\), "mode": "\(0644\|0755\)"},\{0,1\}$#\3 \5 \4 \1#p' \
        "$m" > "$3"
    parsed=$(grep -c '' "$3") || parsed=0
    entries=$(grep -c '"path": ' "$m") || entries=0
    [ "$parsed" = "$count" ] && [ "$entries" = "$count" ] ||
        fail "MANIFEST.json: $count files declared, $entries listed, $parsed well formed"
    # Sorted, no duplicates, MANIFEST.json not listed.
    cut -d ' ' -f 4 "$3" > "$3.paths"
    [ "$(sort -u "$3.paths")" = "$(cat "$3.paths")" ] || fail "MANIFEST.json: paths are not sorted and unique"
    ! grep -qx 'MANIFEST.json' "$3.paths" || fail "MANIFEST.json lists itself"
}

# check_tree ROOT LIST WORK MODE: ROOT holds exactly the files in LIST plus
# MANIFEST.json, no other entry (symlink, device, empty directory), and each
# listed file is a regular file with the listed size and sha256. MODE=set
# gives each file its listed mode and each directory 0755 (install); MODE=check
# requires them (--verify).
check_tree() {
    { cat "$2.paths"; echo MANIFEST.json; } | sort > "$3/want"
    (cd "$1" && find . -mindepth 1 ! -type d -print) | sed 's#^\./##' | sort > "$3/have"
    [ "$(cat "$3/want")" = "$(cat "$3/have")" ] || {
        say "the package's files are not exactly MANIFEST.json's (listed on one side only):"
        sort "$3/want" "$3/have" | uniq -u | sed 's/^/  /' >&2
        exit 1
    }
    [ -z "$(find "$1" -type d -empty -print)" ] || fail "the package has an empty directory"
    while read -r sum mode bytes path; do
        f=$1/$path
        [ -f "$f" ] && [ ! -L "$f" ] || fail "$path: not a regular file"
        [ "$(stat -c %s -- "$f")" = "$bytes" ] || fail "$path: size is not $bytes"
        got=$(sha256sum < "$f")
        [ "${got%% *}" = "$sum" ] || fail "$path: sha256 is not $sum"
        if [ "$4" = set ]; then
            chmod "$mode" -- "$f"
        fi
        [ "$(stat -c %a -- "$f")" = "${mode#0}" ] || fail "$path: mode is not $mode"
    done < "$2"
    if [ "$4" = set ]; then
        find "$1" -type d -exec chmod 0755 {} +
        chmod 0644 -- "$1/MANIFEST.json"
    fi
    [ -z "$(find "$1" -type d ! -perm 0755 -print)" ] || fail "a directory's mode is not 0755"
    [ "$(stat -c %a -- "$1/MANIFEST.json")" = 644 ] || fail "MANIFEST.json: mode is not 0644"
}

if [ "${1-}" = --verify ]; then
    [ $# -eq 2 ] || usage
    prefix=$(cd -P -- "$2" 2>/dev/null && pwd -P) || { say "$2 is not a directory"; exit 2; }
    name=$(sed -n 's/^  "package": "\(saccade-[A-Za-z0-9._+-]*\)",$/\1/p' "$prefix/MANIFEST.json" 2>/dev/null) || name=
    [ -n "$name" ] || fail "$prefix/MANIFEST.json: no package name"
    staging=$(mktemp -d "${TMPDIR:-/tmp}/saccade-verify.XXXXXX") || exit 2
    manifest_list "$prefix" "$name" "$staging/list"
    check_tree "$prefix" "$staging/list" "$staging" check
    say "verified $prefix: $count files match $name's MANIFEST.json"
    exit 0
fi

[ $# -eq 2 ] || usage
tarball=$1
target=$2
case $tarball in
    *.tar.gz) ;;
    *) usage ;;
esac
[ -f "$tarball" ] || { say "$tarball is not a file"; exit 2; }
name=${tarball##*/}
name=${name%.tar.gz}
case $name in
    saccade-*) ;;
    *) say "$tarball is not a saccade package"; exit 2 ;;
esac
case $name in
    *[!A-Za-z0-9._+-]*) say "$tarball: unexpected characters in the package name"; exit 2 ;;
esac
digest=${tarball%.tar.gz}.sha256
[ -f "$digest" ] || { say "no package digest $digest"; exit 2; }

# TARGET: a new entry in an existing directory; the launcher refuses a prefix
# with ':' or ';' (the loader splits --library-path on both).
case $target in
    '' | */ | . | .. | */. | */..) say "TARGET must name a new directory"; exit 2 ;;
esac
case $target in
    */*) parent=${target%/*}; base=${target##*/} ;;
    *) parent=.; base=$target ;;
esac
[ -n "$parent" ] || parent=/
parent=$(cd -P -- "$parent" 2>/dev/null && pwd -P) || { say "the parent of $target is not a directory"; exit 2; }
target=${parent%/}/$base
case $target in
    *:* | *\;*) say "TARGET must not contain ':' or ';' ($target)"; exit 2 ;;
esac
if [ -e "$target" ] || [ -L "$target" ]; then
    say "$target exists; nothing was changed"
    exit 2
fi

# The digest names the tarball exactly once.
sums=$(sed -n 's/^\([0-9a-f]\{64\}\)  '"$(printf '%s' "$name" | sed 's/[.+]/\\&/g')"'\.tar\.gz$/\1/p' "$digest")
case $sums in
    '' | *[!0-9a-f]*) fail "$digest does not name $name.tar.gz exactly once" ;;
esac
got=$(sha256sum < "$tarball")
[ "${got%% *}" = "$sums" ] || fail "$name.tar.gz: sha256 is not the one in $digest"

staging=$(mktemp -d "$parent/.saccade-install.XXXXXX") || { say "cannot create a staging directory in $parent"; exit 2; }
mkdir "$staging/x"
say "extracting $name.tar.gz into $staging"
tar -x -z -f "$tarball" -C "$staging/x" --no-same-owner --no-same-permissions \
    --keep-old-files || fail "extraction failed"
[ "$(cd "$staging/x" && find . -mindepth 1 -maxdepth 1 -print)" = "./$name" ] ||
    fail "the tarball does not hold exactly one directory $name"
[ -d "$staging/x/$name" ] && [ ! -L "$staging/x/$name" ] || fail "$name is not a directory"
manifest_list "$staging/x/$name" "$name" "$staging/list"
check_tree "$staging/x/$name" "$staging/list" "$staging" set

# One rename, no replacement: mv -n -T is renameat2(RENAME_NOREPLACE). Some
# mv versions exit 0 when they skip, so the move is confirmed by the source
# being gone.
mv -n -T -- "$staging/x/$name" "$target" || true
if [ -e "$staging/x/$name" ]; then
    if [ -e "$target" ] || [ -L "$target" ]; then
        say "$target appeared during the installation; it was not replaced"
        exit 2
    fi
    fail "the rename to $target failed"
fi
say "installed $name at $target ($count files)"
