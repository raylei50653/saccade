#!/usr/bin/env bash
# status: diagnostic
# Re-apply graphify CUDA (.cu/.cuh) AST support after a graphify upgrade.
#
# graphify (PyPI: graphifyy) has no built-in .cu support and no user-config for
# custom extensions, so the only way to add it is to edit the installed package:
#   1. detect.py   CODE_EXTENSIONS  += '.cu', '.cuh'   (classify as code)
#   2. extract.py  _DISPATCH        += ".cu"/".cuh" -> extract_cpp
# CUDA is a C++ superset, so we reuse the existing tree-sitter-cpp extractor —
# no new grammar, no pyproject change. watch.py/_WATCHED_EXTENSIONS and
# collect_files/_EXTENSIONS derive from the two sets above, so they need no edit.
#
# A `graphify` upgrade overwrites site-packages and wipes the patch. Re-run this
# script after every upgrade. It is idempotent: already-patched files are skipped.
#
# Usage: bash scripts/tools/graphify_cuda_patch.sh
set -euo pipefail

# ── Resolve the interpreter graphify actually runs under ──────────────────────
PYTHON=""
if [ -f graphify-out/.graphify_python ]; then
    PYTHON="$(cat graphify-out/.graphify_python)"
    [ -x "$PYTHON" ] || PYTHON=""
fi
if [ -z "$PYTHON" ]; then
    GBIN="$(command -v graphify 2>/dev/null || true)"
    if [ -n "$GBIN" ]; then
        SHEBANG="$(head -1 "$GBIN" | tr -d '#!')"
        case "$SHEBANG" in
            *[!a-zA-Z0-9/_.-]*) ;;                       # reject weird shebangs
            *) "$SHEBANG" -c "import graphify" 2>/dev/null && PYTHON="$SHEBANG" ;;
        esac
    fi
fi
[ -z "$PYTHON" ] && PYTHON="python3"

if ! "$PYTHON" -c "import graphify" 2>/dev/null; then
    echo "✗ graphify not importable under $PYTHON — install it first (uv tool install graphifyy)." >&2
    exit 1
fi

# ── Apply the two idempotent edits inside the installed package ───────────────
"$PYTHON" - <<'PY'
import sys, pathlib, graphify

pkg = pathlib.Path(graphify.__file__).parent
print(f"graphify package: {pkg}")
changed = 0
errors = 0

# 1) detect.py — add '.cu', '.cuh' to CODE_EXTENSIONS
d = pkg / "detect.py"
t = d.read_text(encoding="utf-8")
if "'.cu'" in t and "'.cuh'" in t:
    print("• detect.py    : already patched")
elif "'.hpp'," in t:
    t = t.replace("'.hpp',", "'.hpp', '.cu', '.cuh',", 1)
    d.write_text(t, encoding="utf-8")
    print("✓ detect.py    : added '.cu', '.cuh' to CODE_EXTENSIONS")
    changed += 1
else:
    print("✗ detect.py    : anchor \"'.hpp',\" not found — structure changed, patch manually")
    errors += 1

# 2) extract.py — map .cu/.cuh to extract_cpp in _DISPATCH
e = pkg / "extract.py"
t = e.read_text(encoding="utf-8")
if '".cu": extract_cpp' in t:
    print("• extract.py   : already patched")
elif '".hpp": extract_cpp,' in t:
    t = t.replace(
        '".hpp": extract_cpp,',
        '".hpp": extract_cpp,\n    ".cu": extract_cpp,\n    ".cuh": extract_cpp,',
        1,
    )
    e.write_text(t, encoding="utf-8")
    print("✓ extract.py   : mapped .cu/.cuh -> extract_cpp in _DISPATCH")
    changed += 1
else:
    print('✗ extract.py   : anchor \'".hpp": extract_cpp,\' not found — structure changed, patch manually')
    errors += 1

if errors:
    sys.exit(1)
print(f"\n{'patched' if changed else 'no changes (already current)'}.")
PY

# ── Verify the patch is live in a fresh interpreter ──────────────────────────
echo "── verify"
"$PYTHON" - <<'PY'
import importlib, graphify
import graphify.detect as D
import graphify.extract as E
importlib.reload(D); importlib.reload(E)
ok = ".cu" in D.CODE_EXTENSIONS and E._DISPATCH.get(".cu") is not None
print(f"  .cu in CODE_EXTENSIONS : {'.cu' in D.CODE_EXTENSIONS}")
print(f"  .cu -> {getattr(E._DISPATCH.get('.cu'), '__name__', None)}")
print("✓ CUDA AST support active" if ok else "✗ verification failed")
raise SystemExit(0 if ok else 1)
PY

echo
echo "Done. To pull CUDA nodes into an existing graph, re-run code extraction:"
echo "    graphify update ."
