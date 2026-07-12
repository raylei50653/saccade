#!/usr/bin/env bash
# Mirror CI checks locally before pushing.
# Usage: bash scripts/pre_push.sh [--fix]
#   --fix  auto-apply ruff fixes before checking

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }

FIX=0
for arg in "$@"; do [[ "$arg" == "--fix" ]] && FIX=1; done

ERRORS=0

# ── 1. lockfile ──────────────────────────────────────────────────────────────
echo "── uv lock --check"
if uv lock --check 2>&1; then
    ok "lockfile up to date"
else
    fail "uv.lock is out of sync — run: uv lock"
    ERRORS=$((ERRORS + 1))
fi

# ── 2. ruff lint ─────────────────────────────────────────────────────────────
echo "── ruff check"
if [[ $FIX -eq 1 ]]; then
    uv run ruff check --fix . && ok "ruff check (auto-fixed)" || { fail "ruff check"; ERRORS=$((ERRORS + 1)); }
else
    if uv run ruff check . 2>&1; then
        ok "ruff check"
    else
        fail "ruff check — rerun with --fix to auto-fix"
        ERRORS=$((ERRORS + 1))
    fi
fi

# ── 3. ruff format ───────────────────────────────────────────────────────────
echo "── ruff format"
if [[ $FIX -eq 1 ]]; then
    uv run ruff format . && ok "ruff format (auto-fixed)"
else
    if uv run ruff format --check . 2>&1; then
        ok "ruff format"
    else
        fail "ruff format — rerun with --fix to auto-format"
        ERRORS=$((ERRORS + 1))
    fi
fi

# ── 4. mypy ──────────────────────────────────────────────────────────────────
echo "── mypy"
if uv run mypy . 2>&1; then
    ok "mypy"
else
    fail "mypy"
    ERRORS=$((ERRORS + 1))
fi

# ── 4.5 GPU-first contract ───────────────────────────────────────────────────
echo "── GPU-first contract check"
if uv run python3 scripts/tools/check_gpu_contract.py 2>&1; then
    ok "GPU-first contract"
else
    fail "GPU-first contract"
    ERRORS=$((ERRORS + 1))
fi

# ── 4.6 API layer audit ──────────────────────────────────────────────────────
echo "── API layer audit"
uv run python3 scripts/tools/check_api_layers.py 2>&1 || true
ok "API layer audit (warnings only)"

# ── 4.7 doc link check ───────────────────────────────────────────────────────
echo "── doc link check"
if uv run python3 scripts/tools/check_doc_links.py 2>&1; then
    ok "doc links"
else
    fail "doc links — broken relative link(s) in docs/"
    ERRORS=$((ERRORS + 1))
fi

# ── 4.8 stale doc path check ─────────────────────────────────────────────────
echo "── stale doc path check"
if uv run python3 scripts/tools/check_doc_stale_paths.py 2>&1; then
    ok "stale doc paths"
else
    fail "stale doc paths — reference(s) to pre-move Phase 1 doc location(s)"
    ERRORS=$((ERRORS + 1))
fi

# ── 4.9 doc freshness check (warn-only) ──────────────────────────────────────
echo "── doc freshness check (warn-only)"
uv run python3 scripts/tools/check_doc_freshness.py 2>&1 || true
ok "doc freshness (warnings only)"

# ── 4.10 doc structure: index coverage (warn) + C6.4 lifecycle (fail-closed) ─
# Closing a research unit is a merge condition, not a matter of discipline:
# a note that declares itself closed must leave the active path and the active
# index in the same PR that accepted its terminal (Doc Structure C6.3).
echo "── doc structure check (index warn; lifecycle fail-closed)"
if uv run python3 scripts/tools/check_doc_structure.py --strict 2>&1; then
  ok "doc structure"
else
  fail "doc structure — C6.4 lifecycle violation(s); see docs/ownership/doc_structure_contract.md"
  ERRORS=$((ERRORS + 1))
fi

# ── 5. pytest ────────────────────────────────────────────────────────────────
echo "── pytest"
if uv run pytest tests/ -q --ignore=tests/benchmarks 2>&1; then
    ok "pytest"
else
    fail "pytest"
    ERRORS=$((ERRORS + 1))
fi

# ── 6. C++ build check ───────────────────────────────────────────────────────
echo "── C++ change detection"
CHANGED_FILES=$(
    {
        git diff --name-only -- 'src/' 'include/' 'CMakeLists.txt' 2>/dev/null || true
        git diff --cached --name-only -- 'src/' 'include/' 'CMakeLists.txt' 2>/dev/null || true
    } | sort -u
)
# Also check committed changes against main/master.
BASE=$(git rev-parse --verify origin/main 2>/dev/null || git rev-parse --verify origin/master 2>/dev/null || echo "")
if [[ -n "$BASE" ]]; then
    CHANGED_FILES=$(
        {
            printf '%s\n' "$CHANGED_FILES"
            git diff --name-only "$BASE"...HEAD -- 'src/' 'include/' 'CMakeLists.txt' 2>/dev/null || true
        } | sort -u
    )
fi
CPP_CHANGED=$(
    printf '%s\n' "$CHANGED_FILES" |
        grep -E '^(CMakeLists\.txt|src/.*\.(c|cc|cpp|cxx|cu|cuh|h|hh|hpp|hxx)|include/.*\.(cuh|h|hh|hpp|hxx))$' ||
        true
)

if [[ -n "$CPP_CHANGED" ]]; then
    warn "C++ files changed — CI will compile. Verify local build:"
    echo "    cd build && make saccade_tracking_ext -j\$(nproc)"
    if [[ -d build ]]; then
        echo "── make saccade_tracking_ext"
        if (cd build && make saccade_tracking_ext -j"$(nproc)" 2>&1); then
            ok "C++ build"
        else
            fail "C++ build"
            ERRORS=$((ERRORS + 1))
        fi
    else
        warn "build/ dir not found — skipping compile (CI will catch errors)"
    fi
else
    ok "no C++ changes — build check skipped"
fi

# ── 7. determinism-sensitive continuous chain ────────────────────────────────
echo "── determinism-sensitive path check"
if [[ "${SKIP_DETERMINISM_PREPUSH:-0}" == "1" ]]; then
    warn "SKIP_DETERMINISM_PREPUSH=1 — determinism chain SKIPPED"
else
    DETERM_OUTPUT=""
    set +e
    DETERM_OUTPUT=$(uv run python3 scripts/tools/check_determinism_paths.py 2>/dev/null)
    DETERM_RC=$?
    set -e
    if [[ $DETERM_RC -ne 0 ]] && [[ $DETERM_RC -ne 1 ]]; then
        fail "determinism path detection failed (exit $DETERM_RC)"
        ERRORS=$((ERRORS + 1))
    elif [[ $DETERM_RC -eq 0 ]] && [[ "$DETERM_OUTPUT" == "determinism" ]]; then
        echo "── routine continuous chain (A,A,B,A,B,B)"
        echo "    (SKIP_DETERMINISM_PREPUSH=1 to skip)"
        if uv run python scripts/tools/check_decimal_chain_routine.py \
            --preset mamba_whole_graph_m --detector SDP --double-buffer \
            2>&1; then
            ok "routine continuous chain"
        else
            fail "routine continuous chain — divergence detected"
            ERRORS=$((ERRORS + 1))
        fi
    else
        ok "no determinism-sensitive changes"
    fi
fi

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
if [[ $ERRORS -eq 0 ]]; then
    ok "All checks passed — safe to push"
    exit 0
else
    fail "$ERRORS check(s) failed"
    exit 1
fi
