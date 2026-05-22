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
CPP_CHANGED=$(git diff --name-only HEAD -- 'src/**' 'include/**' 'CMakeLists.txt' 2>/dev/null || true)
# Also check against main/master
BASE=$(git rev-parse --verify origin/main 2>/dev/null || git rev-parse --verify origin/master 2>/dev/null || echo "")
if [[ -n "$BASE" ]]; then
    CPP_CHANGED=$(git diff --name-only "$BASE"...HEAD -- 'src/' 'include/' 'CMakeLists.txt' 2>/dev/null || true)
fi

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

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
if [[ $ERRORS -eq 0 ]]; then
    ok "All checks passed — safe to push"
    exit 0
else
    fail "$ERRORS check(s) failed"
    exit 1
fi
