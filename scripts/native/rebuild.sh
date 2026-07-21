#!/usr/bin/env bash
# status: stable
# Saccade C++/CUDA Extension Rebuild Script

set -e

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "🛠️  Rebuilding Saccade C++/CUDA extensions..."
cd "$PROJECT_ROOT"

# 1. Clean and build
mkdir -p build
cd build
cmake ..
make clean
make -j$(nproc)

# 2. Register build/ in venv so Python can find native extensions
cd "$PROJECT_ROOT"
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$PROJECT_ROOT/build" > "$SITE_PACKAGES/saccade_build.pth"
echo "Registered build/ -> $SITE_PACKAGES/saccade_build.pth"

echo "✅ Rebuild complete. You can now run benchmarks or the pipeline."
