#!/usr/bin/env bash
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

# 2. Ensure symlinks in root are correct
echo "🔗 Updating symlinks in project root..."
cd "$PROJECT_ROOT"

# Helper function to create symlink
create_link() {
    local target="build/$1"
    local link="$1"
    if [ -f "$target" ]; then
        ln -sf "$target" "$link"
        echo "  - Linked $link -> $target"
    else
        echo "  - ⚠️  Target $target not found!"
    fi
}

# Find all .so files in build directory and link them to root
find build -maxdepth 1 -name "*.so" -exec basename {} \; | while read -r so_file; do
    create_link "$so_file"
done

echo "✅ Rebuild complete. You can now run benchmarks or the pipeline."
