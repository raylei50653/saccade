#!/usr/bin/env bash
# Saccade 本地 CI 檢查腳本
# 模擬 GitHub Actions 的所有檢查流程

set -e

# 顏色定義
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Local CI Checks...${NC}"

# 1. 檢查 Lockfile
echo -e "\n${BLUE}[1/7] Checking uv.lock...${NC}"
uv lock --check
echo -e "${GREEN}✅ Lockfile is up to date.${NC}"

# 2. 編譯 C++ 擴充套件 (必要，否則測試會報錯)
echo -e "\n${BLUE}[2/7] Compiling C++ extensions...${NC}"
mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(uv run python -c "import pybind11; print(pybind11.get_cmake_dir())") > /dev/null
make -j$(nproc) > /dev/null
cp *.so /app/
cd ..
echo -e "${GREEN}✅ C++ extensions compiled and synced to root.${NC}"

# 3. Ruff Lint 檢查
echo -e "\n${BLUE}[3/7] Running Ruff Lint...${NC}"
uv run ruff check .
echo -e "${GREEN}✅ Lint checks passed.${NC}"

# 4. Ruff 格式檢查
echo -e "\n${BLUE}[4/7] Checking code formatting...${NC}"
uv run ruff format --check .
echo -e "${GREEN}✅ Code formatting is correct.${NC}"

# 5. Mypy 型別檢查
echo -e "\n${BLUE}[5/7] Running Mypy Type Check...${NC}"
uv run mypy .
echo -e "${GREEN}✅ Type safety verified.${NC}"

# 6. Pytest 單元測試與覆蓋率
echo -e "\n${BLUE}[6/7] Running Pytest with Coverage...${NC}"
uv run pytest tests/ -v --ignore=tests/benchmarks
echo -e "${GREEN}✅ All unit tests passed with coverage report.${NC}"

# 7. GPU & TensorRT 健康檢查 (本地專屬)
echo -e "\n${BLUE}[7/7] Running GPU Stack Deep Check...${NC}"
uv run python scripts/gpu_check.py
echo -e "${GREEN}✅ GPU and TensorRT are healthy.${NC}"

echo -e "\n${GREEN}🎉 Local CI Checks Passed! Your GPU-accelerated pipeline is ready.${NC}"
