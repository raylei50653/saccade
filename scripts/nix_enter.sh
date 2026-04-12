#!/usr/bin/env bash
# 一鍵進入 Nix 開發環境並同步 uv 依賴

echo "🚀 Entering Nix development shell..."
nix develop --command bash -c "uv sync && exec bash"
