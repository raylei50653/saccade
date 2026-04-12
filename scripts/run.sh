#!/usr/bin/env bash
# Saccade 統一啟動入口腳本

set -e

# 核心：顯式載入 Nix 與 使用者環境路徑
export PATH="/nix/var/nix/profiles/default/bin:/home/ray/.nix-profile/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
source /home/ray/.bashrc || true

PROJECT_DIR="/home/ray/developer/ai/YOLO_LLM"
MODEL_PATH="/home/ray/developer/ai/models/unsloth/Qwen3-VL-4B-Instruct-GGUF/Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH="/home/ray/developer/ai/models/unsloth/Qwen3-VL-4B-Instruct-GGUF/mmproj-F16.gguf"

cd "$PROJECT_DIR"

case "$1" in
    vlm-backend)
        echo "🚀 Starting VLM Inference Backend (llama-server)..."
        exec /home/ray/.local/bin/llama-server \
            -m "$MODEL_PATH" \
            --mmproj "$MMPROJ_PATH" \
            --host 0.0.0.0 --port 8080 \
            -ngl 24 -c 4096
        ;;
    perception)
        echo "🚀 Starting Perception Pipeline (YOLO)..."
        exec uv run python main.py --mode perception
        ;;
    orchestrator)
        echo "🚀 Starting Orchestrator (Logic & Cognition Loop)..."
        exec uv run python main.py --mode orchestrator
        ;;
    *)
        echo "Usage: $0 {vlm-backend|perception|orchestrator}"
        exit 1
        ;;
esac
