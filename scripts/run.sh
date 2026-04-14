#!/usr/bin/env bash
# Saccade 統一啟動入口腳本

set -e

# 確保環境變量包含當前專案路徑
export PYTHONPATH=$PYTHONPATH:$(pwd)

PROJECT_DIR="/home/ray/developer/ai/YOLO_LLM"
cd "$PROJECT_DIR"

case "$1" in
    perception)
        echo "🚀 Starting Perception Pipeline (YOLO)..."
        exec uv run python main.py --mode perception
        ;;
    orchestrator)
        echo "🚀 Starting Orchestrator (Logic & High-Speed Indexer)..."
        exec uv run python main.py --mode orchestrator
        ;;
    api)
        echo "🌐 Starting Saccade Retrieval API..."
        exec uv run uvicorn api.server:app --host 0.0.0.0 --port 8000
        ;;

    *)
        echo "Usage: $0 {perception|orchestrator|api}"
        exit 1
        ;;
esac
