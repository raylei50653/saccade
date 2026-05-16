#!/usr/bin/env bash
# Saccade 系統管理控制台 (Unified CLI)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVICES=("mediamtx" "yolo-perception" "yolo-orchestrator" "yolo-api")
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
RTSP_PORT="${RTSP_PORT:-8554}"
RTSP_READ_USER="${RTSP_READ_USER:-reader}"
RTSP_READ_PASSWORD="${RTSP_READ_PASSWORD:-readpass123}"
RTSP_PUBLISH_USER="${RTSP_PUBLISH_USER:-publisher}"
RTSP_PUBLISH_PASSWORD="${RTSP_PUBLISH_PASSWORD:-pubpass123}"
RTSP_LIVE_PATH="${RTSP_LIVE_PATH:-live}"
RTSP_READER_URL="rtsp://${RTSP_READ_USER}:${RTSP_READ_PASSWORD}@${RTSP_HOST}:${RTSP_PORT}/${RTSP_LIVE_PATH}"
RTSP_PUBLISHER_URL="rtsp://${RTSP_PUBLISH_USER}:${RTSP_PUBLISH_PASSWORD}@${RTSP_HOST}:${RTSP_PORT}/${RTSP_LIVE_PATH}"

cd "$PROJECT_DIR"

case "$1" in
    up)
        printf "🚀 Starting all Saccade services...\n"
        for s in "${SERVICES[@]}"; do
            printf "  - Starting %s...\n" "$s"
            systemctl --user start "$s"
        done
        printf "✅ All services requested. Use './scripts/ops/saccade_ctl.sh health' to verify.\n"
        ;;
    down)
        printf "🛑 Stopping all Saccade services...\n"
        systemctl --user stop "${SERVICES[@]}" || true
        printf "✅ All services stopped.\n"
        ;;
    restart)
        printf "♻️  Restarting all services...\n"
        for s in "${SERVICES[@]}"; do
            systemctl --user restart "$s"
        done
        ;;
    kill)
        printf "💀 Performing deep cleanup of all Saccade processes...\n"
        systemctl --user stop "${SERVICES[@]}" >/dev/null 2>&1 || true
        pkill -9 -f mediamtx >/dev/null 2>&1 || true
        pkill -9 -f "python main.py" >/dev/null 2>&1 || true
        fuser -k 8080/tcp >/dev/null 2>&1 || true
        fuser -k 8554/tcp >/dev/null 2>&1 || true
        printf "✅ Cleanup complete. All processes terminated and auto-restart disabled.\n"
        ;;
    status)
        printf "📊 Service Status:\n"
        for s in "${SERVICES[@]}"; do
            status=$(systemctl --user is-active "$s" || echo "inactive")
            printf "  - %-20s: %s\n" "$s" "$status"
        done
        # 額外檢查 RTSP 輸入流
        if ffprobe -v error "$RTSP_READER_URL" >/dev/null 2>&1; then
            printf "  - %-20s: ACTIVE\n" "🎥 Camera Input"
        else
            printf "  - %-20s: INACTIVE (Waiting for push...)\n" "🎥 Camera Input"
        fi
        ;;
    camera-on)
        if [ -f "$DUMMY_VIDEO_PATH" ]; then
            printf "🎬 Starting demo video stream (%s) in background...\n" "$DUMMY_VIDEO_PATH"
            nohup ffmpeg -re -i "$DUMMY_VIDEO_PATH" -c copy -f rtsp -rtsp_transport tcp "$RTSP_PUBLISHER_URL" > /tmp/mock_cam.log 2>&1 &
        else
            printf "🎬 Starting mock color-bar stream in background...\n"
            nohup ffmpeg -re -f lavfi -i testsrc=size=640x480:rate=15 -vcodec h264 -f rtsp -rtsp_transport tcp "$RTSP_PUBLISHER_URL" > /tmp/mock_cam.log 2>&1 &
        fi
        printf "✅ Stream is pushing to %s\n" "$RTSP_PUBLISHER_URL"
        ;;
    camera-off)
        printf "🛑 Stopping camera stream...\n"
        pkill -f "ffmpeg.*${RTSP_HOST}:${RTSP_PORT}/${RTSP_LIVE_PATH}" || echo "No stream running."
        ;;
    health)
        printf "🩺 Running internal health check...\n"
        uv run python src/saccade/pipeline/health.py
        ;;
    logs)
        case "$2" in
            backend) journalctl --user -u yolo-vlm-backend -f ;;
            perception) journalctl --user -u yolo-perception -f ;;
            orchestrator) journalctl --user -u yolo-orchestrator -f ;;
            *) echo "Usage: $0 logs {backend|perception|orchestrator}" ;;
        esac
        ;;
    *)
        echo "Saccade - Dual-Track Video Perception Management CLI"
        echo "Usage: $0 {up|down|restart|kill|status|camera-on|camera-off|health|logs [service]}"
        exit 1
        ;;
esac
