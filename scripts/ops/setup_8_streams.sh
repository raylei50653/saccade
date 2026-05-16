#!/usr/bin/env bash
#
# setup_8_streams.sh — Launch 8 RTSP publisher streams (ffmpeg) into MediaMTX.
#
# Usage:
#   ./setup_8_streams.sh          # start 8 streams (daemonizes, exits)
#   ./setup_8_streams.sh stop     # kill all managed streams
#   ./setup_8_streams.sh status   # show running PIDs & stream URLs
#
# Environment variables:
#   RTSP_HOST, RTSP_PORT, RTSP_PUBLISH_USER, RTSP_PUBLISH_PASSWORD
#   RTSP_PATH_PREFIX, PID_FILE
#
# All publishers are written to /tmp/ffmpeg_stream_<N>.log
#
# Daemonization: each ffmpeg runs in its own session via setsid,
# fully isolated from this script's process group.
#

set -euo pipefail

PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
RTSP_HOST="${RTSP_HOST:-127.0.0.1}"
RTSP_PORT="${RTSP_PORT:-8554}"
RTSP_PUBLISH_USER="${RTSP_PUBLISH_USER:-publisher}"
RTSP_PUBLISH_PASSWORD="${RTSP_PUBLISH_PASSWORD:-pubpass123}"
RTSP_PATH_PREFIX="${RTSP_PATH_PREFIX:-stream_}"
PID_FILE="${PID_FILE:-$PROJ/runs/ffmpeg_streams.pid}"

VIDEOS=(
    "$PROJ/datasets/MOT17_videos/MOT17-04-SDP.mp4"
    "$PROJ/datasets/MOT17_videos/MOT17-05-SDP.mp4"
    "$PROJ/datasets/MOT17_videos/MOT17-09-SDP.mp4"
    "$PROJ/datasets/MOT17_videos/MOT17-10-SDP.mp4"
)

mkdir -p "$(dirname "$PID_FILE")"

# ── helpers ──────────────────────────────────────────────────────────
stream_url() {
    echo "rtsp://${RTSP_PUBLISH_USER}:${RTSP_PUBLISH_PASSWORD}@${RTSP_HOST}:${RTSP_PORT}/${RTSP_PATH_PREFIX}${1}"
}

is_alive() {
    kill -0 "$1" 2>/dev/null
}

# ── stop ─────────────────────────────────────────────────────────────
do_stop() {
    local killed=0
    if [[ -f "$PID_FILE" ]]; then
        local pids
        pids=($(cat "$PID_FILE"))
        echo "🛑 Stopping ${#pids[@]} ffmpeg processes from $PID_FILE ..."
        for pid in "${pids[@]}"; do
            if is_alive "$pid"; then
                kill "$pid" 2>/dev/null && killed=$((killed + 1)) || true
            fi
        done
        rm -f "$PID_FILE"
    fi
    # Clean up any stray ffmpeg not in PID file
    local stray
    stray=$(pgrep -f "rtsp://${RTSP_PUBLISH_USER}:${RTSP_PUBLISH_PASSWORD}@${RTSP_HOST}:${RTSP_PORT}/${RTSP_PATH_PREFIX}" || true)
    if [[ -n "$stray" ]]; then
        echo "🛑 Killing stray publishers ..."
        echo "$stray" | xargs kill 2>/dev/null || true
        killed=$((killed + $(echo "$stray" | wc -l)))
    fi
    sleep 0.5
    # Force-kill stragglers
    local remaining
    remaining=$(pgrep -f "rtsp://${RTSP_PUBLISH_USER}:${RTSP_PUBLISH_PASSWORD}@${RTSP_HOST}:${RTSP_PORT}/${RTSP_PATH_PREFIX}" || true)
    if [[ -n "$remaining" ]]; then
        echo "🔨 Force-killing ..."
        echo "$remaining" | xargs kill -9 2>/dev/null || true
        killed=$((killed + $(echo "$remaining" | wc -l)))
    fi
    echo "✅ All ffmpeg stream publishers stopped ($killed processes)."
}

# ── status ───────────────────────────────────────────────────────────
do_status() {
    echo "=== FFmpeg RTSP Publisher Status ==="
    if [[ -f "$PID_FILE" ]]; then
        local pids
        pids=($(cat "$PID_FILE"))
        local alive_count=0
        for i in "${!pids[@]}"; do
            local pid="${pids[$i]}"
            local url
            url=$(stream_url "$i")
            if is_alive "$pid"; then
                echo "  ✅ stream_$i  PID=$pid  URL=$url"
                alive_count=$((alive_count + 1))
            else
                echo "  ❌ stream_$i  PID=$pid  DEAD"
            fi
        done
        echo "  Total: $alive_count/${#pids[@]} alive"
    else
        local running
        running=$(pgrep -f "rtsp://${RTSP_PUBLISH_USER}:${RTSP_PUBLISH_PASSWORD}@${RTSP_HOST}:${RTSP_PORT}/${RTSP_PATH_PREFIX}" || true)
        if [[ -n "$running" ]]; then
            echo "  ⚠️  Found unmanaged ffmpeg processes:"
            echo "$running" | while read -r pid; do
                echo "      PID=$pid"
            done
        else
            echo "  No stream publishers running."
        fi
    fi
}

# ── launch a single stream (called by do_start) ─────────────────────
# Args: $1=stream_index $2=video_path_or_testsrc $3=url $4=log_path
launch_stream() {
    local idx="$1"
    local video="$2"
    local url="$3"
    local log="$4"
    local pidfile="/tmp/ffmpeg_stream_${idx}.pid"

    # Build the ffmpeg command
    local cmd
    if [[ "$video" == testsrc ]]; then
        cmd="ffmpeg -nostdin -re -f lavfi -i testsrc=size=960x960:rate=15 -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -f rtsp -rtsp_transport tcp \"$url\""
    else
        cmd="ffmpeg -nostdin -re -stream_loop -1 -i \"$video\" -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -c:a aac -ac 1 -ar 44100 -f rtsp -rtsp_transport tcp \"$url\""
    fi

    # Run in a new session (setsid) with stdin/stdout/stderr → /dev/null
    # The ffmpeg PID is written to pidfile so the parent can track it.
    setsid bash -c "
        $cmd > \"$log\" 2>&1 </dev/null &
        echo \$! > \"$pidfile\"
        wait
    " &
}

# ── start (daemonizes — exits immediately) ──────────────────────────
do_start() {
    # Kill any existing publishers first (clean start)
    if [[ -f "$PID_FILE" ]] || pgrep -f "rtsp://${RTSP_PUBLISH_USER}:${RTSP_PUBLISH_PASSWORD}@${RTSP_HOST}:${RTSP_PORT}/${RTSP_PATH_PREFIX}" > /dev/null 2>&1; then
        echo "⚠️  Existing publishers detected. Stopping first ..."
        do_stop
        sleep 0.5
    fi

    echo "🎬 Starting 8 background RTSP streams via ffmpeg..."
    local pids=()

    for i in {0..7}; do
        local video_idx=$((i % ${#VIDEOS[@]}))
        local VIDEO="${VIDEOS[$video_idx]}"
        local url
        url=$(stream_url "$i")
        local log="/tmp/ffmpeg_stream_${i}.log"
        local pidfile="/tmp/ffmpeg_stream_${i}.pid"

        launch_stream "$i" "$VIDEO" "$url" "$log"
        pids+=("$i")
        echo "  📡 stream_$i → $url (launching...)"
    done

    # Give setsid + bash time to start ffmpeg and write PID files
    sleep 1

    # Read the actual ffmpeg PIDs from pidfiles
    local ffmpeg_pids=()
    for i in {0..7}; do
        local pidfile="/tmp/ffmpeg_stream_${i}.pid"
        if [[ -f "$pidfile" ]]; then
            local pid
            pid=$(cat "$pidfile")
            # Verify the PID is actually running
            if is_alive "$pid"; then
                ffmpeg_pids+=("$pid")
                echo "  ✅ stream_$i  ffmpeg PID=$pid"
            else
                echo "  ⚠️  stream_$i  PID=$pid not alive, trying pgrep fallback"
                local fallback
                fallback=$(pgrep -f "rtsp://publisher.*stream_${i}\$" | head -1)
                if [[ -n "$fallback" ]]; then
                    ffmpeg_pids+=("$fallback")
                    echo "  ✅ stream_$i  fallback PID=$fallback"
                else
                    echo "  ❌ stream_$i  FAILED to start"
                fi
            fi
        else
            # Fallback: find by pgrep
            local pid
            pid=$(pgrep -f "rtsp://publisher.*stream_${i}\$" | head -1)
            if [[ -n "$pid" ]]; then
                ffmpeg_pids+=("$pid")
                echo "  ✅ stream_$i  PID=$pid (pgrep)"
            else
                echo "  ❌ stream_$i  FAILED to start"
            fi
        fi
    done

    # Write final PID file
    if [[ ${#ffmpeg_pids[@]} -gt 0 ]]; then
        printf '%s\n' "${ffmpeg_pids[@]}" > "$PID_FILE"
    fi

    echo ""
    echo "✅ 8 streams launched in detached sessions (setsid)."
    echo "   PID file: $PID_FILE"
    echo "   Logs:     /tmp/ffmpeg_stream_{0..7}.log"
    echo "   Manage:   $0 stop | status"

    exit 0
}

# ── main ─────────────────────────────────────────────────────────────
case "${1:-start}" in
    start)   do_start   ;;
    stop)    do_stop    ;;
    status)  do_status  ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
