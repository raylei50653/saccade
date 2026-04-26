#!/bin/bash

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
VIDEOS=(
    "$PROJ/assets/videos/MOT20-04.mp4"
    "$PROJ/assets/videos/MOT20-06.mp4"
    "$PROJ/assets/videos/MOT20-07.mp4"
    "$PROJ/assets/videos/MOT20-08.mp4"
)

echo "🎬 Starting 8 background RTSP streams via ffmpeg..."

for i in {0..7}; do
    VIDEO=${VIDEOS[$((i % 4))]}
    URL="rtsp://publisher:pubpass123@127.0.0.1:8554/stream_$i"
    ffmpeg -re -stream_loop -1 -i "$VIDEO" -c copy -f rtsp "$URL" > /tmp/ffmpeg_stream_$i.log 2>&1 &
    echo "📡 stream_$i → $URL (pid $!)"
done

echo "✅ 8 streams pushing in background. Use 'pkill ffmpeg' to stop."
