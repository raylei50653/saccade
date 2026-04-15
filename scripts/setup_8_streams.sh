#!/bin/bash

# MOT20 影片路徑
VIDEOS=("assets/videos/MOT20-04.mp4" "assets/videos/MOT20-06.mp4" "assets/videos/MOT20-07.mp4" "assets/videos/MOT20-08.mp4")

echo "🎬 Starting 8 background RTSP streams via ffmpeg..."

for i in {0..7}; do
    # 循環使用 4 個影片
    VIDEO=${VIDEOS[$((i % 4))]}
    STREAM_NAME="stream_$i"
    URL="rtsp://127.0.0.1:8554/$STREAM_NAME"
    
    echo "📡 Pushing $VIDEO to $URL"
    
    # 使用 -stream_loop -1 進行無限循環推流
    # 使用 -re 以原始影格率讀取
    ffmpeg -re -stream_loop -1 -i "$VIDEO" -c copy -f rtsp "$URL" > /dev/null 2>&1 &
done

echo "✅ 8 streams are pushing in background. Use 'pkill ffmpeg' to stop."
