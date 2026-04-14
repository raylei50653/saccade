#!/bin/bash
# scripts/docker_run.sh
# 用於工業設備的生產環境啟動腳本

CONTAINER_NAME="saccade-perception"
IMAGE_NAME="saccade:latest"

echo "🚀 正在啟動 Saccade 服務: $CONTAINER_NAME..."

# 檢查 Docker 容器是否已存在並運行
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "⚠️ 容器已在運行中，正在停止並移除..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# 執行容器
docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  --restart unless-stopped \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/models:/app/models \
  --network host \
  $IMAGE_NAME \
  uv run main.py --mode perception

echo "✅ 服務已啟動。請使用 'docker logs -f $CONTAINER_NAME' 查看日誌。"
