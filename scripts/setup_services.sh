#!/usr/bin/env bash
# Saccade Systemd 服務安裝腳本

set -e

PROJECT_DIR="/home/ray/developer/ai/YOLO_LLM"
SYSTEMD_DIR="/etc/systemd/system"

echo "🛠️  Setting up Saccade Systemd services..."

# 1. 確保在正確目錄
cd "$PROJECT_DIR"

# 2. 建立連結 (使用絕對路徑)
for service in infra/systemd/*.service; do
    service_name=$(basename "$service")
    echo "  - Linking $service_name..."
    sudo ln -sf "$PROJECT_DIR/$service" "$SYSTEMD_DIR/$service_name"
done

# 3. 清理過時的服務名稱
if [ -f "$SYSTEMD_DIR/yolo-cognition.service" ]; then
    echo "  - Removing legacy yolo-cognition.service..."
    sudo rm -f "$SYSTEMD_DIR/yolo-cognition.service"
fi

# 4. 重新載入 Systemd
echo "♻️  Reloading Systemd daemon..."
sudo systemctl daemon-reload

echo "✅ Systemd services are now linked and ready."
echo "👉 You can now run './scripts/saccade up' again."
