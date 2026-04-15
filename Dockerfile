# 使用 NVIDIA CUDA 13.2 鏡像 (含 cuDNN，對應 2026 年最新感知架構)
FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu22.04

# 避免安裝過程中的互動式詢問
ENV DEBIAN_FRONTEND=noninteractive

# 1. 配置 APT 鏡像源 (使用台灣國網中心 NCHC 加速)
RUN sed -i 's/archive.ubuntu.com/free.nchc.org.tw/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/free.nchc.org.tw/g' /etc/apt/sources.list

# 2. 安裝核心編譯與感知組件 (使用 BuildKit cache mount 加速)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    ninja-build \
    git \
    wget \
    curl \
    python3-pip \
    python3-dev \
    python3-gi \
    libopencv-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libgirepository1.0-dev \
    gobject-introspection \
    gir1.2-glib-2.0 \
    gir1.2-gstreamer-1.0 \
    libcairo2-dev \
    libglib2.0-dev \
    libgl1-mesa-dev \
    zlib1g-dev \
    ffmpeg \
    zsh \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 安裝 UV 與配置環境
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 設定工作目錄
WORKDIR /app

# 配置環境變數，讓 .venv 內的指令優先執行
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib"
ENV UV_LINK_MODE=copy

# 4. 安裝 Python 依賴 (使用 uv sync 確保一致性)
# 將所有依賴 (torch, tensorrt, dali) 放入 pyproject.toml
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# 5. 配置 Shell 環境 (Oh My Zsh)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    echo 'source /app/.venv/bin/activate' >> /root/.zshrc

# 6. 複製原始碼並完成專案安裝
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["/bin/zsh"]
