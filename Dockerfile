# 使用 NVIDIA CUDA 13.2 鏡像 (對應 2026 年最新驅動與 Torch 13.0)
FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

# 避免安裝過程中的互動式詢問
ENV DEBIAN_FRONTEND=noninteractive

# 1. 配置 APT 鏡像源 (使用台灣國網中心 NCHC，加速系統組件安裝)
RUN sed -i 's/archive.ubuntu.com/free.nchc.org.tw/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/free.nchc.org.tw/g' /etc/apt/sources.list

# 2. 安裝核心編譯與感知組件 (使用 BuildKit cache mount 加速)
# 移除 docker-clean 腳本以確保快取掛載點能真正保留 .deb 檔案
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    ninja-build \
    git \
    wget \
    curl \
    python3-pip \
    python3-venv \
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
    && ln -s $(find /usr/lib -name gobject-introspection-1.0.pc | head -n 1) $(dirname $(find /usr/lib -name gobject-introspection-1.0.pc | head -n 1))/girepository-1.0.pc \
    && ln -s $(find /usr/lib -name gobject-introspection-1.0.pc | head -n 1) $(dirname $(find /usr/lib -name gobject-introspection-1.0.pc | head -n 1))/girepository-2.0.pc

# 安裝 Oh My Zsh (無互動模式)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    chsh -s $(which zsh)

# 3. 使用官方推薦方式安裝 Python 依賴管理工具 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 設定工作目錄
WORKDIR /app

# 4. 配置 Python 3.12 並安裝依賴
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./

# 安裝 Python 3.12 並建立虛擬環境
RUN uv python install 3.12 && uv venv

# 第一階段：先安裝最重的核心庫 (利用 Docker 內部快取，下次不用重新下載)
# 這樣這些庫會被緩存在一個獨立的 Layer，減少單一層壓縮的 RAM 壓力
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision onnxruntime-gpu tensorrt nvidia-dali-cuda120

# 第二階段：同步剩下的專案依賴 (同樣利用內部快取)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# 把 Python 虛擬環境加到 zshrc
RUN echo 'source /app/.venv/bin/activate' >> /root/.zshrc


# 複製原始碼
COPY . .

# 環境變量配置
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

CMD ["/bin/zsh"]
