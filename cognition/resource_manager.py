import pynvml
from dataclasses import dataclass


@dataclass
class VRAMStats:
    free_mb: float
    used_mb: float
    total_mb: float


class ResourceManager:
    """
    Saccade 資源管理器 (GPU VRAM 監控)

    負責監控 NVIDIA VRAM 狀態，為系統提供資源決策依據。
    """

    def __init__(self) -> None:
        self._nvml_initialized = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        """初始化 NVML"""
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            print("✅ NVML Initialized successfully.")
        except Exception as e:
            print(
                f"⚠️ Failed to initialize NVML: {e}. VRAM monitoring will be disabled."
            )

    def get_vram_stats(self, gpu_id: int = 0) -> VRAMStats:
        """獲取指定 GPU 的實時記憶體狀態"""
        if not self._nvml_initialized:
            return VRAMStats(0.0, 0.0, 0.0)

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # 轉換為 MB
            return VRAMStats(
                free_mb=info.free / 1024 / 1024,
                used_mb=info.used / 1024 / 1024,
                total_mb=info.total / 1024 / 1024,
            )
        except Exception as e:
            print(f"❌ Error getting VRAM stats: {e}")
            return VRAMStats(0.0, 0.0, 0.0)

    def close(self) -> None:
        """關閉 NVML"""
        if self._nvml_initialized:
            pynvml.nvmlShutdown()


if __name__ == "__main__":

    def main() -> None:
        # 測試執行
        manager = ResourceManager()
        stats = manager.get_vram_stats()
        print(f"Current Stats: {stats}")
        manager.close()

    main()
