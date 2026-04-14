import pynvml
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class DegradationLevel(IntEnum):
    NORMAL = 0      # 正常運行
    REDUCED = 1     # 減少緩衝區 (Level 1, >85% VRAM)
    FAST_PATH = 2   # 僅保留 Perception (Level 2, >92% VRAM)
    EMERGENCY = 3   # 低解析度模式 (Level 3, >96% VRAM)


@dataclass
class VRAMStats:
    free_mb: float
    used_mb: float
    total_mb: float
    usage_percent: float


class ResourceManager:
    """
    Saccade 資源管理器 (Industrial Grade Decision Maker)
    
    1. 實時監控 VRAM 狀態。
    2. 實作階梯式降級 (Stepped Degradation) 邏輯。
    3. 防止系統在臨界點發生抖動 (Hysteresis)。
    """

    def __init__(self, gpu_id: int = 0) -> None:
        self.gpu_id = gpu_id
        self._nvml_initialized = False
        self._init_nvml()
        
        # 狀態紀錄，用於實現 Hysteresis (避免在 89/91% 之間反覆切換)
        self.current_level = DegradationLevel.NORMAL

    def _init_nvml(self) -> None:
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception as e:
            print(f"⚠️ [ResourceManager] NVML Init failed: {e}. Degradation disabled.")

    def get_stats(self) -> VRAMStats:
        if not self._nvml_initialized:
            return VRAMStats(0.0, 0.0, 0.0, 0.0)

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage = (info.used / info.total) * 100
        return VRAMStats(
            free_mb=info.free / 1024 / 1024,
            used_mb=info.used / 1024 / 1024,
            total_mb=info.total / 1024 / 1024,
            usage_percent=usage
        )

    def decide_degradation_level(self) -> DegradationLevel:
        """
        根據當前資源決定降級級別 (具備 Hysteresis 遲滯保護)
        """
        stats = self.get_stats()
        usage = stats.usage_percent
        
        # 門檻值定義 (上升)
        UP_EMERGENCY = 96
        UP_FAST_PATH = 92
        UP_REDUCED = 85
        
        # 遲滯緩衝 (5%)
        HYSTERESIS = 5
        
        new_level = self.current_level
        
        # 1. 昇級邏輯 (負載增加)
        if usage > UP_EMERGENCY:
            new_level = DegradationLevel.EMERGENCY
        elif usage > UP_FAST_PATH:
            new_level = DegradationLevel.FAST_PATH
        elif usage > UP_REDUCED:
            new_level = DegradationLevel.REDUCED
        
        # 2. 降階邏輯 (負載減少) - 只有當負載下降超過 HYSTERESIS 時才回歸
        if self.current_level == DegradationLevel.EMERGENCY and usage < (UP_EMERGENCY - HYSTERESIS):
            new_level = DegradationLevel.FAST_PATH
        elif self.current_level == DegradationLevel.FAST_PATH and usage < (UP_FAST_PATH - HYSTERESIS):
            new_level = DegradationLevel.REDUCED
        elif self.current_level == DegradationLevel.REDUCED and usage < (UP_REDUCED - HYSTERESIS):
            new_level = DegradationLevel.NORMAL
            
        if new_level != self.current_level:
            print(f"⚙️ [ResourceManager] System Scaling: {self.current_level.name} -> {new_level.name} (Usage: {usage:.1f}%)")
            self.current_level = new_level
            
        return self.current_level

    def close(self) -> None:
        if self._nvml_initialized:
            pynvml.nvmlShutdown()


if __name__ == "__main__":
    manager = ResourceManager()
    stats = manager.get_stats()
    level = manager.decide_degradation_level()
    print(f"Stats: {stats}")
    print(f"Selected Level: {level.name}")
    manager.close()
