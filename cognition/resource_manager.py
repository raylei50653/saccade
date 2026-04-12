import os
import yaml
import pynvml
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class VRAMStats:
    free_mb: float
    used_mb: float
    total_mb: float

class ResourceManager:
    """
    Saccade 資源管理器 (Cognition 慢路徑)
    
    負責監控 NVIDIA VRAM 狀態，並根據可用資源動態選擇模型載入配置。
    實踐 DEVELOPMENT.md 中的 Pillar 2: Dynamic Compute Provisioning。
    """
    def __init__(self, config_path: str = "configs/llm_profiles.yaml"):
        self.config_path = config_path
        self.profiles = self._load_profiles()
        self._nvml_initialized = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        """初始化 NVML"""
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            print("✅ NVML Initialized successfully.")
        except Exception as e:
            print(f"⚠️ Failed to initialize NVML: {e}. VRAM monitoring will be disabled.")

    def _load_profiles(self) -> List[Dict[str, Any]]:
        """載入 VRAM 配置設定"""
        if not os.path.exists(self.config_path):
            print(f"⚠️ Config not found: {self.config_path}, using minimal defaults.")
            return [{"name": "default", "vram_limit": 4.0, "n_gpu_layers": 12}]
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            from typing import cast
            return cast(List[Dict[str, Any]], data.get("profiles", []))

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
                total_mb=info.total / 1024 / 1024
            )
        except Exception as e:
            print(f"❌ Error getting VRAM stats: {e}")
            return VRAMStats(0.0, 0.0, 0.0)

    def select_optimal_profile(self, reserve_mb: float = 2048) -> Dict[str, Any]:
        """
        根據當前可用 VRAM 選擇最適合的配置。
        reserve_mb: 預留給 Perception (YOLO) 或系統使用的 VRAM 緩衝 (預設 2GB)。
        """
        stats = self.get_vram_stats()
        available_gb = (stats.free_mb - reserve_mb) / 1024
        
        print(f"📊 Current Free VRAM: {stats.free_mb:.1f} MB (Target available: {available_gb:.1f} GB)")

        # 根據 vram_limit 從高到低排序 profile
        sorted_profiles = sorted(self.profiles, key=lambda x: x['vram_limit'], reverse=True)
        
        for profile in sorted_profiles:
            if available_gb >= profile['vram_limit']:
                print(f"🎯 Selected Profile: {profile['name']} ({profile['description']})")
                return profile
        
        # 如果都不符合，回傳最低配置
        print("⚠️ VRAM critically low, falling back to minimal profile.")
        return sorted_profiles[-1]

    def close(self) -> None:
        """關閉 NVML"""
        if self._nvml_initialized:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    # 測試執行
    manager = ResourceManager()
    profile = manager.select_optimal_profile()
    print(f"Final Recommendation: {profile}")
    manager.close()
