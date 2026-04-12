import os
from typing import List, Any, Optional, Dict, cast
import numpy as np
from ultralytics import YOLO # type: ignore[attr-defined]
from dotenv import load_dotenv

load_dotenv()

class Detector:
    """
    YOLO 偵測器 (Perception 快路徑)
    
    支援 YOLOv8/v10/v11 以及 TensorRT 加速。
    """
    def __init__(self, model_path: str = "./models/yolo/yolo11n.engine", device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model: Optional[YOLO] = None
        
        # 優先搜尋 .engine 檔案
        if not os.path.exists(model_path):
            fallback_pt = model_path.replace(".engine", ".pt")
            if os.path.exists(fallback_pt):
                print(f"💡 Info: TensorRT engine not found, falling back to PyTorch: {fallback_pt}")
                self.model_path = fallback_pt
            else:
                print(f"Warning: Model not found at {model_path}, using default fallback.")
                self.model_path = "models/yolo/yolo11n.pt"
            
        self.load_model()

    def load_model(self) -> None:
        """載入 YOLO 模型權重"""
        try:
            # 對於 .engine 檔案，不需要手動 call .to()
            self.model = YOLO(self.model_path, task="detect")
            if not self.model_path.endswith(".engine"):
                if self.model:
                    self.model.to(self.device)
            print(f"✅ YOLO model loaded: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {str(e)}")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Any]:
        """
        執行目標偵測
        
        :param frame: OpenCV 格式影格 (BGR)
        :param conf_threshold: 置信度閾值
        :return: 偵測結果列表
        """
        if self.model is None:
            return []
            
        results = self.model.track(
            source=frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
            persist=True # 開啟追蹤 (Tracking)
        )
        
        # 目前僅回傳原始 Results 物件，供後續模組處理
        return cast(List[Any], results)

    def get_actionable_labels(self, results: List[Any]) -> List[str]:
        """將偵測結果中的 Class IDs 轉換為標籤名稱"""
        if not results or len(results) == 0:
            return []
        
        if self.model is None:
            return []
            
        names: Dict[int, str] = getattr(self.model, "names", {})
        labels = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))
            labels.append(label)
        return list(set(labels)) # 去重

if __name__ == "__main__":
    # 測試偵測
    detector = Detector()
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    results = detector.detect(dummy_frame)
    if results and len(results) > 0:
        print(f"Detection performed, found {len(results[0].boxes)} objects.")
