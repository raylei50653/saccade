import os
import torch
import numpy as np
from typing import Tuple, Optional, Any, cast
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class DALIVideoPipeline(Pipeline):
    """
    NVIDIA DALI 影像預處理管線
    直接在 GPU 上執行解碼 (nvv4l2decoder)、縮放與歸一化。
    """

    def __init__(
        self,
        video_path: str,
        batch_size: int = 1,
        num_threads: int = 2,
        device_id: int = 0,
    ) -> None:
        super().__init__(batch_size, num_threads, device_id, seed=12)  # type: ignore
        self.video_path = video_path

    def define_graph(self) -> Any:
        # 1. 讀取並在 GPU 上解碼影片
        video = fn.readers.video(
            device="gpu",
            filenames=[self.video_path],
            sequence_length=1,
            step=1,
            stride=1,
            initial_fill=10,
            pad_last_batch=True,
            name="reader",
        )

        # 2. 在 GPU 上進行縮放 (為 YOLO 準備 640x640)
        resized = fn.resize(
            cast(Any, video),
            resize_x=640,
            resize_y=640,
            interp_type=cast(Any, types).INTERP_LINEAR,
        )

        # 3. 歸一化與通道轉換 (HWC -> CHW，除以 255.0)
        # YOLO 通常需要 [0, 1] 的 RGB float32 tensor
        normalized = fn.crop_mirror_normalize(
            cast(Any, resized),
            dtype=cast(Any, types).FLOAT,
            output_layout="FCHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )

        return normalized


class DALIMediaClient:
    """
    Saccade DALI 媒體用戶端
    取代原本的 MediaMTXClient，提供 Zero-Copy, GPU 加速的影像預處理。
    """

    def __init__(self, video_path: str, batch_size: int = 1, device_id: int = 0):
        self.video_path = video_path
        self.batch_size = batch_size
        self.device_id = device_id

        self.pipeline: Optional[DALIVideoPipeline] = None
        self.iterator: Optional[DALIGenericIterator] = None
        self._running = False

    def connect(self) -> bool:
        """初始化 DALI Pipeline 與 Iterator"""
        if not os.path.exists(self.video_path):
            print(f"❌ [DALI] Video file not found: {self.video_path}")
            return False

        try:
            print(f"🚀 [DALI] Initializing pipeline for {self.video_path}...")
            self.pipeline = DALIVideoPipeline(
                video_path=self.video_path,
                batch_size=self.batch_size,
                device_id=self.device_id,
            )
            self.pipeline.build()  # type: ignore

            # 使用 PyTorch Iterator 包裝 DALI Pipeline
            self.iterator = DALIGenericIterator(
                [self.pipeline],
                ["data"],
                reader_name="reader",
                auto_reset=True,
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )
            self._running = True
            return True
        except Exception as e:
            print(f"❌ [DALI] Failed to connect: {e}")
            return False

    def grab_tensor(self) -> Tuple[bool, Optional[torch.Tensor]]:
        """
        獲取下一幀的 Tensor (GPU)
        回傳: (success, tensor)
        """
        if not self.iterator:
            return False, None

        try:
            data = next(self.iterator)
            # DALI 回傳的 data 是 [Batch, Seq_Len, C, H, W]
            # 但因為 sequence_length=1，我們拿到的會是 [Batch, 1, C, H, W]
            # 這裡壓縮掉第 2 維度
            tensor = data[0]["data"].squeeze(1)
            return True, tensor
        except StopIteration:
            self.iterator.reset()
            print("🏁 [DALI] End of stream.")
            return False, None
        except Exception as e:
            print(f"⚠️ [DALI] Grab tensor error: {e}")
            return False, None

    def grab(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        傳統模式抓取 (回傳 CPU numpy array, HWC, uint8)
        僅用於相容性測試。
        """
        success, tensor = self.grab_tensor()
        if success and tensor is not None:
            # Tensor 是 float32 [C, H, W] 值域 [0, 1]
            # 轉回 numpy uint8 [H, W, C]
            cpu_array = tensor.cpu().numpy()
            if self.batch_size == 1:
                cpu_array = cpu_array[0]  # 取出 batch 裡的第一個
            img = (cpu_array.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            return True, img
        return False, None

    def release(self) -> None:
        """釋放 DALI 資源"""
        self._running = False
        if self.iterator:
            del self.iterator
            self.iterator = None
        if self.pipeline:
            del self.pipeline
            self.pipeline = None


if __name__ == "__main__":
    # 簡單測試
    client = DALIMediaClient(video_path="assets/videos/demo.mp4")
    if client.connect():
        print("✅ DALI Media Client connected.")
        for _ in range(5):
            ret, tensor = client.grab_tensor()
            if ret and tensor is not None:
                print(
                    f"Got tensor on {tensor.device}, shape: {tensor.shape}, dtype: {tensor.dtype}"
                )
        client.release()
