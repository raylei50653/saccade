import torch
from typing import List
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class RTSPDALIPipeline(Pipeline):
    def __init__(
        self,
        batch_size=1,
        num_threads=4,
        device_id=0,
        output_size=640,
        prefetch_queue_depth=2,
    ):
        # 增加 prefetch_queue_depth 提高直播流容錯性
        super().__init__(
            batch_size,
            num_threads,
            device_id,
            prefetch_queue_depth=prefetch_queue_depth,
        )
        self.output_size = output_size
        self.input = fn.external_source(device="gpu", name="rtsp_raw", no_copy=True)

    def define_graph(self):
        images = self.input

        # 🚀 影像增強：提升亮度與對比度 (針對夜間街景)
        enhanced = fn.brightness_contrast(images, brightness=1.2, contrast=1.1)

        resized = fn.resize(
            enhanced,
            resize_x=self.output_size,
            resize_y=self.output_size,
            interp_type=types.INTERP_LINEAR,
        )
        normalized = fn.crop_mirror_normalize(
            resized,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return normalized


class DALIRTSPOptimizer:
    def __init__(self, batch_size: int = 1, device_id: int = 0, output_size: int = 640):
        self.batch_size = batch_size
        self.pipeline = RTSPDALIPipeline(
            batch_size=batch_size, device_id=device_id, output_size=output_size
        )
        self.pipeline.build()
        self._primed = False

    def process(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        # 1. 確保 Contiguous
        tensors = [t.contiguous() for t in tensors]

        # 2. 補齊 Batch
        while len(tensors) < self.batch_size:
            tensors.append(tensors[-1])
        tensors = tensors[: self.batch_size]

        # 3. 🚀 預熱邏輯 (Priming)
        # 如果是第一次執行，需要餵入足夠填滿 prefetch 隊列的數據
        if not self._primed:
            # 預設 prefetch_queue_depth=2，所以我們要先餵入 3 次數據 (2 預取 + 1 當前)
            for _ in range(2):
                self.pipeline.feed_input("rtsp_raw", tensors)
            self._primed = True

        # 4. 餵入當前數據
        self.pipeline.feed_input("rtsp_raw", tensors)

        # 5. 執行並取得輸出
        outputs = self.pipeline.run()
        dali_out = outputs[0].as_tensor()

        # 6. 零拷貝介面轉換
        class DALIInterface:
            def __init__(self, dali_tensor):
                self.__cuda_array_interface__ = {
                    "shape": dali_tensor.shape(),
                    "typestr": "<f4",
                    "data": (dali_tensor.data_ptr(), False),
                    "version": 3,
                }

        # ⚠️ 必須 clone()，否則下一輪 DALI 推論會覆蓋當前顯存區域
        return torch.as_tensor(DALIInterface(dali_out), device="cuda").clone()
