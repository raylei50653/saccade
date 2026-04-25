import tensorrt as trt
import os
import torch
import numpy as np
import cv2

class SaccadeInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    Saccade 通用 INT8 校準器 (Pure Torch Edition)
    使用 torch.cuda 取代 pycuda 以減少環境依賴。
    """
    def __init__(self, image_dir, cache_file, batch_size=8, input_shape=(640, 640)):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')][:200]
        self.count = len(self.images)
        self.current_index = 0
        
        # 🚀 使用 torch 分配 CUDA 顯存
        self.device_input = torch.empty(batch_size, 3, *input_shape, device="cuda", dtype=torch.float32).data_ptr()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > self.count:
            return None

        batch_images = []
        for i in range(self.batch_size):
            img_path = self.images[self.current_index + i]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_images.append(img)

        self.current_index += self.batch_size
        batch_data = torch.from_numpy(np.ascontiguousarray(np.stack(batch_images))).to("cuda")
        
        # 🚀 複製數據到預分配的地址
        # 由於我們直接拿 data_ptr，需要確保生命週期
        self.current_batch = batch_data 
        return [int(self.current_batch.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
