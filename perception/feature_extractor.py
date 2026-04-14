import tensorrt as trt
import torch
import time


class TRTFeatureExtractor:
    """
    Saccade TensorRT 特徵提取器 (Phase 2)

    直接讀取 GPU 上的 torch.Tensor 記憶體指標 (Data Pointer) 並餵給 TensorRT Engine，
    產出高維度的語義向量，達成微秒級的無縫交接。
    """

    def __init__(
        self,
        engine_path: str = "models/embedding/google_siglip2-base-patch16-224.engine",
        device: str = "cuda:0",
        max_batch: int = 32
    ) -> None:
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.max_batch = max_batch

        print(f"⏳ Loading TensorRT Engine from {engine_path}...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT Engine.")

        self.context = self.engine.create_execution_context()
        # SigLIP 2 Base (ViT-B) 的特徵維度為 768
        self.feature_dim = 768
        
        # 🛠️ 預先分配 Pinned Buffer Pool (鎖頁記憶體)
        # 這能避免在高頻推理中頻繁配置記憶體導致的 WSL2 抖動
        self.pinned_buffer = torch.empty(
            (self.max_batch, self.feature_dim), 
            device="cpu", 
            pin_memory=True, 
            dtype=torch.float32
        )
        # GPU 端的輸出緩衝區也預先分配 (避免碎片化)
        self.gpu_output_buffer = torch.empty(
            (self.max_batch, self.feature_dim), 
            device=self.device, 
            dtype=torch.float32
        )
        # SigLIP 2 Base 的 last_hidden_state (可選，若不使用可移除以省顯存)
        self.gpu_hidden_buffer = torch.empty(
            (self.max_batch, 196, self.feature_dim),
            device=self.device,
            dtype=torch.float32
        )
        
        print(f"✅ Extractor Ready. Feature Dimension: {self.feature_dim}, Max Batch: {self.max_batch}")

    def extract(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        執行零拷貝推論 (GPU -> GPU)
        """
        batch_size = input_tensor.size(0)
        if batch_size == 0:
            return torch.empty((0, self.feature_dim), device=self.device)

        input_tensor = input_tensor.contiguous()
        self.context.set_input_shape("pixel_values", (batch_size, 3, 224, 224))

        # 複用預分配的 GPU Buffer
        output_tensor = self.gpu_output_buffer[:batch_size]
        hidden_tensor = self.gpu_hidden_buffer[:batch_size]

        self.context.set_tensor_address("pixel_values", input_tensor.data_ptr())
        self.context.set_tensor_address("image_embeds", output_tensor.data_ptr())
        self.context.set_tensor_address("last_hidden_state", hidden_tensor.data_ptr())
        
        # SigLIP 2 推理 (異步)
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)

        return output_tensor

    def extract_to_cpu(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        執行推理並將結果搬移至 Pinned CPU Buffer (D2H 優化)
        """
        batch_size = input_tensor.size(0)
        gpu_features = self.extract(input_tensor)
        
        # 使用 non_blocking=True 進行 DMA 搬運
        cpu_features = self.pinned_buffer[:batch_size]
        cpu_features.copy_(gpu_features, non_blocking=True)
        
        return cpu_features


if __name__ == "__main__":
    # 端到端 效能測試 (Dry Run)
    print("🚀 Testing TRTFeatureExtractor (SigLIP 2)...")
    extractor = TRTFeatureExtractor()

    # 模擬 8 個物件同時進行特徵提取 (224x224)
    dummy_input = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.float32)

    # 預熱
    _ = extractor.extract(dummy_input)
    torch.cuda.synchronize()

    # 測速
    start = time.perf_counter()
    for i in range(100):
        out_features = extractor.extract(dummy_input)
    torch.cuda.synchronize()

    latency = (time.perf_counter() - start) / 100 * 1000

    print(f"✅ Extracted Features Shape: {out_features.shape}")
    print(f"⚡ Average Latency for 8 objects (FP16 TRT): {latency:.2f} ms")
