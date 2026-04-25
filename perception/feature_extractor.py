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
        max_batch: int = 32,
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

        # 從 engine optimization profile 查出真實 max_batch（覆蓋建構參數）
        try:
            _min, _opt, _max = self.engine.get_tensor_profile_shape("pixel_values", 0)
            self.max_batch = int(_max[0])
        except Exception:
            pass  # 保留建構參數預設值

        # 預先分配 GPU 輸出緩衝區（避免碎片化）
        self.gpu_output_buffer = torch.empty(
            (self.max_batch, self.feature_dim),
            device=self.device,
            dtype=torch.float32,
        )
        # SigLIP 2 last_hidden_state（供 ViT attention map 等進階用途）
        self.gpu_hidden_buffer = torch.empty(
            (self.max_batch, 196, self.feature_dim),
            device=self.device,
            dtype=torch.float32,
        )
        # Pinned CPU buffer（DMA D2H 優化）
        self.pinned_buffer = torch.empty(
            (self.max_batch, self.feature_dim),
            device="cpu",
            pin_memory=True,
            dtype=torch.float32,
        )

        print(f"✅ Extractor Ready. Feature Dimension: {self.feature_dim}, Max Batch: {self.max_batch}")

        # 💡 偵測模型是否支援動態 Batch
        self.input_shape = self.engine.get_tensor_shape("pixel_values")
        self.is_dynamic = self.input_shape[0] == -1

        # 🛠️ 自動化 IO 管理
        self.output_buffers = {}
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                shape = self.engine.get_tensor_shape(name)
                # 預先分配空間 (處理動態維度)
                alloc_shape = [self.max_batch if s == -1 else s for s in shape]
                self.output_buffers[name] = torch.empty(
                    tuple(alloc_shape), device=self.device, dtype=torch.float32
                )

        print(
            f"✅ Extractor Ready. Dynamic: {self.is_dynamic}, Outputs: {self.output_names}"
        )

    def extract(
        self, input_tensor: torch.Tensor, stream: torch.cuda.Stream | None = None
    ) -> torch.Tensor:
        """
        執行零拷貝推論 (GPU -> GPU)。
        自動對超過 engine profile 上限的 batch 分塊處理。
        """
        batch_size = input_tensor.size(0)
        if batch_size == 0:
            return torch.empty((0, self.feature_dim), device=self.device)

        # 超過 engine profile 上限時分塊推理
        if batch_size > self.max_batch:
            chunks = input_tensor.split(self.max_batch, dim=0)
            return torch.cat([self._extract_chunk(c) for c in chunks], dim=0)

        return self._extract_chunk(input_tensor)

    def _extract_chunk(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """推理單一合法大小的 chunk（≤ max_batch）。"""
        batch_size = input_tensor.size(0)
        input_tensor = input_tensor.contiguous()
        self.context.set_input_shape("pixel_values", (batch_size, 3, 224, 224))

        # 1. 綁定輸入
        input_tensor = input_tensor.contiguous()
        if self.is_dynamic:
            self.context.set_input_shape("pixel_values", (batch_size, 3, 224, 224))

        self.context.set_tensor_address("pixel_values", input_tensor.data_ptr())
        self.context.set_tensor_address("image_embeds", output_tensor.data_ptr())
        self.context.set_tensor_address("last_hidden_state", hidden_tensor.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)

        return output_tensor.clone()

    def extract_to_cpu(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        執行推理並將結果搬移至 CPU Buffer (D2H 優化)
        """
        batch_size = input_tensor.size(0)
        gpu_features = self.extract(input_tensor)

        # 建立 CPU tensor
        cpu_features = torch.empty(
            (batch_size, self.feature_dim), dtype=torch.float32, pin_memory=True
        )
        # 使用 non_blocking=True 進行 DMA 搬運
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
