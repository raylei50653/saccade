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
        執行零拷貝推論 (GPU -> GPU)
        """
        batch_size = input_tensor.size(0)
        if batch_size == 0:
            return torch.empty((0, self.feature_dim), device=self.device)

        current_stream = stream if stream is not None else torch.cuda.current_stream()

        # 1. 綁定輸入
        input_tensor = input_tensor.contiguous()
        if self.is_dynamic:
            self.context.set_input_shape("pixel_values", (batch_size, 3, 224, 224))

        self.context.set_tensor_address("pixel_values", input_tensor.data_ptr())

        # 2. 綁定所有輸出 (TensorRT 要求所有輸出都必須有位址)
        for name in self.output_names:
            buffer = self.output_buffers[name]
            # 如果是動態 Batch 且緩衝區不夠大，則重新分配
            if self.is_dynamic and buffer.size(0) < batch_size:
                shape = list(self.engine.get_tensor_shape(name))
                shape[0] = batch_size
                buffer = torch.empty(
                    tuple(shape), device=self.device, dtype=torch.float32
                )
                self.output_buffers[name] = buffer

            self.context.set_tensor_address(name, buffer.data_ptr())

        # 3. 執行推論
        if self.is_dynamic:
            self.context.execute_async_v3(current_stream.cuda_stream)
            # 回傳主要的 image_embeds
            return self.output_buffers["image_embeds"][:batch_size]
        else:
            # 🐢 靜態模式：循序處理 (Fallback)
            results = []
            for i in range(batch_size):
                single_input = input_tensor[i].unsqueeze(0).contiguous()
                self.context.set_tensor_address("pixel_values", single_input.data_ptr())
                # 重新綁定所有輸出地址到當前單個 batch
                for name in self.output_names:
                    self.context.set_tensor_address(
                        name, self.output_buffers[name].data_ptr()
                    )

                self.context.execute_async_v3(current_stream.cuda_stream)
                results.append(self.output_buffers["image_embeds"].clone())
            return torch.cat(results, dim=0)

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
