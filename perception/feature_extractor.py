import tensorrt as trt
import torch
import time

class TRTFeatureExtractor:
    """
    Saccade TensorRT 特徵提取器 (Phase 2)
    
    直接讀取 GPU 上的 torch.Tensor 記憶體指標 (Data Pointer) 並餵給 TensorRT Engine，
    產出高維度的語義向量，達成微秒級的無縫交接。
    """
    def __init__(self, engine_path: str = "models/embedding/vit_so400m_patch14_siglip_224.engine", device: str = "cuda:0"):
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        
        print(f"⏳ Loading TensorRT Engine from {engine_path}...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT Engine.")
            
        self.context = self.engine.create_execution_context()
        # 取得輸出維度大小 (通常 SigLIP SO400M 的特徵維度是 1152)
        out_shape = self.engine.get_tensor_shape("output")
        self.feature_dim = out_shape[1]
        print(f"✅ Extractor Ready. Feature Dimension: {self.feature_dim}")
        
    def extract(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        執行零拷貝推論
        
        :param input_tensor: 由 ZeroCopyCropper 產出的連續 Tensor [N, C, H, W]，型別必須是 Float32
        :return: 提取後的語義向量 Tensor [N, Feature_Dim]
        """
        batch_size = input_tensor.size(0)
        if batch_size == 0:
            return torch.empty((0, self.feature_dim), device=self.device)
            
        # 1. 確保張量在連續的記憶體區塊中，否則 TensorRT 無法正確讀取指標
        input_tensor = input_tensor.contiguous()
        
        # 2. 動態設定本次推理的 Batch Size
        self.context.set_input_shape("input", (batch_size, 3, 224, 224))
        
        # 3. 預先分配輸出空間 (全在 GPU 上)
        output_tensor = torch.empty((batch_size, self.feature_dim), device=self.device, dtype=torch.float32)
        
        # 4. 記憶體綁定：直接告訴 TensorRT 從這兩個 PyTorch 指標讀寫數據
        self.context.set_tensor_address("input", input_tensor.data_ptr())
        self.context.set_tensor_address("output", output_tensor.data_ptr())
        
        # 5. 異步觸發推理 (利用當前 PyTorch 的 CUDA Stream)
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        
        return output_tensor

if __name__ == "__main__":
    # 端到端 效能測試 (Dry Run)
    print("🚀 Testing TRTFeatureExtractor...")
    extractor = TRTFeatureExtractor()
    
    # 模擬 8 個物件同時進行特徵提取
    dummy_input = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.float32)
    
    # 預熱
    _ = extractor.extract(dummy_input)
    torch.cuda.synchronize()
    
    # 測速
    start = time.perf_counter()
    for _ in range(100):
        out_features = extractor.extract(dummy_input)
    torch.cuda.synchronize()
    
    latency = (time.perf_counter() - start) / 100 * 1000
    
    print(f"✅ Extracted Features Shape: {out_features.shape}")
    print(f"⚡ Average Latency for 8 objects (FP16 TRT): {latency:.2f} ms")
