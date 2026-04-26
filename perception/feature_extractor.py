import tensorrt as trt
import torch
import torch.nn.functional as F
import time

# ImageNet normalization constants (DINOv2 / TransReID)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_DEFAULT_ENGINE: dict[str, str] = {
    "siglip2":   "models/embedding/google_siglip2-base-patch16-224.engine",
    "dinov2":    "models/embedding/facebook_dinov2-base.engine",
    "transreid": "models/embedding/transreid_256x128.engine",
}


class TRTFeatureExtractor:
    """
    Saccade TensorRT 特徵提取器

    Supports siglip2 (224×224), dinov2 (224×224), and transreid (256×128).
    Input tensors must be float32 in [0, 1] range; normalisation is applied
    inside the extractor according to model_type.
    """

    def __init__(
        self,
        engine_path: str = "",
        device: str = "cuda:0",
        max_batch: int = 32,
        model_type: str = "siglip2",
    ) -> None:
        if model_type not in _DEFAULT_ENGINE:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(_DEFAULT_ENGINE)}")
        self.model_type = model_type
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.max_batch = max_batch

        path = engine_path or _DEFAULT_ENGINE[model_type]
        print(f"Loading TensorRT Engine [{model_type}] from {path}...")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT Engine.")

        self.context = self.engine.create_execution_context()

        # Read actual max_batch and input spatial size from engine profile.
        try:
            _min, _opt, _max = self.engine.get_tensor_profile_shape("pixel_values", 0)
            self.max_batch = int(_max[0])
            self.input_hw: tuple[int, int] = (int(_opt[2]), int(_opt[3]))
        except Exception:
            self.input_hw = (224, 224)

        # Detect dynamic batch.
        raw_shape = self.engine.get_tensor_shape("pixel_values")
        self.is_dynamic = raw_shape[0] == -1

        # Auto-allocate output buffers from engine metadata.
        self.output_buffers: dict[str, torch.Tensor] = {}
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                shape = self.engine.get_tensor_shape(name)
                alloc_shape = [self.max_batch if s == -1 else s for s in shape]
                self.output_buffers[name] = torch.empty(
                    tuple(alloc_shape), device=self.device, dtype=torch.float32
                )

        # Resolve the embedding output key and infer feature_dim.
        self._embed_key = "image_embeds" if "image_embeds" in self.output_buffers else self.output_names[0]
        self.feature_dim: int = int(self.output_buffers[self._embed_key].shape[-1])

        # Pre-cache ImageNet normalization tensors on the target device.
        if model_type in {"dinov2", "transreid"}:
            self._imagenet_mean = torch.tensor(
                _IMAGENET_MEAN, device=device, dtype=torch.float32
            ).view(1, 3, 1, 1)
            self._imagenet_std = torch.tensor(
                _IMAGENET_STD, device=device, dtype=torch.float32
            ).view(1, 3, 1, 1)
        else:
            self._imagenet_mean = None
            self._imagenet_std = None

        print(
            f"Extractor ready — model={model_type}, dim={self.feature_dim}, "
            f"input={self.input_hw}, max_batch={self.max_batch}, dynamic={self.is_dynamic}, "
            f"outputs={self.output_names}"
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

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply model-specific input normalisation (in-place friendly, returns contiguous)."""
        x = x.to(torch.float32)
        if self.model_type == "siglip2":
            # SigLIP vision_model expects pixel_values in [-1, 1].
            return x.mul(2.0).sub(1.0).contiguous()
        # DINOv2 and TransReID use standard ImageNet mean/std.
        return ((x - self._imagenet_mean) / self._imagenet_std).contiguous()  # type: ignore[operator]

    def _extract_chunk(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on a single chunk (≤ max_batch)."""
        batch_size = input_tensor.size(0)
        input_tensor = self._normalize(input_tensor)

        if self.is_dynamic:
            self.context.set_input_shape("pixel_values", (batch_size, 3, *self.input_hw))

        self.context.set_tensor_address("pixel_values", input_tensor.data_ptr())
        for name, buf in self.output_buffers.items():
            self.context.set_tensor_address(name, buf.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)

        return F.normalize(self.output_buffers[self._embed_key][:batch_size].float(), dim=-1).clone()

    def extract_to_cpu(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference and copy result to pinned CPU memory (DMA D2H)."""
        gpu_features = self.extract(input_tensor)
        cpu_features = torch.empty(
            (input_tensor.size(0), self.feature_dim), dtype=torch.float32, pin_memory=True
        )
        cpu_features.copy_(gpu_features, non_blocking=True)
        return cpu_features


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="siglip2", choices=list(_DEFAULT_ENGINE))
    p.add_argument("--engine-path", default="")
    a = p.parse_args()

    extractor = TRTFeatureExtractor(engine_path=a.engine_path, model_type=a.model_type)
    h, w = extractor.input_hw
    dummy_input = torch.randn(8, 3, h, w, device="cuda", dtype=torch.float32)

    _ = extractor.extract(dummy_input)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        out = extractor.extract(dummy_input)
    torch.cuda.synchronize()

    latency = (time.perf_counter() - start) / 100 * 1000
    print(f"Shape: {out.shape}  |  Avg latency (batch=8): {latency:.2f} ms")
