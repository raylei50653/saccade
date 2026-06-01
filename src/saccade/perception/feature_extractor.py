import torch
import tensorrt as trt
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, cast

# ImageNet normalization constants (DINOv2 / TransReID)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_DEFAULT_ENGINE: Dict[str, str] = {
    "siglip2": "models/embedding/google_siglip2-base-patch16-224.engine",
    "siglip2_reid": "models/embedding/marketaju_siglip2_reid_224.engine",
    "dinov2": "models/embedding/facebook_dinov2-base.engine",
    "transreid": "models/embedding/transreid_256x128.engine",
    "osnet": "models/embedding/osnet_x1_0_256x128.engine",
    "fastreid": "models/embedding/fastreid_256x128.engine",
}


try:
    from saccade_perception_ext import (
        FeatureExtractor as FeatureExtractorCpp,
        ModelType,
    )

    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False
    ModelType = Any


def _last_vit_dual(
    lhs: torch.Tensor,
    sigma_embed: float,
    sigma_gate: float,
    top_k_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python reference LaSt-ViT pipeline (V1 per-patch Top-K). [B,N,C] → ([B,C], [B,N])."""
    x = lhs.float()
    B, N, C = x.shape
    K = max(1, int(N * top_k_ratio))

    def _filt(sigma: float) -> torch.Tensor:
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.arange(C // 2 + 1, device=x.device, dtype=torch.float32) / C
        w = torch.exp(-(freqs**2) / (2.0 * sigma**2))
        return torch.fft.irfft(X * w, n=C, dim=-1)  # type: ignore[no-any-return]

    def _scores(xf: torch.Tensor) -> torch.Tensor:
        d = (x - xf).pow(2).sum(dim=-1)
        n2 = x.pow(2).sum(dim=-1).clamp(min=1e-8)
        return (1.0 - d / n2).clamp(0.0, 1.0)

    s_embed = _scores(_filt(sigma_embed))
    s_gate = _scores(_filt(sigma_gate))
    idx = s_embed.topk(K, dim=-1).indices
    feats = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    embed = F.normalize(feats.mean(dim=1), dim=-1)
    return embed, s_gate


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
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose from {list(_DEFAULT_ENGINE.keys())}"
            )

        self.model_type = model_type
        self.device = device
        self.max_batch = max_batch
        self.feature_dim: int = 0
        self.input_hw: Tuple[int, int] = (224, 224)
        self.is_dynamic: bool = False
        self.output_names: List[str] = []
        self.output_buffers: Dict[str, torch.Tensor] = {}
        self._embed_key: str = ""
        self._imagenet_mean: Optional[torch.Tensor] = None
        self._imagenet_std: Optional[torch.Tensor] = None
        self._cpp: Optional[Any] = None  # 使用 Any 避免類型衝突

        path = engine_path or _DEFAULT_ENGINE[model_type]

        if HAS_CPP_EXT:
            cpp_type_map = {
                "siglip2": ModelType.SIGLIP2,
                "siglip2_reid": ModelType.SIGLIP2,
                "dinov2": ModelType.DINOV2,
                "transreid": ModelType.TRANSREID,
                "osnet": ModelType.OSNET,
                "fastreid": ModelType.FASTREID,
            }
            print(f"Loading C++ TensorRT Engine [{model_type}] from {path}...")
            self._cpp = FeatureExtractorCpp(path, cpp_type_map[model_type], max_batch)
            self.feature_dim = self._cpp.feature_dim
            self.input_hw = self._cpp.input_hw
            return

        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT Engine.")

        self.context = self.engine.create_execution_context()

        # Read actual max_batch and input spatial size from engine profile.
        try:
            _min, _opt, _max = self.engine.get_tensor_profile_shape("pixel_values", 0)
            self.max_batch = int(_max[0])
            self.input_hw = (int(_opt[2]), int(_opt[3]))
        except Exception:
            pass

        # Detect dynamic batch.
        raw_shape = self.engine.get_tensor_shape("pixel_values")
        self.is_dynamic = raw_shape[0] == -1

        # Auto-allocate output buffers from engine metadata.
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
        self._embed_key = (
            "image_embeds"
            if "image_embeds" in self.output_buffers
            else self.output_names[0]
        )
        self.feature_dim = int(self.output_buffers[self._embed_key].shape[-1])

        # Pre-cache ImageNet normalization tensors on the target device.
        if model_type in {"dinov2", "transreid", "osnet", "fastreid"}:
            self._imagenet_mean = torch.tensor(
                _IMAGENET_MEAN, device=device, dtype=torch.float32
            ).view(1, 3, 1, 1)
            self._imagenet_std = torch.tensor(
                _IMAGENET_STD, device=device, dtype=torch.float32
            ).view(1, 3, 1, 1)

        print(
            f"Extractor ready — model={model_type}, dim={self.feature_dim}, "
            f"input={self.input_hw}, max_batch={self.max_batch}, dynamic={self.is_dynamic}, "
            f"outputs={self.output_names}"
        )

    def extract(
        self, input_tensor: torch.Tensor, stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """
        執行零拷貝推論 (GPU -> GPU)。
        自動對超過 engine profile 上限的 batch 分塊處理。
        """
        batch_size = input_tensor.size(0)
        if batch_size == 0:
            return torch.empty((0, self.feature_dim), device=self.device)

        if self._cpp is not None:
            out = torch.empty(
                (batch_size, self.feature_dim), device=self.device, dtype=torch.float32
            )
            stream_ptr = (
                stream.cuda_stream
                if stream
                else torch.cuda.current_stream().cuda_stream
            )
            self._cpp.extract(
                input_tensor.data_ptr(), batch_size, out.data_ptr(), stream_ptr
            )
            return out

        # 超過 engine profile 上限時分塊推理
        if batch_size > self.max_batch:
            chunks = torch.split(input_tensor, self.max_batch, dim=0)
            return torch.cat([self._extract_chunk(c) for c in chunks], dim=0)

        return self._extract_chunk(input_tensor)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply model-specific input normalisation (in-place friendly, returns contiguous)."""
        x = x.to(torch.float32)
        if self.model_type in {"siglip2", "siglip2_reid"}:
            # SigLIP vision_model expects pixel_values in [-1, 1].
            return x.mul(2.0).sub(1.0).contiguous()

        # DINOv2 and TransReID use standard ImageNet mean/std.
        if self._imagenet_mean is not None and self._imagenet_std is not None:
            return ((x - self._imagenet_mean) / self._imagenet_std).contiguous()
        return x.contiguous()

    def _extract_chunk(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on a single chunk (≤ max_batch)."""
        batch_size = input_tensor.size(0)
        input_tensor = self._normalize(input_tensor)

        if self.is_dynamic:
            self.context.set_input_shape(
                "pixel_values", (batch_size, 3, *self.input_hw)
            )

        self.context.set_tensor_address("pixel_values", input_tensor.data_ptr())
        for name, buf in self.output_buffers.items():
            self.context.set_tensor_address(name, buf.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)

        return F.normalize(
            self.output_buffers[self._embed_key][:batch_size].float(), dim=-1
        ).clone()

    @property
    def cpp_ptr(self) -> int:
        """Raw pointer to the C++ FeatureExtractor object (for PerceptionPipeline)."""
        if self._cpp is None:
            raise RuntimeError("C++ extension not available")
        return cast(int, self._cpp.cpp_ptr)

    def extract_parts_fused(
        self, input_tensor: torch.Tensor, stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """
        Extract 3-part crops [3*N, 3, H, W], fuse with weights [0.5, 0.3, 0.2], and L2-normalize.
        Returns [N, feature_dim] embeddings.
        """
        num_dets = input_tensor.size(0) // 3
        if num_dets <= 0:
            return torch.empty((0, self.feature_dim), device=self.device)
        out = torch.empty(
            (num_dets, self.feature_dim), device=self.device, dtype=torch.float32
        )
        if self._cpp is not None:
            stream_ptr = (
                stream.cuda_stream
                if stream
                else torch.cuda.current_stream().cuda_stream
            )
            self._cpp.extract_parts_fused(
                input_tensor.data_ptr(), num_dets, out.data_ptr(), stream_ptr
            )
            return out
        # Python fallback: extract all parts then fuse
        part_embeds = self.extract(input_tensor, stream)  # [3*N, D]
        part_embeds = part_embeds.view(3, num_dets, -1)
        weights = torch.tensor(
            [0.5, 0.3, 0.2], device=part_embeds.device, dtype=part_embeds.dtype
        ).view(3, 1, 1)
        return F.normalize((part_embeds * weights).sum(dim=0), dim=-1)

    def extract_with_stability(
        self,
        input_tensor: torch.Tensor,
        sigma_embed: float = 0.015,
        sigma_gate: float = 0.040,
        top_k_ratio: float = 0.5,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LaSt-ViT refined extraction (SigLIP2 only).

        Returns (embeddings [N, D] L2-normalized, stability_scores [N]) both on GPU.
        Falls back to standard extract() with ones stability if model is not SigLIP2
        or C++ extension is unavailable and last_hidden_state not in output_buffers.
        """
        n = input_tensor.size(0)
        if n == 0:
            empty_e = torch.empty((0, self.feature_dim), device=self.device)
            empty_s = torch.empty((0,), device=self.device)
            return empty_e, empty_s

        if self._cpp is not None:
            embed = torch.empty(
                (n, self.feature_dim), device=self.device, dtype=torch.float32
            )
            stab = torch.empty((n,), device=self.device, dtype=torch.float32)
            stream_ptr = (
                stream.cuda_stream
                if stream
                else torch.cuda.current_stream().cuda_stream
            )
            # Chunked over max_batch (C++ handles internally)
            self._cpp.extract_with_stability(
                input_tensor.data_ptr(),
                n,
                embed.data_ptr(),
                stab.data_ptr(),
                stream_ptr,
                sigma_embed,
                sigma_gate,
                top_k_ratio,
            )
            return embed, stab

        # Python TRT path: use last_hidden_state output buffer
        if "last_hidden_state" not in self.output_buffers:
            embed = self.extract(input_tensor, stream)
            return embed, torch.ones(n, device=self.device, dtype=torch.float32)

        # Process in chunks matching max_batch
        all_embeds, all_stab = [], []
        for start in range(0, n, self.max_batch):
            chunk = input_tensor[start : start + self.max_batch]
            bs = chunk.size(0)
            self._extract_chunk(chunk)
            torch.cuda.synchronize()  # saccade-allow-cpu
            lhs = self.output_buffers["last_hidden_state"][:bs].float().clone()
            embed_chunk, scores_gate = _last_vit_dual(
                lhs, sigma_embed, sigma_gate, top_k_ratio
            )
            all_embeds.append(embed_chunk)
            all_stab.append(scores_gate.mean(dim=-1))  # [bs]
        return torch.cat(all_embeds, dim=0), torch.cat(all_stab, dim=0)

    def extract_to_cpu(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference and copy result to pinned CPU memory (DMA D2H)."""
        gpu_features = self.extract(input_tensor)
        cpu_features = torch.empty(
            (input_tensor.size(0), self.feature_dim),
            dtype=torch.float32,
            pin_memory=True,
        )
        cpu_features.copy_(gpu_features, non_blocking=True)
        return cpu_features
