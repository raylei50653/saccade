import os
import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path

from typing import Any, Optional, cast

class JinaVisionWithProjection(nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 取得 pooled_output (通常是第二個回傳值)
        vision_outputs = self.vision_model(pixel_values)
        pooled_output = vision_outputs[1] 
        # 投影至 joint embedding space
        image_embeds = self.visual_projection(pooled_output)
        return cast(torch.Tensor, image_embeds)

def export_jina_clip_onnx(model_name: str = "jinaai/jina-clip-v2", output_dir: str = "models/embedding", img_size: int = 512) -> Optional[str]:
    print(f"🚀 Starting ONNX export for {model_name}...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")
    
    if os.path.exists(onnx_path):
        print(f"✅ ONNX model already exists at {onnx_path}")
        return onnx_path
        
    print("⏳ Downloading and loading PyTorch model (Jina-CLIP-v2)...")
    # 需要 trust_remote_code=True
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    # 封裝以包含 Projection Layer
    vision_wrapper = JinaVisionWithProjection(model)
    vision_wrapper.eval()
    vision_wrapper.to("cpu")

    # 建立虛擬輸入 (Batch Size 設為 2，避免 ONNX 出現 1不支援的 AssertionError)
    dummy_input = torch.randn(2, 3, img_size, img_size, dtype=torch.float32)

    print("⚙️ Exporting to ONNX with Dynamic Batch Size...")
    dynamic_axes = {
        'pixel_values': {0: 'batch_size'},
        'image_embeds': {0: 'batch_size'}
    }

    try:
        torch.onnx.export(
            vision_wrapper,
            (dummy_input,),
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['image_embeds'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"🎉 Successfully exported to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    onnx_file = export_jina_clip_onnx("jinaai/jina-clip-v2", img_size=512)
    
    if onnx_file:
        print("\n💡 [Next Step] Compile to TensorRT FP16 Engine using trtexec:")
        print(f"trtexec --onnx={onnx_file} --saveEngine={onnx_file.replace('.onnx', '.engine')} "
              "--fp16 --minShapes=pixel_values:1x3x512x512 --optShapes=pixel_values:8x3x512x512 --maxShapes=pixel_values:32x3x512x512")