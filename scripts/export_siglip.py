import os
import torch
from transformers import AutoModel
from pathlib import Path

def export_siglip2_onnx(model_name="google/siglip2-so400m-patch14-384", output_dir="models/embedding", img_size=384):
    print(f"🚀 Starting ONNX export for {model_name}...")

    # 建立輸出目錄
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

    if os.path.exists(onnx_path):
        print(f"✅ ONNX model already exists at {onnx_path}")
        return onnx_path

    print("⏳ Downloading and loading PyTorch model (SigLIP 2)...")
    model = AutoModel.from_pretrained(model_name)

    # SigLIP 2 在 transformers 中包含 vision_model 與 text_model
    # 我們只需要視覺編碼器
    vision_model = model.vision_model
    vision_model.eval()
    vision_model.to("cpu")

    # 建立虛擬輸入 (Batch Size 設為 1)
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    print("⚙️ Exporting to ONNX with Dynamic Batch Size...")
    dynamic_axes = {
        'pixel_values': {0: 'batch_size'},
        'last_hidden_state': {0: 'batch_size'},
        'pooler_output': {0: 'batch_size'}
    }

    torch.onnx.export(
        vision_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    print(f"🎉 Successfully exported to {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    # 使用 SigLIP 2 SO400M
    onnx_file = export_siglip2_onnx("google/siglip2-so400m-patch14-384", img_size=384)

    print("\n💡 [Next Step] Compile to TensorRT FP16 Engine using trtexec:")
    print(f"trtexec --onnx={onnx_file} --saveEngine={onnx_file.replace('.onnx', '.engine')} "
          "--fp16 --minShapes=pixel_values:1x3x384x384 --optShapes=pixel_values:8x3x384x384 --maxShapes=pixel_values:32x3x384x384")
