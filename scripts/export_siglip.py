import os
import torch
import timm
from pathlib import Path

def export_siglip_onnx(model_name="vit_so400m_patch14_siglip_224", output_dir="models/embedding", img_size=224):
    print(f"🚀 Starting ONNX export for {model_name}...")
    
    # 建立輸出目錄
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    if os.path.exists(onnx_path):
        print(f"✅ ONNX model already exists at {onnx_path}")
        return onnx_path
        
    print("⏳ Downloading and loading PyTorch model (this may take a minute)...")
    try:
        # num_classes=0 表示我們只要特徵向量 (Features)，不要最後的分類層
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    except Exception as e:
        print(f"⚠️ Failed to load {model_name}. Trying base model as fallback.")
        model_name = "vit_base_patch16_siglip_224"
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    model.eval()
    # 移至 CPU 進行導出較不易發生 OOM (如果是 400M 參數)
    model.to("cpu")

    # 建立虛擬輸入 (Batch Size 設為 1，但稍後會指定為動態)
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    print("⚙️ Exporting to ONNX with Dynamic Batch Size...")
    # 設定動態 Batch Size，讓未來的 TensorRT Engine 能一次處理 1~N 個物件
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"🎉 Successfully exported to {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    # 使用 SO400M 作為首選
    onnx_file = export_siglip_onnx("vit_so400m_patch14_siglip_224")
    
    print("\n💡 [Next Step] Compile to TensorRT FP16 Engine using trtexec:")
    print(f"trtexec --onnx={onnx_file} --saveEngine={onnx_file.replace('.onnx', '.engine')} "
          "--fp16 --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:32x3x224x224")
