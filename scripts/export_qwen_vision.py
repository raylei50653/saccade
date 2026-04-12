import os
import torch
from transformers import Qwen2VLForConditionalGeneration
from pathlib import Path

def export_qwen_vision(model_name="Qwen/Qwen2-VL-2B-Instruct", output_dir="models/embedding", img_size=224):
    print(f"🚀 Initializing Qwen-VL Vision Encoder export for {model_name}...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    onnx_path = os.path.join(output_dir, "qwen_vl_vision_2b.onnx")
    
    print("⏳ Downloading and loading the Qwen2-VL model (this will take a while)...")
    # 只載入半精度以節省記憶體，並禁用 Flash Attention 以便於 ONNX 轉換 (FlashAttn 通常不支援直接匯出)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="cpu", 
        attn_implementation="eager"
    )
    
    # 單獨擷取視覺編碼器 (Vision Tower)
    vision_model = model.model.visual
    vision_model.eval()
    
    print("⚙️ Preparing dummy inputs for the Vision Tower...")
    
    # Qwen-VL 視覺編碼器接收的輸入格式相當特殊，它期望的是被處理過的 patch_features 與網格結構
    # 假設我們將圖片 resize 到 224x224, Patch Size 通常是 14，那麼我們會有 16x16 = 256 個 Patches
    # 但在 Qwen 中，它還考慮到時間維度 (Video) 和空間網格 (grid_thw)
    
    # [T, H, W] - 假設是單張圖片 T=1, H=16, W=16
    t, h, w = 1, img_size // 14, img_size // 14
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long)
    
    # 視覺特徵的維度：通常輸入的 pixel_values 已經展平成 [T * H * W, Channel * Patch * Patch]
    # 在 Qwen2-VL 中，視覺特徵維度 (in_channels) 通常為 1176 或 3*14*14
    num_patches = t * h * w
    in_channels = vision_model.patch_embed.proj.in_channels
    dummy_pixel_values = torch.randn(num_patches, in_channels, dtype=torch.float16)

    print(f"  - Dummy pixel_values shape: {dummy_pixel_values.shape}")
    print(f"  - Dummy grid_thw shape: {grid_thw.shape}")
    
    print("⚙️ Exporting to ONNX...")
    
    # 設定動態維度 (可接受不同數量的 Patches)
    dynamic_axes = {
        'pixel_values': {0: 'num_patches'},
        'output': {0: 'num_patches'}
    }

    try:
        torch.onnx.export(
            vision_model,
            (dummy_pixel_values, grid_thw),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['pixel_values', 'grid_thw'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"🎉 Successfully exported Qwen Vision Encoder to {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX Export Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 注意：這裡使用 Qwen2-VL-2B 作為範例，因為目前官方還沒發布獨立的 3-VL-Embedding-2B
    export_qwen_vision("Qwen/Qwen2-VL-2B-Instruct")
