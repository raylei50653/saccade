import asyncio
import torch
import time
from perception.feature_extractor import TRTFeatureExtractor
from perception.embedding_dispatcher import AsyncEmbeddingDispatcher
from perception.cropper import ZeroCopyCropper
from perception.tracking.tracker import SmartTracker
from perception.feature_bank import FeatureBank

import pytest

@pytest.mark.anyio
async def test_parallel_pipeline():
    print("🧪 Starting Saccade Parallel Re-ID Pipeline Test (CPU Logic Mode)...")
    
    # 強制使用 CPU 進行邏輯驗證，避開硬體不相容問題
    device = "cpu"

    # 1. 初始化組件 (Mock Extractor)
    class MockExtractor:
        def __init__(self):
            self.device = "cpu"
            self.feature_dim = 768
        def extract(self, input_tensor):
            # 模擬推論延遲 (5ms)
            time.sleep(0.005) 
            return torch.randn((input_tensor.size(0), 768), device=self.device)
            
    extractor = MockExtractor()
    dispatcher = AsyncEmbeddingDispatcher(extractor)
    dispatcher.start()
    
    cropper = ZeroCopyCropper(output_size=(224, 224))
    bank = FeatureBank(max_ids=100, device=device)
    tracker = SmartTracker(feature_bank=bank, dispatcher=dispatcher)

    # 2. 模擬偵測序列
    dummy_frame = torch.rand((1, 3, 480, 640), device=device, dtype=torch.float32)
    dummy_boxes = torch.tensor([
        [10, 10, 50, 50], 
        [100, 100, 150, 150],
    ], device=device, dtype=torch.float32)
    
    dummy_ids = torch.tensor([1, 2], device=device, dtype=torch.int32)

    print("🚀 Simulating 10 frames of tracking...")
    start_time = time.perf_counter()

    dummy_scores = torch.tensor([0.9, 0.8], device=device, dtype=torch.float32)
    dummy_classes = torch.tensor([0, 0], device=device, dtype=torch.int32)

    for frame_id in range(10):
        # 執行並行追蹤
        tracked_ids, tracked_boxes, tracked_classes = await tracker.update(
            timestamp=frame_id * 33,
            boxes=dummy_boxes,
            scores=dummy_scores,
            classes=dummy_classes,
            frame_tensor=dummy_frame,
            cropper=cropper,
            stream_id=1
        )
        
        # 讓 asyncio 有機會執行背景 Worker
        await asyncio.sleep(0.001)

    print("⏳ Waiting for background Re-ID tasks to settle...")
    await asyncio.sleep(0.2) 

    # 3. 驗證結果
    print("\n📊 Verification:")
    success_count = 0
    for tid in [1, 2]:
        idx = (bank.id_map == tid).nonzero()
        if idx.numel() > 0:
            print(f"✅ ID {tid}: Feature successfully extracted and stored in Bank.")
            success_count += 1
        else:
            print(f"❌ ID {tid}: Feature NOT found in Bank.")

    if success_count == 2:
        print("\n🎉 PASS: Parallel Re-ID logic is functional.")
    else:
        print("\nFAIL: Re-ID features missing.")

    dispatcher.stop()

if __name__ == "__main__":
    asyncio.run(test_parallel_pipeline())
