import asyncio
import torch
from perception.cropper import ZeroCopyCropper
from perception.tracking.tracker import SmartTracker
from perception.feature_bank import FeatureBank

import pytest


@pytest.mark.anyio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
async def test_parallel_pipeline():
    print("🧪 Starting Saccade Parallel Re-ID Pipeline Test (GPU Mode)...")

    device = "cuda"

    # 1. 初始化組件 (Mock Extractor)
    class MockExtractor:
        def __init__(self):
            self.device = "cuda"
            self.feature_dim = 768

        def extract(self, input_tensor):
            # 模擬推論
            return torch.randn((input_tensor.size(0), 768), device=self.device)

    extractor = MockExtractor()
    # dispatcher = AsyncEmbeddingDispatcher(extractor) # SmartTracker has its own stream

    cropper = ZeroCopyCropper(output_size=(224, 224))
    bank = FeatureBank(max_ids=100, device=device)
    tracker = SmartTracker(feature_bank=bank, extractor=extractor, cropper=cropper)

    # 2. 模擬偵測序列
    dummy_frame = torch.rand((3, 480, 640), device=device, dtype=torch.float32)
    dummy_boxes = torch.tensor(
        [
            [10, 10, 50, 50],
            [100, 100, 150, 150],
        ],
        device=device,
        dtype=torch.float32,
    )

    print("🚀 Simulating 15 frames of tracking...")

    dummy_scores = torch.tensor([0.9, 0.8], device=device, dtype=torch.float32)
    dummy_classes = torch.tensor([0, 0], device=device, dtype=torch.int32)

    for frame_id in range(15):
        # 執行追蹤 (SmartTracker.update 是同步的)
        tracked_ids, tracked_boxes, tracked_classes = tracker.update(
            boxes=dummy_boxes,
            scores=dummy_scores,
            classes=dummy_classes,
            frame_tensor=dummy_frame,
            stream_id=1,
        )
        # 讓 asyncio 有機會執行
        await asyncio.sleep(0.001)

    # 3. 驗證結果
    print("\n📊 Verification:")
    # 由於有 confirm_streak=3，在 15 幀後應該已經有軌跡
    assert tracked_ids.numel() > 0
    print(f"✅ Tracked {tracked_ids.numel()} objects.")


if __name__ == "__main__":
    asyncio.run(test_parallel_pipeline())
