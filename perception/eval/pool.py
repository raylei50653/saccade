import torch


class AdaptiveFramePool:
    def __init__(self, h, w, device='cuda'):
        print(f"🕯️ Allocating VRAM Buffers for adaptive 960 tiled eval ({w}x{h})...")
        self.frame_buffer = torch.zeros((3, h, w), device=device, dtype=torch.float32)
        self.canvas_640p = torch.zeros((3, 640, 640), device=device, dtype=torch.float32)
        self.canvas_960p = torch.zeros((3, 960, 960), device=device, dtype=torch.float32)
        self.tiles_batch4 = torch.zeros((4, 3, 640, 640), device=device, dtype=torch.float32)
        # Pre-allocated tile x/y offsets — avoids per-frame GPU tensor creation.
        self.tile_dx = torch.tensor([0.0, 320.0, 0.0, 320.0], device=device, dtype=torch.float32).view(4, 1, 1)
        self.tile_dy = torch.tensor([0.0, 0.0, 320.0, 320.0], device=device, dtype=torch.float32).view(4, 1, 1)
