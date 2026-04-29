import torch
from typing import Union


class AdaptiveFramePool:
    def __init__(
        self, h: int, w: int, device: Union[str, torch.device] = "cuda"
    ) -> None:
        print(f"🕯️ Allocating VRAM Buffers for adaptive 960 tiled eval ({w}x{h})...")
        self.frame_buffer = torch.zeros((3, h, w), device=device, dtype=torch.float32)
        self.canvas_640p = torch.zeros(
            (3, 640, 640), device=device, dtype=torch.float32
        )
        self.canvas_960p = torch.zeros(
            (3, 960, 960), device=device, dtype=torch.float32
        )
        self.tiles_batch4 = torch.zeros(
            (4, 3, 640, 640), device=device, dtype=torch.float32
        )
        self.tiles_batch6 = torch.zeros(
            (6, 3, 640, 640), device=device, dtype=torch.float32
        )

        # Pre-allocated tile x/y offsets — avoids per-frame GPU tensor creation.
        self.tile_dx = torch.tensor(
            [0.0, 320.0, 0.0, 320.0], device=device, dtype=torch.float32
        ).view(4, 1, 1)
        self.tile_dy = torch.tensor(
            [0.0, 0.0, 320.0, 320.0], device=device, dtype=torch.float32
        ).view(4, 1, 1)

        # 3×2 tiling on 960p canvas (same scale as 2×2, adds a middle column).
        # x: 3 cols at stride=160 on 960p → [0:640],[160:800],[320:960] (75% x-overlap)
        # y: 2 rows at stride=320 on 960p → [0:640],[320:960]  (50% y-overlap, same as 2×2)
        self.tile_3x2_dx = torch.tensor(
            [0.0, 160.0, 320.0, 0.0, 160.0, 320.0], device=device, dtype=torch.float32
        ).view(6, 1, 1)
        self.tile_3x2_dy = torch.tensor(
            [0.0, 0.0, 0.0, 320.0, 320.0, 320.0], device=device, dtype=torch.float32
        ).view(6, 1, 1)
