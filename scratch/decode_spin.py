import time
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file

SEQ = Path("datasets/MOT17/train/MOT17-04-SDP/img1")
RGB = ImageReadMode.RGB
raws = [read_file(str(p)) for p in sorted(SEQ.glob("*.jpg"))[:200]]
t_end = time.time() + 12
i = 0
while time.time() < t_end:
    decode_jpeg(raws[i % len(raws)], device="cuda", mode=RGB)
    i += 1
torch.cuda.synchronize()
print("decoded", i)
