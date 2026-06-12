import sys
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file

SEQ = Path("datasets/MOT17/train/MOT17-04-SDP/img1")
RGB = ImageReadMode.RGB
raws = [read_file(str(p)) for p in sorted(SEQ.glob("*.jpg"))[:120]]
for d in raws[:20]:
    decode_jpeg(d, device="cuda", mode=RGB)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("decode100")
for d in raws[20:120]:
    decode_jpeg(d, device="cuda", mode=RGB)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
