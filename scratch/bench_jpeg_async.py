"""Is nvJPEG decode_jpeg actually async? Measure host-submit vs GPU time."""

import time
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file

SEQ = Path("datasets/MOT17/train/MOT17-04-SDP/img1")
RGB = ImageReadMode.RGB
files = sorted(str(p) for p in SEQ.glob("*.jpg"))[:320]
raws = [read_file(f) for f in files]

# warmup
for d in raws[:20]:
    decode_jpeg(d, device="cuda", mode=RGB)
torch.cuda.synchronize()

work = raws[20:320]
n = len(work)

# Measure: time to RETURN from all decode calls (host submit), then time to sync.
t0 = time.perf_counter()
outs = [decode_jpeg(d, device="cuda", mode=RGB) for d in work]
t_submit = time.perf_counter()
torch.cuda.synchronize()
t_sync = time.perf_counter()

print(f"host submit (loop return): {(t_submit - t0)*1e3/n:7.3f} ms/frame")
print(f"extra GPU after sync:      {(t_sync - t_submit)*1e3/n:7.3f} ms/frame")
print(f"total:                     {(t_sync - t0)*1e3/n:7.3f} ms/frame")
print()
print("If host-submit ~= total -> decode BLOCKS on CPU (host-side huffman), not async.")
print("If host-submit << total -> decode is async, GPU-bound.")
