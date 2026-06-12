"""Break down the nvJPEG path: file-read vs decode, serial vs batched."""

import time
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file

SEQ = Path("datasets/MOT17/train/MOT17-04-SDP/img1")
RGB = ImageReadMode.RGB
files = sorted(str(p) for p in SEQ.glob("*.jpg"))[:320]


def time_it(name, fn, sync=True):
    # warmup
    fn(files[:20])
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = fn(files[20:320])
    if sync:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{name:34s} {dt*1e3/n:7.3f} ms/frame  {n/dt:8.1f} fps")


# 1. just CPU file read
def just_read(fs):
    for f in fs:
        _ = read_file(f)
    return len(fs)


# 2. read + serial decode (current code path)
def read_decode_serial(fs):
    for f in fs:
        data = read_file(f)
        _ = decode_jpeg(data, device="cuda", mode=RGB)
    return len(fs)


# 3. preloaded bytes, serial decode (isolate pure decode)
def decode_only_serial(fs):
    raws = [read_file(f) for f in fs]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for d in raws:
        _ = decode_jpeg(d, device="cuda", mode=RGB)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{'  (pure decode, preloaded)':34s} {dt*1e3/len(fs):7.3f} ms/frame  {len(fs)/dt:8.1f} fps")
    return len(fs)


# 4. batched decode (nvJPEG batched API)
def decode_batched(fs, bs=16):
    raws = [read_file(f) for f in fs]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(raws), bs):
        _ = decode_jpeg(raws[i : i + bs], device="cuda", mode=RGB)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{f'  (batched bs={bs}, preloaded)':34s} {dt*1e3/len(fs):7.3f} ms/frame  {len(fs)/dt:8.1f} fps")
    return len(fs)


if __name__ == "__main__":
    time_it("CPU file read only", just_read, sync=False)
    time_it("read + serial decode (current)", read_decode_serial)
    decode_only_serial(files[:300])
    decode_batched(files[:300], bs=8)
    decode_batched(files[:300], bs=16)
    decode_batched(files[:300], bs=32)
