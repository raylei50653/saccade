"""Does decode_jpeg release the GIL? If so, a background decode thread overlaps
the CPU huffman with the main pipeline -> 'async in the main flow'."""

import threading
import time
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file

SEQ = Path("datasets/MOT17/train/MOT17-04-SDP/img1")
RGB = ImageReadMode.RGB
raws = [read_file(str(p)) for p in sorted(SEQ.glob("*.jpg"))[:420]]
for d in raws[:20]:
    decode_jpeg(d, device="cuda", mode=RGB)
torch.cuda.synchronize()


def decode_chunk(chunk):
    for d in chunk:
        decode_jpeg(d, device="cuda", mode=RGB)


def run(nthreads):
    work = raws[20:420]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if nthreads == 1:
        decode_chunk(work)
    else:
        parts = [work[i::nthreads] for i in range(nthreads)]
        ts = [threading.Thread(target=decode_chunk, args=(p,)) for p in parts]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = len(work)
    print(f"{nthreads} thread(s): {dt*1e3/n:7.3f} ms/frame  {n/dt:8.1f} fps")


if __name__ == "__main__":
    run(1)
    run(2)
    run(4)
