import torch
import numpy as np
import time
import saccade_tracking_ext
import argparse

def run_benchmark(max_objs=1024, num_dets=100, embed_dim=768, num_frames=500, warmup=50):
    print(f"--- Benchmarking Association: MaxObjs={max_objs}, Dets={num_dets}, EmbedDim={embed_dim} ---")
    
    # 1. Setup Tracker
    tracker = saccade_tracking_ext.GPUByteTracker(max_objs, embed_dim)
    tracker.set_params(0.1, 0.5, 0.8, 30)
    
    # 2. Prepare Synthetic Data on GPU
    # boxes: [num_dets, 4] (x1, y1, x2, y2)
    # scores: [num_dets]
    # classes: [num_dets]
    # embeds: [num_dets, embed_dim]
    
    # Create random but valid boxes
    h_boxes = np.random.rand(num_dets, 4).astype(np.float32) * 1000
    h_boxes[:, 2:] += h_boxes[:, :2] # ensure x2 > x1, y2 > y1
    h_scores = np.random.rand(num_dets).astype(np.float32)
    h_classes = np.random.randint(0, 80, size=num_dets).astype(np.int32)
    h_embeds = np.random.rand(num_dets, embed_dim).astype(np.float32)
    # Normalize embeddings for CosSim
    h_embeds /= np.linalg.norm(h_embeds, axis=1, keepdims=True)
    
    d_boxes = torch.from_numpy(h_boxes).cuda()
    d_scores = torch.from_numpy(h_scores).cuda()
    d_classes = torch.from_numpy(h_classes).cuda()
    d_embeds = torch.from_numpy(h_embeds).cuda()
    
    # 3. Warm-up
    print(f"Warming up for {warmup} frames...")
    for _ in range(warmup):
        _ = tracker.update(
            d_boxes.data_ptr(), 
            d_scores.data_ptr(), 
            d_classes.data_ptr(), 
            num_dets, 
            torch.cuda.current_stream().cuda_stream, 
            d_embeds.data_ptr(),
            0, # No GMC for simple bench
            1.0 # light_factor
        )
    torch.cuda.synchronize()

    # 4. Statistics Collection
    print(f"Collecting data for {num_frames} frames...")
    latencies = []
    
    for i in range(num_frames):
        torch.cuda.nvtx.range_push(f"Frame_{i}")
        start = time.perf_counter()
        
        # Launch kernels
        _ = tracker.update(
            d_boxes.data_ptr(), 
            d_scores.data_ptr(), 
            d_classes.data_ptr(), 
            num_dets, 
            torch.cuda.current_stream().cuda_stream, 
            d_embeds.data_ptr(),
            0,
            1.0
        )
        
        # We synchronize here ONLY for measurement accuracy in this benchmark
        torch.cuda.synchronize() 
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms
        torch.cuda.nvtx.range_pop()

    # 5. Report Results
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    
    print(f"Results (ms):")
    print(f"  Avg: {avg:.4f}")
    print(f"  P50: {p50:.4f}")
    print(f"  P95: {p95:.4f}")
    print(f"  P99: {p99:.4f}")
    print(f"  FPS: {1000.0/avg:.2f}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dets", type=int, default=100)
    parser.add_argument("--max_objs", type=int, default=1024)
    args = parser.parse_args()
    
    # Stress test levels
    test_scenarios = [100, 300, 500, 1000]
    for n in test_scenarios:
        if n <= args.max_objs:
            run_benchmark(max_objs=args.max_objs, num_dets=n)
