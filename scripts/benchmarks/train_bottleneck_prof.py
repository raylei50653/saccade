import torch
import time
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from saccade.perception.temporal_yolo.yolo_joint import (
    JointConfig,
    build_temporal_yolo_joint,
)
from saccade.perception.temporal_yolo.yolo_conditioned import (
    ConditionedConfig,
    TrackerGateInput,
    build_temporal_yolo_conditioned,
)
from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader
from saccade.perception.temporal_yolo.loss import TemporalTrackingLoss

# reuse proxy step from train_conditioned
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "train" / "temporal_yolo"))
from train_conditioned import _proxy_step


def benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scales = tuple(s.strip() for s in args.scales.split(","))
    loss_label = (
        "proxy(no-decoder)"
        if args.proxy_loss
        else ("fast(greedy)" if args.fast_loss else "auction(CPU)")
    )
    model_label = "conditioned(phase1,frozen)" if args.conditioned else "joint"
    print(
        f"Model: {model_label}  Loss: {loss_label}  B={args.batch_size} T={args.clip_len} scales={scales}"
    )

    loader = build_mot17_dataloader(
        data_root=args.data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seqs=args.seqs.split(",") if args.seqs else None,
        preload_to_ram=True,
    )

    if args.conditioned:
        cfg = ConditionedConfig(
            embed_dim=256,
            num_heads=8,
            num_queries=args.num_queries,
            num_decoder_layers=args.num_decoder_layers,
            ffn_dim=1024,
            score_threshold=0.3,
            scales=scales,
            freeze_backbone=True,
            freeze_decoder=True,
        )
        model = build_temporal_yolo_conditioned(args.yolo_weights, cfg, device)
        param_groups = [
            g for g in model.parameter_groups(1e-5, 5e-5, 1e-4) if g["name"] == "gate"
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        cfg = JointConfig(
            embed_dim=256,
            num_heads=8,
            num_queries=args.num_queries,
            num_decoder_layers=args.num_decoder_layers,
            ffn_dim=1024,
            score_threshold=0.3,
            scales=scales,
            freeze_backbone=False,
            use_checkpointing=False,
        )
        model = build_temporal_yolo_joint(args.yolo_weights, cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    criterion = TemporalTrackingLoss()

    t_data = t_device = t_forward = t_loss = t_backward = t_step = 0.0
    n_warmup, n_iters = 3, 15

    print(f"Warmup {n_warmup} + measure {n_iters} iterations...")
    data_iter = iter(loader)

    for i in range(n_warmup + n_iters):
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        t1 = time.perf_counter()

        frames = batch["frames"].to(device, non_blocking=True).float() / 255.0
        gt_boxes_batch = batch["gt_boxes"]
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        B, T, _, H, W = frames.shape

        if args.proxy_loss and args.conditioned:
            # Proxy path: backbone no_grad + gate only (no decoder forward or backward)
            batch_loss = _proxy_step(model, frames, gt_boxes_batch, device)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            t4 = t3  # forward and loss are merged in _proxy_step
        else:
            pred_boxes_b_t = [[] for _ in range(B)]
            pred_scores_b_t = [[] for _ in range(B)]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model.reset_sequence()
                prev_queries = None
                for t in range(T):
                    if args.conditioned:
                        gate_inputs = [
                            TrackerGateInput.from_boxes_scores(
                                gt_boxes_batch[b][t], None, (H, W)
                            ).to(device)
                            for b in range(B)
                        ]
                        out = model(
                            frames[:, t],
                            prev_queries=prev_queries,
                            gate_inputs=gate_inputs,
                        )
                    else:
                        out = model(frames[:, t], prev_queries)
                    prev_queries = out["updated_queries"].detach()
                    for b in range(B):
                        pred_boxes_b_t[b].append(out["boxes"][b])
                        pred_scores_b_t[b].append(out["scores"][b])
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            loss_fn = criterion.forward_fast if args.fast_loss else criterion
            losses = loss_fn(pred_boxes_b_t, pred_scores_b_t, gt_boxes_batch, (H, W))
            batch_loss = losses["loss_total"]
            torch.cuda.synchronize()
            t4 = time.perf_counter()

        batch_loss.backward()
        torch.cuda.synchronize()
        t5 = time.perf_counter()

        optimizer.step()
        torch.cuda.synchronize()
        t6 = time.perf_counter()

        if i >= n_warmup:
            t_data += t1 - t0
            t_device += t2 - t1
            t_forward += t3 - t2
            t_loss += t4 - t3
            t_backward += t5 - t4
            t_step += t6 - t5

    def ms(t):
        return t / n_iters * 1000

    avg = dict(
        data=ms(t_data),
        device=ms(t_device),
        forward=ms(t_forward),
        loss=ms(t_loss),
        backward=ms(t_backward),
        step=ms(t_step),
    )
    total = sum(avg.values())

    print(f"\n{'─' * 52}")
    for k, v in avg.items():
        print(f"  {k:12s}  {v:7.1f} ms  ({v / total * 100:4.1f}%)")
    print(f"  {'total':12s}  {total:7.1f} ms")
    print(f"  throughput    {1000 / total:7.2f} it/s")
    print(f"{'─' * 52}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--scales", default="p3,p4,p5")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seqs", default="MOT17-02-SDP")
    parser.add_argument(
        "--conditioned", action="store_true", help="Option D Phase 1 setup"
    )
    parser.add_argument(
        "--fast-loss", action="store_true", help="GPU greedy matching loss"
    )
    parser.add_argument(
        "--proxy-loss",
        action="store_true",
        help="Feature sharpening proxy (no decoder)",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile backbone")
    args = parser.parse_args()
    benchmark(args)
