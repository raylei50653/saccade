"""
multistream_mamba_server.py
───────────────────────────
Dynamic cross-stream batching for the Option-F Mamba tracking pipeline.

Multiple independent video streams each run on their own worker thread and call
``submit(stream_id, frame)``. A single server thread coalesces the *current
frame* of up to ``max_batch`` streams into one batch-N TRT backbone call
(GIL-free in C++), then runs the Mamba SSM head per stream using that stream's
own temporal window. This exploits the ~83% GPU idle measured by nsys: while
stream A does its CPU-bound work (GMC, tracker update), stream B's frame can be
in the backbone batch.

                Stream 0 ─submit(0,f)─┐
                Stream 1 ─submit(1,f)─┼─► server thread
                Stream 2 ─submit(2,f)─┤     │ 1× C++ BatchedBackbone (batch-N)
                Stream 3 ─submit(3,f)─┘     │ per-stream Mamba head + post-process
                                            ↓ promises fulfilled
                                      each stream runs tracker.update in its thread

Correctness is anchored on ``MambaGatedDetector._detect_from_feats``: the server
calls that exact code path with a per-stream ``StreamState``, so each stream's
detections are identical to running the sequence single-stream. Trackers live in
the caller (the stream's own thread), not the server.

P1 (this module): backbone is batched across streams; the head runs per stream
(B=1). P2 will batch the head across streams sharing the same temporal depth.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from torch import Tensor

from saccade.perception.temporal_yolo.mamba_gated_detector import (
    StreamState,
    build_mamba_gated_detector,
)


class _StreamCtx:
    """Per-stream detection state held by the server (not the tracker)."""

    __slots__ = ("state", "last_fpn")

    def __init__(self, state: StreamState) -> None:
        self.state = state
        self.last_fpn: list[Tensor] | None = None


class _Request:
    """A single submitted frame and its result promise."""

    __slots__ = ("stream_id", "frame", "event", "result", "error")

    def __init__(self, stream_id: Any, frame: Tensor) -> None:
        self.stream_id = stream_id
        self.frame = frame
        self.event = threading.Event()
        # Tensor of detections, or (feats, backbone_event) in event-handoff mode.
        self.result: Any = None
        self.error: BaseException | None = None


class MultiStreamMambaServer:
    """Cross-stream batched Mamba detector.

    Parameters
    ----------
    backbone_engine : str
        Path to the batch-N TRT backbone engine (.engine).
    mamba_ckpt : str
        Path to the Mamba head checkpoint (Option-F v14).
    yolo_pt_path : str
        YOLO backbone .pt (used only to build the head wrapper / gate config).
    teacher_ckpt : str
        Optional gated-YOLO teacher checkpoint (gate is unused when gate_input
        is None, which is the serving path).
    img_size : int
        Input spatial resolution (default 640).
    conf_thr, max_det : float, int
        Post-process thresholds. Defaults match the ``mamba_optimal`` eval path.
    max_batch : int
        Max frames coalesced into one backbone call (default 4, must match the
        engine's max batch).
    flush_timeout_s : float
        Max seconds the server waits to grow a partial batch before flushing.
    device : str
        CUDA device.
    """

    def __init__(
        self,
        backbone_engine: str,
        mamba_ckpt: str,
        *,
        yolo_pt_path: str = "models/yolo/yolo26s.pt",
        teacher_ckpt: str = "",
        img_size: int = 640,
        conf_thr: float = 0.001,
        max_det: int = 300,
        max_batch: int = 4,
        flush_timeout_s: float = 0.002,
        device: str = "cuda",
        mamba_head_script: str = "models/yolo/mamba_head_best.pt",
        use_cpp_head: bool = True,
        event_handoff: bool = False,
    ) -> None:
        from saccade_perception_ext import BatchedBackbone  # C++ binding

        # Shared model: provides the Mamba head, strides, conf/max_det and the
        # _detect_from_feats code path. trt_backbone_engine="" — the C++
        # BatchedBackbone owns backbone inference instead.
        self._model = build_mamba_gated_detector(
            yolo_pt_path=yolo_pt_path,
            teacher_ckpt=teacher_ckpt,
            mamba_ckpt=mamba_ckpt,
            img_size=img_size,
            device=device,
            conf_thr=conf_thr,
            max_det=max_det,
            trt_backbone_engine="",
            use_cuda_graph=False,
        )
        self._temporal_T: int = self._model._temporal_T

        self._backbone = BatchedBackbone(backbone_engine, max_batch, img_size)

        # GIL-free C++ head: runs the TorchScript Mamba head + decode on
        # pre-computed feats, releasing the GIL so concurrent stream workers'
        # Python postprocess overlaps. Only valid for the single-frame (v14,
        # temporal_T==0) path; temporal streams fall back to the Python head.
        self._cpp_head: Any = None
        if use_cpp_head and self._temporal_T == 0:
            import os

            if os.path.exists(mamba_head_script):
                from saccade_perception_ext import (  # C++ binding
                    MambaGatedDetector as _CppMambaGatedDetector,
                )

                self._cpp_head = _CppMambaGatedDetector(
                    trt_backbone_path=backbone_engine,
                    mamba_head_script_path=mamba_head_script,
                    img_size=img_size,
                    conf_thr=conf_thr,
                )
        self._conf_thr = conf_thr
        self._max_det = max_det

        # Event-handoff mode: the server thread does the (batched) backbone only,
        # records a CUDA event, and hands feats + event to each worker. Each
        # worker then runs the GIL-free C++ head on its *own* CUDA stream (after
        # waiting on the event), so per-stream GPU work overlaps instead of
        # serializing on the default stream. Requires the C++ head.
        self.event_handoff: bool = event_handoff and self._cpp_head is not None

        self.img_size = img_size
        self.max_batch = max_batch
        self.flush_timeout_s = flush_timeout_s
        self.device = torch.device(device)

        # Per-stream detection state.
        self._streams: dict[Any, _StreamCtx] = {}
        self._streams_lock = threading.Lock()

        # Diagnostics: histogram of realized batch sizes (index = batch size).
        self.batch_hist: list[int] = [0] * (max_batch + 1)

        # Request queue + server thread.
        self._queue: list[_Request] = []
        self._cv = threading.Condition(threading.Lock())
        self._running = True
        self._server = threading.Thread(
            target=self._server_loop, daemon=True, name="MultiStreamMamba-Server"
        )
        self._server.start()

    # ──────────────────────────────────────────────── public API

    def register_stream(self, stream_id: Any) -> None:
        """Allocate independent temporal state for a new stream."""
        with self._streams_lock:
            self._streams[stream_id] = _StreamCtx(StreamState.create(self._temporal_T))

    def reset_stream(self, stream_id: Any) -> None:
        """Clear a stream's temporal/GMC buffers (e.g. at sequence start)."""
        ctx = self._get_ctx(stream_id)
        ctx.state.reset()
        ctx.last_fpn = None

    def set_gmc_warp(
        self, stream_id: Any, warp: Tensor | None, orig_h: int = 0, orig_w: int = 0
    ) -> None:
        """Push this frame's GMC affine warp into the stream's buffer."""
        if warp is not None:
            self._get_ctx(stream_id).state.push_gmc(warp)

    def submit(self, stream_id: Any, frame: Tensor) -> Any:
        """Submit one frame for batched detection. Blocks until ready.

        Parameters
        ----------
        frame : Tensor
            ``[1, 3, H, W]`` or ``[3, H, W]`` float32 CUDA tensor in [0, 1].

        Returns
        -------
        Tensor or tuple
            Detections ``[1, max_det, 6]``; or, in event-handoff mode,
            ``(feats, backbone_event)`` for the worker to run the head itself.
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        if stream_id not in self._streams:
            raise KeyError(
                f"stream {stream_id!r} not registered; call register_stream()"
            )

        req = _Request(stream_id, frame.contiguous())
        with self._cv:
            self._queue.append(req)
            self._cv.notify()
        req.event.wait()
        if req.error is not None:
            raise req.error
        assert req.result is not None
        return req.result

    def extract_fpn_embeddings(
        self, stream_id: Any, frame_bchw: Tensor | None, boxes_xyxy: Tensor
    ) -> Tensor:
        """Zero-training ReID: center-pool the stream's last FPN at each box.

        boxes_xyxy are in model (img_size) pixel coords. Returns L2-normalized
        [N, 896] (128+256+512) embeddings on CUDA. Empty boxes -> [0, 896].
        """
        opts: dict[str, Any] = dict(dtype=torch.float32, device="cuda")
        if boxes_xyxy.size(0) == 0:
            return torch.zeros(0, 896, **opts)
        feats = self._get_ctx(stream_id).last_fpn
        if feats is None:
            return torch.zeros(0, 896, **opts)
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5
        parts = []
        for f in feats:
            fh, fw = f.size(2), f.size(3)
            cxi = (cx / self.img_size * fw).long().clamp(0, fw - 1)
            cyi = (cy / self.img_size * fh).long().clamp(0, fh - 1)
            parts.append(f[0, :, cyi, cxi].T)  # [N, C]
        cat = torch.cat(parts, 1)
        normed: Tensor = cat / (cat.norm(2, 1, keepdim=True) + 1e-12)
        return normed

    def shutdown(self) -> None:
        """Stop the server thread cleanly."""
        with self._cv:
            self._running = False
            self._cv.notify_all()
        self._server.join(timeout=5.0)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    # ──────────────────────────────────────────────── internals

    def _get_ctx(self, stream_id: Any) -> _StreamCtx:
        try:
            return self._streams[stream_id]
        except KeyError:
            raise KeyError(
                f"stream {stream_id!r} not registered; call register_stream()"
            ) from None

    def _server_loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue and self._running:
                    self._cv.wait(timeout=0.1)
                if not self._running and not self._queue:
                    return
                # Grow the batch up to max_batch within the flush window.
                if len(self._queue) < self.max_batch and self._running:
                    self._cv.wait_for(
                        lambda: len(self._queue) >= self.max_batch or not self._running,
                        timeout=self.flush_timeout_s,
                    )
                n = min(len(self._queue), self.max_batch)
                batch = self._queue[:n]
                del self._queue[:n]

            if batch:
                self._run_batch(batch)

    def _run_batch(self, reqs: list[_Request]) -> None:
        self.batch_hist[len(reqs)] += 1
        try:
            frames = torch.cat([r.frame for r in reqs], dim=0)  # [N, 3, H, W]
            with torch.no_grad():
                p3_b, p4_b, p5_b = self._backbone.batch_infer(frames)

                # Event-handoff: mark backbone completion on the server stream so
                # each worker stream can wait on it before reading feats. One
                # event covers the whole batch (batch_infer produced all feats).
                backbone_ev = None
                if self.event_handoff:
                    backbone_ev = torch.cuda.Event()  # type: ignore[no-untyped-call]
                    backbone_ev.record()

                # Slice per stream + record last FPN for ReID.
                sliced: list[tuple[_Request, _StreamCtx, list[Tensor]]] = []
                for i, req in enumerate(reqs):
                    ctx = self._streams[req.stream_id]
                    feats = [p3_b[i : i + 1], p4_b[i : i + 1], p5_b[i : i + 1]]
                    ctx.last_fpn = feats
                    sliced.append((req, ctx, feats))

                # Group by the temporal depth each stream will reach after this
                # frame, so the head runs once per depth (batch-major B=group).
                groups: dict[int, list[tuple[_Request, _StreamCtx, list[Tensor]]]] = {}
                for req, ctx, feats in sliced:
                    tb = ctx.state.temporal_buffer
                    depth = (
                        min(len(tb[0]) + 1, ctx.state.temporal_T)
                        if tb is not None
                        else 0
                    )
                    groups.setdefault(depth, []).append((req, ctx, feats))

                for grp in groups.values():
                    if self.event_handoff:
                        # Hand feats + backbone event to each worker; the worker
                        # runs the C++ head on its own CUDA stream (see
                        # MambaStreamProxy.detect_raw).
                        for req, _ctx, feats in grp:
                            req.result = (feats, backbone_ev)
                            req.event.set()
                    elif self._cpp_head is not None:
                        # GIL-free C++ head per stream (temporal_T==0 path), run
                        # on the server thread. The GIL is released inside
                        # forward_feats_padded_ptr.
                        for req, _ctx, feats in grp:
                            req.result = self._cpp_detect(feats)
                            req.event.set()
                    else:
                        feats_list = [f for (_, _, f) in grp]
                        states = [c.state for (_, c, _) in grp]
                        outs = self._model._detect_batch(feats_list, states)
                        for (req, _, _), (dets, _extra) in zip(grp, outs):
                            req.result = dets
                            req.event.set()
        except BaseException as e:  # noqa: BLE001 — propagate to every waiter
            for req in reqs:
                if not req.event.is_set():
                    req.error = e
                    req.event.set()

    def _cpp_detect(self, feats: list[Tensor]) -> Tensor:
        """Run the GIL-free C++ head on one stream's feats.

        Returns the padded ``[1, max_det, 6]`` serving tensor (batch dim kept to
        match ``_detect_batch``), byte-for-byte equivalent to
        ``_postprocess_mamba`` (conf-threshold + topk, no NMS).
        """
        p3, p4, p5 = (f.contiguous() for f in feats)
        # Contiguous [1, max_det, 6] — the C++ fill writes max_det*6 floats, so
        # the buffer layout is identical whether viewed as [max_det, 6] or
        # [1, max_det, 6].
        out = torch.empty(1, self._max_det, 6, dtype=torch.float32, device=self.device)
        self._cpp_head.forward_feats_padded_ptr(
            p3.data_ptr(),
            p4.data_ptr(),
            p5.data_ptr(),
            self._conf_thr,
            self._max_det,
            out.data_ptr(),
        )
        return out


class MambaStreamProxy:
    """Per-stream detector adapter over a shared MultiStreamMambaServer.

    Implements the (Python) detector interface ``run_eval`` expects so that the
    existing evaluation pipeline can drive one stream while detection is
    coalesced across streams by the shared server. Mirrors the role of
    ``ConcurrentDetectorProxy`` but routes through the batched server instead of
    a per-thread TRT context. Each proxy owns its own tracker.

    Intentionally exposes **no** ``cpp_ptr`` so ``run_eval`` uses the Python
    detection path (and never the C++ pool).
    """

    def __init__(self, server: MultiStreamMambaServer, stream_id: Any) -> None:
        from saccade.perception.tracking import GPUByteTracker

        self.server = server
        self.stream_id = stream_id
        server.register_stream(stream_id)
        self.tracker = GPUByteTracker(max_objects=2048)
        # Dedicated CUDA stream so this stream's head + postprocess + tracker run
        # concurrently with other streams instead of serializing on the default
        # stream. Only used when the server runs in event-handoff mode; the
        # driver thread should make this the thread's current stream
        # (torch.cuda.set_stream) so run_eval's GPU work lands on it.
        self.stream: torch.cuda.Stream | None = (
            torch.cuda.Stream()  # type: ignore[no-untyped-call]
            if server.event_handoff
            else None
        )

    @property
    def device(self) -> torch.device:
        return self.server.device

    def reset_tracker(self) -> None:
        from saccade.perception.tracking import GPUByteTracker

        self.tracker = GPUByteTracker(max_objects=2048)
        self.server.reset_stream(self.stream_id)

    def set_gmc_warp(
        self, warp: Tensor | None, orig_h: int = 0, orig_w: int = 0
    ) -> None:
        self.server.set_gmc_warp(self.stream_id, warp, orig_h, orig_w)

    def detect_raw(self, input_tensor: Tensor) -> Tensor:
        res = self.server.submit(self.stream_id, input_tensor)
        if not self.server.event_handoff:
            return res  # type: ignore[no-any-return]
        # Event-handoff: res is (feats, backbone_event). Wait for the backbone on
        # this worker's current stream, then run the GIL-free C++ head on it so
        # the head overlaps with other streams' work.
        feats, backbone_ev = res
        cur = torch.cuda.current_stream()
        if backbone_ev is not None:
            cur.wait_event(backbone_ev)
        for f in feats:
            f.record_stream(cur)  # mark cross-stream use for the caching allocator
        return self.server._cpp_detect(feats)

    def extract_fpn_embeddings(
        self, frame_bchw: Tensor | None, boxes_xyxy: Tensor
    ) -> Tensor:
        return self.server.extract_fpn_embeddings(
            self.stream_id, frame_bchw, boxes_xyxy
        )

    def set_use_cuda_graph(self, enabled: bool) -> None:
        # CUDA-graph capture is managed (or disabled) by the server.
        pass
