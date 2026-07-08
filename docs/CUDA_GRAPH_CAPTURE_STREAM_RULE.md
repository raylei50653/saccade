# CUDA Graph Capture Stream Rule

## Rule

Inside `with torch.cuda.graph(graph):`, **always** resolve the CUDA stream via
`torch.cuda.current_stream().cuda_stream` — do **not** reuse a `stream_ptr`
captured before entering the context.

## Why

`torch.cuda.graph()` creates a dedicated capture stream and makes it the
current stream for the duration of the context.  PyTorch uses
`cudaStreamCaptureModeGlobal`, which **rejects** CUDA operations submitted to
any stream that is not the capture stream itself.

If you pass a `stream_ptr` obtained *outside* the context (the "original"
stream), the C++ extension launches kernels on that non-capture stream.
Global mode treats this as a capture violation — operations are silently
dropped or the graph is left empty/broken.  The graph object is created
without error, but `replay()` produces garbage (all-zeros or unchanged input).

## Correct Pattern

```python
stream_ptr = torch.cuda.current_stream().cuda_stream  # for warmup (outside)

# Warmup (eager, outside graph context) — use stream_ptr
perception_pipeline.process_detections_graph(..., stream_ptr)
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    capture_stream = torch.cuda.current_stream().cuda_stream  # resolve INSIDE
    perception_pipeline.process_detections_graph(..., capture_stream)
```

## Incorrect Pattern (produces broken graphs)

```python
stream_ptr = torch.cuda.current_stream().cuda_stream  # WRONG: outside context

with torch.cuda.graph(graph):
    perception_pipeline.process_detections_graph(..., stream_ptr)  # rejected
```

## Affected Sites

All `torch.cuda.graph()` capture sites in the codebase were audited and
fixed in commit `5bd9a634`:

- `_capture_main_nms_graph_nocopyback` (stages.py)
- `_capture_main_nms_graph` (stages.py)
- Monolithic `process_detections_graph` capture in `_run_nms` (stages.py)

Future graph captures (GMC, post, detect, tracker) must follow the same rule.
