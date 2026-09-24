"""Native TRTEngine enqueues on the caller's stream, including the null stream.

Regression for #457.  ``torch.cuda.current_stream().cuda_stream`` is ``0``
outside a stream context; ``TRTEngine`` used to read a null handle as "no
stream" and enqueue on a private non-blocking stream instead, which drops
ordering against the kernels the caller queued on the legacy default stream.
On ``--preset baseline`` that let TRT read the detector input before its
resize finished and changed MOT output run to run.

The test makes the ordering observable without relying on timing luck: the
producer queues a long device-side sleep and then writes the input on stream
0.  An engine that runs on stream 0 must see the written value; an engine
that runs on any stream not ordered after stream 0 reads the stale value
while the sleep is still in flight.

Skipped without CUDA, TensorRT, or the native perception extension.
"""

# scope: perception
# function: regression
# lifecycle: active

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

trt = pytest.importorskip("tensorrt")
_ext = pytest.importorskip("saccade_perception_ext")

_N = 1 << 16
# Long enough that the producer is still sleeping when a mis-ordered enqueue
# runs; torch.cuda._sleep counts GPU clock cycles.
_SLEEP_CYCLES = 200_000_000


def _build_identity_engine(path: Path) -> None:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    x = network.add_input("x", trt.float32, (1, _N))
    # x * 1 keeps a real kernel reading the input binding.
    one = network.add_constant(
        (1, 1), trt.Weights(torch.ones(1, dtype=torch.float32).numpy())
    )
    y = network.add_elementwise(
        x, one.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    y.name = "y"
    network.mark_output(y)
    config = builder.create_builder_config()
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None
    path.write_bytes(bytes(serialized))


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    path = tmp_path_factory.mktemp("trt") / "identity.engine"
    _build_identity_engine(path)
    engine = _ext.TRTEngine(str(path))
    # The default context is created lazily on the first infer(); its
    # allocation synchronizes the device and would hide a mis-ordered enqueue.
    x = torch.zeros((1, _N), device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    assert engine.infer([x.data_ptr(), y.data_ptr()], 0)
    torch.cuda.synchronize()
    return engine


def _stale_input_then_produce() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros((1, _N), device="cuda", dtype=torch.float32)
    y = torch.full((1, _N), -1.0, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    assert torch.cuda.current_stream().cuda_stream == 0
    torch.cuda._sleep(_SLEEP_CYCLES)
    x.fill_(1.0)
    return x, y


def _assert_saw_produced_input(y: torch.Tensor) -> None:
    torch.cuda.synchronize()
    assert torch.equal(y, torch.ones_like(y)), (
        "TRT read the input before the stream-0 producer finished: "
        f"unique outputs {torch.unique(y).tolist()}"
    )


def test_infer_null_stream_orders_after_legacy_default_stream(engine) -> None:
    x, y = _stale_input_then_produce()
    assert engine.infer([x.data_ptr(), y.data_ptr()], 0)
    _assert_saw_produced_input(y)


def test_enqueue_v3_null_stream_orders_after_legacy_default_stream(engine) -> None:
    x, y = _stale_input_then_produce()
    assert engine.set_tensor_address("x", x.data_ptr())
    assert engine.set_tensor_address("y", y.data_ptr())
    assert engine.enqueue_v3(0)
    _assert_saw_produced_input(y)


def test_infer_with_context_null_stream_orders_after_legacy_default_stream(
    engine,
) -> None:
    ctx = engine.create_context()
    try:
        x, y = _stale_input_then_produce()
        assert engine.infer_with_context(ctx, [x.data_ptr(), y.data_ptr()], 0)
        _assert_saw_produced_input(y)
    finally:
        torch.cuda.synchronize()
        _ext.TRTEngine.delete_context(ctx)
