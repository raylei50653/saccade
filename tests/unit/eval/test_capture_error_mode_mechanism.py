"""What ``capture_error_mode`` actually buys the graphed-callables entrance (#340).

``graphed_callables`` moved off ``torch.cuda.make_graphed_callables`` onto a
vendored port so the capture can run at ``thread_local`` instead of torch's
hard-coded ``global``.  That is a claim about CUDA behaviour, so it is measured
here against a device rather than argued from the CUDA documentation.

Four cases, each in its own process because a failed capture leaves the CUDA
context unfit to be reused:

* **positive control** -- upstream at ``global``, a second thread forcing a
  fresh ``cudaMalloc`` while the capture is open.  The capture must die.  A
  green suite here with a control that never fires would be worthless.
* **equivalence control** -- the port at ``global``, same workload.  It must die
  the same way, which is what makes the port a transcription in behaviour and
  not only in text (see ``test_torch_graphs_port.py`` for the text half).
* **the fix** -- the port at ``thread_local``, same workload.  The capture must
  survive *and* the other thread's allocation must really have happened.
* **the named limit** -- the port at ``thread_local``, but the second thread
  calls ``torch.cuda.synchronize()`` instead.  This still kills the capture.
  ``thread_local`` narrows the exposure to another thread's allocator; it does
  not make cross-thread capture invalidation impossible.

None of this identifies the historical #340 CUDA 901.  It shows one mechanism
that can invalidate this entrance's capture and that the patch closes one half
of it; the campaign's unattributed failure is a separate open question.

Every worker re-checks that its unsafe call was issued while
``is_current_stream_capturing()`` was true.  An earlier hand probe passed all
three modes because the barrier fired during a warmup iteration instead, so
that check is a hard failure, not a diagnostic.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[3]

_WORKER = """
import json, sys, threading
import torch

sys.path.insert(0, {src!r})

MODE, UNSAFE = sys.argv[1], sys.argv[2]
DEV = "cuda"


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8).to(DEV)

    def forward(self, x):
        return self.lin(x).relu()


def segments_allocated():
    return torch.cuda.memory_stats().get("segment.all.allocated", -1)


def main():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    module = Tiny()
    args = (torch.randn(4, 8, device=DEV, requires_grad=True),)

    open_capture = threading.Event()
    other_done = threading.Event()
    note = {{"issued_while_capturing": False, "other_error": None, "segment_delta": 0}}

    def other_thread():
        if not open_capture.wait(60):
            other_done.set()
            return
        before = segments_allocated()
        try:
            if UNSAFE == "malloc":
                # Large and fresh: the caching allocator must reach cudaMalloc.
                del_me = torch.empty((512 * 1024 * 1024,), dtype=torch.uint8, device=DEV)
                del del_me
            elif UNSAFE == "sync":
                torch.cuda.synchronize()
            else:
                raise AssertionError(UNSAFE)
        except Exception as exc:
            note["other_error"] = type(exc).__name__
        finally:
            note["segment_delta"] = segments_allocated() - before
            other_done.set()

    inner_forward = module.forward

    def forward(x):
        # Warmup iterations run this too; only the captured one may signal.
        if torch.cuda.is_current_stream_capturing() and not open_capture.is_set():
            note["issued_while_capturing"] = True
            open_capture.set()
            other_done.wait(90)
        return inner_forward(x)

    module.forward = forward

    worker = threading.Thread(target=other_thread, name="allocator-thread")
    worker.start()
    try:
        if MODE == "upstream":
            torch.cuda.make_graphed_callables(module, args, num_warmup_iters=3)
        else:
            from saccade.perception.eval import _torch_graphs

            error_mode = "global" if MODE == "port-global" else "thread_local"

            def graph_context(graph, *, pool):
                return torch.cuda.graph(graph, pool=pool, capture_error_mode=error_mode)

            _torch_graphs.make_graphed_callables(
                module, args, num_warmup_iters=3, graph_context=graph_context
            )
        note["capture"] = "ok"
    except Exception as exc:
        note["capture"] = "raised"
        note["capture_error"] = type(exc).__name__
    finally:
        open_capture.set()
        worker.join(120)

    print("RESULT " + json.dumps(note), flush=True)


main()
"""


def _run_case(tmp_path: Path, mode: str, unsafe: str) -> dict:
    worker = tmp_path / f"worker_{mode}_{unsafe}.py".replace("-", "_")
    worker.write_text(_WORKER.format(src=str(ROOT / "src")))

    completed = subprocess.run(
        [sys.executable, str(worker), mode, unsafe],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(tmp_path),
    )
    lines = [ln for ln in completed.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, (
        f"worker produced no result line (rc={completed.returncode})\n"
        f"stdout:\n{completed.stdout[-2000:]}\nstderr:\n{completed.stderr[-2000:]}"
    )
    note = json.loads(lines[-1][len("RESULT ") :])

    # Not a soft diagnostic: a case whose unsafe call missed the capture window
    # proves nothing, and must not be allowed to read as a pass.
    assert note["issued_while_capturing"], (
        f"{mode}/{unsafe}: the unsafe call was never issued inside the capture"
    )
    return note


def test_another_thread_allocating_kills_an_upstream_capture(tmp_path) -> None:
    """Positive control: the hazard this patch is about is real on this device."""
    note = _run_case(tmp_path, "upstream", "malloc")

    assert note["capture"] == "raised"


def test_the_port_at_global_fails_exactly_where_upstream_does(tmp_path) -> None:
    """Equivalence control: the seam alone changes no behaviour."""
    note = _run_case(tmp_path, "port-global", "malloc")

    assert note["capture"] == "raised"


def test_thread_local_survives_another_thread_allocating(tmp_path) -> None:
    """The fix -- and the allocation must really have reached cudaMalloc."""
    note = _run_case(tmp_path, "port-thread_local", "malloc")

    assert note["capture"] == "ok"
    assert note["other_error"] is None
    assert note["segment_delta"] >= 1, (
        "the allocation was served from the caching allocator, so this run did "
        "not exercise the unsafe API it claims to"
    )


def test_thread_local_does_not_survive_another_thread_synchronising(tmp_path) -> None:
    """Named limit: the patch narrows the exposure, it does not remove it."""
    note = _run_case(tmp_path, "port-thread_local", "sync")

    assert note["capture"] == "raised"
