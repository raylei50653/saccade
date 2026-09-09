"""The vendored ``make_graphed_callables`` port stays a transcription (#340).

``cuda_capture.graphed_callables`` cannot pass ``capture_error_mode`` through
``torch.cuda.make_graphed_callables`` -- PyTorch does not expose the argument --
so ``_torch_graphs`` carries a copy of that function with one seam added: a
required ``graph_context`` factory used for the forward and backward captures.

A copy of upstream code is only as trustworthy as the evidence that it *is* a
copy.  These tests are that evidence, and they are deliberately fail-closed in
both directions.  The hash pin fires when the pinned torch dependency moves, so
a torch bump forces a deliberate re-port rather than silently leaving us on a
stale transcription.  The seam test fires when our copy drifts from the upstream
text for any reason other than the two documented call sites.

What is *not* asserted here: that the port behaves like upstream at runtime.
Text equality cannot show that.  The behavioural equivalence control lives in
``test_capture_error_mode_mechanism.py``, which runs the port at
``capture_error_mode="global"`` and requires it to fail exactly where upstream
fails.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import difflib
import hashlib
import inspect

import pytest
import torch.cuda.graphs

from saccade.perception.eval import _torch_graphs

# The two graph contexts named in the module docstring, and nothing else.
_EXPECTED_REMOVED = [
    "        with torch.cuda.graph(fwd_graph, pool=mempool):",
    "            with torch.cuda.graph(bwd_graph, pool=mempool):",
]
_EXPECTED_ADDED = [
    "    *,",
    "    graph_context: Callable[..., typing.Any],",
    "        with graph_context(fwd_graph, pool=mempool):",
    "            with graph_context(bwd_graph, pool=mempool):",
]


def _upstream_source() -> str:
    return inspect.getsource(torch.cuda.graphs.make_graphed_callables)


def test_the_pinned_upstream_hash_matches_the_installed_torch() -> None:
    """A torch bump must break this, not silently strand us on an old copy."""
    digest = hashlib.sha256(_upstream_source().encode()).hexdigest()

    assert digest == _torch_graphs.UPSTREAM_FUNCTION_SHA256, (
        "torch.cuda.make_graphed_callables has changed since the port was taken. "
        "Re-port src/saccade/perception/eval/_torch_graphs.py against the new "
        "upstream text deliberately, then update UPSTREAM_FUNCTION_SHA256."
    )


def test_the_port_differs_from_upstream_only_at_the_graph_context_seams() -> None:
    """The copy is a transcription plus one seam -- not a fork."""
    upstream = _upstream_source().splitlines()
    ported = inspect.getsource(_torch_graphs.make_graphed_callables).splitlines()

    diff = list(difflib.unified_diff(upstream, ported, n=0, lineterm=""))
    removed = [line[1:] for line in diff if line.startswith("-") and line[1:2] != "-"]
    added = [line[1:] for line in diff if line.startswith("+") and line[1:2] != "+"]

    assert removed == _EXPECTED_REMOVED
    assert added == _EXPECTED_ADDED


def test_the_graph_context_factory_is_required() -> None:
    """Omitting it must be an error, never a silent fall back to torch's global."""
    signature = inspect.signature(_torch_graphs.make_graphed_callables)
    parameter = signature.parameters["graph_context"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        _torch_graphs.make_graphed_callables(object(), ())
