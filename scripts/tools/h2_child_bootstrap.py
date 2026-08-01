#!/usr/bin/env python3
"""Start the import recorder, then load the measurement child.

This file exists because of an ordering problem that cannot be solved inside the
child.  A recorder installed in `run_h2_measurement_child.main()` would already
have missed that module's own top-level imports — the behavioral-identity module
that chooses which evaluator stack loads, the RunSpec resolver, the packet
verifier — because those resolve while the module object is being built, long
before `main` is called.  The blind spot would cover exactly the code that
decides what the run does.

So the recorder is installed by a separate process entry point that imports
nothing of the repository beyond the recorder itself, and only then imports the
child.  Everything the child pulls in is therefore witnessed, and the two files
that could not be — this one and the recorder — are named in
`h2_import_witness.BOOTSTRAP_SELF_PATHS`, recorded in the witness, and bounded by
the verifier rather than assumed away.
"""
# status: stable

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_import_witness  # noqa: E402

ENTRY_MODULE = "run_h2_measurement_child"


def main(argv: Sequence[str] | None = None) -> int:
    h2_import_witness.install(entry_module=ENTRY_MODULE)
    entry = importlib.import_module(ENTRY_MODULE)
    return int(entry.main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
