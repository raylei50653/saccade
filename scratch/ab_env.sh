#!/usr/bin/env bash
# Sets up LD_LIBRARY_PATH (torch cu13 + nvidia libs) and PYTHONPATH for the
# prebuilt saccade_tracking_ext, then execs whatever command is passed.
cd /home/ray/developer/ai/saccade || exit 1
TORCHLIB=$(.venv/bin/python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
NVLIBS=$(.venv/bin/python -c "import glob,os,nvidia; base=os.path.dirname(nvidia.__file__); print(':'.join(sorted(set(os.path.dirname(p) for p in glob.glob(base+'/*/lib/*.so*')))))")
export LD_LIBRARY_PATH="$TORCHLIB:$NVLIBS:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/build:$PYTHONPATH"
exec "$@"
