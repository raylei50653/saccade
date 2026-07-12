# R1 device bridge replay (research-only)

`r1_bridge_replay.cu` is a line-for-line copy of Consumer-A `bridge_vel4` /
`bridge_linres4` / `bridge_anchor4` from `src/tracking/tracker_gpu.cu`, exposed
as a tiny batch device API for host R0 replay.

Build (`.so` is gitignored):

```bash
bash scripts/tools/build_r1_bridge_replay.sh
```

Used by `saccade.perception.eval.consumer_a_bridge_fidelity` when
`libr1_bridge_replay.so` is present. Without it, a host float32+FMA fallback
runs; adaptive-anchor residuals may then exceed the sealed R1 `1e-5` budget.
