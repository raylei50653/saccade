# #465 compiled-head block-scope (B) Inductor mechanism declaration

Status: **declaration, draft (not frozen); diagnostic only; no measurement
authorized by this document** (formal runs need a separate runner-PR freeze, §8)

## 0. Question

The compiled-head reverse-engineering result
([result](native_runtime_compiled_head_reverse_engineering_result.md), under the
[declaration](native_runtime_compiled_head_reverse_engineering_declaration.md)
blob `565b4a1d`) left the block scope at:

> B = `UNRESOLVED` (first divergence at AOT, not sufficient): D_B = E on all
> 5,316 frames; A_B ≠ E and A_B ∉ class(C_B).

It also recorded, descriptively, that C_B ∈ class(C).

This declaration asks the owner's question (09-29), verbatim:

> Which post-Dynamo/AOT-to-Inductor mechanism introduces the additional drift
> that makes C_B enter the C-like drift class, given that A_B already differs
> from E but remains outside class(C_B)?

It does **not** presume that A_B's drift is the precursor of C_B's drift, or
that C_B's drift grows out of A_B's. A_B is used only as a validity anchor and
as a descriptive landmark (§4, §6.4). Every decision in §6 is taken against the
reference R = C_B.

The study is explanatory. It does not reopen the PR-2L parity verdict, does not
change #465's runtime-route status, and does not touch the head scope (H).

## 1. Non-goals

This study does not:

- change the detector, tracker, preset, thresholds, model weights, or benchmark claims;
- optimize performance or select a production compile configuration;
- study the head scope. H stays `INDUCTOR_LOWERING_BOUNDARY`; its §6 stage-2
  remains eligible under the earlier declaration and is not started here;
- re-derive the scope cut or the D_B / A_B ladder; those are carried in (§2);
- explain the L2 whole-graph arm `A_C` or any end-to-end MOT difference;
- test backend-state / execution-policy dependence (cuDNN / TF32 / benchmark
  flags are held fixed, §4);
- name a causal operator or kernel. A mechanism label (§3) is a compiler
  configuration class, not an operator (§7);
- attribute a cause from a numerical correlation alone.

Any production change discovered from this work requires a separate PR.

## 2. Evidence carried in

All numbers come from the formal RE packet
`results/compiled_head_reverse_engineering_465/20260929T115258Z/`
(`packet.json` sha256
`3aa775e41ea8849bf651323ed5ebf184609d9b45a67ef59712feac750be151c8`), run at the
r2 execution freeze `51e34c39` (tag `freeze/465-compiled-head-re-r2`). They
motivate this study; they are not new measurements.

1. Validity of that run held (V1 24/24, V-anchor exact, repeat identity 140
   frames, 0 input mutations, 0 non-finite, Dynamo counters unchanged).
2. Block wrap sites: 3 (`block_layout = [1, 1, 1]`), each a `MambaBlock`
   wrapped individually. The executed R1 graph at every block site is compile
   id `0/1` (the dynamic-length recompile), compiled by arm C and reused by C_B.
3. **A_B and C_B lower the same captured graph.** For both compile ids, the
   Dynamo output graph and the AOT inference graph fingerprints of A_B and C_B
   are identical:

   | compile id | `dynamo_output_graph` | `aot_inference_graph` | C_B `inductor_output_code` |
   | --- | --- | --- | --- |
   | `0/0` | `0c1be424b444a717b3389ef3de118e9616e56e37074ea20b0a59e63a8383942f` | `673b49a05b42ee12bbd29f4e99a16fb4a9274d865b2982e071ad22d96c5b0aab` | `2942d1fdca1d9430e9429af7edb771bc8516106a53d885705305a969029a4dea` |
   | `0/1` | `f22dbab96fa916b6d0fb1fe2401cefd0bf75790a0e2392d80b9b8a84e840f2e5` | `9b1e8ef473c0c987e108ee93f1e20613af9595c08642ef57227c09d9948dfe9c` | `ca0c0c388ca22c2ef80b41c5c5266ba81af2639f7d96b110374a43e8a5612fb4` |

   So every difference between A_B and C_B is introduced **after** the AOT
   inference graph, inside Inductor (post-grad / joint-graph passes, lowering,
   scheduling, code generation, and the compiled-kernel options). This is the
   interval this declaration studies.
4. The shared AOT graph already contains the decompositions that A_B executes
   with eager ATen kernels: each `nn.Linear` on a 3-D input as `expand` + `bmm`
   (the output projection as a 2-D `mm`), `F.silu` as `x / (1 + exp(-x))`,
   `dt_proj` as `bmm` + separate bias `add`, and the depthwise `conv1d` as one
   `aten.convolution` **with** its bias. `selective_scan_fwd` is an opaque
   custom op.
5. C_B's generated code (compile id `0/1`), read from R0 only, differs from the
   AOT graph as follows:
   - `bmm` / `mm` stay extern kernels (`extern_kernels.bmm` / `.mm`);
   - the convolution is an extern call with `bias=None`, fed a Triton-materialized
     contiguous input; its bias is added inside a Triton pointwise kernel;
   - both `silu` sites run as Triton pointwise kernels
     (`libdevice.exp`, then `(a / b)` with Triton's default fp32 division);
   - `-exp(A_log)` and the `dt` bias add run as Triton pointwise kernels;
   - Inductor counters: `pattern_matcher_count = 10`, `pad_mm_bench = 2`,
     `extern_calls = 14`. The `mm`/`bmm` shapes in the code are unpadded
     (for example the `x_proj` output keeps width 33).
6. Descriptive drift vs E (score entries / box-coordinate entries over the fixed
   E-mask / crossing support |K|):
   - C_B: 571,362,101 / 545,379 / 5; score max-abs 1.3653e-3, box max-abs 0.4657 px;
   - A_B: 46,223,554 / 92,921 / 0; score max-abs 1.3677e-3, box max-abs 0.2423 px;
   - |D_{A_B} ∩ D_{C_B}|: 36,170,503 score entries, 65,090 box entries.

   No decision here depends on the ratio between these numbers.
7. C_B ∈ class(C) (score recall 0.987 / precision 0.9995, same max-abs, K = the
   same 5 crossings). R = C_B therefore stands for the C-like class in the
   owner's question.

## 2.1 Toolchain facts the design relies on

Read from the installed PyTorch `2.11.0+cu130` source, not measured:

- `torch.compile` refuses `mode` and `options` together. The PR-2L construction
  used `mode="default"`, which is the same as passing neither.
- `torch._inductor.config.eager_numerics.division_rounding` (default `False`)
  makes Triton fp32 `x / y` codegen use `div_rn` instead of the approximate
  default division.
- `torch._inductor.config.eager_numerics.disable_ftz` (default `False`) is
  passed to Triton as `disable_ftz`.
- `torch._inductor.config.emulate_precision_casts` (default `False`) sets the
  Triton compile option `enable_fp_fusion = not emulate_precision_casts`; its
  other effects concern fp16/bf16 casts and a few decomposition choices.
- `torch._inductor.config.pattern_matcher` (default `True`) gates the
  joint-graph and post-grad pattern passes. `pad_mm` is registered inside the
  joint-graph pattern pass, so `pattern_matcher=False` also disables it.
- `torch._inductor.config.fallback_by_default` (default `False`) makes Inductor
  fall back to ATen for every node that is not explicitly annotated for
  regional compile.

If the runner PR finds any of these facts false for the pinned toolchain, the
affected arm is omitted before measurement (§5.3), and the rows that need it
become unreachable.

## 3. Mechanism classes

This study separates C_B's post-AOT interval into four **declared mechanism
classes**, each switched by one Inductor configuration flag, plus one coarse
coverage cut. The list is closed; nothing may be added after reading data.

| Id | Mechanism class | Knock-out setting (vs C_B) |
| --- | --- | --- |
| `div` | approximate fp32 division in generated Triton code | `eager_numerics.division_rounding = True` |
| `fma` | floating-point contraction (FMA) when compiling generated Triton code | `emulate_precision_casts = True` (admissible only if §5.3 G2 holds) |
| `ftz` | flush-to-zero in generated Triton code | `eager_numerics.disable_ftz = True` |
| `pm` | joint-graph and post-grad pattern rewrites, including `pad_mm` | `pattern_matcher = False` |

Coverage cut `fb`: `fallback_by_default = True`, which keeps the Inductor
pipeline but executes every node as an ATen fallback (no generated Triton
pointwise code). It is not one mechanism. It separates "the class-defining
drift needs Inductor-generated kernels" from "it survives with ATen kernels
under Inductor's graph passes and wrapper".

Mechanisms outside the four classes (for example `libdevice.exp` vs the eager
`exp` kernel, the conv-bias split, the materialized conv input layout, or buffer
reuse) are **not** separately testable here. If they carry the class, the result
is `UNCOVERED_MECHANISM` (§6.2), with the `fb` cut as a qualifier.

## 4. Arms

Scope: block sites only. Head sites stay eager in every arm. Every compiled arm
wraps exactly the three block sites of §2 item 2, one `torch.compile` call per
`MambaBlock`, with backend `inductor`, and the configuration shown. Every arm is
built from its own `copy.deepcopy` of the same pre-compile head, as in the
earlier declaration §4. All arms are run through `_forward_eager(feats,
return_embeddings=False)` on the same 7 sequences / 5,316 frames and the same
backbone features as PR-2L L1.

Backend policy is fixed for all arms at the PR-2L L1 values:
`cudnn.benchmark=False`, `cudnn.allow_tf32=True`,
`cuda.matmul.allow_tf32=False`, graph-executor optimize off.

Let M = {`div`, `fma`, `ftz`, `pm`}.

| Arm | Construction |
| --- | --- |
| E | eager (reference for all counts) |
| A_B | block sites `aot_eager_decomp_partition` (validity anchor, descriptive landmark) |
| C_B | block sites `inductor`, default configuration (reference R) |
| K_x, x ∈ M | C_B with only mechanism x knocked out (4 arms) |
| K_all | C_B with all four mechanisms knocked out at once |
| J_x, x ∈ M | K_all with only mechanism x restored (4 arms) |
| K_fb | C_B with `fallback_by_default = True` (coverage cut) |

13 arms in total. No other arm may be added after the runner freeze.

A configuration may be applied through `torch.compile(..., options=...)`, a
scoped `torch._inductor.config.patch`, or an equivalent that the runner PR
names. Whatever route is used, §5.2 and §5.3 must hold for it.

**Per arm X, record against E and against R = C_B**, exactly as in the earlier
declaration §4: `n_score(X)`, `n_box(X)` over the fixed E-mask, the difference
sets D_X and their overlaps with D_R, score and box max-abs, crossing support
K_X and |K_X ∩ K_R|, non-finite count, bit-identity vs E and vs R per frame,
compile graph / code identity, and repeat identity. In addition, record
bit-identity vs A_B per frame and whether X ∈ class(A_B) (descriptive, §6.4).

## 5. Validity

### 5.1 Carried-in anchors (V-anchor-B)

In every process that runs frames:

- E must be bit-identical to E in every other process, frame by frame;
- C_B vs E must reproduce the packet's `r1.arms.C_B` record exactly, and A_B
  vs E the `r1.arms.A_B` record: score and box counts and max-abs, crossing
  count, non-finite count, and `equal_E_frames` (§2 item 6). These aggregate
  records are what the packet persists for these arms;
- the pair A_B | C_B must reproduce the packet's `r1.pairs["A_B|C_B"]` record
  exactly (36,170,503 score entries, 65,090 box entries, 0 crossings,
  `equal_R_frames` 0).

Any mismatch makes the run `UNRESOLVED`: the drift being decomposed would not
be the drift the earlier result recorded.

### 5.2 Code anchor

C_B's executed graph at every block site must have the Dynamo, AOT and Inductor
output-code fingerprints of §2 item 3 (compile id `0/1`), and its isolated R0
compile must reproduce both rows. This covers any benchmark-driven choice inside
Inductor (for example the two `pad_mm` benchmarks): if C_B compiles to different
code, the run is `UNRESOLVED`.

### 5.3 Knob-applied gates (R0, before any frame is read)

For each non-reference compiled arm, the isolated R0 compile must show, per
compile id:

- **G1 (pre-Inductor identity).** The Dynamo output graph and AOT inference
  graph fingerprints equal C_B's (§2 item 3). A knob that changes the captured
  or AOT graph is not a post-AOT mechanism, and the arm is omitted.
- **G2 (declared effect only).**
  - `K_div` / `J_div`: the generated source differs from its base arm only in
    fp32 division expressions (`div_rn` vs `/`); the compile options are equal.
  - `K_fma` / `J_fma`: the generated source text is identical to its base arm;
    the Triton compile options differ only in `enable_fp_fusion`.
  - `K_ftz` / `J_ftz`: the generated source text is identical to its base arm;
    the Triton compile options differ only in `disable_ftz`.
  - `K_pm` / `J_pm`: the recorded post-grad graph or generated code shows the
    pattern-pass difference. The runner records which patterns matched in C_B
    and whether any `pad_mm` rewrite was applied.
  - `K_all`: the effects of all four, and nothing else.
  - `K_fb`: no generated Triton kernel remains in the output code.

  The base arm of K_x is C_B; the base arm of J_x is K_all.
- **G3 (engagement).** If an arm's generated source and compile options are
  identical to its base arm's, the mechanism is **not engaged** at these sites.
  The arm still runs, and it must then be bit-identical to its base arm on every
  frame (otherwise `UNRESOLVED`: something outside the generated code differs).
  A not-engaged mechanism counts as "K_x ≡ C_B" in §6.

An arm that fails G1 or G2 is **omitted before measurement**, and that is
recorded with the reason. It is never replaced by an approximate arm. An omitted
K_x or K_all makes every §6.2 row that needs it unreachable, and the result falls
through to `UNRESOLVED`. An omitted J_x only removes x from the sufficiency
readout (§6.2 rows 4 and 6), which must then say so.

### 5.4 Execution gates

Carried over from the earlier declaration §7 and its runner: repeat identity for
every arm; 0 input mutations; 0 non-finite; Dynamo counters unchanged across the
frame pass; recompile limit never hit; every compiled call attributed to a graph
whose fingerprint equals the arm's isolated R0 compile; one fixed compile-cache
policy, stated in the runner PR. R1 may be split across several processes (for
example to stay under the recompile limit); §5.1 then applies to each process.

## 6. Decision rules

### 6.1 Drift class

The class predicate is the earlier declaration's §5.1, **unchanged**: overlap
recall and precision ≥ 0.9 and max-abs within `[0.5, 2] ×` for both scores and
boxes, and crossing-support overlap ≥ 0.9 both ways for scores (with `|K_R| = 5`,
this requires `K_X = K_R` exactly). X ≡ R, X = E and "partial" are as defined
there. The thresholds are fixed; they must not be re-chosen after reading data.

For this study, R = C_B. "x is **class-necessary**" means K_x ∉ class(C_B).
"x is **class-sufficient over K_all**" means J_x ∈ class(C_B).

### 6.2 Mechanism result (reference R = C_B)

The first satisfied row is the result:

| # | Observation | Result |
| --- | --- | --- |
| 1 | a validity failure (§5.1, §5.2, §5.3 G3, §5.4), or a required K arm omitted | `UNRESOLVED` |
| 2 | K_all ∈ class(C_B) | `UNCOVERED_MECHANISM` (qualifier from §6.3) |
| 3 | exactly one x ∈ M is class-necessary, and J_x ∈ class(C_B) | `SINGLE_MECHANISM(x)` |
| 4 | exactly one x ∈ M is class-necessary, and J_x ∉ class(C_B) | `NECESSARY_NOT_SUFFICIENT(x)` |
| 5 | two or more x ∈ M are class-necessary | `MULTIPLE_NECESSARY(S_nec)`, S_nec = the class-necessary set; each J_x reported |
| 6 | no x ∈ M is class-necessary (K_all ∉ class(C_B) by row 2) | `REDUNDANT_MECHANISMS(S_suf)`, S_suf = {x : J_x ∈ class(C_B)}; if S_suf is empty, `COMBINATION_ONLY` |

Rows 2–6 are exhaustive once row 1 does not hold. There is no silent
fallthrough. When J_x was omitted (§5.3), row 3 cannot hold for that x; the
result is then row 4's label with the qualifier "sufficiency not measured", and
it must not be read as a negative sufficiency result.

### 6.3 Coverage cut (K_fb)

Reported with every non-`UNRESOLVED` result, and required as the qualifier of
row 2:

| Observation | Coverage label |
| --- | --- |
| K_fb omitted (§5.3) | `COVERAGE_NOT_MEASURED` |
| K_fb ∈ class(C_B) | `SURVIVES_ATEN_FALLBACK`: the class-defining drift is present without Inductor-generated kernels |
| K_fb ∉ class(C_B) | `NEEDS_GENERATED_KERNELS`: removing Inductor-generated kernels takes X out of class(C_B) |

Neither label names a mechanism. `SURVIVES_ATEN_FALLBACK` together with
`UNCOVERED_MECHANISM` points to graph passes, wrapper, or extern-call layout;
`NEEDS_GENERATED_KERNELS` together with `UNCOVERED_MECHANISM` points to
generated-kernel numerics outside `div` / `fma` / `ftz`. These pointers are
readings for the next declaration, not results.

### 6.4 Descriptive only (no row depends on these)

- for every arm: X = E, X = A_B, X ∈ class(A_B), X ≡ C_B, X ∈ class(C);
- K_all = A_B bit-for-bit, if it holds (the four classes then account for every
  difference between A_B and C_B at these sites, not only for class membership);
- per-site readouts are not collected: all three block sites are switched
  together in every arm.

## 7. What a result may and may not say

- A mechanism label says that, under the fixed backend policy and toolchain,
  switching that one Inductor flag at the three block sites moves the head
  output into or out of class(C_B). It does not say which kernel, which of the
  two `silu` sites, or which block site carries it.
- `SINGLE_MECHANISM(x)` is "necessary (single knock-out) and sufficient over
  the K_all base". It is not "the only source of C_B's drift": the other
  classes may still change bits without changing class membership.
- Nothing here says that A_B's drift grows into C_B's.
- A later operator-level localization must, as in the earlier declaration §6,
  distinguish the first observed divergent boundary from a causal
  operator/kernel, and needs its own PR. It is allowed only after this study
  returns a result other than `UNRESOLVED`.

## 8. Requirements for the runner PR

The runner PR must be independently reviewable and must, before any formal run:

- hard-bind this declaration's blob at its freeze commit, the RE packet sha256
  of §2, and the fingerprints of §2 item 3;
- pin the toolchain identity as the earlier runner did (PyTorch, Triton, CUDA
  runtime, driver, cuDNN, the single mapped `libcudart`);
- use one declared output root and one fixed compile-cache policy;
- verify a clean tree, the expected commit, and its own freeze tag through its
  own tag parser (the r1 lesson);
- implement §5.3 as an R0 step that reads no MOT17 frames and writes, per arm,
  the gate outcome and the omission reason if any;
- verify that all arms receive identical backbone features;
- include synthetic tests that read no MOT17 frames for the §6.1 predicate,
  every §6.2 row (including the omitted-J qualifier), every §6.3 label, and
  every §5.3 gate;
- declare the exact configuration route used per arm (§4).

No formal numerical measurement is authorized until this declaration and the
runner are frozen separately.

## 9. Expected deliverable

> Under the fixed PR-2L L1 backend policy, with head sites eager and the three
> block sites compiled, C_B's membership in its C-like drift class is
> **<§6.2 result>**, with coverage **<§6.3 label>**.

If the evidence cannot decide, report `UNRESOLVED`. Do not add arms.

## 10. Relationship to #465

PR-2L's frozen parity result remains authoritative for the runtime-route
decision. This is a diagnostic side branch of the compiled-head reverse
engineering study.

Refs #465.
