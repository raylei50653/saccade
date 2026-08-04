# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/ray/h2_w6_formal_measurement_20260804T052122Z.incomplete/runs/MOT17-04-SDP/02_capture_on/_env/tmp/torchinductor_ray/pe/cpe43jk4zsfmful7brpyxel6fupvmn3kkv6zbmvoeida72v5xyb3.py
# Topologically Sorted Source Nodes: [flatten, flatten_1, flatten_2, cls_all, scores, max_1], Original ATen: [aten.view, aten.cat, aten.sigmoid, aten.max]
# Source node to ATen node mapping:
#   cls_all => cat
#   flatten => view
#   flatten_1 => view_1
#   flatten_2 => view_2
#   max_1 => max_1
#   scores => sigmoid
# Graph fragment:
#   %arg0_1 : Tensor "f32[1, 80, 80, 80][512000, 6400, 80, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[1, 80, 40, 40][128000, 1600, 40, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[1, 80, 20, 20][32000, 400, 20, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %view : Tensor "f32[1, 80, 6400][512000, 6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [1, 80, 6400]), kwargs = {})
#   %view_1 : Tensor "f32[1, 80, 1600][128000, 1600, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg1_1, [1, 80, 1600]), kwargs = {})
#   %view_2 : Tensor "f32[1, 80, 400][32000, 400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [1, 80, 400]), kwargs = {})
#   %cat : Tensor "f32[1, 80, 8400][672000, 8400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %view_1, %view_2], 2), kwargs = {})
#   %sigmoid : Tensor "f32[1, 80, 8400][672000, 8400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%cat,), kwargs = {})
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%sigmoid, 1), kwargs = {})
#   return %getitem_2,%getitem_3
triton_red_fused_cat_max_sigmoid_view_0 = async_compile.triton('triton_red_fused_cat_max_sigmoid_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_max_sigmoid_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': '0E9C14D2DCB14DC08B3936EF85F3E9F70CDD496F6AE3BEDCF3EAE2E95FA709DF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2889600, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_cat_max_sigmoid_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8400
    r0_numel = 80
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp21_index = tl.full([XBLOCK, R0_BLOCK], 2147483647, tl.int32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 6400, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (6400*r0_1 + (x0)), r0_mask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 8000, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tmp6 & tmp8
        tmp10 = tl.load(in_ptr1 + (1600*r0_1 + ((-6400) + x0)), r0_mask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 >= tmp7
        tmp12 = tl.full([1, 1], 8400, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tl.load(in_ptr2 + (400*r0_1 + ((-8000) + x0)), r0_mask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.where(tmp9, tmp10, tmp14)
        tmp16 = tl.where(tmp4, tmp5, tmp15)
        tmp17 = tl.sigmoid(tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = triton_helpers.maximum(_tmp19, tmp18)
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
        _tmp21_next, _tmp21_index_next = triton_helpers.maximum_with_index(
            _tmp21, _tmp21_index, tmp18, rindex
        )
        _tmp21 = tl.where(r0_mask & xmask, _tmp21_next, _tmp21)
        _tmp21_index = tl.where(r0_mask & xmask, _tmp21_index_next, _tmp21_index)
    tmp19 = triton_helpers.max2(_tmp19, 1)[:, None]
    tmp21_val, tmp21_idx = triton_helpers.max_with_index(_tmp21, _tmp21_index, 1)
    tmp21 = tmp21_idx[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tl.store(out_ptr1 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: /home/ray/h2_w6_formal_measurement_20260804T052122Z.incomplete/runs/MOT17-04-SDP/02_capture_on/_env/tmp/torchinductor_ray/ad/cadcp2xsjbinmceoyieyxuliusxg5vwjuyx4dzbexafey4i6t245.py
# Topologically Sorted Source Nodes: [flatten_3, flatten_4, flatten_5, reg_all, chunk, getattr_1, unsqueeze, x1y1, x2y2, add_1, wh], Original ATen: [aten.view, aten.cat, aten.split, aten.permute, aten.unsqueeze, aten.sub, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_1
#   chunk => split
#   flatten_3 => view_3
#   flatten_4 => view_4
#   flatten_5 => view_5
#   getattr_1 => permute
#   reg_all => cat_1
#   unsqueeze => unsqueeze
#   wh => sub_1
#   x1y1 => sub
#   x2y2 => add
# Graph fragment:
#   %arg6_1 : Tensor "f32[8400, 2][2, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg3_1 : Tensor "f32[1, 4, 80, 80][25600, 6400, 80, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[1, 4, 40, 40][6400, 1600, 40, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[1, 4, 20, 20][1600, 400, 20, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %view_3 : Tensor "f32[1, 4, 6400][25600, 6400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg3_1, [1, 4, 6400]), kwargs = {})
#   %view_4 : Tensor "f32[1, 4, 1600][6400, 1600, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg4_1, [1, 4, 1600]), kwargs = {})
#   %view_5 : Tensor "f32[1, 4, 400][1600, 400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg5_1, [1, 4, 400]), kwargs = {})
#   %cat_1 : Tensor "f32[1, 4, 8400][33600, 8400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_3, %view_4, %view_5], 2), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%cat_1, 2, 1), kwargs = {})
#   %permute : Tensor "f32[2, 8400][1, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg6_1, [1, 0]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 2, 8400][2, 1, 2]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 0), kwargs = {})
#   %sub : Tensor "f32[1, 2, 8400][2, 1, 2]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %getitem), kwargs = {})
#   %add : Tensor "f32[1, 2, 8400][2, 1, 2]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[1, 2, 8400][2, 1, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, %add), kwargs = {})
#   %sub_1 : Tensor "f32[1, 2, 8400][2, 1, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %sub), kwargs = {})
#   return %add_1,%sub_1
triton_poi_fused_add_cat_permute_split_sub_unsqueeze_view_1 = async_compile.triton('triton_poi_fused_add_cat_permute_split_sub_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_permute_split_sub_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0E9C14D2DCB14DC08B3936EF85F3E9F70CDD496F6AE3BEDCF3EAE2E95FA709DF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 336000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_permute_split_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2
    x0 = (xindex % 2)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 6400, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (6400*x0 + (x1)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp1 >= tmp4
    tmp8 = tl.full([1], 8000, tl.int64)
    tmp9 = tmp1 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tl.load(in_ptr2 + (1600*x0 + ((-6400) + x1)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp1 >= tmp8
    tmp13 = tl.full([1], 8400, tl.int64)
    tmp14 = tmp1 < tmp13
    tmp15 = tl.load(in_ptr3 + (400*x0 + ((-8000) + x1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp10, tmp11, tmp15)
    tmp17 = tl.where(tmp5, tmp6, tmp16)
    tmp18 = tmp0 - tmp17
    tmp19 = tl.load(in_ptr1 + (12800 + 6400*x0 + (x1)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr2 + (3200 + 1600*x0 + ((-6400) + x1)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr3 + (800 + 400*x0 + ((-8000) + x1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp10, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp19, tmp22)
    tmp24 = tmp0 + tmp23
    tmp25 = tmp18 + tmp24
    tmp26 = tmp24 - tmp18
    tl.store(out_ptr0 + (x2), tmp25, xmask)
    tl.store(out_ptr1 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: /home/ray/h2_w6_formal_measurement_20260804T052122Z.incomplete/runs/MOT17-04-SDP/02_capture_on/_env/tmp/torchinductor_ray/ll/cll3a55tcqeebvpxcdprdsxzz4s2vihw2xn6togtkoha23laxmqz.py
# Topologically Sorted Source Nodes: [c_xy, bboxes, squeeze, strides_t, bboxes_1, xywh, getitem_2, getitem_3, truediv_1, x1y1_1, getitem_4, getitem_5, truediv_2, x2y2_1], Original ATen: [aten.div, aten.cat, aten.squeeze, aten.unsqueeze, aten.mul, aten.permute, aten.slice, aten.sub, aten.add]
# Source node to ATen node mapping:
#   bboxes => cat_2
#   bboxes_1 => mul
#   c_xy => div
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   getitem_4 => slice_3
#   getitem_5 => slice_4
#   squeeze => squeeze
#   strides_t => unsqueeze_1
#   truediv_1 => div_1
#   truediv_2 => div_2
#   x1y1_1 => sub_2
#   x2y2_1 => add_2
#   xywh => permute_1
# Graph fragment:
#   %add_1 : Tensor "f32[1, 2, 8400][16800, 1, 2]cuda:0" = PlaceHolder[target=add_1]
#   %sub_1 : Tensor "f32[1, 2, 8400][16800, 1, 2]cuda:0" = PlaceHolder[target=sub_1]
#   %arg7_1 : Tensor "f32[8400, 1][1, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %div : Tensor "f32[1, 2, 8400][2, 1, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1, 2), kwargs = {})
#   %cat_2 : Tensor "f32[1, 4, 8400][33600, 8400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%div, %sub_1], 1), kwargs = {})
#   %squeeze : Tensor "f32[8400][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 8400][8400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze, 0), kwargs = {})
#   %mul : Tensor "f32[1, 4, 8400][33600, 8400, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_2, %unsqueeze_1), kwargs = {})
#   %permute_1 : Tensor "f32[1, 8400, 4][33600, 1, 8400]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %slice_1 : Tensor "f32[1, 8400, 2][33600, 1, 8400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_1, 2, 0, 2), kwargs = {})
#   %slice_2 : Tensor "f32[1, 8400, 2][33600, 1, 8400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_1, 2, 2, 4), kwargs = {})
#   %div_1 : Tensor "f32[1, 8400, 2][16800, 1, 8400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_2, 2), kwargs = {})
#   %sub_2 : Tensor "f32[1, 8400, 2][16800, 1, 8400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %div_1), kwargs = {})
#   %slice_3 : Tensor "f32[1, 8400, 2][33600, 1, 8400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_1, 2, 0, 2), kwargs = {})
#   %slice_4 : Tensor "f32[1, 8400, 2][33600, 1, 8400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_1, 2, 2, 4), kwargs = {})
#   %div_2 : Tensor "f32[1, 8400, 2][16800, 1, 8400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_4, 2), kwargs = {})
#   %add_2 : Tensor "f32[1, 8400, 2][16800, 1, 8400]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_3, %div_2), kwargs = {})
#   return %sub_2,%add_2
triton_poi_fused_add_cat_div_mul_permute_slice_squeeze_sub_unsqueeze_2 = async_compile.triton('triton_poi_fused_add_cat_div_mul_permute_slice_squeeze_sub_unsqueeze_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_div_mul_permute_slice_squeeze_sub_unsqueeze_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '0E9C14D2DCB14DC08B3936EF85F3E9F70CDD496F6AE3BEDCF3EAE2E95FA709DF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 537600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_div_mul_permute_slice_squeeze_sub_unsqueeze_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full([1], 0.5, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 4, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (2*x1 + ((-2) + x0)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = 2 + x0
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp3
    tmp20 = tl.load(in_ptr0 + (2*x1 + (2 + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full([1], 0.5, tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tmp17 >= tmp3
    tmp26 = tmp17 < tmp11
    tmp27 = tl.load(in_ptr1 + (2*x1 + (x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp19, tmp24, tmp27)
    tmp29 = tmp28 * tmp15
    tmp30 = tl.full([1], 0.5, tl.float32)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp16 - tmp31
    tmp33 = tmp16 + tmp31
    tl.store(out_ptr0 + (x0 + 4*x1), tmp32, xmask)
    tl.store(out_ptr1 + (x0 + 4*x1), tmp33, xmask)
''', device_str='cuda')


# kernel path: /home/ray/h2_w6_formal_measurement_20260804T052122Z.incomplete/runs/MOT17-04-SDP/02_capture_on/_env/tmp/torchinductor_ray/ir/cirqbcskuimbdhcem2lnqxyvvszd6hh5h65ncdpcf4ss4g4l6mti.py
# Topologically Sorted Source Nodes: [results, setitem, getitem_11, getitem_12, setitem_1, setitem_2, getitem_13, getitem_14, float_1], Original ATen: [aten.new_zeros, aten.select, aten.slice, aten.index, aten.copy, aten._to_copy]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   getitem_11 => select_1
#   getitem_12 => index
#   getitem_13 => select_13
#   getitem_14 => index_1
#   results => full_default
#   setitem => copy, select_2, slice_5
#   setitem_1 => copy_1, select_8, select_9
#   setitem_2 => copy_2, select_17, select_18
# Graph fragment:
#   %getitem_5 : Tensor "i64[300][1]cuda:0" = PlaceHolder[target=getitem_5]
#   %getitem_3 : Tensor "i64[1, 8400][8400, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem_4 : Tensor "f32[300][1]cuda:0" = PlaceHolder[target=getitem_4]
#   %cat_3 : Tensor "f32[1, 8400, 4][33600, 4, 1]cuda:0" = PlaceHolder[target=cat_3]
#   %full_default : Tensor "f32[1, 300, 6][1800, 6, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([1, 300, 6], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_2 : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%full_default, 0, 0), kwargs = {})
#   %slice_5 : Tensor "f32[300, 4][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%select_2, 1, 0, 4), kwargs = {})
#   %select_1 : Tensor "f32[8400, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cat_3, 0, 0), kwargs = {})
#   %index : Tensor "f32[300, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%select_1, [%getitem_5]), kwargs = {})
#   %copy : Tensor "f32[300, 4][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_5, %index), kwargs = {})
#   %select_int : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%full_default, 0, 0), kwargs = {})
#   %slice_scatter_default : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int, %copy, 1, 0, 4), kwargs = {})
#   %select_scatter_default : Tensor "f32[1, 300, 6][1800, 6, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %slice_scatter_default, 0, 0), kwargs = {})
#   %select_8 : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default, 0, 0), kwargs = {})
#   %select_9 : Tensor "f32[300][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_8, 1, 4), kwargs = {})
#   %copy_1 : Tensor "f32[300][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_9, %getitem_4), kwargs = {})
#   %select_int_1 : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default, 0, 0), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_1, %copy_1, 1, 4), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[1, 300, 6][1800, 6, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %select_scatter_default_1, 0, 0), kwargs = {})
#   %select_17 : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default_2, 0, 0), kwargs = {})
#   %select_18 : Tensor "f32[300][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_17, 1, 5), kwargs = {})
#   %select_13 : Tensor "i64[8400][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%getitem_3, 0, 0), kwargs = {})
#   %index_1 : Tensor "i64[300][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%select_13, [%getitem_5]), kwargs = {})
#   %convert_element_type : Tensor "f32[300][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_1, torch.float32), kwargs = {})
#   %copy_2 : Tensor "f32[300][6]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_18, %convert_element_type), kwargs = {})
#   %select_int_2 : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%select_scatter_default_2, 0, 0), kwargs = {})
#   %select_scatter_default_3 : Tensor "f32[300, 6][6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %copy_2, 1, 5), kwargs = {})
#   %select_scatter_default_4 : Tensor "f32[1, 300, 6][1800, 6, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %select_scatter_default_3, 0, 0), kwargs = {})
#   return %select_scatter_default_4
triton_poi_fused__to_copy_copy_index_new_zeros_select_slice_3 = async_compile.triton('triton_poi_fused__to_copy_copy_index_new_zeros_select_slice_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_index_new_zeros_select_slice_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '0E9C14D2DCB14DC08B3936EF85F3E9F70CDD496F6AE3BEDCF3EAE2E95FA709DF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 14400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_copy_index_new_zeros_select_slice_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tmp0 == tmp0
    tmp2 = x0
    tmp3 = tl.full([1], 5, tl.int32)
    tmp4 = tmp2 == tmp3
    tmp6 = tl.full([XBLOCK], 8400, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 8400)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 8400")
    tmp11 = tl.load(in_ptr1 + (tmp9), xmask, eviction_policy='evict_last')
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.full([1], 4, tl.int32)
    tmp14 = tmp2 == tmp13
    tmp16 = tl.full([1], 4, tl.int64)
    tmp17 = tmp2 < tmp16
    tmp18 = tl.load(in_ptr0 + (x1), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full([XBLOCK], 8400, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert(((0 <= tl.broadcast_to(tmp22, [XBLOCK])) & (tl.broadcast_to(tmp22, [XBLOCK]) < 8400)) | ~(tmp17 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp22, [XBLOCK]) < 8400")
    tmp24 = tl.load(in_ptr3 + (x0 + 4*tmp22), tmp17 & xmask, other=0.0)
    tmp25 = tl.full([1], 0.0, tl.float32)
    tmp26 = tl.where(tmp17, tmp24, tmp25)
    tmp27 = tl.where(tmp1, tmp26, tmp25)
    tmp28 = tl.where(tmp14, tmp15, tmp27)
    tmp29 = tl.where(tmp1, tmp28, tmp27)
    tmp30 = tl.where(tmp4, tmp12, tmp29)
    tmp31 = tl.where(tmp1, tmp30, tmp29)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 80, 80, 80), (512000, 6400, 80, 1))
        assert_size_stride(arg1_1, (1, 80, 40, 40), (128000, 1600, 40, 1))
        assert_size_stride(arg2_1, (1, 80, 20, 20), (32000, 400, 20, 1))
        assert_size_stride(arg3_1, (1, 4, 80, 80), (25600, 6400, 80, 1))
        assert_size_stride(arg4_1, (1, 4, 40, 40), (6400, 1600, 40, 1))
        assert_size_stride(arg5_1, (1, 4, 20, 20), (1600, 400, 20, 1))
        assert_size_stride(arg6_1, (8400, 2), (2, 1))
        assert_size_stride(arg7_1, (8400, 1), (1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 8400), (8416, 1), torch.float32)
            buf1 = empty_strided_cuda((1, 8400), (8400, 1), torch.int64)
            # Topologically Sorted Source Nodes: [flatten, flatten_1, flatten_2, cls_all, scores, max_1], Original ATen: [aten.view, aten.cat, aten.sigmoid, aten.max]
            stream0 = get_raw_stream(0)
            triton_red_fused_cat_max_sigmoid_view_0.run(arg0_1, arg1_1, arg2_1, buf0, buf1, 8400, 80, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            # Topologically Sorted Source Nodes: [getitem_8, topk], Original ATen: [aten.select, aten.topk]
            buf2 = torch.ops.aten.topk.default(reinterpret_tensor(buf0, (8400, ), (1, ), 0), 300)
            del buf0
            buf3 = buf2[0]
            assert_size_stride(buf3, (300, ), (1, ), 'torch.ops.aten.topk.default')
            assert_alignment(buf3, 16, 'torch.ops.aten.topk.default')
            buf4 = buf2[1]
            assert_size_stride(buf4, (300, ), (1, ), 'torch.ops.aten.topk.default')
            assert_alignment(buf4, 16, 'torch.ops.aten.topk.default')
            del buf2
            buf5 = empty_strided_cuda((1, 2, 8400), (16800, 1, 2), torch.float32)
            buf6 = empty_strided_cuda((1, 2, 8400), (16800, 1, 2), torch.float32)
            # Topologically Sorted Source Nodes: [flatten_3, flatten_4, flatten_5, reg_all, chunk, getattr_1, unsqueeze, x1y1, x2y2, add_1, wh], Original ATen: [aten.view, aten.cat, aten.split, aten.permute, aten.unsqueeze, aten.sub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_permute_split_sub_unsqueeze_view_1.run(arg6_1, arg3_1, arg4_1, arg5_1, buf5, buf6, 16800, stream=stream0)
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            buf9 = empty_strided_cuda((1, 8400, 4), (33600, 4, 1), torch.float32)
            buf7 = reinterpret_tensor(buf9, (1, 8400, 2), (33600, 4, 1), 0)  # alias
            buf8 = reinterpret_tensor(buf9, (1, 8400, 2), (33600, 4, 1), 2)  # alias
            # Topologically Sorted Source Nodes: [c_xy, bboxes, squeeze, strides_t, bboxes_1, xywh, getitem_2, getitem_3, truediv_1, x1y1_1, getitem_4, getitem_5, truediv_2, x2y2_1], Original ATen: [aten.div, aten.cat, aten.squeeze, aten.unsqueeze, aten.mul, aten.permute, aten.slice, aten.sub, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_div_mul_permute_slice_squeeze_sub_unsqueeze_2.run(buf5, buf6, arg7_1, buf7, buf8, 16800, stream=stream0)
            del arg7_1
            del buf5
            del buf6
            buf10 = empty_strided_cuda((1, 300, 6), (1800, 6, 1), torch.float32)
            # Topologically Sorted Source Nodes: [results, setitem, getitem_11, getitem_12, setitem_1, setitem_2, getitem_13, getitem_14, float_1], Original ATen: [aten.new_zeros, aten.select, aten.slice, aten.index, aten.copy, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_copy_index_new_zeros_select_slice_3.run(buf4, buf1, buf3, buf9, buf10, 1800, stream=stream0)
            del buf1
            del buf3
            del buf4
            del buf7
            del buf8
            del buf9
        return (buf10, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 80, 80, 80), (512000, 6400, 80, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 80, 40, 40), (128000, 1600, 40, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 80, 20, 20), (32000, 400, 20, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 4, 80, 80), (25600, 6400, 80, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 4, 40, 40), (6400, 1600, 40, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 4, 20, 20), (1600, 400, 20, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((8400, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((8400, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
