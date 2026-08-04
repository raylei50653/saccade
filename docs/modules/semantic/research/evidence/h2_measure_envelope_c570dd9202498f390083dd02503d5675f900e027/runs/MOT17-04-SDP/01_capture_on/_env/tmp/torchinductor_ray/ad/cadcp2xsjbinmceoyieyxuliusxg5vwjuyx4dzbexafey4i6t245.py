
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
