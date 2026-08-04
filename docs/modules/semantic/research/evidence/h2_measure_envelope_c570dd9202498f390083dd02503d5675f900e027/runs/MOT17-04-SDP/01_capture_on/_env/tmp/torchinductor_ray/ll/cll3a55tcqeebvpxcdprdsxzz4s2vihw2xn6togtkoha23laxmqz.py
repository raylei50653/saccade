
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
