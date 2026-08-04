
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
