
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
