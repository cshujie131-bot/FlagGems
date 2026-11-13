# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetaX FP32 full-vocabulary sort with native generic fallback.

The histogram is the generic FG histogram with only int64 -> int32 local
accumulation. Conversion and stable sweep are reused from generic FG sort.
"""

import hashlib
import inspect
import logging
from pathlib import Path

import torch
import triton
import triton.language as tl

from flag_gems.ops.sort import convert_to_uint_preverse_order
from flag_gems.ops.sort import sort as _generic_sort
from flag_gems.ops.sort import sort_stable as _generic_sort_stable
from flag_gems.ops.sort import sweep
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


def supports_candidate(inp, dim=-1, descending=False):
    """Host metadata only; no tensor values, device query or synchronization."""
    return (
        inp.dtype == torch.float32
        and inp.ndim == 2
        and inp.is_cuda
        and inp.is_contiguous()
        and 1 <= inp.shape[0] <= 64
        and inp.shape[1] == 248320
        and dim in (-1, 1)
        and descending is False
    )


@triton.jit
def compute_global_hist_int32_kernel(
    arr_ptr,
    out_ptr,
    num_passes,
    m,
    n,
    tiles_n_per_cta,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    num_bits_per_pass: tl.constexpr,
    descending: tl.constexpr,
):
    # Minimal upstream compute_global_hist_kernel; only accumulator dtype differs.
    pid = tl.program_id(0)
    pid_n = pid // m
    pid_m = pid % m
    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)
    for p in range(0, num_passes):
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R):
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int32)
            for n_start in range(cta_n_start, cta_n_end, TILE_N):
                n_offsets = n_start + tl.arange(0, TILE_N)
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask)
                arr = convert_to_uint_preverse_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask
                matches = tl.where(mask, (bin_indices[:, None] == key), False)
                acc += matches
            local_sum = tl.sum(acc, axis=1)
            tl.atomic_add(
                out_ptr + pid_m * num_passes * r + p * r + bin_indices,
                local_sum,
                sem="relaxed",
            )


def _radix_sort_fp32(inp):
    """Selected hist32/k2 full sort. Both native entrypoints call this global."""
    m, n = inp.shape
    num_bins, num_passes = 4, 16
    with torch_device_fn.device(inp.device):
        histogram = torch.zeros(
            (m, num_passes, num_bins), device=inp.device, dtype=torch.int32
        )
        compute_global_hist_int32_kernel[(m * triton.cdiv(n, 1024 * 8), 1, 1)](
            inp,
            histogram,
            num_passes,
            m,
            n,
            8,
            1024,
            4,
            2,
            False,
            num_warps=4,
        )
        offsets = (torch.cumsum(histogram, -1) - histogram).to(torch.uint32)
        values_in = torch.clone(inp)
        indices_in = (
            torch.arange(n, dtype=torch.int64, device=inp.device)
            .broadcast_to(inp.shape)
            .contiguous()
        )
        values_out = torch.empty_like(inp)
        indices_out = torch.empty_like(indices_in)
        grid_n = triton.cdiv(n, 2048)
        status = torch.empty(
            (m, num_bins, grid_n), device=inp.device, dtype=torch.uint32
        )
        for index in range(num_passes):
            status.zero_()
            sweep[(m * grid_n, 1)](
                values_in,
                indices_in,
                values_out,
                indices_out,
                offsets,
                status,
                num_passes,
                index,
                index * 2,
                m,
                n,
                grid_n,
                2048,
                4,
                2,
                False,
                num_warps=4,
            )
            values_in, values_out = values_out, values_in
            indices_in, indices_out = indices_out, indices_in
    return values_in, indices_in


def sort(inp, dim=-1, descending=False):
    if supports_candidate(inp, dim, descending):
        logger.debug("GEMS_METAX SORT_K2_HIST32")
        return _radix_sort_fp32(inp)
    return _generic_sort(inp, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    # Generic FG radix is stable for either stable flag; retain that behavior.
    if supports_candidate(inp, dim, descending):
        logger.debug("GEMS_METAX SORT_STABLE_K2_HIST32")
        return _radix_sort_fp32(inp)
    return _generic_sort_stable(inp, stable=stable, dim=dim, descending=descending)


def runtime_info():
    """Read-only audit metadata; never called by sort or its hot-path guard."""
    source = Path(__file__).resolve()
    generic = Path(inspect.getsourcefile(_generic_sort_stable)).resolve()
    helper = inspect.unwrap(_radix_sort_fp32)
    return {
        "candidate": "hist32_k2",
        "provider_module": __name__,
        "provider_source": str(source),
        "provider_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generic_source": str(generic),
        "generic_source_sha256": hashlib.sha256(generic.read_bytes()).hexdigest(),
        "entrypoints": {"sort": sort.__module__, "sort.stable": sort_stable.__module__},
        "helper_symbol": "_radix_sort_fp32",
        "helper_original_module": helper.__module__,
        "scope_guard_symbol": "supports_candidate",
        "scope": {
            "dtype": "float32",
            "ndim": 2,
            "contiguous": True,
            "device_type": "cuda",
            "batch_min": 1,
            "batch_max": 64,
            "width": 248320,
            "dims": [-1, 1],
            "last_dim": True,
            "ascending": True,
            "descending": False,
        },
        "effective_config": {
            "k_bits": 2,
            "hist_acc_bits": 32,
            "hist_tile_n": 1024,
            "hist_tiles_per_cta": 8,
            "hist_tile_r": 4,
            "hist_num_warps": 4,
            "sweep_tile_n": 2048,
            "sweep_tile_r": 4,
            "sweep_num_warps": 4,
            "num_bins": 4,
            "num_passes": 16,
        },
        "fallback": "Unmodified flag_gems.ops.sort sort/sort_stable with original arguments",
        "tensor_values_read": False,
        "audit_counter_in_operator": False,
        "sampling_or_rng_changed": False,
    }
