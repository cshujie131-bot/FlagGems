# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Boundary and changed-input coverage for the fixed MetaX index paths."""

import importlib

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name != "metax", reason="MetaX index paths"
)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
@pytest.mark.parametrize("full_mask", [False, True])
def test_masked_scalar_changed_input_graph(dtype, full_mask):
    put = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.index_put"
    ).index_put_
    x = torch.zeros(4, 17, device="cuda", dtype=dtype)
    mask = (
        torch.zeros_like(x, dtype=torch.bool)
        if full_mask
        else torch.zeros(4, device="cuda", dtype=torch.bool)
    )
    scalar = torch.tensor(1, device="cuda", dtype=dtype)
    mask[0] = True
    for _ in range(2):
        put(x, [mask], scalar)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = put(x, [mask], scalar)
    assert result.data_ptr() == x.data_ptr()
    x.zero_()
    mask.logical_not_()
    scalar.fill_(1 if dtype == torch.bool else 3)
    graph.replay()
    expected = torch.zeros_like(x)
    expected[mask] = scalar
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


def test_index_stride_product_above_int32():
    read = importlib.import_module("flag_gems.runtime.backend._metax.ops.index").index
    put = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.index_put"
    ).index_put_
    # Only two elements are initialized/read; the 3 GiB allocation crosses int32.
    storage = torch.empty(3, 2**30 + 1, device="cuda", dtype=torch.int8)
    x = storage[:, :1]
    x[0, 0], x[2, 0] = 11, 33
    ids = torch.tensor([2], device="cuda")
    torch.testing.assert_close(
        read(x, [ids]), torch.tensor([[33]], device="cuda", dtype=x.dtype)
    )
    put(x, [ids], torch.tensor([[55]], device="cuda", dtype=x.dtype))
    assert x[2, 0].item() == 55 and x[0, 0].item() == 11
