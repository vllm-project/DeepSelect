from builtins import sorted as builtins_sorted
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


MODES = [
    pytest.param(False, False, False, id="indices"),
    pytest.param(False, False, True, id="values"),
    pytest.param(False, True, False, id="sorted-indices"),
    pytest.param(False, True, True, id="sorted-indices-values"),
    pytest.param(True, False, True, id="sorted-values"),
]
DTYPES = [pytest.param(torch.float32, id="fp32"), pytest.param(torch.bfloat16, id="bf16")]
INDEX_DTYPES = [pytest.param(torch.int32, id="i32"), pytest.param(torch.int64, id="i64")]
TOPKS = [1, 7, 512, 1024, 4096]


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        pytest.skip("native SM120/SM121 tests require compute capability 12.0 or 12.1")
    return device


class NativeAPI:
    @staticmethod
    def get_stride_requirement():
        return torch.ops.deep_select.get_alignment_requirement()

    @staticmethod
    def topk(input, topk, sorted=False, begin=None, end=None, indices_type=torch.int64,
             sorted_index=False, output_idx=None, output_idx_offset=None,
             idx_oob_fill_value=2147483647, value_oob_fill_value=float("-inf"),
             return_value=True, abort_when_nan_found=True):
        values = (aligned_tensor(NativeAPI, input.shape[0], topk, input.dtype,
                                 input.device, output=True)[0] if return_value else None)
        if output_idx is None:
            output_idx = aligned_tensor(NativeAPI, input.shape[0], topk, indices_type,
                                        input.device, output=True)[0]
        result = torch.ops.deep_select.topk(
            input, topk, begin, end, sorted, sorted_index, values, output_idx,
            output_idx_offset, idx_oob_fill_value, value_oob_fill_value,
            return_value, abort_when_nan_found)
        assert result is None
        return values, output_idx


@pytest.fixture(scope="module", params=["wrapper", "native"])
def api(device, request):
    import deep_select

    return deep_select if request.param == "wrapper" else NativeAPI


def aligned_tensor(api, batch, width, dtype, device, output=False, offset=False):
    alignment = api.get_stride_requirement()[int(output)] // dtype.itemsize
    stride = max(alignment, (width + alignment - 1) // alignment * alignment)
    if offset:
        stride += 2 * alignment
    storage = torch.empty((batch + int(offset), stride), dtype=dtype, device=device)
    row = int(offset)
    column = alignment if offset else 0
    view = storage[row:row + batch, column:column + width]
    assert view.stride(0) * dtype.itemsize % api.get_stride_requirement()[int(output)] == 0
    return view, storage


def check_result(api, x, k, result, *, indices_type=torch.int64, return_value=True,
                 sorted=False, sorted_index=False, end=None, output_idx_offset=None,
                 idx_oob_fill_value=-1, value_oob_fill_value=float("-inf")):
    values, indices = result
    assert indices.shape == (x.shape[0], k)
    assert indices.dtype == indices_type
    assert indices.device == x.device
    assert indices.stride(1) == 1
    assert indices.stride(0) * indices.element_size() % api.get_stride_requirement()[1] == 0
    if return_value:
        assert values is not None
        assert values.shape == indices.shape and values.dtype == x.dtype
        assert values.device == x.device and values.stride(1) == 1
        assert values.stride(0) * values.element_size() % api.get_stride_requirement()[1] == 0
    else:
        assert values is None
    source = x.cpu()
    actual_indices = indices.cpu().to(torch.int64)
    actual_values = values.cpu() if values is not None else None
    ends = end.cpu().tolist() if end is not None else [x.shape[1]] * x.shape[0]
    offsets = output_idx_offset.cpu().tolist() if output_idx_offset is not None else [0] * x.shape[0]
    for row, (length, offset) in enumerate(zip(ends, offsets)):
        count = min(k, length)
        chosen = actual_indices[row, :count] - offset
        assert bool(((chosen >= 0) & (chosen < length)).all()), (row, chosen)
        assert chosen.unique().numel() == count, (row, "duplicate indices")
        assert bool((actual_indices[row, count:] == idx_oob_fill_value).all())
        gathered = source[row, chosen]
        if actual_values is not None:
            bits = torch.int32 if x.dtype == torch.float32 else torch.int16
            assert torch.equal(actual_values[row, :count].view(bits), gathered.view(bits))
            assert bool((actual_values[row, count:] == value_oob_fill_value).all())
        if count:
            reference = torch.topk(source[row, :length].float(), count).values
            torch.testing.assert_close(gathered.float().sort(descending=True).values,
                                       reference, rtol=0, atol=0)
        if sorted_index:
            assert bool((chosen[1:] > chosen[:-1]).all())
        if sorted:
            assert bool((gathered[:-1] >= gathered[1:]).all())


def run_checked(api, x, k, **kwargs):
    kwargs.setdefault("idx_oob_fill_value", -1)
    result = api.topk(x, k, **kwargs)
    check_result(api, x, k, result, **{key: value for key, value in kwargs.items()
                                     if key not in ("output_idx", "abort_when_nan_found")})
    return result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_zero_batch(api, device, dtype, indices_type, return_value):
    x, _ = aligned_tensor(api, 0, 65536, dtype, device)
    end = torch.empty(0, dtype=torch.int32, device=device)
    offsets = torch.empty(0, dtype=torch.int32, device=device)
    assert x.shape == (0, 65536)
    for metadata in [{}, dict(end=end, output_idx_offset=offsets)]:
        result = run_checked(api, x, 512, indices_type=indices_type,
                             return_value=return_value, **metadata)
        assert result[1].numel() == 0
        if return_value:
            assert result[0].numel() == 0
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("sorted,sorted_index,return_value", MODES)
@pytest.mark.parametrize("k", TOPKS)
@pytest.mark.parametrize("width_kind", ["empty", "short", "equal", "16384", "129280"])
def test_matrix(api, device, dtype, indices_type, sorted, sorted_index, return_value, k, width_kind):
    if dtype == torch.bfloat16 and sorted:
        pytest.skip("sorted values are FP32-only")
    widths = [0, max(0, k - 1), k, 16384, 129280]
    width_index = ["empty", "short", "equal", "16384", "129280"].index(width_kind)
    width = widths[width_index]
    batch = [1, 4, 16][(TOPKS.index(k) + width_index) % 3]
    x, _ = aligned_tensor(api, batch, width, dtype, device)
    generator = torch.Generator(device=device).manual_seed(1234 + k + width)
    x.copy_(torch.randn(x.shape, dtype=dtype, device=device, generator=generator))
    run_checked(api, x, k, indices_type=indices_type, sorted=sorted,
                sorted_index=sorted_index, return_value=return_value)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("sorted,sorted_index,return_value", MODES)
@pytest.mark.parametrize("k", TOPKS)
def test_end_offset_and_storage(api, device, dtype, indices_type, sorted, sorted_index, return_value, k):
    if dtype == torch.bfloat16 and sorted:
        pytest.skip("sorted values are FP32-only")
    width = 16384
    x, input_storage = aligned_tensor(api, 4, width, dtype, device, offset=True)
    input_storage.fill_(float("nan"))
    x.copy_(((torch.arange(width, device=device) * 71) % 997).to(dtype).expand_as(x))
    before = input_storage.clone()
    end_storage = torch.tensor([99, 0, k - 1, k, width], dtype=torch.int32, device=device)
    end = end_storage[1:]
    offset_storage = torch.tensor([99, 100, 200, -300, 400], dtype=torch.int32, device=device)
    offsets = offset_storage[1:]
    out, output_storage = aligned_tensor(api, 4, k, indices_type, device, output=True, offset=True)
    output_storage.fill_(-987654)
    assert not x.is_contiguous() and not out.is_contiguous()
    assert x.storage_offset() > 0 and out.storage_offset() > 0
    result = run_checked(api, x, k, indices_type=indices_type, sorted=sorted,
                         sorted_index=sorted_index, return_value=return_value, end=end,
                         output_idx_offset=offsets, output_idx=out)
    assert result[1] is out
    untouched = torch.ones_like(output_storage, dtype=torch.bool)
    alignment = api.get_stride_requirement()[1] // indices_type.itemsize
    untouched[1:, alignment:alignment + k] = False
    assert bool((output_storage[untouched] == -987654).all())
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(input_storage.view(bits), before.view(bits))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("pattern", ["zeros", "equal", "infinities", "negative-infinity"])
@pytest.mark.parametrize("sorted,sorted_index,return_value", MODES)
def test_ties_and_infinities(api, device, dtype, indices_type, pattern, sorted, sorted_index, return_value):
    if dtype == torch.bfloat16 and sorted:
        pytest.skip("sorted values are FP32-only")
    x, _ = aligned_tensor(api, 4, 16384, dtype, device)
    if pattern == "zeros":
        x.fill_(0.0)
        x[:, ::2] = -0.0
    elif pattern == "equal":
        x.fill_(3.5)
    elif pattern == "infinities":
        x.fill_(-float("inf"))
        x[:, ::3] = float("inf")
        x[:, 1::3] = 0.0
    else:
        x.fill_(-float("inf"))
    run_checked(api, x, 512, indices_type=indices_type, sorted=sorted,
                sorted_index=sorted_index, return_value=return_value)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_nan_nonabort_and_short_rows(api, device, dtype, indices_type, return_value):
    x, _ = aligned_tensor(api, 4, 16384, dtype, device)
    x.fill_(1)
    x[0, 1000] = float("nan")
    x[1, 0] = float("nan")
    x[2, 6] = float("nan")
    x[3, 1000] = float("nan")
    end = torch.tensor([16384, 3, 7, 100], dtype=torch.int32, device=device)
    offsets = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=device)
    values, indices = api.topk(x, 7, end=end, indices_type=indices_type,
                               output_idx_offset=offsets, return_value=return_value,
                               idx_oob_fill_value=-1, abort_when_nan_found=False)
    assert indices[0, 0].item() == 0x3F3F3F3F
    for row, count in [(1, 3), (2, 7)]:
        torch.testing.assert_close(indices[row, :count].cpu().to(torch.int64),
                                   torch.arange(count) + int(offsets[row]), rtol=0, atol=0)
        assert bool((indices[row, count:] == -1).all())
        if return_value:
            bits = torch.int32 if dtype == torch.float32 else torch.int16
            assert torch.equal(values[row, :count].view(bits), x[row, :count].view(bits))
            assert bool(torch.isneginf(values[row, count:]).all())
    check_result(api, x[3:], 7, (values[3:] if values is not None else None, indices[3:]),
                 indices_type=indices_type, return_value=return_value, end=end[3:],
                 output_idx_offset=offsets[3:])
    for row in [1, 2]:
        safe_values, safe_indices = api.topk(x[row:row + 1], 7, end=end[row:row + 1],
                                             return_value=return_value, indices_type=indices_type,
                                             idx_oob_fill_value=-1)
        torch.cuda.synchronize(device)
        assert safe_indices[0, 0].item() == 0
        if return_value:
            assert torch.isnan(safe_values[0]).any().item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("k", [7, 513, 2048, 2049])
def test_unsorted_i32_selection_tails(api, device, dtype, return_value, k):
    warp_end = (k // 32 + 1) * 32
    block_end = (k // 256 + 1) * 256
    lengths = builtins_sorted({0, k - 1, k, k + 1, warp_end - 1, warp_end,
                              warp_end + 1, block_end - 1, block_end, block_end + 1})
    width = max(lengths)
    x, storage = aligned_tensor(api, len(lengths), width, dtype, device, offset=True)
    storage.fill_(float("nan"))
    for row, length in enumerate(lengths):
        x[row, :length].copy_(((torch.arange(length, device=device) * 71) % 997).to(dtype))
    end = torch.tensor(lengths, dtype=torch.int32, device=device)
    offsets = torch.arange(len(lengths), dtype=torch.int32, device=device) * 100 - 300
    out, output_storage = aligned_tensor(api, len(lengths), k, torch.int32, device,
                                         output=True, offset=True)
    output_storage.fill_(-987654)
    result = run_checked(api, x, k, end=end, output_idx_offset=offsets, output_idx=out,
                         indices_type=torch.int32, return_value=return_value,
                         value_oob_fill_value=12345.0)
    assert result[1] is out
    untouched = torch.ones_like(output_storage, dtype=torch.bool)
    alignment = api.get_stride_requirement()[1] // torch.int32.itemsize
    untouched[1:, alignment:alignment + k] = False
    assert bool((output_storage[untouched] == -987654).all())


@pytest.mark.parametrize("dtype,raw", [
    pytest.param(torch.float32, [0xC0A00000, 0x40400000, 0x41200000], id="fp32"),
    pytest.param(torch.bfloat16, [0xC0A0, 0x4040, 0x4120], id="bf16"),
    pytest.param(torch.bfloat16, [0xBF81, 0xBF80, 0xBF7F], id="bf16-negative-finite"),
    pytest.param(torch.bfloat16, [0x8003, 0x8002, 0x8001], id="bf16-negative-subnormal"),
])
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("k,width", [(511, 4097), (513, 4097), (2048, 8193)])
def test_unsorted_i32_cross_warp_threshold_quota(api, device, dtype, raw, return_value, k, width):
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    sign = 1 << (dtype.itemsize * 8 - 1)
    signed = [value - 2 * sign if value & sign else value for value in raw]
    greater = [index for index in range(width) if index % 37 == 0]
    equal = [index for index in range(width) if index % 3 == 1 and index % 37 != 0]
    quota = k - len(greater)
    assert 0 < quota < len(equal)
    chunk = ((width + 255) // 256) * 32
    assert equal[quota - 1] // chunk >= 2
    expected = builtins_sorted(greater + equal[:quota])
    x, _ = aligned_tensor(api, 1, width, dtype, device)
    x.view(bits).fill_(signed[0])
    x.view(bits)[0, equal] = signed[1]
    x.view(bits)[0, greater] = signed[2]
    for _ in range(3):
        values, indices = run_checked(api, x, k, indices_type=torch.int32,
                                      return_value=return_value)
        actual = indices[0].cpu().tolist()
        assert builtins_sorted(actual) == expected
        if return_value:
            expected_bits = [signed[2] if index % 37 == 0 else signed[1] for index in actual]
            assert values[0].view(bits).cpu().tolist() == expected_bits


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("k", [513, 2048])
def test_unsorted_i32_nan_at_end(api, device, dtype, return_value, k):
    x, _ = aligned_tensor(api, 4, k + 2, dtype, device)
    x.fill_(1)
    lengths = [k + 1, k + 1, k - 1, k]
    for row, nan_index in enumerate([k, k + 1, k - 2, k - 1]):
        x[row, nan_index] = float("nan")
    end = torch.tensor(lengths, dtype=torch.int32, device=device)
    offsets = torch.tensor([100, -200, 300, -400], dtype=torch.int32, device=device)
    values, indices = api.topk(x, k, end=end, output_idx_offset=offsets,
                               indices_type=torch.int32, return_value=return_value,
                               idx_oob_fill_value=-1, abort_when_nan_found=False)
    assert indices.dtype == torch.int32
    assert indices[0, 0].item() == 0x3F3F3F3F
    check_result(api, x[1:2], k, (values[1:2] if values is not None else None, indices[1:2]),
                 indices_type=torch.int32, return_value=return_value, end=end[1:2],
                 output_idx_offset=offsets[1:2])
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    for row in [2, 3]:
        length = lengths[row]
        offset = offsets[row].item()
        assert indices[row, :length].cpu().tolist() == list(range(offset, offset + length))
        assert bool((indices[row, length:] == -1).all())
        if return_value:
            assert torch.equal(values[row, :length].view(bits), x[row, :length].view(bits))
            assert bool(torch.isneginf(values[row, length:]).all())
    if not return_value:
        assert values is None
    safe = api.topk(x[1:], k, end=end[1:], output_idx_offset=offsets[1:],
                    indices_type=torch.int32, return_value=return_value, idx_oob_fill_value=-1)
    torch.cuda.synchronize(device)
    check_result(api, x[1:2], k, (safe[0][:1] if return_value else None, safe[1][:1]),
                 indices_type=torch.int32, return_value=return_value, end=end[1:2],
                 output_idx_offset=offsets[1:2])
    assert torch.equal(safe[1][1:], indices[2:])
    if return_value:
        assert torch.equal(safe[0][1:].view(bits), values[2:].view(bits))
    else:
        assert safe[0] is None


def run_checked_low_index(api, x, k, **kwargs):
    result = run_checked(api, x, k, **kwargs)
    bits = torch.int32 if x.dtype == torch.float32 else torch.int16
    sign = 1 << (x.element_size() * 8 - 1)
    mask = 2 * sign - 1
    source = x.view(bits).cpu().tolist()
    ends = kwargs["end"].cpu().tolist() if "end" in kwargs else [x.shape[1]] * x.shape[0]
    offsets = kwargs["output_idx_offset"].cpu().tolist() if "output_idx_offset" in kwargs else [0] * x.shape[0]
    for row, (length, offset) in enumerate(zip(ends, offsets)):
        def key(index):
            value = source[row][index] & mask
            if value in (0, sign):
                value = 0
            return (~value & mask) if value & sign else value ^ sign
        count = min(k, length)
        expected = builtins_sorted(range(length), key=lambda index: (-key(index), index))[:count]
        actual = (result[1][row, :count].cpu().to(torch.int64) - offset).tolist()
        assert builtins_sorted(actual) == builtins_sorted(expected), row
    return result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("width", [2048, 2049, 4097, 131072, 131073])
def test_streaming_k512_end_boundaries(api, device, dtype, return_value, width):
    lengths = builtins_sorted({0, width, *[boundary + delta
                              for boundary in [512, 1024, 1536, 2048, 2560, 4096, 131072]
                              for delta in [-1, 0, 1] if boundary + delta <= width]})
    x, storage = aligned_tensor(api, len(lengths), width, dtype, device, offset=True)
    storage.fill_(float("nan"))
    for row, length in enumerate(lengths):
        x[row, :length].copy_(((torch.arange(length, device=device) * 71) % 997).to(dtype))
    end = torch.tensor(lengths, dtype=torch.int32, device=device)
    offsets = torch.arange(len(lengths), dtype=torch.int32, device=device) * 100 - 700
    out, output_storage = aligned_tensor(api, len(lengths), 512, torch.int32, device,
                                         output=True, offset=True)
    output_storage.fill_(-987654)
    result = run_checked_low_index(api, x, 512, end=end, output_idx_offset=offsets,
                                   output_idx=out, indices_type=torch.int32,
                                   return_value=return_value, value_oob_fill_value=12345.0)
    assert result[1] is out
    untouched = torch.ones_like(output_storage, dtype=torch.bool)
    alignment = api.get_stride_requirement()[1] // torch.int32.itemsize
    untouched[1:, alignment:alignment + 512] = False
    assert bool((output_storage[untouched] == -987654).all())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("width", [8193, 131072])
def test_streaming_k512_repeated_compaction_patterns(api, device, dtype, return_value, width):
    x, _ = aligned_tensor(api, 8, width, dtype, device)
    position = torch.arange(width, device=device)
    x[0].copy_(position.to(dtype))
    x[1].copy_((width - position).to(dtype))
    x[2].fill_(3.5)
    x[3].fill_(-torch.finfo(dtype).max)
    x[3, ::2] = torch.finfo(dtype).max
    x[4].copy_((position // 1024).to(dtype))
    x[4, -512:] = 1024
    x[5].fill_(-5)
    x[5, position % 1024 < 64] = 3
    x[5, position % 1024 == 1023] = 10
    x[6].copy_((position // 1024).to(dtype))
    x[6, :2048] = 0
    x[6, 2048:3072] = 1
    x[6, 3071] = -1
    tile_count = (width + 1023) // 1024
    tile_bits = (tile_count - 1).bit_length()
    scan_order_values = [-1] * width
    full_tile_rank = 0
    for visit in range(1 << tile_bits):
        tile = sum(((visit >> bit) & 1) << (tile_bits - 1 - bit) for bit in range(tile_bits))
        if tile >= tile_count or (tile + 1) * 1024 > width:
            continue
        value = 0 if full_tile_rank < 2 else 1 if full_tile_rank == 2 else full_tile_rank
        start = tile * 1024
        scan_order_values[start:start + 1024] = [value] * 1024
        if full_tile_rank == 2:
            scan_order_values[start + 1023] = -1
        full_tile_rank += 1
    x[7].copy_(torch.tensor(scan_order_values, dtype=dtype, device="cpu"))
    offsets = torch.tensor([-100, 200, -300, 400, -500, 600, -700, 800], dtype=torch.int32, device=device)
    run_checked_low_index(api, x, 512, indices_type=torch.int32,
                           return_value=return_value, output_idx_offset=offsets)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_streaming_k512_special_value_ties(api, device, dtype, return_value):
    width = 16385
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    sign = 1 << (dtype.itemsize * 8 - 1)
    raw = ([0xBF800001, 0xBF800000, 0xBF7FFFFF] if dtype == torch.float32
           else [0xBF81, 0xBF80, 0xBF7F])
    patterns = [raw, [sign | 3, sign | 2, sign | 1]]
    x, _ = aligned_tensor(api, 4, width, dtype, device)
    position = torch.arange(width, device=device)
    equal = position % 1024 < 64
    greater = position % 1024 == 1023
    for row, raw in enumerate(patterns):
        signed = [value - 2 * sign if value & sign else value for value in raw]
        x.view(bits)[row].fill_(signed[0])
        x.view(bits)[row, equal] = signed[1]
        x.view(bits)[row, greater] = signed[2]
    x[2].fill_(-float("inf"))
    x[2, equal] = 0.0
    x[2, equal & (position % 2 == 0)] = -0.0
    x[2, greater] = float("inf")
    x[3].fill_(-float("inf"))
    run_checked_low_index(api, x, 512, indices_type=torch.int32, return_value=return_value)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_streaming_k512_late_nan_and_excluded_suffix(api, device, dtype, return_value):
    width = 8193
    length = 7169
    x, _ = aligned_tensor(api, 4, width, dtype, device)
    x[:2].copy_(torch.arange(width, device=device).to(dtype).expand(2, -1))
    x[2:].fill_(float("inf"))
    x[:, length:] = float("nan")
    x[0, length - 1] = float("nan")
    x[2, length - 1] = float("nan")
    end = torch.full((4,), length, dtype=torch.int32, device=device)
    offsets = torch.tensor([100, -200, 300, -400], dtype=torch.int32, device=device)
    values, indices = api.topk(x, 512, end=end, output_idx_offset=offsets,
                               indices_type=torch.int32, return_value=return_value,
                               idx_oob_fill_value=-1, abort_when_nan_found=False)
    for row in [0, 2]:
        assert indices[row, 0].item() == 0x3F3F3F3F
    for row in [1, 3]:
        result = (values[row:row + 1] if return_value else None, indices[row:row + 1])
        check_result(api, x[row:row + 1], 512, result, indices_type=torch.int32,
                     return_value=return_value, end=end[row:row + 1],
                     output_idx_offset=offsets[row:row + 1])
        run_checked_low_index(api, x[row:row + 1], 512, indices_type=torch.int32,
                               return_value=return_value, end=end[row:row + 1],
                               output_idx_offset=offsets[row:row + 1])
    assert builtins_sorted((indices[3].cpu() + 400).tolist()) == list(range(512))
    if not return_value:
        assert values is None


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("batch,width", [(1, 65535), (1, 65536), (3, 65537),
                                         (16, 131072), (3, 131073), (17, 65536)])
def test_segmented_p8_dispatch_boundaries(api, device, dtype, return_value, batch, width):
    x, _ = aligned_tensor(api, batch, width, dtype, device)
    position = torch.arange(width, device=device)
    for row in range(batch):
        x[row].copy_(((position * 71 + row * 13) % 997).to(dtype))
    offsets = torch.arange(batch, dtype=torch.int32, device=device) * 100 - 700
    run_checked_low_index(api, x, 512, indices_type=torch.int32, return_value=return_value,
                           output_idx_offset=offsets)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_segmented_p8_ragged_end_and_storage(api, device, dtype, return_value):
    width = 65537
    lengths = [0, 1, 7, 511, 512, 513, 514, 1023, 1024, 4095, 4096, 4097,
               8193, width - 7, width - 1, width]
    x, storage = aligned_tensor(api, len(lengths), width, dtype, device, offset=True)
    storage.fill_(float("nan"))
    for row, length in enumerate(lengths):
        x[row, :length].copy_(((torch.arange(length, device=device) * 71) % 997).to(dtype))
    before = storage.clone()
    end = torch.tensor(lengths, dtype=torch.int32, device=device)
    offsets = torch.arange(len(lengths), dtype=torch.int32, device=device) * 100 - 700
    out, output_storage = aligned_tensor(api, len(lengths), 512, torch.int32, device,
                                         output=True, offset=True)
    output_storage.fill_(-987654)
    result = run_checked_low_index(api, x, 512, end=end, output_idx_offset=offsets,
                                   output_idx=out, indices_type=torch.int32,
                                   return_value=return_value, value_oob_fill_value=12345.0)
    assert result[1] is out
    untouched = torch.ones_like(output_storage, dtype=torch.bool)
    alignment = api.get_stride_requirement()[1] // torch.int32.itemsize
    untouched[1:, alignment:alignment + 512] = False
    assert bool((output_storage[untouched] == -987654).all())
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(storage.view(bits), before.view(bits))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("length", [513, 4097, 65529])
def test_segmented_p8_nan_in_each_segment(api, device, dtype, return_value, length):
    x, _ = aligned_tensor(api, 8, 65536, dtype, device)
    for row in range(8):
        x[row].fill_(float("inf") if row % 2 else 1)
    x[:, length:] = float("nan")
    nan_indices = [(row + 1) * length // 8 - 1 for row in range(8)]
    for row, index in enumerate(nan_indices):
        x[row, index] = float("nan")
    end = torch.full((8,), length, dtype=torch.int32, device=device)
    offsets = torch.arange(8, dtype=torch.int32, device=device) * 100 - 300
    values, indices = api.topk(x, 512, end=end, output_idx_offset=offsets,
                               indices_type=torch.int32, return_value=return_value,
                               abort_when_nan_found=False)
    assert indices.dtype == torch.int32
    assert indices[:, 0].cpu().tolist() == [0x3F3F3F3F] * 8
    if not return_value:
        assert values is None
    for row, index in enumerate(nan_indices):
        x[row, index] = float("inf") if row % 2 else 1
    run_checked_low_index(api, x, 512, end=end, output_idx_offset=offsets,
                           indices_type=torch.int32, return_value=return_value)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_segmented_p8_cross_segment_original_index_ties(api, device, dtype, return_value):
    width, length = 65537, 65529
    x, _ = aligned_tensor(api, 3, width, dtype, device)
    x.fill_(float("nan"))
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    sign = 1 << (dtype.itemsize * 8 - 1)
    finite = ([0xBF800001, 0xBF800000, 0xBF7FFFFF] if dtype == torch.float32
              else [0xBF81, 0xBF80, 0xBF7F])
    equal = [index for segment in range(8)
             for index in range(segment * length // 8 + 17, segment * length // 8 + 113)]
    greater = [(segment + 1) * length // 8 - 1 for segment in range(8)]
    x[0, :length] = -float("inf")
    x[0, equal] = 0.0
    x[0, equal[::2]] = -0.0
    x[0, greater] = float("inf")
    for row, raw in enumerate([finite, [sign | 3, sign | 2, sign | 1]], start=1):
        signed = [value - 2 * sign if value & sign else value for value in raw]
        x.view(bits)[row, :length] = signed[0]
        x.view(bits)[row, equal] = signed[1]
        x.view(bits)[row, greater] = signed[2]
    end = torch.full((3,), length, dtype=torch.int32, device=device)
    offsets = torch.tensor([-100, 200, -300], dtype=torch.int32, device=device)
    result = run_checked_low_index(api, x, 512, end=end, output_idx_offset=offsets,
                                   indices_type=torch.int32, return_value=return_value)
    expected = builtins_sorted(greater + equal[:504])
    for row, offset in enumerate(offsets.cpu().tolist()):
        assert builtins_sorted((result[1][row].cpu() - offset).tolist()) == expected


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_segmented_p8_graph_scratch_reuse(api, device, dtype, return_value):
    x, _ = aligned_tensor(api, 3, 65536, dtype, device)
    x.fill_(1)
    end = torch.full((3,), x.shape[1], dtype=torch.int32, device=device)
    offsets = torch.zeros(3, dtype=torch.int32, device=device)
    kwargs = dict(end=end, output_idx_offset=offsets, indices_type=torch.int32,
                  return_value=return_value, idx_oob_fill_value=-1, abort_when_nan_found=False)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            api.topk(x, 512, **kwargs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        values, indices = api.topk(x, 512, **kwargs)
    states = [([65536, 4097, 513], [-100, 200, -300], False),
              ([513, 4097, 65529], [400, -500, 600], True),
              ([0, 511, 512], [-700, 800, -900], True),
              ([4097, 65536, 513], [1000, -1100, 1200], False)]
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    for lengths, new_offsets, have_nan in states:
        x.fill_(float("nan"))
        for row, length in enumerate(lengths):
            x[row, :length] = 1
            x[row, max(0, length - 512):length] = 10
            if have_nan and length and row < 2:
                x[row, length - 1] = float("nan")
        end.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        offsets.copy_(torch.tensor(new_offsets, dtype=torch.int32, device=device))
        graph.replay()
        torch.cuda.synchronize(device)
        for row, (length, offset) in enumerate(zip(lengths, new_offsets)):
            if have_nan and row < 2 and length > 512:
                assert indices[row, 0].item() == 0x3F3F3F3F
                continue
            count = min(length, 512)
            actual = (indices[row, :count].cpu().to(torch.int64) - offset).tolist()
            assert builtins_sorted(actual) == list(range(max(0, length - 512), length))
            assert bool((indices[row, count:] == -1).all())
            if return_value:
                assert torch.equal(values[row, :count].view(bits), x[row, actual].view(bits))
                assert bool(torch.isneginf(values[row, count:]).all())
        if not return_value:
            assert values is None


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_segmented_p8_nondefault_stream_allocation_churn(api, device, dtype, return_value):
    x, _ = aligned_tensor(api, 3, 65536, dtype, device)
    end = torch.empty(3, dtype=torch.int32, device=device)
    offsets = torch.empty(3, dtype=torch.int32, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    snapshots = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        for iteration, lengths in enumerate([[65536, 4097, 513], [0, 511, 512],
                                              [513, 65529, 4096], [4097, 513, 65536]]):
            x.fill_(float("nan"))
            for row, length in enumerate(lengths):
                x[row, :length] = -1
                x[row, max(0, length - 512):length] = 10 + iteration
            end.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            offsets.copy_(torch.tensor([100 + iteration, -200 - iteration, 300 + iteration],
                                       dtype=torch.int32, device=device))
            result = api.topk(x, 512, end=end, output_idx_offset=offsets,
                              indices_type=torch.int32, return_value=return_value, idx_oob_fill_value=-1)
            snapshots.append((x.clone(), end.clone(), offsets.clone(),
                              (result[0].clone() if return_value else None, result[1].clone())))
            del result
            churn_keys = torch.empty((3, 8, 512), dtype=torch.int64, device=device)
            churn_counts = torch.empty((3, 8), dtype=torch.int32, device=device)
            churn_keys.fill_(-987654)
            churn_counts.fill_(-1)
            del churn_keys, churn_counts
    stream.synchronize()
    for source, lengths, row_offsets, result in snapshots:
        check_result(api, source, 512, result, end=lengths, output_idx_offset=row_offsets,
                     indices_type=torch.int32, return_value=return_value)
        for row, (length, offset) in enumerate(zip(lengths.cpu().tolist(), row_offsets.cpu().tolist())):
            count = min(length, 512)
            actual = (result[1][row, :count].cpu().to(torch.int64) - offset).tolist()
            assert builtins_sorted(actual) == list(range(max(0, length - 512), length))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("indices_type,k", [
    pytest.param(torch.int64, 7, id="fallback-i64"),
    pytest.param(torch.int32, 7, id="fast-512"),
    pytest.param(torch.int32, 513, id="fallback-k-above-512"),
    pytest.param(torch.int32, 512, id="streaming-k512"),
])
def test_nondefault_stream(api, device, dtype, return_value, indices_type, k):
    x, _ = aligned_tensor(api, 4, 16384, dtype, device)
    x.fill_(-1)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        x[:, -k:] = 10
        result = api.topk(x, k, indices_type=indices_type, return_value=return_value,
                          idx_oob_fill_value=-1)
        copied = (result[0].clone() if return_value else None, result[1].clone())
    stream.synchronize()
    check_result(api, x, k, result, indices_type=indices_type, return_value=return_value)
    assert torch.equal(copied[1], result[1])
    if return_value:
        assert torch.equal(copied[0], result[0])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("sorted_index,k", [
    pytest.param(True, 7, id="fallback-sorted-index"),
    pytest.param(False, 7, id="fast-512"),
    pytest.param(False, 2048, id="fallback-k-above-512"),
    pytest.param(False, 512, id="streaming-k512"),
])
def test_graph_replay_updates_input_and_end(api, device, dtype, return_value, sorted_index, k):
    x, _ = aligned_tensor(api, 4, 16384, dtype, device)
    x.fill_(0)
    end = torch.full((4,), x.shape[1], dtype=torch.int32, device=device)
    offsets = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device=device)
    kwargs = dict(end=end, output_idx_offset=offsets, sorted_index=sorted_index,
                  return_value=return_value, indices_type=torch.int32, idx_oob_fill_value=-1)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            api.topk(x, k, **kwargs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = api.topk(x, k, **kwargs)
    for ends, new_offsets, hot in [
        ([16384, 0, k - 1, k], [-100, 20, -30, 40], 100),
        ([k, 16384, k + 1, k - 1], [500, -600, 700, -800], 1000),
        ([k + 1, k - 1, 0, 16384], [-90, 80, -70, 60], 4000),
    ]:
        x.fill_(-5)
        x[:, hot:hot + k] = 10
        end.copy_(torch.tensor(ends, dtype=torch.int32, device=device))
        offsets.copy_(torch.tensor(new_offsets, dtype=torch.int32, device=device))
        graph.replay()
        torch.cuda.synchronize(device)
        check_result(api, x, k, result, **kwargs)


def test_input_device_guard(api, device):
    if torch.cuda.device_count() < 2:
        pytest.skip("two visible CUDA devices are required")
    other = next(index for index in range(torch.cuda.device_count()) if index != device.index)
    x, _ = aligned_tensor(api, 4, 16384, torch.float32, device)
    x.copy_(torch.arange(x.shape[1], device=device).expand_as(x))
    torch.cuda.synchronize(device)
    with torch.cuda.device(other):
        result = api.topk(x, 7, sorted=True, idx_oob_fill_value=-1)
        assert torch.cuda.current_device() == other
    check_result(api, x, 7, result, sorted=True)
    if torch.cuda.get_device_capability(other) in ((12, 0), (12, 1)):
        target = torch.device("cuda", other)
        x_other, _ = aligned_tensor(api, 4, 16384, torch.bfloat16, target)
        x_other.fill_(0)
        x_other[:, -7:] = 10
        torch.cuda.synchronize(target)
        with torch.cuda.device(device):
            other_result = api.topk(x_other, 7, idx_oob_fill_value=-1)
            assert torch.cuda.current_device() == device.index
        check_result(api, x_other, 7, other_result)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("batch,width,k", [
    (16, 2048, 511), (16, 2048, 512), (17, 2048, 513),
    (17, 2049, 511), (16, 2049, 512), (17, 2049, 512), (16, 2049, 513),
    (16, 65535, 512), (17, 65535, 512),
    (16, 65536, 511), (16, 65536, 512), (17, 65536, 512), (16, 65536, 513),
    (16, 65537, 512), (17, 65537, 512),
    (16, 131072, 511), (16, 131072, 512), (17, 131072, 512), (16, 131072, 513),
    (16, 131073, 512), (17, 131073, 512),
])
def test_dispatch_k_batch_width_boundaries(api, device, dtype, return_value, batch, width, k):
    x, _ = aligned_tensor(api, batch, width, dtype, device)
    position = torch.arange(width, device=device)
    for row in range(batch):
        x[row].copy_(((position * 71 + row * 13) % 997).to(dtype))
    run_checked_low_index(api, x, k, indices_type=torch.int32, return_value=return_value)


INVALID_CASES = [
    "input-fp16", "input-fp64", "input-i32", "input-rank1", "input-rank3", "input-cpu",
    "input-last-stride", "input-row-alignment", "input-pointer-alignment", "input-storage",
    "k-zero", "k-negative", "k-too-large", "indices-dtype", "output-dtype", "output-shape",
    "output-last-stride", "output-row-alignment", "output-pointer-alignment", "output-cpu", "output-overlap",
    "end-dtype", "end-shape", "end-stride", "end-cpu",
    "offset-dtype", "offset-shape", "offset-stride", "offset-cpu",
    "sorted-bf16", "both-sorts", "sorted-no-values", "begin", "hint",
]


@pytest.mark.parametrize("case", INVALID_CASES)
def test_invalid_arguments(api, device, case):
    if api is NativeAPI and case in ("hint", "output-dtype"):
        pytest.skip("wrapper-only arguments are absent from the registered schema")
    x, _ = aligned_tensor(api, 4, 16384, torch.float32, device)
    x.fill_(0)
    k = 7
    kwargs = {}
    if case.startswith("input-"):
        kind = case.removeprefix("input-")
        if kind in ("fp16", "fp64", "i32"):
            x = x.to({"fp16": torch.float16, "fp64": torch.float64, "i32": torch.int32}[kind])
        elif kind == "rank1":
            x = x[0]
        elif kind == "rank3":
            x = x.unsqueeze(0)
        elif kind == "cpu":
            x = x.cpu()
        elif kind == "last-stride":
            x = x[:, ::2]
        elif kind == "row-alignment":
            x = torch.zeros((4, 16385), device=device)[:, :16384]
        elif kind == "pointer-alignment":
            x = x[:, 1:]
        elif kind == "storage":
            storage = torch.zeros((7,), dtype=torch.float32, device=device)
            row_stride = api.get_stride_requirement()[0] // storage.element_size()
            x = storage.as_strided((1, 7), (row_stride, 1))
    elif case.startswith("k-"):
        k = {"k-zero": 0, "k-negative": -1, "k-too-large": 4097}[case]
    elif case == "indices-dtype":
        kwargs["indices_type"] = torch.int16
    elif case.startswith("output-"):
        out, _ = aligned_tensor(api, 4, k, torch.int64, device, output=True)
        kind = case.removeprefix("output-")
        if kind == "dtype":
            out = out.to(torch.int32)
        elif kind == "shape":
            out = out[:, :k - 1]
        elif kind == "last-stride":
            out, _ = aligned_tensor(api, 4, 2 * k, torch.int64, device, output=True)
            out = out[:, ::2]
        elif kind == "row-alignment":
            out = torch.empty((4, k), dtype=torch.int64, device=device)
        elif kind == "pointer-alignment":
            out, _ = aligned_tensor(api, 4, k + 1, torch.int64, device, output=True)
            out = out[:, 1:]
        elif kind == "cpu":
            out = out.cpu()
        elif kind == "overlap":
            out = out[:1].expand(4, -1)
        kwargs["output_idx"] = out
    elif case.startswith(("end-", "offset-")):
        name, kind = case.split("-")
        argument = torch.full((4,), 10, dtype=torch.int32, device=device)
        if kind == "dtype":
            argument = argument.to(torch.int64)
        elif kind == "shape":
            argument = argument[:3]
        elif kind == "stride":
            argument = torch.full((8,), 10, dtype=torch.int32, device=device)[::2]
        elif kind == "cpu":
            argument = argument.cpu()
        kwargs["end" if name == "end" else "output_idx_offset"] = argument
    elif case == "sorted-bf16":
        x = x.to(torch.bfloat16)
        kwargs["sorted"] = True
    elif case == "both-sorts":
        kwargs.update(sorted=True, sorted_index=True)
    elif case == "sorted-no-values":
        kwargs.update(sorted=True, return_value=False)
    else:
        kwargs[case] = torch.zeros((4,), dtype=torch.int32, device=device)
    with pytest.raises((RuntimeError, AssertionError, ValueError, IndexError)):
        api.topk(x, k, **kwargs)
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("argument", ["end", "output_idx", "output_idx_offset"])
def test_reject_cross_device_arguments(api, device, argument):
    if torch.cuda.device_count() < 2:
        pytest.skip("two visible CUDA devices are required")
    other = next(index for index in range(torch.cuda.device_count()) if index != device.index)
    x, _ = aligned_tensor(api, 4, 16384, torch.float32, device)
    x.fill_(0)
    if argument == "output_idx":
        value, _ = aligned_tensor(api, 4, 7, torch.int64, torch.device("cuda", other), output=True)
    else:
        value = torch.full((4,), 7, dtype=torch.int32, device=torch.device("cuda", other))
    with pytest.raises(RuntimeError, match="device"):
        api.topk(x, 7, **{argument: value})
    torch.cuda.synchronize(device)


def shared_storage_view(storage, dtype, width, byte_offset, alignment):
    row_bytes = (width * dtype.itemsize + alignment - 1) // alignment * alignment
    return storage[byte_offset:byte_offset + row_bytes].view(dtype).reshape(1, -1)[:, :width]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("argument", ["input", "end", "output_idx_offset"])
@pytest.mark.parametrize("disjoint", [False, True], ids=["overlap", "disjoint-shared-storage"])
@pytest.mark.parametrize("return_value", [False, True])
def test_reject_output_idx_alias(api, device, dtype, indices_type, argument, disjoint, return_value):
    storage = torch.zeros(8192, dtype=torch.uint8, device=device)
    x, _ = aligned_tensor(api, 1, 128, dtype, device)
    x.fill_(0)
    output = shared_storage_view(storage, indices_type, 7, 1024 if disjoint else 0,
                                 api.get_stride_requirement()[1])
    kwargs = dict(output_idx=output, indices_type=indices_type, return_value=return_value)
    if argument == "input":
        x = shared_storage_view(storage, dtype, 128, 0, api.get_stride_requirement()[0])
    else:
        metadata = storage[:4].view(torch.int32)
        metadata.fill_(128 if argument == "end" else 100)
        kwargs[argument] = metadata
    before = storage.clone()
    with pytest.raises(RuntimeError, match="overlap.*storage|storage.*overlap"):
        api.topk(x, 7, **kwargs)
    torch.cuda.synchronize(device)
    assert torch.equal(storage, before)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("argument", ["input", "end", "output_idx_offset", "output_idx"])
@pytest.mark.parametrize("disjoint", [False, True], ids=["overlap", "disjoint-shared-storage"])
def test_reject_backend_output_value_alias(api, device, dtype, indices_type, argument, disjoint):
    storage = torch.zeros(8192, dtype=torch.uint8, device=device)
    x, _ = aligned_tensor(api, 1, 128, dtype, device)
    x.fill_(0)
    indices, _ = aligned_tensor(api, 1, 7, indices_type, device, output=True)
    values = shared_storage_view(storage, dtype, 7, 1024 if disjoint else 0,
                                 api.get_stride_requirement()[1])
    end = offsets = None
    if argument == "input":
        x = shared_storage_view(storage, dtype, 128, 0, api.get_stride_requirement()[0])
    elif argument == "output_idx":
        indices = shared_storage_view(storage, indices_type, 7, 0, api.get_stride_requirement()[1])
    elif argument == "end":
        end = storage[:4].view(torch.int32)
        end.fill_(128)
    else:
        offsets = storage[:4].view(torch.int32)
        offsets.fill_(100)
    before = storage.clone()
    with pytest.raises(RuntimeError, match="overlap.*storage|storage.*overlap"):
        torch.ops.deep_select.topk(x, 7, None, end, False, False, values, indices,
                                   offsets, -1, float("-inf"), True, True)
    torch.cuda.synchronize(device)
    assert torch.equal(storage, before)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("sorted,sorted_index,return_value", MODES)
@pytest.mark.parametrize("k", [7, 512])
@pytest.mark.parametrize("pattern", ["equal", "threshold"])
def test_low_index_tie_selection(api, device, dtype, indices_type, sorted, sorted_index, return_value, k, pattern):
    if dtype == torch.bfloat16 and sorted:
        pytest.skip("sorted values are FP32-only")
    x, _ = aligned_tensor(api, 1, 16384, dtype, device)
    x.fill_(3)
    expected = list(range(k))
    if pattern == "threshold":
        x[:, -3:] = 10
        expected = list(range(k - 3)) + list(range(x.shape[1] - 3, x.shape[1]))
    for _ in range(3):
        _, indices = run_checked(api, x, k, indices_type=indices_type, sorted=sorted,
                                 sorted_index=sorted_index, return_value=return_value)
        actual = indices[0].cpu().tolist()
        assert builtins_sorted(actual) == expected
        if sorted:
            assert actual == (expected[-3:] + expected[:-3] if pattern == "threshold" else expected)
        elif sorted_index:
            assert actual == expected


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("sorted,sorted_index,return_value", MODES)
@pytest.mark.parametrize("k", [1, 7, 12])
def test_subnormal_and_signed_zero_order(api, device, dtype, indices_type, sorted, sorted_index, return_value, k):
    if dtype == torch.bfloat16 and sorted:
        pytest.skip("sorted values are FP32-only")
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    bit_count, mantissa = (32, 23) if dtype == torch.float32 else (16, 7)
    sign = 1 << (bit_count - 1)
    tiny = 1 << mantissa
    raw = [sign | 1, 0, sign, 1, tiny - 1, sign | (tiny - 1),
           2, sign | 2, tiny, sign | tiny, 0, sign]
    signed = [value - 2 * sign if value & sign else value for value in raw]
    x, _ = aligned_tensor(api, 1, 16384, dtype, device)
    x.fill_(-float("inf"))
    x.view(bits)[0, :len(raw)].copy_(torch.tensor(signed, dtype=bits, device=device))
    def key(index):
        value = raw[index]
        if value in (0, sign):
            value = 0
        return ((~value & (2 * sign - 1)) if value & sign else value ^ sign)
    ranked = builtins_sorted(range(len(raw)), key=lambda index: (-key(index), index))[:k]
    values, indices = api.topk(x, k, sorted=sorted, sorted_index=sorted_index,
                               return_value=return_value, indices_type=indices_type)
    actual = indices[0].cpu().tolist()
    assert builtins_sorted(actual) == builtins_sorted(ranked)
    if sorted:
        assert actual == ranked
    elif sorted_index:
        assert actual == builtins_sorted(ranked)
    if return_value:
        assert values[0].view(bits).cpu().tolist() == [signed[index] for index in actual]
    else:
        assert values is None


@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("length", [5, 7])
@pytest.mark.parametrize("fill", [-float("inf"), float("inf"), 12345.0], ids=["fill-neginf", "fill-posinf", "fill-finite"])
@pytest.mark.parametrize("abort_when_nan_found", [False, True])
def test_sorted_short_row_nan_payload_and_fill(api, device, indices_type, length, fill, abort_when_nan_found):
    x, _ = aligned_tensor(api, 1, 128, torch.float32, device)
    x.fill_(0)
    unsigned = [0xffc01234, 0x3f800000, 0x7fc05678, 0xff800000,
                0x80000000, 0x00000000, 0xff801234]
    signed = [value if value < 0x80000000 else value - 0x100000000 for value in unsigned]
    x.view(torch.int32)[0, :7].copy_(torch.tensor(signed, dtype=torch.int32, device=device))
    end = torch.tensor([length], dtype=torch.int32, device=device)
    offsets = torch.tensor([100], dtype=torch.int32, device=device)
    values, indices = api.topk(x, 7, sorted=True, end=end, output_idx_offset=offsets,
                               indices_type=indices_type, idx_oob_fill_value=-1,
                               value_oob_fill_value=fill, abort_when_nan_found=abort_when_nan_found)
    expected = [0, 2, 1, 4, 3] if length == 5 else [0, 2, 6, 1, 4, 5, 3]
    assert indices[0, :length].cpu().tolist() == [index + 100 for index in expected]
    assert indices[0, length:].cpu().tolist() == [-1] * (7 - length)
    assert values[0, :length].view(torch.int32).cpu().tolist() == [signed[index] for index in expected]
    fill_bits = torch.tensor(fill, dtype=torch.float32).view(torch.int32).item()
    assert values[0, length:].view(torch.int32).cpu().tolist() == [fill_bits] * (7 - length)


def assert_trap_child(api, device, dtype_name, indices_name, k, return_value, case):
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--trap-child",
                            dtype_name, str(device.index), indices_name, str(k),
                            str(int(return_value)), case, "native" if api is NativeAPI else "wrapper"],
                           capture_output=True, text=True, timeout=120)
    output = child.stdout + child.stderr
    assert "TRAP_ARMED" in output, output
    assert child.returncode != 0, output
    assert any(text in output.lower() for text in
               ["illegal instruction", "device-side assert", "unspecified launch failure",
                "cudaerrorillegalinstruction"]), output


@pytest.mark.skipif(os.environ.get("DEEP_SELECT_TEST_NAN_TRAP") != "1",
                    reason="opt-in only: traps a child CUDA context; set DEEP_SELECT_TEST_NAN_TRAP=1")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("indices_name,k", [
    pytest.param("int64", 7, id="fallback-i64"),
    pytest.param("int32", 7, id="fast-512"),
    pytest.param("int32", 2048, id="fallback-k-above-512"),
    pytest.param("int32", 512, id="streaming-k512"),
])
def test_nan_default_traps_in_child(api, device, dtype_name, return_value, indices_name, k):
    assert_trap_child(api, device, dtype_name, indices_name, k, return_value, "nan")


@pytest.mark.skipif(os.environ.get("DEEP_SELECT_TEST_NAN_TRAP") != "1",
                    reason="opt-in only: traps a child CUDA context; set DEEP_SELECT_TEST_NAN_TRAP=1")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("case", ["end-negative", "end-too-large"])
@pytest.mark.parametrize("k,prefix", [
    pytest.param(513, "", id="fallback-k-above-512"),
    pytest.param(7, "whole-row-", id="whole-row-k7"),
    pytest.param(512, "streaming-", id="streaming-k512"),
])
def test_unsorted_i32_invalid_end_traps_in_child(api, device, dtype_name, return_value, case, k, prefix):
    assert_trap_child(api, device, dtype_name, "int32", k, return_value, prefix + case)


@pytest.mark.skipif(os.environ.get("DEEP_SELECT_TEST_NAN_TRAP") != "1",
                    reason="opt-in only: traps a child CUDA context; set DEEP_SELECT_TEST_NAN_TRAP=1")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("case", ["nan", "end-negative", "end-too-large"])
def test_segmented_p8_traps_in_child(api, device, dtype_name, return_value, case):
    assert_trap_child(api, device, dtype_name, "int32", 512, return_value, "segmented-p8-" + case)


def trap_child(dtype_name, device_index, indices_name, k, return_value, case, caller):
    import resource
    import deep_select

    api = NativeAPI if caller == "native" else deep_select
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    torch.cuda.set_device(int(device_index))
    device = torch.device("cuda", int(device_index))
    assert torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
    k = int(k)
    width = 8193 if k == 512 else k + 2
    length = 7169 if k == 512 else k + 1
    if case.startswith("segmented-p8-"):
        width, length = 65536, 513
        case = case.removeprefix("segmented-p8-")
    elif case.startswith("whole-row-"):
        width = length = 4096
        case = case.removeprefix("whole-row-")
    elif case.startswith("streaming-"):
        width = length = 16384
        case = case.removeprefix("streaming-")
    x, _ = aligned_tensor(api, 1, width, getattr(torch, dtype_name), device)
    x.fill_(float("inf") if k == 512 else 0)
    end = torch.tensor([length], dtype=torch.int32, device=device)
    kwargs = dict(end=end, indices_type=getattr(torch, indices_name), return_value=bool(int(return_value)))
    if case == "nan":
        x[0, length:] = float("nan")
    api.topk(x, k, **kwargs)
    torch.cuda.synchronize(device)
    if case == "nan":
        x[0, length - 1] = float("nan")
    elif case == "end-negative":
        end.fill_(-1)
    elif case == "end-too-large":
        end.fill_(x.shape[1] + 1)
    else:
        raise ValueError(case)
    torch.cuda.synchronize(device)
    print("TRAP_ARMED", flush=True)
    api.topk(x, k, **kwargs)
    torch.cuda.synchronize(device)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--trap-child":
        trap_child(*sys.argv[2:])
    else:
        raise SystemExit(pytest.main([str(Path(__file__).resolve()), *sys.argv[1:]]))
