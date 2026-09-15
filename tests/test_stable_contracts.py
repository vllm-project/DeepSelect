import importlib

import pytest
import torch

from .test_sm120 import DTYPES, INDEX_DTYPES, NativeAPI, aligned_tensor, check_result, shared_storage_view


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3), (12, 0), (12, 1)):
        pytest.skip("DeepSelect requires SM100, SM103, SM120, or SM121")
    return device


@pytest.fixture(scope="module")
def registered():
    import deep_select

    return deep_select


@pytest.fixture(scope="module", params=["wrapper", "native"])
def api(device, registered, request):
    return registered if request.param == "wrapper" else NativeAPI


def test_registered_ops_once(registered):
    module = importlib.import_module("deep_select.deep_select_cuda")
    assert module.__name__ == "deep_select.deep_select_cuda"
    topk = torch.ops.deep_select.topk.default
    alignment = torch.ops.deep_select.get_alignment_requirement.default
    expected = alignment()
    assert tuple(registered.get_stride_requirement()) == tuple(expected)
    assert len(expected) == 2 and all(value > 0 and value % 32 == 0 for value in expected)
    for _ in range(3):
        assert importlib.import_module("deep_select.deep_select_cuda") is module
        assert importlib.import_module("deep_select") is registered
        assert torch.ops.deep_select.topk.default is topk
        assert torch.ops.deep_select.get_alignment_requirement.default is alignment
    assert torch.ops.deep_select.topk.overloads() == ["default"]
    assert torch.ops.deep_select.get_alignment_requirement.overloads() == ["default"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("batch,width,k", [(0, 0, 7), (0, 65536, 512), (3, 0, 7), (3, 16384, 513)])
def test_shared_empty_and_varlen(api, device, dtype, indices_type, return_value, batch, width, k):
    x, _ = aligned_tensor(api, batch, width, dtype, device)
    x.copy_(((torch.arange(width, device=device) * 71) % 997).to(dtype).expand_as(x))
    end = torch.tensor(([0, min(k - 1, width), width] if batch else []),
                       dtype=torch.int32, device=device)
    offsets = torch.tensor(([-100, 200, -300] if batch else []), dtype=torch.int32, device=device)
    if batch > 0 and width == 0 and torch.cuda.get_device_capability(device)[0] == 10:
        with pytest.raises(RuntimeError, match="SM100/SM103 require vocab_size > 0 for nonempty batches"):
            api.topk(x, k, end=end, output_idx_offset=offsets, indices_type=indices_type,
                     return_value=return_value, idx_oob_fill_value=-1, value_oob_fill_value=12345.0)
        return
    result = api.topk(x, k, end=end, output_idx_offset=offsets, indices_type=indices_type,
                      return_value=return_value, idx_oob_fill_value=-1, value_oob_fill_value=12345.0)
    check_result(api, x, k, result, end=end, output_idx_offset=offsets, indices_type=indices_type,
                 return_value=return_value, idx_oob_fill_value=-1, value_oob_fill_value=12345.0)
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("case", [
    "idx-fill-low", "idx-fill-high", "value-fill-low", "value-fill-high",
    "input-rank0", "input-rank1", "input-rank3", "input-dtype", "input-cpu",
    "input-pointer", "input-stride", "input-storage", "width-limit", "batch-limit",
    "k-zero", "k-negative", "k-limit", "k-int64", "begin",
    "end-dtype", "end-shape", "end-stride", "end-cpu",
    "offset-dtype", "offset-shape", "offset-stride", "offset-cpu",
    "index-dtype", "index-shape", "index-pointer", "index-row-stride", "index-last-stride", "index-overlap", "index-cpu",
    "value-missing", "value-dtype", "value-shape", "value-pointer", "value-row-stride", "value-last-stride", "value-overlap", "value-cpu",
    "both-sorts", "sorted-no-values",
])
def test_native_rejects_invalid_contract(registered, device, dtype, indices_type, case):
    x, _ = aligned_tensor(NativeAPI, 3, 128, dtype, device)
    x.fill_(0)
    values, _ = aligned_tensor(NativeAPI, 3, 7, dtype, device, output=True)
    indices, _ = aligned_tensor(NativeAPI, 3, 7, indices_type, device, output=True)
    args = dict(input=x, topk=7, begin=None, end=None, sorted_value=False, sorted_index=False,
                output_value=values, output_index=indices, output_idx_offset=None,
                idx_oob_fill_value=-1, value_oob_fill_value=float("-inf"),
                return_value=True, abort_when_nan_found=True)
    if case.startswith("idx-fill-"):
        args["idx_oob_fill_value"] = -(1 << 31) - 1 if case.endswith("low") else 1 << 31
    elif case.startswith("value-fill-"):
        args["value_oob_fill_value"] = -1e300 if case.endswith("low") else 1e300
    elif case.startswith("input-"):
        kind = case.removeprefix("input-")
        if kind == "rank0":
            x = x[0, 0]
        elif kind == "rank1":
            x = x[0]
        elif kind == "rank3":
            x = x.unsqueeze(0)
        elif kind == "dtype":
            x = x.to(torch.float16)
        elif kind == "cpu":
            x = x.cpu()
        elif kind == "pointer":
            x = x[:, 1:]
        elif kind == "stride":
            x = x[:, ::2]
        else:
            storage = torch.empty(7, dtype=dtype, device=device)
            x = storage.as_strided((1, 7), (128 // dtype.itemsize, 1))
            args["output_value"] = values[:1]
            args["output_index"] = indices[:1]
        args["input"] = x
    elif case in ("width-limit", "batch-limit"):
        if case == "width-limit":
            args["input"] = x.as_strided((0, 1 << 23), (1 << 23, 1))
            args["output_value"] = values[:0]
            args["output_index"] = indices[:0]
        else:
            args["input"] = x.as_strided((1 << 31, 0), (0, 1))
            args["output_value"] = values[:1].expand(1 << 31, -1)
            args["output_index"] = indices[:1].expand(1 << 31, -1)
    elif case.startswith("k-"):
        args["topk"] = {"k-zero": 0, "k-negative": -1, "k-limit": 4097, "k-int64": 1 << 32}[case]
    elif case == "begin":
        args["begin"] = torch.zeros(3, dtype=torch.int32, device=device)
    elif case.startswith(("end-", "offset-")):
        name, kind = case.split("-")
        metadata = torch.full((3,), 7, dtype=torch.int32, device=device)
        if kind == "dtype":
            metadata = metadata.to(torch.int64)
        elif kind == "shape":
            metadata = metadata[:2]
        elif kind == "stride":
            metadata = torch.full((6,), 7, dtype=torch.int32, device=device)[::2]
        else:
            metadata = metadata.cpu()
        args["end" if name == "end" else "output_idx_offset"] = metadata
    elif case.startswith(("index-", "value-")):
        name, kind = case.split("-", 1)
        output = indices if name == "index" else values
        if kind == "missing":
            output = None
        elif kind == "dtype":
            output = output.to(torch.int16 if name == "index" else torch.float64)
        elif kind == "shape":
            output = output[:, :6]
        elif kind == "pointer":
            output, _ = aligned_tensor(NativeAPI, 3, 8, output.dtype, device, output=True)
            output = output[:, 1:]
        elif kind == "row-stride":
            output = torch.empty((3, 7), dtype=output.dtype, device=device)
        elif kind == "last-stride":
            output, _ = aligned_tensor(NativeAPI, 3, 14, output.dtype, device, output=True)
            output = output[:, ::2]
        elif kind == "overlap":
            output = output[:1].expand(3, -1)
        else:
            output = output.cpu()
        args["output_index" if name == "index" else "output_value"] = output
    elif case == "both-sorts":
        args.update(sorted_value=True, sorted_index=True)
    else:
        args.update(sorted_value=True, return_value=False, output_value=None)
    with pytest.raises(RuntimeError):
        torch.ops.deep_select.topk(**args)
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
def test_native_output_padding_and_bounded_tail(registered, device, dtype, indices_type, return_value):
    x, input_storage = aligned_tensor(NativeAPI, 3, 129, dtype, device, offset=True)
    input_storage.fill_(float("nan"))
    x.copy_(torch.arange(129, device=device).to(dtype).expand_as(x))
    before = input_storage.clone()
    indices, index_storage = aligned_tensor(NativeAPI, 3, 7, indices_type, device, output=True, offset=True)
    values, value_storage = aligned_tensor(NativeAPI, 3, 7, dtype, device, output=True, offset=True)
    index_storage.fill_(-987654)
    value_storage.fill_(12345)
    end = torch.tensor([0, 5, 129], dtype=torch.int32, device=device)
    offsets = torch.tensor([-100, 200, -300], dtype=torch.int32, device=device)
    result = torch.ops.deep_select.topk(x, 7, None, end, False, False,
                                        values if return_value else None, indices, offsets,
                                        -1, float("-inf"), return_value, True)
    assert result is None
    check_result(NativeAPI, x, 7, (values if return_value else None, indices),
                 indices_type=indices_type, return_value=return_value, end=end, output_idx_offset=offsets)
    for storage, output, fill in [(index_storage, indices, -987654), (value_storage, values, 12345)]:
        untouched = torch.ones_like(storage, dtype=torch.bool)
        if output is indices or return_value:
            alignment = NativeAPI.get_stride_requirement()[1] // output.element_size()
            untouched[1:, alignment:alignment + 7] = False
        assert bool((storage[untouched] == fill).all())
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(input_storage.view(bits), before.view(bits))
    for output_dtype in [dtype, indices_type]:
        storage = torch.empty(7, dtype=output_dtype, device=device)
        tail = storage.as_strided((1, 7), (32 // output_dtype.itemsize, 1))
        out_values = tail if output_dtype == dtype else values[:1]
        out_indices = tail if output_dtype == indices_type else indices[:1]
        assert torch.ops.deep_select.topk(x[:1], 7, None, None, False, False, out_values,
                                          out_indices, None, -1, float("-inf"), True, True) is None
        check_result(NativeAPI, x[:1], 7, (out_values, out_indices), indices_type=indices_type)
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("output_name,argument", [
    ("output_index", "input"), ("output_index", "end"), ("output_index", "output_idx_offset"),
    ("output_value", "input"), ("output_value", "end"),
    ("output_value", "output_idx_offset"), ("output_value", "output_index"),
])
@pytest.mark.parametrize("disjoint", [False, True], ids=["overlap", "disjoint-shared-storage"])
@pytest.mark.parametrize("return_value", [False, True])
def test_native_output_storage_alias(registered, device, dtype, indices_type, output_name, argument, disjoint, return_value):
    storage = torch.zeros(8192, dtype=torch.uint8, device=device)
    x, _ = aligned_tensor(NativeAPI, 1, 128, dtype, device)
    values, _ = aligned_tensor(NativeAPI, 1, 7, dtype, device, output=True)
    indices, _ = aligned_tensor(NativeAPI, 1, 7, indices_type, device, output=True)
    args = dict(input=x, topk=7, begin=None, end=None, sorted_value=False, sorted_index=False,
                output_value=values, output_index=indices, output_idx_offset=None,
                idx_oob_fill_value=-1, value_oob_fill_value=float("-inf"),
                return_value=return_value, abort_when_nan_found=True)
    args[output_name] = shared_storage_view(storage, args[output_name].dtype, 7,
                                            1024 if disjoint else 0, NativeAPI.get_stride_requirement()[1])
    if argument == "input":
        args[argument] = shared_storage_view(storage, dtype, 128, 0, NativeAPI.get_stride_requirement()[0])
    elif argument == "output_index":
        args[argument] = shared_storage_view(storage, indices_type, 7, 0, NativeAPI.get_stride_requirement()[1])
    else:
        args[argument] = storage[:4].view(torch.int32)
        args[argument].fill_(128 if argument == "end" else 100)
    before = storage.clone()
    with pytest.raises(RuntimeError, match="overlap.*storage|storage.*overlap"):
        torch.ops.deep_select.topk(**args)
    torch.cuda.synchronize(device)
    assert torch.equal(storage, before)


@pytest.mark.parametrize("argument", ["output_value", "output_index", "end", "output_idx_offset"])
def test_native_cross_device(registered, device, argument):
    if torch.cuda.device_count() < 2:
        pytest.skip("two visible CUDA devices are required")
    other = next(index for index in range(torch.cuda.device_count()) if index != device.index)
    x, _ = aligned_tensor(NativeAPI, 1, 128, torch.float32, device)
    values, _ = aligned_tensor(NativeAPI, 1, 7, torch.float32, device, output=True)
    indices, _ = aligned_tensor(NativeAPI, 1, 7, torch.int64, device, output=True)
    args = dict(input=x, topk=7, begin=None, end=None, sorted_value=False, sorted_index=False,
                output_value=values, output_index=indices, output_idx_offset=None,
                idx_oob_fill_value=-1, value_oob_fill_value=float("-inf"),
                return_value=True, abort_when_nan_found=True)
    if argument.startswith("output_") and argument != "output_idx_offset":
        args[argument], _ = aligned_tensor(NativeAPI, 1, 7, args[argument].dtype,
                                           torch.device("cuda", other), output=True)
    else:
        args[argument] = torch.full((1,), 7, dtype=torch.int32, device=torch.device("cuda", other))
    with pytest.raises(RuntimeError, match="device"):
        torch.ops.deep_select.topk(**args)
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices_type", INDEX_DTYPES)
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("width", [16384, 524288])
def test_sm100_sm103_optimized_nan_nonabort(api, device, dtype, indices_type, return_value, width):
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("optimized SM100/SM103 NaN paths")
    x, _ = aligned_tensor(api, 2, width, dtype, device)
    x.fill_(1)
    x[0, width - 2] = float("nan")
    x[1, width - 1] = float("nan")
    end = torch.full((2,), width - 1, dtype=torch.int32, device=device)
    offsets = torch.tensor([100, -200], dtype=torch.int32, device=device)
    values, indices = api.topk(x, 512, end=end, output_idx_offset=offsets,
                               indices_type=indices_type, return_value=return_value,
                               idx_oob_fill_value=-1, abort_when_nan_found=False)
    assert indices[0, 0].item() == 0x3F3F3F3F
    check_result(api, x[1:], 512, (values[1:] if return_value else None, indices[1:]),
                 end=end[1:], output_idx_offset=offsets[1:], indices_type=indices_type,
                 return_value=return_value)
    torch.cuda.synchronize(device)
