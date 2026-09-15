// This file contains only the host-side topk() function, torch library registration
// (stable ABI) and the Python module init.
// Kernel template instantiations are in separate files for parallel compilation.
#include <Python.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime_api.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <tuple>

#include "stable_tensor_checks.h"
#include "structs.h"

#ifndef DEEP_SELECT_BUILD_SM100
#define DEEP_SELECT_BUILD_SM100 1
#endif
#ifndef DEEP_SELECT_BUILD_SM120
#define DEEP_SELECT_BUILD_SM120 0
#endif
#ifndef DEEP_SELECT_BUILD_SM121
#define DEEP_SELECT_BUILD_SM121 0
#endif

#if DEEP_SELECT_BUILD_SM100
#include <cuda_bf16.h>
#include "dispatch_utils.h"
#include "cuda_kernels/config.h"
#include "cuda_kernels/v3/topk_select.h"
#include "cuda_kernels/v3_fp32/topk_select.h"
#include "cuda_kernels/v3_cluster/topk_select.h"
#endif
#if DEEP_SELECT_BUILD_SM120 || DEEP_SELECT_BUILD_SM121
#include "cuda_kernels/sm120/topk_select.h"
#endif

using deep_select::Tensor;
using torch::headeronly::ScalarType;

void topk(
    const Tensor& input,
    int64_t topk,
    const std::optional<Tensor>& begin,
    const std::optional<Tensor>& end,
    bool sorted_value,
    bool sorted_index,
    const std::optional<Tensor>& output_value,
    const Tensor& output_index,
    const std::optional<Tensor>& output_idx_offset,
    int64_t idx_oob_fill_value,
    double value_oob_fill_value,
    bool return_value,
    bool abort_when_nan_found
) {
    STD_TORCH_CHECK(input.dim() == 2, "input must have two dimensions");
    const int64_t batch_size = input.size(0);
    const int64_t vocab_size = input.size(1);
    const ScalarType value_t = input.scalar_type();
    const ScalarType output_index_t = output_index.scalar_type();

    STD_TORCH_CHECK(batch_size >= 0 && batch_size <= INT32_MAX, "batch_size must fit int32");
    STD_TORCH_CHECK(vocab_size >= 0 && vocab_size < MAX_VOCAB_SIZE, "vocab_size must be < 2^23");
    STD_TORCH_CHECK(topk > 0 && topk <= 4096, "topk must be > 0 and <= 4096");
    STD_TORCH_CHECK(value_t == ScalarType::BFloat16 || value_t == ScalarType::Float, "input dtype must be bfloat16 or float32");
    STD_TORCH_CHECK(output_index_t == ScalarType::Int || output_index_t == ScalarType::Long, "indices dtype must be int32 or int64");
    STD_TORCH_CHECK(idx_oob_fill_value >= INT32_MIN && idx_oob_fill_value <= INT32_MAX, "idx_oob_fill_value must fit int32");
    STD_TORCH_CHECK(!std::isfinite(value_oob_fill_value) || std::abs(value_oob_fill_value) <= std::numeric_limits<float>::max(),
                    "finite value_oob_fill_value must fit float32");
    STD_TORCH_CHECK(!(sorted_value && !return_value), "`return_value` must be enabled when `sorted_value` is True");
    STD_TORCH_CHECK(!(sorted_value && sorted_index), "`sorted_value` and `sorted_index` cannot be used at the same time");
    STD_TORCH_CHECK(!(sorted_value && value_t == ScalarType::BFloat16), "`sorted_value` is only supported for float32 input");
    STD_TORCH_CHECK(!begin.has_value(), "`begin` is not supported currently");
    if (return_value) {
        STD_TORCH_CHECK(output_value.has_value(), "`output_value` must not be `None` when `return_value` is True");
    }

    DS_CHECK_DEVICE(input);
    const auto device_index = input.get_device_index();
    auto check_device = [&](const char* name, const auto& tensor) {
        STD_TORCH_CHECK(deep_select::_check_optional_tensor(tensor, [&](const Tensor& t) {
            return t.is_cuda() && t.get_device_index() == device_index;
        }), "`", name, "` must be on the same CUDA device as `input`");
    };
    check_device("begin", begin);
    check_device("end", end);
    check_device("output_value", output_value);
    check_device("output_index", output_index);
    check_device("output_idx_offset", output_idx_offset);

    DS_CHECK_SHAPE(input, batch_size, vocab_size);
    DS_CHECK_SHAPE(begin, batch_size);
    DS_CHECK_SHAPE(end, batch_size);
    DS_CHECK_SHAPE(output_value, batch_size, topk);
    DS_CHECK_SHAPE(output_index, batch_size, topk);
    DS_CHECK_SHAPE(output_idx_offset, batch_size);

    DS_CHECK_DTYPE(begin, ScalarType::Int);
    DS_CHECK_DTYPE(end, ScalarType::Int);
    DS_CHECK_DTYPE(output_value, value_t);
    DS_CHECK_DTYPE(output_idx_offset, ScalarType::Int);

    DS_CHECK_LAST_DIM_CONTIGUOUS(input);
    DS_CHECK_CONTIGUOUS(begin);
    DS_CHECK_CONTIGUOUS(end);
    DS_CHECK_LAST_DIM_CONTIGUOUS(output_value);
    DS_CHECK_LAST_DIM_CONTIGUOUS(output_index);
    DS_CHECK_CONTIGUOUS(output_idx_offset);

    auto check_dim0_stride = [&](const char tensor_name[], const Tensor& tensor, uint32_t alignment_requirement_bytes) {
        const int64_t cur_stride = tensor.stride(0);
        const uint64_t itemsize = tensor.element_size();
        STD_TORCH_CHECK(cur_stride >= 0, tensor_name, ".stride(0) must be nonnegative");
        STD_TORCH_CHECK(deep_select::checked_mul(cur_stride, itemsize) % alignment_requirement_bytes == 0,
            std::format("{}.stride(0) (currently {} numbers) must be a multiple of {} Bytes ({} numbers)",
                tensor_name, cur_stride,
                alignment_requirement_bytes, alignment_requirement_bytes / itemsize
            )
        );
        deep_select::check_pointer_alignment(tensor_name, tensor);
    };
    check_dim0_stride("input", input, INPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    check_dim0_stride("output_index", output_index, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    if (output_value.has_value()) {
        check_dim0_stride("output_value", *output_value, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    }
    deep_select::check_storage_bounds("input", input, 128);
    if (end.has_value()) deep_select::check_storage_bounds("end", *end);
    if (output_idx_offset.has_value()) deep_select::check_storage_bounds("output_idx_offset", *output_idx_offset);
    auto check_output = [&](const char* name, const Tensor& output) {
        STD_TORCH_CHECK(batch_size <= 1 || output.stride(0) >= topk, name, " rows must not overlap");
        deep_select::check_storage_bounds(name, output);
        deep_select::check_storage_disjoint(name, output, "input", input);
        if (end.has_value()) deep_select::check_storage_disjoint(name, output, "end", *end);
        if (output_idx_offset.has_value()) deep_select::check_storage_disjoint(name, output, "output_idx_offset", *output_idx_offset);
    };
    check_output("output_index", output_index);
    if (output_value.has_value()) {
        check_output("output_value", *output_value);
        deep_select::check_storage_disjoint("output_index", output_index, "output_value", *output_value);
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);
    cudaDeviceProp device_prop;
    STD_TORCH_CHECK(cudaGetDeviceProperties(&device_prop, device_index) == cudaSuccess,
                    "failed to get CUDA device properties");
    const bool use_sm100 = DEEP_SELECT_BUILD_SM100 && device_prop.major == 10;
    const bool use_native_sm12 = device_prop.major == 12 &&
        ((DEEP_SELECT_BUILD_SM120 && device_prop.minor == 0) ||
         (DEEP_SELECT_BUILD_SM121 && device_prop.minor == 1));
    STD_TORCH_CHECK(use_sm100 || use_native_sm12, "DeepSelect was not built for this CUDA capability: ",
                    device_prop.major, ".", device_prop.minor);
    if (batch_size == 0) return;
    STD_TORCH_CHECK(!use_sm100 || vocab_size > 0,
                    "SM100/SM103 require vocab_size > 0 for nonempty batches");
    STD_TORCH_CHECK(batch_size <= device_prop.maxGridSize[0], "batch_size exceeds the CUDA grid limit");

    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));

    TopkSelectArgs args = {
        (uint32_t)batch_size,
        (uint32_t)vocab_size,
        (uint32_t)topk,

        input.data_ptr(),
        deep_select::get_optional_tensor_ptr<void>(output_value),
        output_index.data_ptr(),
        deep_select::get_optional_tensor_ptr<int>(begin),
        deep_select::get_optional_tensor_ptr<int>(end),
        deep_select::get_optional_tensor_ptr<int>(output_idx_offset),

        (uint64_t)input.stride(0),
        output_value.has_value() ? (uint64_t)output_value->stride(0) : 0,
        (uint64_t)output_index.stride(0),

        sorted_value,
        sorted_index,
        return_value,
        (int)idx_oob_fill_value,
        (float)value_oob_fill_value,
        abort_when_nan_found,

        device_prop.sharedMemPerBlockOptin,
        static_cast<cudaStream_t>(stream_ptr)
    };

#if DEEP_SELECT_BUILD_SM120 || DEEP_SELECT_BUILD_SM121
    if (use_native_sm12) {
        const bool segmented = topk_select_sm120::use_segmented_topk(args, output_index_t == ScalarType::Long);
        if (segmented) {
            const uint64_t blocks = deep_select::checked_mul(batch_size, topk_select_sm120::SEGMENTED_PARTITIONS);
            STD_TORCH_CHECK(blocks <= static_cast<uint64_t>(device_prop.maxGridSize[0]),
                            "segmented batch_size exceeds the CUDA grid limit");
            const uint64_t workspace_elements = deep_select::checked_mul(blocks, topk_select_sm120::SEGMENTED_WORKSPACE_STRIDE);
            STD_TORCH_CHECK(workspace_elements <= UINT32_MAX, "segmented workspace indexing exceeds uint32");
            deep_select::checked_mul(workspace_elements, sizeof(int32_t));
            auto workspace = torch::stable::empty(
                {batch_size, topk_select_sm120::SEGMENTED_PARTITIONS, topk_select_sm120::SEGMENTED_WORKSPACE_STRIDE},
                ScalarType::Int, std::nullopt, input.device());
            topk_select_sm120::run_segmented_topk_select_kernel(
                args, value_t == ScalarType::BFloat16, static_cast<int32_t*>(workspace.data_ptr()));
        } else {
            topk_select_sm120::run_topk_select_kernel(args, value_t == ScalarType::BFloat16, output_index_t == ScalarType::Long);
        }
        return;
    }
#endif
#if DEEP_SELECT_BUILD_SM100
    STD_TORCH_CHECK(device_prop.multiProcessorCount > 0, "CUDA device must have at least one multiprocessor");
    uint32_t num_sm = device_prop.multiProcessorCount;
    uint32_t num_waves = (batch_size + num_sm-1) / num_sm;

    STD_TORCH_CHECK(value_t == ScalarType::BFloat16 || value_t == ScalarType::Float, "input dtype must be bfloat16 or float32");
    if (value_t == ScalarType::BFloat16) {
        STD_TORCH_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                    "vocab_size must be < 2^23 for bfloat16 input");
        STD_TORCH_CHECK(topk <= 4096, "topk must be <= 4096");
        if (batch_size <= 6 && (uint32_t)vocab_size >= 512u * 1024u && topk <= 1024) {  // TODO Tune
            INTEGER_TYPE_SWITCH(output_index_t, OutIdxT, [&]() {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        topk_select_bf16_cluster::run_topk_select_kernel<
                            TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, 1024, 256, 1, 4096, 4096, 16, 512, 16>>(args);
                    });
                });
            });
        } else {
            INTEGER_TYPE_SWITCH(output_index_t, OutIdxT, [&]() {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        //   wave == 1 -> occ1 (512t / B8192 / B2 4096 / TMA5 rounds)
                        //   otherwise -> occ2 (256t / B4096 / B2 4096 / TMA3 or 4)
                        auto dispatch = [&]<uint32_t MAX_TOPK>() {
                            if (num_waves == 1)
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 512, 1, 8192, 4096, 5>>(args);
                            else if constexpr (MAX_TOPK <= 512)
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 256, 2, 4096, 4096, 4>>(args);
                            else
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 256, 2, 4096, 4096, 3>>(args);
                        };
                        if (topk <= 512) {
                            dispatch.template operator()<512>();
                        } else if (topk <= 1024) {
                            dispatch.template operator()<1024>();
                        } else {
                            // Big-topk coverage tier, topk in (1024, 4096]: one correctness-only tuple
                            // (512t / occ1 / B8192 / B2 4096 / TMA3 / max_topk 4096), no wave split.
                            topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, 4096, 512, 1, 8192, 4096, 3>>(args);
                        }
                    });
                });
            });
        }
    } else {
        STD_TORCH_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                    "vocab_size must be < 2^23 for float32 input");
        STD_TORCH_CHECK(topk <= 4096, "topk must be <= 4096");

        INTEGER_TYPE_SWITCH(output_index_t, OutIdxT, [&]() {
            //   topk <= 1024        -> 512t / B8192 / B2 4096 / TMA3
            //   topk in (1024,4096] -> 256t / B4096 / B2 4096 / TMA3 (correctness-only coverage tier)
            auto dispatch = [&]<bool SORTED_VALUE, bool SORTED_INDEX, bool RETURN_VALUE>() {
                auto launch = [&]<uint32_t MAX_TOPK, uint32_t NUM_THREADS, uint32_t B>() {
                    topk_select_fp32::run_topk_select_kernel<TopkSelectConfig<float, OutIdxT, SORTED_VALUE, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, NUM_THREADS, 1, B, 4096, 3>>(args);
                };
                if (topk <= 512) {
                    launch.template operator()<512, 512, 8192>();
                } else if (topk <= 1024) {
                    launch.template operator()<1024, 512, 8192>();
                } else {
                    launch.template operator()<4096, 256, 4096>();
                }
            };
            if (sorted_value) {
                dispatch.template operator()<true, false, true>();
            } else {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        dispatch.template operator()<false, SORTED_INDEX, RETURN_VALUE>();
                    });
                });
            }
        });
    }
#endif
}

std::tuple<int64_t, int64_t> get_alignment_requirement() {
    return {INPUT_STRIDE_ALIGNMENT_REQUIREMENT, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT};
}

STABLE_TORCH_LIBRARY(deep_select, m) {
    m.def(
        "topk(Tensor input, int topk, Tensor? begin, Tensor? end, "
        "bool sorted_value, bool sorted_index, "
        "Tensor(a!)? output_value, Tensor(b!) output_index, Tensor? output_idx_offset, "
        "int idx_oob_fill_value, float value_oob_fill_value, "
        "bool return_value, bool abort_when_nan_found) -> ()");
    m.def("get_alignment_requirement() -> (int, int)");
}

STABLE_TORCH_LIBRARY_IMPL(deep_select, CUDA, m) {
    m.impl("topk", TORCH_BOX(&topk));
}

STABLE_TORCH_LIBRARY_IMPL(deep_select, CompositeExplicitAutograd, m) {
    m.impl("get_alignment_requirement", TORCH_BOX(&get_alignment_requirement));
}

// The module/init names follow TORCH_EXTENSION_NAME (defined by the build
// system, e.g. deep_select_cuda for the standalone package, _deepselect_C
// when built inside vLLM).
#define _DS_CONCAT_IMPL(A, B) A##B
#define _DS_CONCAT(A, B) _DS_CONCAT_IMPL(A, B)
#define _DS_STRINGIFY_IMPL(A) #A
#define _DS_STRINGIFY(A) _DS_STRINGIFY_IMPL(A)

static struct PyModuleDef deep_select_cuda_module = {
    PyModuleDef_HEAD_INIT,
    _DS_STRINGIFY(TORCH_EXTENSION_NAME),
    nullptr,
    -1,
    nullptr,
};

PyMODINIT_FUNC _DS_CONCAT(PyInit_, TORCH_EXTENSION_NAME)(void) {
    return PyModule_Create(&deep_select_cuda_module);
}
