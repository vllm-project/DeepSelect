// Stable-ABI replacements for the tensor check helpers previously provided by
// kerutils/supplemental/torch_tensors.h (which requires the unstable libtorch
// C++ ABI and pybind11).
#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <type_traits>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

namespace deep_select {

using torch::stable::Tensor;

// Check whether the given tensor or optional tensor satisfies the given condition
// If tensor_or_opt is a tensor, check_fn is applied directly
// If tensor_or_opt is an optional tensor, check_fn is applied only when the optional has value
template <typename T>
static inline bool _check_optional_tensor(const T& tensor_or_opt, const std::function<bool(const Tensor&)>& check_fn) {
    if constexpr (std::is_same_v<T, Tensor>) {
        return check_fn(tensor_or_opt);
    } else {
        if (tensor_or_opt.has_value()) {
            return check_fn(tensor_or_opt.value());
        } else {
            return true;
        }
    }
}

static inline bool _check_sizes(const Tensor& tensor, std::initializer_list<int64_t> expected) {
    if (tensor.dim() != (int64_t)expected.size()) {
        return false;
    }
    int64_t i = 0;
    for (int64_t e : expected) {
        if (tensor.size(i++) != e) {
            return false;
        }
    }
    return true;
}

// Get the pointer of the given optional tensor
// Return (PtrT*)tensor.data_ptr() if the optional has value, nullptr otherwise
template <typename PtrT>
static inline PtrT* get_optional_tensor_ptr(const std::optional<Tensor>& tensor_or_opt) {
    if (tensor_or_opt.has_value()) {
        return static_cast<PtrT*>(tensor_or_opt->data_ptr());
    } else {
        return nullptr;
    }
}

static inline uint64_t checked_add(uint64_t a, uint64_t b) {
    STD_TORCH_CHECK(a <= uint64_t(std::numeric_limits<int64_t>::max()) &&
                    b <= uint64_t(std::numeric_limits<int64_t>::max()) - a,
                    "tensor address arithmetic exceeds int64");
    return a + b;
}

static inline uint64_t checked_mul(uint64_t a, uint64_t b) {
    STD_TORCH_CHECK(a == 0 || b <= uint64_t(std::numeric_limits<int64_t>::max()) / a,
                    "tensor address arithmetic exceeds int64");
    return a * b;
}

static inline uint64_t storage_size_bytes(const Tensor& tensor) {
    int64_t bytes = 0;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(tensor.get(), &bytes));
    STD_TORCH_CHECK(bytes >= 0, "tensor storage size must be nonnegative");
    return static_cast<uint64_t>(bytes);
}

static inline uint64_t storage_offset_bytes(const Tensor& tensor) {
    int64_t offset = 0;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_offset(tensor.get(), &offset));
    STD_TORCH_CHECK(offset >= 0, "tensor storage offset must be nonnegative");
    return checked_mul(static_cast<uint64_t>(offset), tensor.element_size());
}

static inline void check_storage_bounds(const char* name, const Tensor& tensor, uint64_t row_padding = 1) {
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
        STD_TORCH_CHECK(tensor.size(dim) >= 0 && tensor.stride(dim) >= 0,
                        name, " sizes and strides must be nonnegative");
    }
    if (tensor.numel() == 0) return;
    uint64_t last = 0;
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
        last = checked_add(last, checked_mul(tensor.size(dim) - 1, tensor.stride(dim)));
    }
    const uint64_t itemsize = tensor.element_size();
    uint64_t end = checked_add(storage_offset_bytes(tensor), checked_mul(checked_add(last, 1), itemsize));
    if (row_padding != 1) {
        const uint64_t row_bytes = checked_mul(tensor.size(-1), itemsize);
        const uint64_t padded = checked_mul(checked_add(row_bytes, row_padding - 1) / row_padding, row_padding);
        end = checked_add(end, padded - row_bytes);
    }
    STD_TORCH_CHECK(end <= storage_size_bytes(tensor), name,
                    " backing storage must include all elements and the final row padded to a multiple of ",
                    row_padding, " bytes");
}

struct StorageInterval {
    uintptr_t start;
    uint64_t bytes;
};

static inline StorageInterval storage_interval(const Tensor& tensor) {
    void* data = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(tensor.get(), &data));
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(data);
    const uint64_t offset = storage_offset_bytes(tensor);
    const uint64_t bytes = storage_size_bytes(tensor);
    STD_TORCH_CHECK(offset <= bytes && offset <= pointer, "invalid tensor storage offset");
    const uintptr_t start = pointer - offset;
    STD_TORCH_CHECK(bytes <= std::numeric_limits<uintptr_t>::max() - start,
                    "tensor storage address range overflows uintptr_t");
    return {start, bytes};
}

static inline void check_storage_disjoint(const char* output_name, const Tensor& output,
                                          const char* other_name, const Tensor& other) {
    if (output.numel() == 0 || other.numel() == 0) return;
    const auto a = storage_interval(output);
    const auto b = storage_interval(other);
    const bool overlaps = a.start <= b.start ? b.start - a.start < a.bytes : a.start - b.start < b.bytes;
    STD_TORCH_CHECK(!overlaps, "`", output_name, "` must not overlap storage with `", other_name,
                    "` (including disjoint views of shared storage)");
}

static inline void check_pointer_alignment(const char* name, const Tensor& tensor) {
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 32 == 0,
                    name, ".data_ptr() must be 32-byte aligned");
}

} // namespace deep_select

// Check whether the given tensor (or optional<tensor>) is on CUDA GPU
#define DS_CHECK_DEVICE(tensor) STD_TORCH_CHECK(deep_select::_check_optional_tensor(tensor, [](const deep_select::Tensor& t) { return t.is_cuda(); }), #tensor " must be on CUDA")

// Check whether the given tensor (or optional<tensor>) has the given shape
#define DS_CHECK_SHAPE(tensor, ...) STD_TORCH_CHECK(deep_select::_check_optional_tensor(tensor, [&](const deep_select::Tensor& t) { return deep_select::_check_sizes(t, {__VA_ARGS__}); }), #tensor " must have shape (" #__VA_ARGS__ ")")

// Check whether the given tensor (or optional<tensor>) is contiguous
#define DS_CHECK_CONTIGUOUS(tensor) STD_TORCH_CHECK(deep_select::_check_optional_tensor(tensor, [](const deep_select::Tensor& t) { return t.is_contiguous(); }), #tensor " must be contiguous")

// Check whether the last dimension of the given tensor (or optional<tensor>) is contiguous
#define DS_CHECK_LAST_DIM_CONTIGUOUS(tensor) STD_TORCH_CHECK(deep_select::_check_optional_tensor(tensor, [](const deep_select::Tensor& t) { return t.size(-1) == 1 || t.stride(-1) == 1; }), #tensor " must have contiguous last dimension")

// Check whether the given tensor (or optional<tensor>) has the specified dtype
#define DS_CHECK_DTYPE(tensor, target_dtype) STD_TORCH_CHECK(deep_select::_check_optional_tensor(tensor, [&](const deep_select::Tensor& t) { return t.scalar_type() == (target_dtype); }), #tensor " must have dtype " #target_dtype)
