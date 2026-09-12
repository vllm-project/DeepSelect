// Stable-ABI replacements for the tensor check helpers previously provided by
// kerutils/supplemental/torch_tensors.h (which requires the unstable libtorch
// C++ ABI and pybind11).
#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <type_traits>

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

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
