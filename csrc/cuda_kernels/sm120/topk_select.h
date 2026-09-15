#pragma once

#include "structs.h"

namespace topk_select_sm120 {
inline constexpr int SEGMENTED_TOPK = 512;
inline constexpr int SEGMENTED_PARTITIONS = 8;
inline constexpr int SEGMENTED_WORKSPACE_STRIDE = 2 * SEGMENTED_TOPK + 1;

bool use_segmented_topk(const TopkSelectArgs& args, bool indices_int64);
void run_segmented_topk_select_kernel(const TopkSelectArgs& args, bool is_bfloat16, int32_t* workspace);
void run_topk_select_kernel(const TopkSelectArgs& args, bool is_bfloat16, bool indices_int64);
}
