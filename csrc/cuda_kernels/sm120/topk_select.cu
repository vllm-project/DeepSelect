#include "topk_select.h"

#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <stdexcept>
#include <string>

namespace topk_select_sm120 {
namespace {

void check_kernel_launch() {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("SM120 topk kernel launch failed: ") + cudaGetErrorString(error));
}

constexpr int NUM_THREADS = 512;
constexpr int ITEMS_PER_THREAD = 8;
constexpr int MAX_TOPK = NUM_THREADS * ITEMS_PER_THREAD;
using BlockSort = cub::BlockRadixSort<uint64_t, NUM_THREADS, ITEMS_PER_THREAD, cub::NullType, 4, false>;
using BlockScan = cub::BlockScan<uint32_t, NUM_THREADS>;

struct SharedMemory {
    union {
        uint32_t indices[MAX_TOPK];
        BlockSort::TempStorage sort;
        BlockScan::TempStorage scan;
    } work;
    uint32_t histogram[256];
    uint32_t prefix;
    uint32_t rank;
    uint32_t count;
    uint32_t have_nan;
};
static_assert(sizeof(SharedMemory) <= 48 * 1024,
              "SM120 selector must fit the default per-block shared-memory budget");

__device__ __forceinline__ uint32_t ordered_key(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0xffffffffU;
    return bits ^ ((bits & 0x80000000U) ? 0xffffffffU : 0x80000000U);
}

template<typename ValueT, typename IndexT>
__global__ __launch_bounds__(NUM_THREADS) void topk_kernel(TopkSelectArgs args) {
    __shared__ SharedMemory smem;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const ValueT* input = static_cast<const ValueT*>(args.input) + row * args.stride_input_batch;
    IndexT* output_index = static_cast<IndexT*>(args.output_index) + row * args.stride_output_index_batch;
    ValueT* output_value = args.return_value
        ? static_cast<ValueT*>(args.output_value) + row * args.stride_output_value_batch : nullptr;
    const int32_t offset = args.output_idx_offset ? args.output_idx_offset[row] : 0;
    const int32_t end = args.end_ptr ? args.end_ptr[row] : static_cast<int32_t>(args.vocab_size);
    if (end < 0 || static_cast<uint32_t>(end) > args.vocab_size) {
        if (tid == 0) asm volatile("trap;");
        return;
    }
    const uint32_t length = static_cast<uint32_t>(end);
    const uint32_t valid_count = min(length, args.topk);
    if (tid == 0) {
        smem.prefix = 0;
        smem.rank = args.topk;
        smem.count = 0;
        smem.have_nan = 0;
    }
    __syncthreads();

    if (length <= args.topk) {
        for (uint32_t i = tid; i < length; i += NUM_THREADS) smem.work.indices[i] = i;
        __syncthreads();
    } else {
        uint32_t mask = 0;
        for (int shift = 24; shift >= 0; shift -= 8) {
            if (tid < 256) smem.histogram[tid] = 0;
            __syncthreads();
            const uint32_t prefix = smem.prefix;
            for (uint32_t i = tid; i < length; i += NUM_THREADS) {
                const float value = static_cast<float>(input[i]);
                const uint32_t bits = __float_as_uint(value);
                if (shift == 24 && (bits & 0x7fffffffU) > 0x7f800000U)
                    atomicExch(&smem.have_nan, 1U);
                const uint32_t key = ordered_key(value);
                if ((key & mask) == prefix) atomicAdd(&smem.histogram[(key >> shift) & 255U], 1U);
            }
            __syncthreads();
            if (smem.have_nan) {
                if (tid == 0) {
                    if (args.abort_when_nan_found) asm volatile("trap;");
                    else output_index[0] = static_cast<IndexT>(0x3f3f3f3f);
                }
                return;
            }
            if (tid == 0) {
                uint32_t rank = smem.rank;
                for (int bucket = 255; bucket >= 0; --bucket) {
                    const uint32_t count = smem.histogram[bucket];
                    if (rank <= count) {
                        smem.prefix |= static_cast<uint32_t>(bucket) << shift;
                        smem.rank = rank;
                        break;
                    }
                    rank -= count;
                }
            }
            mask |= 255U << shift;
            __syncthreads();
        }

        const uint32_t threshold = smem.prefix;
        const uint32_t quota = smem.rank;
        const uint32_t chunk = (length + NUM_THREADS - 1) / NUM_THREADS;
        const uint32_t lo = min(tid * chunk, length);
        const uint32_t hi = min(lo + chunk, length);
        uint32_t equal_count = 0;
        for (uint32_t i = lo; i < hi; ++i)
            equal_count += ordered_key(static_cast<float>(input[i])) == threshold;
        uint32_t equal_before;
        BlockScan(smem.work.scan).ExclusiveSum(equal_count, equal_before);
        __syncthreads();
        for (uint32_t i = lo; i < hi; ++i) {
            const uint32_t key = ordered_key(static_cast<float>(input[i]));
            const bool take_equal = key == threshold && equal_before++ < quota;
            if (key > threshold || take_equal) {
                const uint32_t slot = atomicAdd(&smem.count, 1U);
                smem.work.indices[slot] = i;
            }
        }
        __syncthreads();
    }

    if (args.sorted_value || args.sorted_index) {
        uint64_t keys[ITEMS_PER_THREAD];
        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            const uint32_t slot = tid * ITEMS_PER_THREAD + j;
            uint64_t key = 0;
            if (slot < valid_count) {
                const uint32_t index = smem.work.indices[slot];
                key = static_cast<uint64_t>(~index);
                if (args.sorted_value)
                    key |= static_cast<uint64_t>(ordered_key(static_cast<float>(input[index]))) << 32;
            }
            keys[j] = key;
        }
        __syncthreads();
        BlockSort(smem.work.sort).SortDescending(keys);
        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            const uint32_t slot = tid * ITEMS_PER_THREAD + j;
            if (slot < args.topk) {
                const uint32_t index = ~static_cast<uint32_t>(keys[j]);
                const bool valid = slot < valid_count;
                output_index[slot] = valid ? static_cast<IndexT>(static_cast<int64_t>(index) + offset)
                                           : static_cast<IndexT>(args.idx_oob_fill_value);
                if (args.return_value)
                    output_value[slot] = valid ? input[index] : static_cast<ValueT>(args.value_oob_fill_value);
            }
        }
    } else {
        for (uint32_t slot = tid; slot < args.topk; slot += NUM_THREADS) {
            const bool valid = slot < valid_count;
            const uint32_t index = valid ? smem.work.indices[slot] : 0;
            output_index[slot] = valid ? static_cast<IndexT>(static_cast<int64_t>(index) + offset)
                                       : static_cast<IndexT>(args.idx_oob_fill_value);
            if (args.return_value)
                output_value[slot] = valid ? input[index] : static_cast<ValueT>(args.value_oob_fill_value);
        }
    }
}

constexpr int FAST_THREADS = 256;
using FastScan = cub::BlockScan<uint32_t, FAST_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;

constexpr int STREAM_TOPK = 512;
constexpr int STREAM_TILE_ITEMS = 4;
constexpr int STREAM_CANDIDATE_ITEMS = 10;
constexpr int STREAM_CAPACITY = FAST_THREADS * STREAM_CANDIDATE_ITEMS;
constexpr int STREAM_COMPACT_COUNT = 1536;
static_assert(STREAM_COMPACT_COUNT - 1 + FAST_THREADS * STREAM_TILE_ITEMS < STREAM_CAPACITY);

struct StreamingSharedMemory {
    uint32_t keys[STREAM_CAPACITY];
    uint32_t indices[STREAM_CAPACITY];
    uint32_t histogram[FAST_THREADS / 32][257];
    FastScan::TempStorage scan;
    uint32_t worst_keys[FAST_THREADS / 32];
    uint32_t worst_indices[FAST_THREADS / 32];
    uint32_t bucket;
    uint32_t rank;
    uint32_t complete;
    uint32_t threshold_key;
    uint32_t threshold_index;
};
static_assert(sizeof(StreamingSharedMemory) <= 32 * 1024);

__device__ __forceinline__ bool streaming_better(
    uint32_t key, uint32_t index, uint32_t other_key, uint32_t other_index) {
    return key > other_key || (key == other_key && index < other_index);
}

template<bool IndexKey, int FirstShift, int LastShift>
__device__ __forceinline__ void streaming_radix(
    StreamingSharedMemory& smem,
    const uint32_t (&keys)[STREAM_CANDIDATE_ITEMS],
    const uint32_t (&indices)[STREAM_CANDIDATE_ITEMS],
    uint32_t& active, uint32_t& selected, uint32_t& rank) {
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid / 32;
    for (int shift = FirstShift; shift >= LastShift; shift -= 8) {
        #pragma unroll
        for (int w = 0; w < FAST_THREADS / 32; ++w) smem.histogram[w][tid] = 0;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < STREAM_CANDIDATE_ITEMS; ++j) {
            const uint32_t key = IndexKey ? ~indices[j] : keys[j];
            if (active & (1U << j))
                atomicAdd(&smem.histogram[warp][(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        const uint32_t bucket = 255U - tid;
        uint32_t bucket_count = 0;
        #pragma unroll
        for (int w = 0; w < FAST_THREADS / 32; ++w)
            bucket_count += smem.histogram[w][bucket];
        uint32_t greater;
        FastScan(smem.scan).ExclusiveSum(bucket_count, greater);
        __syncthreads();
        if (greater < rank && rank <= greater + bucket_count) {
            smem.bucket = bucket;
            smem.rank = rank - greater;
            smem.complete = rank - greater == bucket_count;
        }
        __syncthreads();
        const uint32_t pivot = smem.bucket;
        const bool complete = smem.complete != 0;
        rank = smem.rank;
        uint32_t next_active = 0;
        #pragma unroll
        for (int j = 0; j < STREAM_CANDIDATE_ITEMS; ++j) {
            const uint32_t key = IndexKey ? ~indices[j] : keys[j];
            const uint32_t digit = (key >> shift) & 255U;
            if (active & (1U << j)) {
                if (digit > pivot || (complete && digit == pivot)) selected |= 1U << j;
                else if (digit == pivot) next_active |= 1U << j;
            }
        }
        active = next_active;
        if (complete) break;
    }
}

template<typename ValueT>
__device__ __forceinline__ void streaming_compact(StreamingSharedMemory& smem, uint32_t count) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31U;
    const uint32_t warp = tid / 32;
    uint32_t keys[STREAM_CANDIDATE_ITEMS];
    uint32_t indices[STREAM_CANDIDATE_ITEMS];
    uint32_t active = 0;
    #pragma unroll
    for (int j = 0; j < STREAM_CANDIDATE_ITEMS; ++j) {
        const uint32_t slot = tid + j * FAST_THREADS;
        keys[j] = slot < count ? smem.keys[slot] : 0;
        indices[j] = slot < count ? smem.indices[slot] : 0;
        if (slot < count) active |= 1U << j;
    }
    __syncthreads();
    uint32_t rank = STREAM_TOPK;
    uint32_t selected = 0;
    streaming_radix<false, 24, sizeof(ValueT) == 2 ? 16 : 0>(
        smem, keys, indices, active, selected, rank);
    if (!smem.complete)
        streaming_radix<true, 16, 0>(smem, keys, indices, active, selected, rank);

    uint32_t slot;
    FastScan(smem.scan).ExclusiveSum(static_cast<uint32_t>(__popc(selected)), slot);
    __syncthreads();
    uint32_t worst_key = 0xffffffffU;
    uint32_t worst_index = 0;
    #pragma unroll
    for (int j = 0; j < STREAM_CANDIDATE_ITEMS; ++j) {
        if (selected & (1U << j)) {
            const uint32_t key = keys[j];
            const uint32_t index = indices[j];
            smem.keys[slot] = key;
            smem.indices[slot++] = index;
            if (streaming_better(worst_key, worst_index, key, index)) {
                worst_key = key;
                worst_index = index;
            }
        }
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
        const uint32_t key = __shfl_down_sync(0xffffffffU, worst_key, delta);
        const uint32_t index = __shfl_down_sync(0xffffffffU, worst_index, delta);
        if (streaming_better(worst_key, worst_index, key, index)) {
            worst_key = key;
            worst_index = index;
        }
    }
    if (lane == 0) {
        smem.worst_keys[warp] = worst_key;
        smem.worst_indices[warp] = worst_index;
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int w = 1; w < FAST_THREADS / 32; ++w) {
            const uint32_t key = smem.worst_keys[w];
            const uint32_t index = smem.worst_indices[w];
            if (streaming_better(worst_key, worst_index, key, index)) {
                worst_key = key;
                worst_index = index;
            }
        }
        smem.threshold_key = worst_key;
        smem.threshold_index = worst_index;
    }
    __syncthreads();
}

template<typename ValueT>
struct StreamingInputSource {
    const ValueT* input;
    uint32_t begin;
    uint32_t length;

    __device__ __forceinline__ bool load(uint32_t slot, uint32_t& key, uint32_t& index) const {
        if (slot >= length) return false;
        index = begin + slot;
        key = ordered_key(static_cast<float>(input[index]));
        return true;
    }
};

struct StreamingWorkspaceSource {
    const int32_t* workspace;

    __device__ __forceinline__ bool load(uint32_t slot, uint32_t& key, uint32_t& index) const {
        if (slot >= SEGMENTED_PARTITIONS * STREAM_TOPK) return false;
        const uint32_t partition = slot / STREAM_TOPK;
        const uint32_t candidate = slot % STREAM_TOPK;
        const int32_t* segment = workspace + partition * SEGMENTED_WORKSPACE_STRIDE;
        if (candidate >= static_cast<uint32_t>(segment[2 * STREAM_TOPK])) return false;
        key = reinterpret_cast<const uint32_t*>(segment)[candidate];
        index = static_cast<uint32_t>(segment[STREAM_TOPK + candidate]);
        return true;
    }
};

template<typename ValueT, typename Source>
__device__ __forceinline__ bool streaming_select(
    StreamingSharedMemory& smem, const Source& source, uint32_t length, uint32_t& count) {
    const uint32_t tid = threadIdx.x;
    count = 0;
    if (length == 0) return true;
    bool have_threshold = false;
    constexpr uint32_t tile_size = FAST_THREADS * STREAM_TILE_ITEMS;
    const uint32_t tiles = (length + tile_size - 1) / tile_size;
    const uint32_t tile_bits = tiles > 1 ? 32 - __clz(tiles - 1U) : 0;
    const uint32_t tile_domain = 1U << tile_bits;
    for (uint32_t step = 0; step < tile_domain; ++step) {
        const uint32_t tile = tile_bits ? __brev(step) >> (32 - tile_bits) : 0;
        if (tile >= tiles) continue;
        const uint32_t base = tile * tile_size;
        uint32_t keys[STREAM_TILE_ITEMS];
        uint32_t indices[STREAM_TILE_ITEMS];
        uint32_t keep = 0;
        bool have_nan = false;
        #pragma unroll
        for (int j = 0; j < STREAM_TILE_ITEMS; ++j) {
            keys[j] = 0;
            indices[j] = 0;
            if (source.load(base + tid + j * FAST_THREADS, keys[j], indices[j])) {
                have_nan |= keys[j] == 0xffffffffU;
                if (!have_threshold || streaming_better(
                        keys[j], indices[j], smem.threshold_key, smem.threshold_index))
                    keep |= 1U << j;
            }
        }
        if (__syncthreads_or(have_nan)) return false;
        uint32_t before;
        uint32_t tile_count;
        FastScan(smem.scan).ExclusiveSum(static_cast<uint32_t>(__popc(keep)), before, tile_count);
        __syncthreads();
        uint32_t slot = count + before;
        #pragma unroll
        for (int j = 0; j < STREAM_TILE_ITEMS; ++j) {
            if (keep & (1U << j)) {
                smem.keys[slot] = keys[j];
                smem.indices[slot++] = indices[j];
            }
        }
        count += tile_count;
        __syncthreads();
        if (count >= STREAM_COMPACT_COUNT) {
            streaming_compact<ValueT>(smem, count);
            count = STREAM_TOPK;
            have_threshold = true;
        }
    }
    if (count > STREAM_TOPK) {
        streaming_compact<ValueT>(smem, count);
        count = STREAM_TOPK;
    }
    return true;
}

template<typename ValueT, bool ReturnValue>
__global__ __launch_bounds__(FAST_THREADS) void topk_streaming_kernel(TopkSelectArgs args) {
    __shared__ StreamingSharedMemory smem;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const ValueT* input = static_cast<const ValueT*>(args.input) + row * args.stride_input_batch;
    int32_t* output_index = static_cast<int32_t*>(args.output_index) + row * args.stride_output_index_batch;
    ValueT* output_value = nullptr;
    if constexpr (ReturnValue)
        output_value = static_cast<ValueT*>(args.output_value) + row * args.stride_output_value_batch;
    const int32_t offset = args.output_idx_offset ? args.output_idx_offset[row] : 0;
    const int32_t end = args.end_ptr ? args.end_ptr[row] : static_cast<int32_t>(args.vocab_size);
    if (end < 0 || static_cast<uint32_t>(end) > args.vocab_size) {
        if (tid == 0) asm volatile("trap;");
        return;
    }
    const uint32_t length = static_cast<uint32_t>(end);
    if (length <= STREAM_TOPK) {
        for (uint32_t slot = tid; slot < STREAM_TOPK; slot += FAST_THREADS) {
            output_index[slot] = slot < length
                ? static_cast<int32_t>(static_cast<int64_t>(slot) + offset) : args.idx_oob_fill_value;
            if constexpr (ReturnValue)
                output_value[slot] = slot < length ? input[slot] : static_cast<ValueT>(args.value_oob_fill_value);
        }
        return;
    }

    uint32_t count;
    if (!streaming_select<ValueT>(smem, StreamingInputSource<ValueT>{input, 0, length}, length, count)) {
        if (tid == 0) {
            if (args.abort_when_nan_found) asm volatile("trap;");
            else output_index[0] = 0x3f3f3f3f;
        }
        return;
    }
    for (uint32_t slot = tid; slot < STREAM_TOPK; slot += FAST_THREADS) {
        const uint32_t index = smem.indices[slot];
        output_index[slot] = static_cast<int32_t>(static_cast<int64_t>(index) + offset);
        if constexpr (ReturnValue) output_value[slot] = input[index];
    }
}

template<typename ValueT>
__global__ __launch_bounds__(FAST_THREADS) void topk_segmented_local_kernel(
    TopkSelectArgs args, int32_t* workspace) {
    __shared__ StreamingSharedMemory smem;
    const uint32_t row = blockIdx.x / SEGMENTED_PARTITIONS;
    const uint32_t partition = blockIdx.x % SEGMENTED_PARTITIONS;
    const uint32_t tid = threadIdx.x;
    int32_t* segment = workspace + blockIdx.x * SEGMENTED_WORKSPACE_STRIDE;
    const int32_t end = args.end_ptr ? args.end_ptr[row] : static_cast<int32_t>(args.vocab_size);
    if (end < 0 || static_cast<uint32_t>(end) > args.vocab_size) {
        if (tid == 0) asm volatile("trap;");
        return;
    }
    const uint32_t length = static_cast<uint32_t>(end);
    if (length <= STREAM_TOPK) {
        if (tid == 0) segment[2 * STREAM_TOPK] = 0;
        return;
    }
    const uint32_t begin = partition * length / SEGMENTED_PARTITIONS;
    const uint32_t limit = (partition + 1) * length / SEGMENTED_PARTITIONS;
    const ValueT* input = static_cast<const ValueT*>(args.input) + row * args.stride_input_batch;
    uint32_t count;
    if (!streaming_select<ValueT>(
            smem, StreamingInputSource<ValueT>{input, begin, limit - begin}, limit - begin, count)) {
        if (tid == 0) segment[2 * STREAM_TOPK] = -1;
        return;
    }
    for (uint32_t slot = tid; slot < count; slot += FAST_THREADS) {
        reinterpret_cast<uint32_t*>(segment)[slot] = smem.keys[slot];
        segment[STREAM_TOPK + slot] = static_cast<int32_t>(smem.indices[slot]);
    }
    __syncthreads();
    if (tid == 0) segment[2 * STREAM_TOPK] = static_cast<int32_t>(count);
}

template<typename ValueT, bool ReturnValue>
__global__ __launch_bounds__(FAST_THREADS) void topk_segmented_merge_kernel(
    TopkSelectArgs args, const int32_t* workspace) {
    __shared__ StreamingSharedMemory smem;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const int32_t end = args.end_ptr ? args.end_ptr[row] : static_cast<int32_t>(args.vocab_size);
    if (end < 0 || static_cast<uint32_t>(end) > args.vocab_size) {
        if (tid == 0) asm volatile("trap;");
        return;
    }
    const uint32_t length = static_cast<uint32_t>(end);
    const int32_t* row_workspace = workspace + row * SEGMENTED_PARTITIONS * SEGMENTED_WORKSPACE_STRIDE;
    int32_t* output_index = static_cast<int32_t*>(args.output_index) + row * args.stride_output_index_batch;
    const bool have_nan = tid < SEGMENTED_PARTITIONS &&
        row_workspace[tid * SEGMENTED_WORKSPACE_STRIDE + 2 * STREAM_TOPK] == -1;
    if (__syncthreads_or(have_nan)) {
        if (tid == 0) {
            if (args.abort_when_nan_found) asm volatile("trap;");
            else output_index[0] = 0x3f3f3f3f;
        }
        return;
    }
    const ValueT* input = static_cast<const ValueT*>(args.input) + row * args.stride_input_batch;
    ValueT* output_value = nullptr;
    if constexpr (ReturnValue)
        output_value = static_cast<ValueT*>(args.output_value) + row * args.stride_output_value_batch;
    const int32_t offset = args.output_idx_offset ? args.output_idx_offset[row] : 0;
    if (length <= STREAM_TOPK) {
        for (uint32_t slot = tid; slot < STREAM_TOPK; slot += FAST_THREADS) {
            output_index[slot] = slot < length
                ? static_cast<int32_t>(static_cast<int64_t>(slot) + offset) : args.idx_oob_fill_value;
            if constexpr (ReturnValue)
                output_value[slot] = slot < length ? input[slot] : static_cast<ValueT>(args.value_oob_fill_value);
        }
        return;
    }
    uint32_t count;
    if (!streaming_select<ValueT>(smem, StreamingWorkspaceSource{row_workspace},
                                 SEGMENTED_PARTITIONS * STREAM_TOPK, count)) {
        if (tid == 0) {
            if (args.abort_when_nan_found) asm volatile("trap;");
            else output_index[0] = 0x3f3f3f3f;
        }
        return;
    }
    for (uint32_t slot = tid; slot < STREAM_TOPK; slot += FAST_THREADS) {
        const uint32_t index = smem.indices[slot];
        output_index[slot] = static_cast<int32_t>(static_cast<int64_t>(index) + offset);
        if constexpr (ReturnValue) output_value[slot] = input[index];
    }
}

template<typename ValueT>
void launch_segmented(const TopkSelectArgs& args, int32_t* workspace) {
    topk_segmented_local_kernel<ValueT>
        <<<args.batch_size * SEGMENTED_PARTITIONS, FAST_THREADS, 0, args.stream>>>(args, workspace);
    check_kernel_launch();
    if (args.return_value)
        topk_segmented_merge_kernel<ValueT, true>
            <<<args.batch_size, FAST_THREADS, 0, args.stream>>>(args, workspace);
    else
        topk_segmented_merge_kernel<ValueT, false>
            <<<args.batch_size, FAST_THREADS, 0, args.stream>>>(args, workspace);
    check_kernel_launch();
}

template<int Capacity>
struct FastSharedMemory {
    uint32_t indices[Capacity];
    uint32_t histogram[256];
    FastScan::TempStorage scan;
    uint32_t prefix;
    uint32_t rank;
    uint32_t complete;
    uint32_t have_nan;
};
static_assert(sizeof(FastSharedMemory<512>) <= 4 * 1024);
static_assert(sizeof(FastSharedMemory<2048>) <= 10 * 1024);

template<typename ValueT, int Capacity, bool ReturnValue>
__global__ __launch_bounds__(FAST_THREADS) void topk_unsorted_kernel(TopkSelectArgs args) {
    __shared__ FastSharedMemory<Capacity> smem;
    constexpr bool is_bfloat16 = sizeof(ValueT) == 2;
    constexpr uint32_t full_warp = 0xffffffffU;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31U;
    const uint32_t warp = tid / 32;
    const uint32_t lower_lanes = (1U << lane) - 1U;
    const ValueT* input = static_cast<const ValueT*>(args.input) + row * args.stride_input_batch;
    int32_t* output_index = static_cast<int32_t*>(args.output_index) + row * args.stride_output_index_batch;
    ValueT* output_value = nullptr;
    if constexpr (ReturnValue)
        output_value = static_cast<ValueT*>(args.output_value) + row * args.stride_output_value_batch;
    const int32_t offset = args.output_idx_offset ? args.output_idx_offset[row] : 0;
    const int32_t end = args.end_ptr ? args.end_ptr[row] : static_cast<int32_t>(args.vocab_size);
    if (end < 0 || static_cast<uint32_t>(end) > args.vocab_size) {
        if (tid == 0) asm volatile("trap;");
        return;
    }
    const uint32_t length = static_cast<uint32_t>(end);
    if (length <= args.topk) {
        for (uint32_t slot = tid; slot < args.topk; slot += FAST_THREADS) {
            output_index[slot] = slot < length
                ? static_cast<int32_t>(static_cast<int64_t>(slot) + offset) : args.idx_oob_fill_value;
            if constexpr (ReturnValue)
                output_value[slot] = slot < length ? input[slot] : static_cast<ValueT>(args.value_oob_fill_value);
        }
        return;
    }
    if (tid == 0) {
        smem.prefix = 0;
        smem.rank = args.topk;
        smem.complete = 0;
        smem.have_nan = 0;
    }
    __syncthreads();

    uint32_t mask = 0;
    for (int shift = 24; shift >= (is_bfloat16 ? 16 : 0); shift -= 8) {
        smem.histogram[tid] = 0;
        __syncthreads();
        const uint32_t prefix = smem.prefix;
        const uint32_t rank = smem.rank;
        for (uint32_t base = 0; base < length; base += FAST_THREADS) {
            const uint32_t i = base + tid;
            uint32_t key = 0;
            if (i < length) key = ordered_key(static_cast<float>(input[i]));
            if (shift == 24) {
                const uint32_t nan_lanes = __ballot_sync(full_warp, i < length && key == 0xffffffffU);
                if (lane == 0 && nan_lanes) atomicExch(&smem.have_nan, 1U);
            }
            if (i < length && (key & mask) == prefix)
                atomicAdd(&smem.histogram[(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (smem.have_nan) {
            if (tid == 0) {
                if (args.abort_when_nan_found) asm volatile("trap;");
                else output_index[0] = 0x3f3f3f3f;
            }
            return;
        }
        const uint32_t bucket = 255U - tid;
        const uint32_t count = smem.histogram[bucket];
        uint32_t greater;
        FastScan(smem.scan).ExclusiveSum(count, greater);
        __syncthreads();
        if (greater < rank && rank <= greater + count) {
            smem.prefix = prefix | (bucket << shift);
            smem.rank = rank - greater;
            smem.complete = rank - greater == count;
        }
        __syncthreads();
        mask |= 255U << shift;
        if (smem.complete) break;
    }

    const bool complete = smem.complete != 0;
    uint32_t threshold = smem.prefix;
    if constexpr (is_bfloat16) {
        if (!complete && !(threshold & 0x80000000U)) threshold |= 0xffffU;
    }
    const uint32_t quota = smem.rank;
    const uint32_t chunk = ((length + FAST_THREADS - 1) / FAST_THREADS) * 32;
    const uint32_t lo = min(warp * chunk, length);
    const uint32_t hi = min(lo + chunk, length);
    uint32_t greater_count = 0;
    uint32_t equal_count = 0;
    for (uint32_t i = lo + lane; i < hi; i += 32) {
        const uint32_t key = ordered_key(static_cast<float>(input[i]));
        greater_count += complete ? key >= threshold : key > threshold;
        if (!complete) equal_count += key == threshold;
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
        greater_count += __shfl_down_sync(full_warp, greater_count, delta);
        if (!complete) equal_count += __shfl_down_sync(full_warp, equal_count, delta);
    }
    uint32_t equal_before = 0;
    if (!complete) {
        FastScan(smem.scan).ExclusiveSum(lane == 0 ? equal_count : 0U, equal_before);
        __syncthreads();
        equal_before = __shfl_sync(full_warp, equal_before, 0);
    }
    const uint32_t take_equal = complete ? 0U : min(equal_count, quota - min(quota, equal_before));
    uint32_t slot;
    FastScan(smem.scan).ExclusiveSum(lane == 0 ? greater_count + take_equal : 0U, slot);
    __syncthreads();
    slot = __shfl_sync(full_warp, slot, 0);
    for (uint32_t base = lo; base < hi; base += 32) {
        const uint32_t i = base + lane;
        uint32_t key = 0;
        if (i < hi) key = ordered_key(static_cast<float>(input[i]));
        bool take = i < hi && (complete ? key >= threshold : key > threshold);
        if (!complete) {
            const uint32_t equals = __ballot_sync(full_warp, i < hi && key == threshold);
            if ((equals & (1U << lane)) && equal_before + __popc(equals & lower_lanes) < quota)
                take = true;
            equal_before += __popc(equals);
        }
        const uint32_t winners = __ballot_sync(full_warp, take);
        if (take) smem.indices[slot + __popc(winners & lower_lanes)] = i;
        slot += __popc(winners);
    }
    __syncthreads();
    for (uint32_t slot = tid; slot < args.topk; slot += FAST_THREADS) {
        const uint32_t index = smem.indices[slot];
        output_index[slot] = static_cast<int32_t>(static_cast<int64_t>(index) + offset);
        if constexpr (ReturnValue) output_value[slot] = input[index];
    }
}

template<typename ValueT, int Capacity>
void launch_unsorted(const TopkSelectArgs& args) {
    if (args.return_value)
        topk_unsorted_kernel<ValueT, Capacity, true><<<args.batch_size, FAST_THREADS, 0, args.stream>>>(args);
    else
        topk_unsorted_kernel<ValueT, Capacity, false><<<args.batch_size, FAST_THREADS, 0, args.stream>>>(args);
}

template<typename ValueT>
void launch(const TopkSelectArgs& args, bool indices_int64) {
    if (!indices_int64 && !args.sorted_value && !args.sorted_index && args.topk <= 512) {
        if (args.topk == STREAM_TOPK && args.vocab_size > 2048 && args.vocab_size <= 131072) {
            if (args.return_value)
                topk_streaming_kernel<ValueT, true><<<args.batch_size, FAST_THREADS, 0, args.stream>>>(args);
            else
                topk_streaming_kernel<ValueT, false><<<args.batch_size, FAST_THREADS, 0, args.stream>>>(args);
        } else launch_unsorted<ValueT, 512>(args);
    } else if (indices_int64)
        topk_kernel<ValueT, int64_t><<<args.batch_size, NUM_THREADS, 0, args.stream>>>(args);
    else
        topk_kernel<ValueT, int32_t><<<args.batch_size, NUM_THREADS, 0, args.stream>>>(args);
    check_kernel_launch();
}

} // namespace

bool use_segmented_topk(const TopkSelectArgs& args, bool indices_int64) {
    return !indices_int64 && !args.sorted_value && !args.sorted_index &&
        args.topk == SEGMENTED_TOPK && args.vocab_size >= 65536 && args.vocab_size <= 131072 &&
        args.batch_size >= 1 && args.batch_size <= 16;
}

void run_segmented_topk_select_kernel(const TopkSelectArgs& args, bool is_bfloat16, int32_t* workspace) {
    if (is_bfloat16) launch_segmented<nv_bfloat16>(args, workspace);
    else launch_segmented<float>(args, workspace);
}

void run_topk_select_kernel(const TopkSelectArgs& args, bool is_bfloat16, bool indices_int64) {
    if (args.batch_size == 0) return;
    if (is_bfloat16) launch<nv_bfloat16>(args, indices_int64);
    else launch<float>(args, indices_int64);
}

} // namespace topk_select_sm120
